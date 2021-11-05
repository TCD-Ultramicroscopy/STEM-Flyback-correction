###############################################################################
#
# process_images.py
#
# Created by Jonathan J. P. Peters
#
# This is the hub of the processing, though mostly this is data wrangling.
# This is where the inputs become outputs!
#
###############################################################################

import os
import numpy as np
from tifffile import imsave
from lmfit import Model, Parameters
import matplotlib.pyplot as plt

# These are from this project
from generate_reference import gen_ref
from fit_hysteresis import fit_hysteresis
from read_dm3 import DM3


###############################################################################
# process_images
#
# Loop over images in the data folder and fit the hysteresis.
# outputs are savd to "fit.txt" in the data folder
###############################################################################
def process_images(data_folder: str, redo_fits: bool, reuse_ref: bool, force_new_ref: bool, show_plot: bool, crop_time: float):
    # These are used to store data for each image, to be processed afterwards
    outputs = []

    out_file_path = os.path.join(data_folder, 'fits.txt')
    for subdir, dirs, files in os.walk(data_folder):

        # not the best way of doing this, but I want to avoid further indentation
        if not redo_fits and os.path.exists(out_file_path):
            continue

        for file in files:

            #
            # Open the actual files
            #

            # get the file name and extension separetly (file name used late for output)
            file_body, ext = os.path.splitext(file)

            # Only care about DM files
            if ext not in ['.dm3', '.dm4']:
                continue

            full_path = os.path.join(subdir, file)
            dmf = DM3(full_path)

            # Extract the data we need
            img = dmf.image.astype(np.float64)
            fb = float(dmf.tags['root.ImageList.1.ImageTags.DigiScan.Flyback'])
            dt = float(dmf.tags['root.ImageList.1.ImageTags.DigiScan.Sample Time'])
            rot = float(dmf.tags['root.ImageList.1.ImageTags.DigiScan.Rotation'])

            # this just helps debugging by not fitting everything
            # if np.random.rand() > 0.1:
            #     continue

            # important to do some sort of normalisation, otherwise correlations can overflow
            img = img - np.mean(img)
            img = img / np.std(img)

            #
            # Define the reference image
            #

            # here I am just using the right half of the image,
            # but could have a static image and set it as the ref variable
            sz = img.shape[1]
            ref = img[:, -int(0.5 * sz):]

            #
            # Making the reference is a bit slow, so we save the generated reference for reuse
            # use .npy files for speed and to retain full float64 precision
            #
            if reuse_ref:
                ref_path = os.path.join(subdir, file_body + "_ref")

                if os.path.exists(ref_path + ".npy") and not force_new_ref:
                    ref = np.load(ref_path + ".npy")
                else:
                    print(f"Making reference for: {full_path}")

                    ref = gen_ref(img, ref)

                    np.save(ref_path, ref)
            else:
                ref = gen_ref(img, ref)

            #
            # Do the fitting
            #

            f_img, f_ref, f_fit, qual, (xx, tt), ab, ab_err, fit_full = fit_hysteresis(img, ref, dt, fb,
                                                                                       crop_time=crop_time)

            # save the fit data to our list
            outputs.append({'fb': fb,
                            'dt': dt,
                            'a': ab[0],
                            'b': ab[1],
                            'a_err': ab_err[0],
                            'b_err': ab_err[1],
                            'quality': qual,
                            'file': full_path,
                            'rotation': rot})

            # save fit images as .tif files
            norm_path = os.path.join(subdir, file_body + "_norm.tif")
            ref_path = os.path.join(subdir, file_body + "_ref.tif")
            corrected_path = os.path.join(subdir, file_body + "_corrected.tif")

            imsave(norm_path, fit_full[0].astype(np.float32))
            imsave(ref_path, fit_full[1].astype(np.float32))
            imsave(corrected_path, fit_full[2].astype(np.float32))

            #
            # Plot the fit outputs
            #

            plot_fit_images(img, f_img, f_ref, xx, f_fit, ab, dt, fb, file_body, subdir, show_plot)

    #
    # save our outputs (if we generated something)
    #
    if outputs:
        with open(out_file_path, 'w') as out_f:

            print(f'Writing fit data to: {out_file_path}')

            out_f.write(f'flyback (us),dwell time (us), a, b,fit quality,rotation (deg),file name\n')

            for fit in outputs:

                o1 = fit['fb']
                o2 = fit['dt']
                o3 = fit['a']
                o4 = fit['b']
                o5 = fit['quality']
                o6 = fit['rotation']
                o7 = fit['file']
                o8 = fit['a_err']
                o9 = fit['b_err']

                out_f.write(f'{o1},{o2},{o3},{o4},{o5},{o6},{o7},{o8},{o9}\n')


###############################################################################
# plot_fit_images
#
# Plots the hysteresis fit for each image. by default just saves the plot
# without displaying it
###############################################################################
def plot_fit_images(img, fit_img, fit_ref, x, f_fit, ab, dt, fb, file_body, output_dir, show_plot=False):
    diff_start = np.mean(np.abs(fit_ref - fit_img), axis=0)
    diff_end = np.mean(np.abs(fit_ref - f_fit), axis=0)

    im_min = img.min()
    im_max = img.max()

    diff_min = -2
    diff_max = 2

    # calculate the y axis scale needed to keep pixel ration as 1
    x_px_scale = x[1] - x[0]
    y_px_range = fit_img.shape[0] * x_px_scale

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(13, 10))

    fig.suptitle(f'Dwell time: {dt} μs, flyback: {fb} μs', fontsize=14)
    plt.figtext(.5, .94, f'A: {ab[0]}, b: {ab[1]}', fontsize=14, ha='center')
    plt.figtext(.5, .92, file_body, fontsize=8, ha='center')

    im_kwargs = {'vmin': im_min, 'vmax': im_max, 'cmap': 'magma', 'extent': [x[0], x[-1], y_px_range, 0]}
    diff_kwargs = {'vmin': diff_min, 'vmax': diff_max, 'cmap': 'Spectral', 'extent': [x[0], x[-1], y_px_range, 0]}

    ax1.imshow(fit_img, **im_kwargs)

    ax2.imshow(fit_ref, **im_kwargs)
    ax3.imshow(fit_ref - fit_img, **diff_kwargs)
    ax4.imshow(f_fit, **im_kwargs)
    ax5.imshow(fit_ref - f_fit, **diff_kwargs)
    ax6.plot(x, diff_start, label='Original')
    ax6.plot(x, diff_end, label='Corrected')

    ax6.legend()
    ax6.set_ylabel("Mean absolute difference (arb. units)")
    ax6.set_xlabel("x (px)")

    asp = np.diff(ax6.get_xlim())[0] / np.diff(ax6.get_ylim())[0]
    asp /= np.abs(np.diff(ax3.get_xlim())[0] / np.diff(ax3.get_ylim())[0])
    ax6.set_aspect(asp)

    ax1.set_title("Original image")
    ax2.set_title("Generated reference")
    ax3.set_title("Original difference")
    ax4.set_title("Corrected image")
    ax5.set_title("Corrected difference")
    ax6.set_title("Average difference")

    # hide y axis, it is bollocks anyway
    ax1.axes.yaxis.set_visible(False)
    ax2.axes.yaxis.set_visible(False)
    ax3.axes.yaxis.set_visible(False)
    ax4.axes.yaxis.set_visible(False)
    ax5.axes.yaxis.set_visible(False)

    ax1.axes.xaxis.set_visible(False)
    ax2.axes.xaxis.set_visible(False)
    ax3.axes.xaxis.set_visible(False)
    ax4.axes.xaxis.set_visible(False)
    ax5.axes.xaxis.set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_fig_path = ref_path = os.path.join(output_dir, file_body + "_plot.pdf")
    plt.savefig(save_fig_path)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


###############################################################################
# read_output
#
# Reads in the output generated by process_images
###############################################################################
def read_output(data_file_path):
    if not os.path.exists(data_file_path):
        exit(0)

    data = []

    # read data in again as we may be using pre-analysed data
    with open(data_file_path, 'r') as in_f:

        lines = in_f.readlines()

        for l in lines[1:]:
            l_spl = l.split(',')

            data.append({'fb': float(l_spl[0]),
                         'dt': float(l_spl[1]),
                         'a': float(l_spl[2]),
                         'b': float(l_spl[3]),
                         'quality': float(l_spl[4]),
                         'rotation': float(l_spl[5]),
                         'file': l_spl[6],
                         'a_err': float(l_spl[7]),
                         'b_err': float(l_spl[8])
                         })

    return data


###############################################################################
# process_fb_dt
#
# Takes the output and plots a and b parameters as a function of dwell time
# (and for different flybacks)
###############################################################################
def process_fb_dt(data_folder):
    #
    # Here I read in the data from the file, so the fit doesn't have to be repeated each time
    #

    out_file_path = os.path.join(data_folder, 'fits.txt')
    data = read_output(out_file_path)

    #
    # Arrange data into more usable structures
    #

    # allow limiting flyback and dwell time ranges
    # long flyback images often produce bad fits as there is not distortion
    fb_lim = (0.0, 100.0)
    dt_lim = (0.0, np.inf)

    dt_vals = []
    fb_vals = []
    a_vals = []
    b_vals = []
    a_err_vals = []
    b_err_vals = []

    for d in data:
        fb = d['fb']
        dt = d['dt']

        if fb_lim[0] < fb < fb_lim[1] and dt_lim[0] < dt < dt_lim[1]:
            dt_vals.append(dt)
            fb_vals.append(fb)
            a_vals.append(d['a'])
            b_vals.append(d['b'])
            a_err_vals.append(d['a_err'])
            b_err_vals.append(d['b_err'])

    # make 'em numpy arrays
    dt_vals = np.array(dt_vals)
    fb_vals = np.array(fb_vals)
    a_vals = np.array(a_vals)
    b_vals = np.array(b_vals)
    a_err_vals = np.array(a_err_vals)
    b_err_vals = np.array(b_err_vals)

    #
    # First I want to fit my data, I do this ignoring flyback (it doesn't matter anyway)
    #

    def fit_a_func(x, a, b, c):
        return a / (x + c) + b

    f_a_model = Model(fit_a_func)
    params_a = Parameters()
    params_a.add('a', value=10)
    params_a.add('b', value=10)
    params_a.add('c', value=0, vary=True)

    mth = 'leastsq'
    result_a = f_a_model.fit(a_vals, params_a, x=dt_vals, method=mth, nan_policy='omit')

    def fit_b_func(x, m, c):
        return m * x + c

    f_b_model = Model(fit_b_func)
    params_b = Parameters()
    params_b.add('m', value=50, min=0)
    params_b.add('c', value=50, min=0)

    mth = 'leastsq'
    result_b = f_b_model.fit(b_vals, params_b, x=dt_vals, method=mth, nan_policy='omit')

    #
    # Plot data!
    #

    # make plot now
    fig = plt.figure(figsize=(10, 4.5))

    ax_a = fig.add_subplot(121)
    ax_b = fig.add_subplot(122)

    # get data sorted by flyback time (so I can separate it on the plot)
    unique_fb = np.sort(np.unique(fb_vals))

    # use these to make everything unique on plot (to a reasonable extent)
    mkr = ['o', 's', '^', 'v', '*', 'o', 's', '^', 'v']
    col = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
    mfc = ['C0', 'C1', 'C2', 'C3', 'C4', 'w', 'w', 'w', 'w']

    cntr = 0
    for u_fb in unique_fb:
        fb_inds = np.where(fb_vals == u_fb)

        dt_plot = dt_vals[fb_inds]
        a_plot = a_vals[fb_inds]
        b_plot = b_vals[fb_inds]
        a_err_plot = a_err_vals[fb_inds]
        b_err_plot = b_err_vals[fb_inds]

        plot_args = {'color': col[cntr], 'marker': mkr[cntr],
                     'markerfacecolor': mfc[cntr], 'linewidth': 0,
                     'label': "Flyback = " + str(u_fb)}

        ax_a.errorbar(dt_plot, a_plot, yerr=a_err_plot, **plot_args)
        ax_b.errorbar(dt_plot, b_plot, yerr=b_err_plot, **plot_args)

        cntr += 1
        if cntr > 8:
            cntr = 0

    # plot our fits on top now"
    max_dt = np.max(dt_vals)
    min_dt = np.min(dt_vals)
    sample_dt = np.linspace(min_dt, max_dt, 1000)

    ax_a.plot(sample_dt, f_a_model.eval(result_a.params, x=sample_dt), '-')
    ax_b.plot(sample_dt, f_b_model.eval(result_b.params, x=sample_dt), '-')

    #
    # Finish off the plot
    #
    ax_a.set_title("A parameter")
    ax_b.set_title("B parameter")

    ax_a.set_xlabel('Dwell time (μs)')
    ax_a.set_ylabel('A (pixels)')

    ax_b.set_xlabel('Dwell time (μs)')
    ax_b.set_ylabel('b (μs)')

    ax_a.legend()
    ax_b.legend()

    plt.tight_layout()

    save_fig2_path = os.path.join(data_folder, 'ab_plots.pdf')
    plt.savefig(save_fig2_path)

    plt.show()


###############################################################################
# process_fb_dt
#
# Takes the output and plots a and b parameters as a function scan rotation
###############################################################################
def process_rotation(data_folder):
    #
    # Here I read in the data from the file, so the fit doesn't have to be repeated each time
    #

    out_file_path = os.path.join(data_folder, 'fits.txt')
    data = read_output(out_file_path)

    #
    # This block is used to plot the rotation dependence of the parameters
    #

    data_rot_dic = {}
    data_a_dic = {}
    data_b_dic = {}

    for d in data:
        dt_str = str(d['dt'])

        if dt_str not in data_rot_dic.keys():
            data_rot_dic[dt_str] = []
            data_a_dic[dt_str] = []
            data_b_dic[dt_str] = []

        data_rot_dic[dt_str].append(d['rotation'])
        data_a_dic[dt_str].append(d['a'])
        data_b_dic[dt_str].append(d['b'])

    def sin_func(x, a, c, x0):
        return a * np.cos(2 * np.deg2rad((x + x0))) + c

    fig = plt.figure(figsize=(10, 4))
    ax_r1 = fig.add_subplot(121)
    ax_r2 = fig.add_subplot(122)

    mkr = ['o', 's', '^', 'o', 's', '^', 'v']
    col = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
    mfc = ['w', 'w', 'w', 'w', 'w', 'w', 'w']

    cntr = 0
    for dt_k in data_rot_dic.keys():
        data_rot = np.array(data_rot_dic[dt_k])
        data_a = np.array(data_a_dic[dt_k])
        data_b = np.array(data_b_dic[dt_k])

        f_model = Model(sin_func)

        params = Parameters()
        params.add('a', value=1)
        params.add('c', value=np.mean(data_a))
        params.add('x0', value=-35, vary=False, max=0)

        mth = 'leastsq'
        result = f_model.fit(data_a, params, method=mth, x=data_rot)

        plot_args = {'color': col[cntr], 'marker': mkr[cntr],
                     'markerfacecolor': mfc[cntr], 'linewidth': 0,
                     'label': dt_k + ' (μs)'}

        ax_r1.plot(data_rot, data_a, **plot_args)
        ax_r2.plot(data_rot, data_b, **plot_args)

        # plot fit
        max_dr = np.max(data_rot)
        min_dr = np.min(data_rot)
        sample_dr = np.linspace(min_dr, max_dr, 1000)

        ax_r1.plot(sample_dr, f_model.eval(result.params, x=sample_dr), '-')

        # just for picking plot colours etc
        cntr += 1

    ax_r1.set_xlabel('Scan Rotation (°)')
    ax_r1.set_ylabel('A (pixels)')

    ax_r2.set_xlabel('Scan Rotation (°)')
    ax_r2.set_ylabel('b (μs)')

    # # for when normalised
    # ax_r1.set_ylim([0, 2.5])
    # ax_r2.set_ylim([0, 2.7])

    ax_r1.legend()
    fig_path = os.path.join(data_folder, "FIG.pdf")
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.show()
