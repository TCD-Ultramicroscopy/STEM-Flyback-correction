###############################################################################
#
# generate_reference.py
#
# Created by Jonathan J. P. Peters
#
# The functions in this file are used to generate a reference image
# mostly it is a lot of cropping and self tiling, followed by aligning
# Could definitely be improved in performance, but output can be reused
#
# gen_ref is the only function that needs to be called externally
#
###############################################################################

import numpy as np
from scipy.ndimage import shift as sp_shift
from scipy.signal import correlate as sp_correlate

from scipy.ndimage import median_filter as sp_med
from scipy.ndimage import gaussian_filter as sp_gauss

from .phase_correlate import phase_cross_correlation


###############################################################################
# _get_tile_vector
#
# Get tiling vector from different halves of an image
# aligns top half with bottom or left with right
###############################################################################
def _get_tile_vector(img, ax: int):
    # shift axes is the opposite of the selected axis
    other_ax = np.abs(ax - 1)
    # image shape as an array
    shp = np.array(img.shape)

    #
    # first split the image into halves
    #

    # get the size, use int conversion to floor value
    ax_sz = int(shp[ax] / 2)

    # first reference is first half
    half_1_inds = np.arange(0, ax_sz)
    half_1 = np.take(img, half_1_inds, ax)

    # second reference is second half
    ax_diff = int(ax_sz) # - 20
    half_2_inds = np.arange(ax_diff, ax_diff + ax_sz)
    half_2 = np.take(img, half_2_inds, ax)

    #
    # optional filtering
    # This sometimes makes things worse!
    #

    half_1 = sp_med(half_1, 5)
    half_2 = sp_med(half_2, 5)

    #
    # Now do the correlation
    #

    # get the sub-pixel shift from correlation between these halves
    # TODO: find/make version that clips this to some sensible region (i.e. not edge of the image)
    shift_full = phase_cross_correlation(half_1, half_2, upsample_factor=100, max_shift=100)

    # correct the shifts back to full reference coords
    im_shifts = np.zeros(img.ndim)
    im_shifts[ax] = ax_diff - shift_full[ax]
    im_shifts[other_ax] = -shift_full[other_ax]

    return im_shifts


###############################################################################
# _tile_ref
#
# Tile image using vectors from get_tile_vector()
###############################################################################
def _tile_ref(img, ref):
    # get the output image size (relative to input image)
    # scale a bit so we have room to align images (only need to be one unit cell wiggle room)
    over_scale = 1.2

    # image sizes as array for some maths!
    img_shp = np.array(img.shape)
    ref_shp = np.array(ref.shape)

    # get output shape with over scaling
    out_shp = np.ceil(img_shp * over_scale).astype(np.int)
    # get tiling needed to cover this
    # 2 is because the overlap is approximately half
    # +1 to account for edge if overlap is slightly more than half
    tile_n = 2 * np.ceil(out_shp / ref_shp).astype(np.int) + 1

    # get the vectors we should tile this image by
    tile_vec_y = _get_tile_vector(ref, 0)
    tile_vec_x = _get_tile_vector(ref, 1)

    #
    out_ref = np.zeros(out_shp)
    out_msk = np.zeros_like(out_ref)

    shift_img = np.zeros_like(out_ref)
    shift_msk = np.zeros_like(out_msk)

    # copy in initial image (this is not fully n dimensional)
    shp = np.array(ref.shape)
    shift_img[0:shp[0], 0:shp[1]] = ref
    shift_msk[0:shp[0], 0:shp[1]] = 1

    # start from -2 to not get different tiling at edge (probably doesnt matter)
    for j in range(-2, tile_n[0]):
        for i in range(-2, tile_n[1]):

            shift_ref_d = sp_shift(shift_img, j * tile_vec_y + i * tile_vec_x, order=1)
            shift_msk_d = sp_shift(shift_msk, j * tile_vec_y + i * tile_vec_x, order=1)

            # # mask off points where mask is less than one, can have errors?
            # mm = np.where(shift_msk_d >= 0.99999)
            #
            # out_ref[mm] += shift_ref_d[mm]
            # out_msk[mm] += shift_msk_d[mm]

            out_ref += shift_ref_d
            out_msk += shift_msk_d

    norm_img = np.divide(out_ref, out_msk, out=np.zeros_like(out_ref), where=out_msk > 1e-10)

    return norm_img


###############################################################################
# _crop_ref
#
# Crops tiled reference image from tile_reference() to be same size as the input image
###############################################################################
def _crop_ref(img, ref):

    off = 100

    # we want to align to right hand half
    img_half = img[:, int(img.shape[1] / 2)-off:-off]

    # correlate
    corr = sp_correlate(ref, img_half, method='fft', mode='same')

    # crop down correlation to make sure our images fully overlap
    lc = int(0.75 * img.shape[1] - off)
    rc = int(img.shape[1] - lc)

    bc = int(img.shape[0] / 2)
    tc = int(img.shape[0] / 2)

    corr_cropped = corr[bc:-tc, lc:-rc]

    # get teh correlation maximum
    shft = np.array(np.unravel_index(corr_cropped.argmax(), corr_cropped.shape))

    # crop out the reference
    l = shft[1]
    b = shft[0]
    t = b + img.shape[0]
    r = l + img.shape[1]

    out_ref = ref[b:t, l:r]

    return out_ref


###############################################################################
# gen_ref
#
# Convenience function to call all the functions above
# It is important to normalise data before it comes here, otherwise you will get overflows
# I used the top an bottom halves of the ref (which can be the right half of img) to avoid artefacts in the
# difference where the right half would be a lot better than the left (with an obvious discontinuity)
###############################################################################
def gen_ref(img, ref):
    # I'm assuming the reference is the right half
    sz = int(ref.shape[0] / 2)
    ref_1 = ref[:sz, :]
    ref_2 = ref[sz:, :]

    # make over sized reference
    full_ref_1 = _tile_ref(img, ref_1)
    full_ref_2 = _tile_ref(img, ref_2)

    shift_full = phase_cross_correlation(full_ref_1, full_ref_2, upsample_factor=100, max_shift=100)
    shift_ref_d = sp_shift(full_ref_2, shift_full, order=1)

    full_ref = shift_ref_d + full_ref_1

    shift_int = (np.ceil(np.abs(shift_full)) * np.sign(shift_full)).astype(np.int)
    if shift_int[0] < 0 and shift_int[1] < 0:
        full_ref = full_ref[:shift_int[0], :shift_int[1]]
    elif shift_int[0] < 0 and shift_int[1] > 0:
        full_ref = full_ref[:shift_int[0], shift_int[1]:]
    elif shift_int[0] > 0 and shift_int[1] < 0:
        full_ref = full_ref[shift_int[0]:, :shift_int[1]]
    else:
        full_ref = full_ref[shift_int[0]:, shift_int[1]:]

    # crop reference down
    rv = _crop_ref(img, full_ref)
    return rv
