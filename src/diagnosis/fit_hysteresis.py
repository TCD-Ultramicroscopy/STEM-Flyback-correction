###############################################################################
#
# fit_hysteresis.py
#
# Created by Jonathan J. P. Peters
#
# This is where the magic happens ;)
#
###############################################################################

import numpy as np
from lmfit import Model, Parameters
from math import isnan

from scipy.ndimage import median_filter as sp_med
from scipy.ndimage import gaussian_filter as sp_gauss

# this is where the correction method is chosen
from unbenders import UnbendManualInterp as unbend_class


###############################################################################
# quality_func
#
# this function gives a measure of the quality of he fit
# This was just used as an output of the fit quality.
# It is NOT used in the actual fit
###############################################################################
def quality_func(img, ref):
    sz_y = img.shape[0]
    im_f = img[~np.isnan(img)].reshape(sz_y, -1)
    ref_f = ref[~np.isnan(img)].reshape(sz_y, -1)

    residual = ref_f - im_f

    return _reduce_fcn(residual) / residual.size


###############################################################################
# _reduce_fcn
#
# This IS used in the actual fit
###############################################################################
def _reduce_fcn(r):
    return (r*r).sum()


###############################################################################
# fit_hysteresis
#
# Everything happens here, mostly settings up coordinates and data wrangling,
# then using lmfit to perform the fit
###############################################################################
def fit_hysteresis(img_in, ref_in, dt, fb, crop_time=0):
    #
    # process image and reference
    #

    # copy and make float64
    img = img_in.astype(np.float64)
    ref = ref_in.astype(np.float64)

    # optional filter to reduce noise effects
    img = sp_med(img, 5)
    ref = sp_med(ref, 5)

    # normalise histograms (again?)
    img = img - np.mean(img)
    img = img / np.std(img)

    ref = ref - np.mean(ref)
    ref = ref / np.std(ref)

    #
    # Crop out any unusable part of the image
    # calculated as ~60 microseconds for our STEM
    #

    if fb > crop_time:
        crop_px = 0
    else:
        crop_px = int(np.ceil(crop_time - fb) / dt)

    img = img[:, crop_px:]
    ref = ref[:, crop_px:]

    #
    # Estimate some initial parameters with limits
    #

    initial_a = 25
    min_a = 0.0
    max_a = img.shape[1]
    fit_a = True

    initial_b = img.shape[1] / (2 * 5)
    min_b = 0.0
    max_b = 1500
    fit_b = True

    #
    # Doing the actual fit
    #

    # create coordinates of image for fit (we don't change y, so just make it and forget it)
    yy = np.arange(img.shape[0]).astype(np.float32)

    #
    # First, I define an xx and tt so you could have one as time, one as position
    # see the 'exp_func' below to see how they are used
    #

    # get the pixel numbers of the cropped image
    tt = np.arange(img.shape[1]).astype(np.float32)
    # add croped pixel on to get pixel value of uncropped image
    tt += crop_px
    # convert to time
    tt *= dt
    # correct for flyback
    tt += fb

    # define x in image coords
    # doesnt matter about offsets and so an as it the the difference that matters
    xx = np.arange(img.shape[1]).astype(np.float32)

    #
    # This is doing the actual actual fit
    #

    # create an instance of our fit class
    fit_class = unbend_class(img, xx, yy, tt)

    # this is our unused 'x' for fit_model that we need to pass to the fitting functions
    fit_x = np.zeros_like(ref).ravel()
    fit_y = ref.ravel()

    f_model = Model(fit_class.fit_model)
    # params = f_model.make_params(a=initial_a, b=initial_b)
    params = Parameters()
    params.add('a', value=initial_a, vary=fit_a, min=min_a, max=max_a)
    params.add('b', value=initial_b, vary=fit_b, min=min_b, max=max_b)

    mth = 'Nelder-Mead'
    result = f_model.fit(fit_y.astype(np.float32), params, x=fit_x.astype(np.float32),
                         nan_policy='omit', method=mth,
                         fit_kws={'reduce_fcn': _reduce_fcn})

    #
    # get fit results
    #

    # print(result.fit_report())

    a = result.params['a'].value
    b = result.params['b'].value
    a_e = result.params['a'].stderr
    b_e = result.params['b'].stderr

    # depending on the fit method, the errors will not exist, so handle that
    if a_e is None or isnan(a_e):
        a_e = 0.0

    if b_e is None or isnan(b_e):
        b_e = 0.0

    # generate fitted image for output
    fitted_img = fit_class.fit_func(a, b)

    # determine quality
    fit_qual = quality_func(fitted_img, ref)

    #
    # Process outputs and return
    #

    print("=-----------------------------------------------=")
    print(f"Flyback: {fb}, Dwell: {dt}")
    print(f"A: {a} ± {a_e}")
    print(f"b: {b} ± {b_e}")
    print("=-----------------------------------------------=")
    print("")

    img_in_norm = img_in.astype(np.float64)
    img_in_norm = img_in_norm - np.mean(sp_med(img_in_norm, 5))
    img_in_norm = img_in_norm / np.std(sp_med(img_in_norm, 5))

    ref_in_norm = ref_in.astype(np.float64)
    ref_in_norm = ref_in_norm - np.mean(sp_med(ref_in_norm, 5))
    ref_in_norm = ref_in_norm / np.std(sp_med(ref_in_norm, 5))

    xx_full = np.arange(img_in_norm.shape[1]).astype(np.float32) - crop_px
    tt_full = np.arange(img_in_norm.shape[1]).astype(np.float32)
    tt_full *= dt
    tt_full += fb

    fit_full_class = unbend_class(img_in_norm, xx_full, yy, tt_full)
    fitted_img_full = fit_full_class.fit_func(a, b)

    fitted_img_full[np.isnan(fitted_img_full)] = 0.0
    fitted_img[np.isnan(fitted_img)] = 0.0

    return img, ref, fitted_img, fit_qual, (xx, tt), (a, b), (a_e, b_e), (img_in_norm, ref_in_norm, fitted_img_full)
