###############################################################################
#
# diagnose.py
#
# Created by Jonathan J. P. Peters
#
# The entry point of the flyback fitting, designed to be a clean and calming
# space to set the needed parameters without the clutter of the main script.
#
###############################################################################

from diagnosis.process_images import process_images, process_fb_dt, process_rotation


# The folder containing the data to fit flyback for
# Currently files must be .dm3 or .dm4 with digiscan tags
data_folder = r""

# if false, the script will look for a previous fit output and use that
redo_fits = True

# reuses a reference if one is found, otherwise it creates and saves a new one
# if False, a reference will be made each time AND NOT saved
reuse_ref = True

# make a new reference AND save it, even if one is found. Ignored if reuse_ref is false
force_new_ref = False

# show plot of image, reference and fit (only if redo_fits is True)
# this figure is saved even if this is false, if true it will display and block the script
show_plot = True

# amount of time to crop from start of the image
# this data is unusable at low flyback times and can interfere with the fit
# though the shifts will move this data out of the field of view (at least partially)
crop_time = 60  # micro seconds

# do the fitting for all images (and save outputs)
process_images(data_folder, redo_fits, reuse_ref, force_new_ref, show_plot, crop_time)

# plot data as function of dwell time
process_fb_dt(data_folder)

# plot dat as a function of scan rotation
process_rotation(data_folder)