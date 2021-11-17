###############################################################################
#
# correct.py
#
# Created by Jonathan J. P. Peters
#
# Script used to apply the corretion to an image
#
###############################################################################

from utils import DM3
from unbenders import UnbendManualInterp as unbend_class

# a and b parameters from fit
a = 1.0
b = 1.0

# The full path the the .dm3 or .dm4 image to be corrected
image_path = r""

# Open the image
dmf = DM3(image_path)

# Extract the data we need
img = dmf.image.astype(np.float64)
fb = float(dmf.tags['root.ImageList.1.ImageTags.DigiScan.Flyback'])
dt = float(dmf.tags['root.ImageList.1.ImageTags.DigiScan.Sample Time'])

# Generate a set of coordinates for our image (see diagnosis/fit_hysteresis.py for more detail)
yy = np.arange(img.shape[0]).astype(np.float32)

tt = np.arange(img.shape[1]).astype(np.float32)
tt *= dt
tt += fb

xx = np.arange(img.shape[1]).astype(np.float32)

# Do the actual unbending
unbender = unbend_class(img, xx, yy, tt)
unbent_img = unbender.fit_func(a, b)

# Display results
plt.imshow(unbent_img)
plt.show()