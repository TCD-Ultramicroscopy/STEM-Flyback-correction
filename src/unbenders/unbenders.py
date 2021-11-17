###############################################################################
#
# unbenders.py
#
# Created by Jonathan J. P. Peters
#
# These classes are to help create functions to unbend STEM images. The uses
# of the class is simple to facilitate testing multiple unbend functions.
# Particularly for testing for speed
#
# Simple create a class that inherist BaseUnbend with a method called:
# fit_func that takes an a and b value (both floats)
#
###############################################################################

import numpy as np


class BaseUnbend:
    def __init__(self, im: np.ndarray, x: np.ndarray, y: np.ndarray, t: np.ndarray):
        # still need zeros on 1d arrays as output is still a tuple
        assert im.shape[1] == x.shape[0] == t.shape[0]
        assert im.shape[0] == y.shape[0]

        self.image = im

        self.x_coords = x
        self.y_coords = y
        self.t_coords = t

    def fit_func(self, a: float, b: float):
        print("Calling _fit_func on BaseUnbend! This function is designed to be overridden.")
        return self.image

    def fit_model(self, x: np.ndarray, a: float, b: float):
        unbend = self.fit_func(a, b)
        return unbend.ravel()


class UnbendManualSlowInterp(BaseUnbend):
    def fit_func(self, a: float, b: float):
        # make x values of current positions
        if b == 0.0:  # not a useful case, but allows the fit to use this without errors
            new_xx = self.x_coords
        else:
            new_xx = self.x_coords - a * np.exp(- self.t_coords / b)

        # find where the gridded values fit into this
        unbent_image = np.full_like(self.image, np.nan)
        st_i = 1
        for i in range(self.x_coords.size):

            x_loc = st_i + np.searchsorted(new_xx[st_i:], self.x_coords[i], side='right')
            if x_loc >= new_xx.size:
                break

            st_i = x_loc - 1

            bin_w = new_xx[x_loc] - new_xx[x_loc - 1]
            frac = (self.x_coords[i] - new_xx[x_loc - 1]) / bin_w

            unbent_image[:, i] = frac * self.image[:, x_loc] + (1 - frac) * self.image[:, x_loc - 1]

        return unbent_image


class UnbendManualInterp(BaseUnbend):
    def fit_func(self, a: float, b: float):
        # make x values of current positions
        if b == 0.0:  # not a useful case, but allows the fit to use this without errors
            new_xx = self.x_coords
        else:
            new_xx = self.x_coords - a * np.exp(- self.t_coords / b)

        x_loc = np.searchsorted(new_xx, self.x_coords, side='right')
        x_loc[x_loc == x_loc.size] = x_loc.size - 1
        bin_w = new_xx[x_loc] - new_xx[x_loc - 1]
        frac = (self.x_coords - new_xx[x_loc - 1]) / bin_w

        unbent_image = frac * self.image[:, x_loc] + (1 - frac) * self.image[:, x_loc - 1]

        return unbent_image
