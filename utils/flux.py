# Copyright (C) 2021  Daniel Magro and Renato Sortino
# Full License at: https://github.com/DanielMagro97/CalculateAstroSNR/blob/main/LICENSE

import numpy as np                  # for numpy arrays and other operations


def compute_peak_flux(masked_image: np.ma.MaskedArray) -> float:
    objects: np.ma.MaskedArray = masked_image[masked_image.mask]
    return objects.data.max()
