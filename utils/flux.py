# Copyright (C) 2021  Daniel Magro and Renato Sortino
# Full License at: https://github.com/DanielMagro97/CalculateAstroSNR/blob/main/LICENSE

import numpy as np                  # for numpy arrays and other operations


def compute_peak_flux(masked_image: np.ma.MaskedArray) -> float:
    objects: np.ma.MaskedArray = masked_image[masked_image.mask]
    return objects.data.max()


def compute_integrated_flux(masked_image: np.ma.MaskedArray, header: np.array) -> float:
    objects: np.ma.MaskedArray = masked_image[masked_image.mask]
    flux_sum = objects.sum()
    cdelt1 = header['CDELT1']
    cdelt2 = header['CDELT2']
    bmaj = header['BMAJ']
    bmin = header['BMIN']
    beam_area = (cdelt1 * cdelt2) / (bmaj * bmin)
    integrated_flux = flux_sum / beam_area
    return integrated_flux