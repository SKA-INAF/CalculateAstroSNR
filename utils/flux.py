def compute_peak_flux(masked_image):
    objects = masked_image[masked_image.mask]
    return objects.data.max()
