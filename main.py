from typing import List             # for type annotation

import argparse                     # for parsing command line arguments
import os                           # for working with files and directories
from pathlib import Path
from tqdm import tqdm               # for progress bars
import numpy as np                  # for numpy arrays and other operations
import astropy.stats                # for 3-sigma clipping and MAD
from utils.flux import compute_peak_flux

from utils.load_fits import load_fits_image, read_samples     # for loading fits images with optional normalisation


def get_args_parser():
    parser = argparse.ArgumentParser('SNR Calculation', add_help=False)

    parser.add_argument('--json_list_path', default="../MLDataset_cleaned/trainset.dat", type=str)

    return parser


def main(args):
    # path of the file which points to the paths of the JSONs of all the images for which the SNR should be calculated
    image_json_list_path = Path(args.json_list_path)

    samples: List = read_samples(image_json_list_path)

    # Look at each JSON file in image_json_path_list
    for sample in tqdm(samples, desc='Going through images and their object masks, and calculating MAD'):
        # check that the image fits file actually exists on disk
        if not os.path.isfile(sample['img']):
            # if not, print a message and skip it
            print('File ' + sample['img'] + ' not found on disk.')
            continue
        # load the image and headers from disk
        fits_image = load_fits_image(sample['img'])
        image = fits_image[0].data
        image_header = fits_image[0].header

        # generate an aggregate of all the individual object masks
        combined_mask: np.ndarray = np.full((image.shape[0], image.shape[1]), fill_value=False, dtype=bool)
        # Load each mask associated with the current image, as specified in the JSON File
        for obj in sample['objs']:
            # Take the containing folder for masks and images (up to sampleX)
            image_dir: List[str] = sample['img'].split(os.sep)[:-2]

            if os.name == 'nt':
                # Hack for Windows path
                image_dir: str = os.sep.join(image_dir)
                mask_path: str = os.path.join(image_dir, 'masks', obj['mask'])
            elif os.name == 'posix':
                # Linux paths
                mask_path = os.path.join(os.path.sep, *image_dir, 'masks', obj['mask'])
            else:
                raise NotImplementedError(f'Operating System {os.name} not supported')

            # check that the mask fits file actually exists on disk
            if not os.path.isfile(mask_path):
                # if not, print a message and skip it
                print('File ' + mask_path + ' not found on disk.')
                continue

            # load the mask and headers from disk
            fits_mask: np.ndarray = load_fits_image(mask_path)
            mask = fits_mask[0].data
            mask_header = fits_mask[0].header

            # convert the masks into boolean (for logical operations)
            boolean_mask: np.ndarray = (mask != 0)
            # Carry out an OR between the current and aggregated mask, to combine them
            np.bitwise_or(boolean_mask, combined_mask, out=combined_mask)

        masked_image = np.ma.array(image, mask=combined_mask)

        # https://docs.astropy.org/en/stable/api/astropy.stats.sigma_clipped_stats.html
        three_sigma_clip = astropy.stats.sigma_clipped_stats(masked_image, sigma=3)
        print('3 Sigma Clip (mean, median, standard deviation): ' + str(three_sigma_clip))

        # https://docs.astropy.org/en/stable/api/astropy.stats.median_absolute_deviation.html
        mad: float = astropy.stats.median_absolute_deviation(masked_image)
        print('MAD: ' + str(mad))


        peak_flux = compute_peak_flux(masked_image)
        print(f'Peak Flux: {peak_flux}')


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)
