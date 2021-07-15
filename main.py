from typing import List         # for type annotation

import os                                       # for working with files and directories
import json                                     # for reading JSON files from disk
import numpy as np                              # for numpy arrays and other operations
import astropy.stats                            # for 3-sigma clipping
from scipy import stats
import argparse

from utils.load_fits import load_fits_image, read_samples     # for loading fits images with optional normalisation
def get_args_parser():
    parser = argparse.ArgumentParser('SNR Calculation', add_help=False)

    parser.add_argument('--json_list_path', default="/home/rensortino/MLDataset_cleaned/trainset.dat", type=str)

    return parser


def main(args):
    # path of the file which points to the paths of the JSONs of all the images for which the SNR should be calculated
    image_json_list_path: str = args.json_list_path

    samples = read_samples(image_json_list_path)

    # TODO Maybe remove up to...
    # Look at each JSON file in image_json_path_list
    for sample in samples:
        if not os.path.isfile(sample['img']):
            # if not, print a message and skip it
            print('File ' + sample['img'] + ' not found on disk.')
            continue
        image = load_fits_image(sample['img'], 'none')
        # Take the containing folder for masks and images (up to sampleX)
        image_dir = sample['img'].split(os.sep)[:-2]

        # generate an aggregate of all the individual object masks
        combined_mask: np.ndarray = np.zeros((image.shape[0], image.shape[1]))
        # Load each mask associated with the current image, as specified in the JSON File
        for obj in sample['objs']:
            mask_path = os.path.join(os.path.sep, *image_dir, 'masks', obj['mask'])

            # check that the mask fits file actually exists on disk
            if not os.path.isfile(mask_path):
                # if not, print a message and skip it
                print('File ' + mask_path + ' not found on disk.')
                continue

            # load the mask from disk
            # TODO!!!! check whether to load images with normalisation or not for SNR
            mask: np.ndarray = load_fits_image(mask_path, 'none')

            # convert the masks into boolean (for logical operations)
            # boolean_mask: np.ndarray = (mask == 1)
            # Some masks have values different from 1
            boolean_mask: np.ndarray = (mask != 0)
            boolean_combined_mask: np.ndarray = (combined_mask != 0)
            # Carry out an OR between the current and aggregated mask, to combine them
            # by storing the output in combined_mask, the output is broadcasted into floats
            np.bitwise_or(boolean_mask, boolean_combined_mask, out=combined_mask)

        # calculate the image with masked areas zeroed out
        # TODO might not be needed
        image_masked = image - (image * combined_mask)

        boolean_combined_mask: np.ndarray = (combined_mask != 0)
        masked_image = np.ma.array(image, mask=boolean_combined_mask)
        # print(masked_image)

        # mean calculation for unmasked pixels
        # sum: float = 0.0
        # count: int = 0
        # boolean_combined_mask: np.ndarray = (combined_mask == 1)
        # for i, axis_one in enumerate(image):
        #     for j, axis_two in enumerate(axis_one):
        #         # if the current element is marked 'True', skip it since it contains an object
        #         if boolean_combined_mask[i][j]:
        #             continue
        #         sum += image[i][j]
        #         count += 1
        # print('Average pixel ' + str((sum/count)))

        three_sigma_clip = astropy.stats.sigma_clip(image_masked, sigma=3)
        # print(three_sigma_clip)

        mad = stats.median_absolute_deviation(masked_image)
        # print(mad)

        astro_mad: float = astropy.stats.median_absolute_deviation(masked_image)
        print(astro_mad)

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)