from typing import List         # for type annotation

import os                                       # for working with files and directories
import json                                     # for reading JSON files from disk
import numpy as np                              # for numpy arrays and other operations
import astropy.stats                            # for 3-sigma clipping
from scipy import stats

from utils.load_fits import load_fits_image     # for loading fits images with optional normalisation


if __name__ == '__main__':
    # path of the file which points to the paths of the JSONs of all the images for which the SNR should be calculated
    # image_json_list_path: str = r'D:\datasets\rg-dataset\crossval.dat'
    # image_json_list_path: str = r'D:\datasets\rg-dataset\crossval_local.dat'
    # TODO use argparse
    image_json_list_path: str = r'D:\datasets\rg-dataset\crossval_local_sample.dat'

    # List storing the path of each image's JSON
    image_json_path_list: List[str] = []
    # populate the list by going through the file pointed to in image_json_list_path
    with open(image_json_list_path, 'r') as image_json_path_list_file:
        # go through the file line by line
        for image_json_path in image_json_path_list_file:
            image_json_path_list.append(image_json_path.strip())
        # print(image_json_path_list)

    # Look at each JSON file in image_json_path_list
    for image_json_path in image_json_path_list:
        # check that the json file actually exists on disk
        if not os.path.isfile(image_json_path):
            # if not, print a message and skip it
            print('File ' + image_json_path + ' not found on disk.')
            continue
        with open(image_json_path, 'r') as image_json:
            image_mask_paths = json.load(image_json)

        # Load the Image Fits file
        image_path: str = image_mask_paths['img']

        # set the absolute image path to the path of the image json file minus the last 2 directories ('masks','mask_source1234.json')
        full_image_path: List[str] = image_json_path.split(os.sep)[:-2]
        # and add to it the last 2 directories of the image path ('imgs', 'source1234.fits')
        # full_image_path.extend(image_path.split(os.sep)[-2:])
        # TODO use the above!!!
        full_image_path.extend(image_path.split('/')[-2:])

        # if the system is Windows, the path will be D:datasets\\rg-dataset..., to fix this
        if os.name == 'nt':
            full_image_path[0] = full_image_path[0] + '\\'

        # reconstruct the list of directories into a path
        full_image_path: str = os.path.join(*full_image_path)


        # check that the image fits file actually exists on disk
        if not os.path.isfile(full_image_path):
            # if not, print a message and skip it
            print('File ' + full_image_path + ' not found on disk.')
            continue

        # load the image from disk
        # TODO!!!! check whether to load images with normalisation or not for SNR
        # TODO check whether to do anything with NaNs
        image = load_fits_image(full_image_path, 'none')

        # generate an aggregate of all the individual object masks
        combined_mask: np.ndarray = np.zeros((image.shape[0], image.shape[1]))
        # Load each mask associated with the current image, as specified in the JSON File
        for mask_path in image_mask_paths['objs']:
            # set the absolute image path to the path of the image json file minus the file name of the JSON file ('mask_source1234.json')
            full_mask_path: List[str] = image_json_path.split(os.sep)[:-1]
            # and add to it the file name of the mask path ('mask_source1231_obj1.fits')
            full_mask_path.append(mask_path['mask'])

            # if the system is Windows, the path will be D:datasets\\rg-dataset..., to fix this
            if os.name == 'nt':
                full_mask_path[0] = full_mask_path[0] + '\\'

            # reconstruct the list of directories into a path
            full_mask_path: str = os.path.join(*full_mask_path)


            # check that the mask fits file actually exists on disk
            if not os.path.isfile(full_mask_path):
                # if not, print a message and skip it
                print('File ' + full_mask_path + ' not found on disk.')
                continue

            # load the mask from disk
            # TODO!!!! check whether to load images with normalisation or not for SNR
            mask: np.ndarray = load_fits_image(full_mask_path, 'none')

            # convert the masks into boolean (for logical operations)
            boolean_mask: np.ndarray = (mask == 1)
            boolean_combined_mask: np.ndarray = (combined_mask == 1)
            # Carry out an OR between the current and aggregated mask, to combine them
            # by storing the output in combined_mask, the output is broadcasted into floats
            np.bitwise_or(boolean_mask, boolean_combined_mask, out=combined_mask)

        # calculate the image with masked areas zeroed out
        # TODO might not be needed
        image_masked = image - (image * combined_mask)
        # print(image_masked)

        boolean_combined_mask: np.ndarray = (combined_mask == 1)
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
