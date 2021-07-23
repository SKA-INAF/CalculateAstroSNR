# Copyright (C) 2021  Daniel Magro and Renato Sortino
# Full License at: https://github.com/DanielMagro97/CalculateAstroSNR/blob/main/LICENSE

from typing import List, Dict       # for type annotation

import argparse                     # for parsing command line arguments
import os                           # for working with files and directories
from pathlib import Path
from tqdm import tqdm               # for progress bars
import numpy as np                  # for numpy arrays and other operations
import astropy.stats                # for 3-sigma clipping and MAD

from utils.data import read_samples, load_fits_image, get_output_path, save_to_json
from utils.flux import compute_peak_flux

# Suppress `Invalid 'BLANK' keyword in header.` warnings
import warnings
from astropy.io.fits.verify import VerifyWarning
warnings.simplefilter('ignore', category=VerifyWarning)


def get_args_parser():
    parser = argparse.ArgumentParser('SNR Calculation', add_help=False)

    parser.add_argument('--json_list_path', default="../MLDataset_cleaned/trainset.dat", type=str)

    noise_estimator_group = parser.add_mutually_exclusive_group()
    noise_estimator_group.add_argument('--3sigma_clip', dest='noise_estimator', action='store_const', const='3sigma_clip')
    noise_estimator_group.add_argument('--mad', dest='noise_estimator', action='store_const', const='mad')
    noise_estimator_group.set_defaults(noise_estimator='3sigma_clip')

    return parser


def main(args):
    # path of the file which points to the paths of the JSONs of all the images for which the SNR should be calculated
    image_json_list_path = Path(args.json_list_path)

    # retrieve what noise_estimator was specified (or the default) in the command line (3sigma_clip or mad)
    noise_estimator: str = args.noise_estimator
    print(f'Using {noise_estimator} for Background Noise Estimation.\n')

    samples: List = read_samples(image_json_list_path)

    # Declare an empty Dict which will store each image's SNR
    images_to_snr: Dict = {}
    # Declare an empty Dict which will store each JSON path and its SNR
    json_to_snr: Dict = {}

    # Look at each JSON file in image_json_path_list
    for sample in tqdm(samples, desc='Going through images and their object masks, and calculating SNR'):
        # check that the image fits file actually exists on disk
        if not os.path.isfile(sample['img']):
            # if not, print a message and skip it
            print('File ' + sample['img'] + ' not found on disk.')
            continue
        # load the image and headers from disk
        fits_image = load_fits_image(sample['img'])
        image: np.ndarray = fits_image[0].data
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
            fits_mask = load_fits_image(mask_path)
            mask: np.ndarray = fits_mask[0].data

            # convert the masks into boolean (for logical operations)
            boolean_mask: np.ndarray = (mask != 0)
            # Carry out an OR between the current and aggregated mask, to combine them
            np.bitwise_or(boolean_mask, combined_mask, out=combined_mask)

        masked_image: np.ma.MaskedArray = np.ma.array(image, mask=combined_mask)

        if noise_estimator == '3sigma_clip':
            # https://docs.astropy.org/en/stable/api/astropy.stats.sigma_clipped_stats.html
            three_sigma_clip = astropy.stats.sigma_clipped_stats(masked_image, sigma=3)
            # sigma_clipped_stats returns (mean, median, standard deviation), use the standard deviation
            background_noise: float = three_sigma_clip[2]
        elif noise_estimator == 'mad':
            # https://docs.astropy.org/en/stable/api/astropy.stats.median_absolute_deviation.html
            mad: float = astropy.stats.median_absolute_deviation(masked_image)
            background_noise: float = mad
        else:
            raise NotImplementedError(f'Specified Noise Estimator {noise_estimator} not supported')
        print(f'Background Noise: {background_noise}')

        peak_flux: float = compute_peak_flux(masked_image)
        print(f'Peak Flux: {peak_flux}')

        snr: float = peak_flux / background_noise
        print(f'SNR: {snr}')

        # Add entry for each image path
        img_path = get_output_path(sample)
        images_to_snr[img_path] = snr
        # Add entry for each JSON file
        json_to_snr[sample['json']] = snr

    # Save images_to_snr dictionary in a JSON format
    save_to_json(images_to_snr, 'images_to_snr.json')

    # Save each json path and its associated path to a text file, in the format: <json_path> <snr>
    # Also calculate how many of the images have an SNR less than 2, 5, 10, and more than 10
    with open('./json_snr_list.txt', 'w') as output_file:
        less_two: int = 0
        less_five: int = 0
        less_ten: int = 0
        more_ten: int = 0

        for json_path in json_to_snr:
            output_file.write(json_path + ' ' + str(json_to_snr[json_path]) + '\n')

            snr = json_to_snr[json_path]
            if snr < 2:
                less_two += 1
            elif snr < 5:
                less_five += 1
            elif snr < 10:
                less_ten += 1
            else:
                more_ten += 1

        print(f'Images with an SNR<2: {less_two}')
        print(f'Images with an SNR<5: {less_five}')
        print(f'Images with an SNR<10: {less_ten}')
        print(f'Images with an SNR>=10: {more_ten}')

    # Save each json path and its associated path to a text file, depending on its SNR
    # This split is intended for comparison with other solutions, as these are typical SNR values for which performances
    # are reported, however these boundaries can be adjusted as necessary
    with open('./snr_less_5.txt', 'w') as snr_less_five, open('./snr_less_10.txt', 'w') as snr_less_ten, \
            open('./snr_more_10.txt', 'w') as snr_more_ten:
        for json_path in json_to_snr:
            snr: float = json_to_snr[json_path]

            if snr < 5:
                snr_less_five.write(json_path + '\n')

            if snr < 10:
                snr_less_ten.write(json_path + '\n')
            else:
                snr_more_ten.write(json_path + '\n')

    # Save each json path and its associated path to a text file, depending on its SNR
    # This split is intended to produce a graph of performance against SNR, with binned SNRs. Again, these
    # boundaries can be adjusted as necessary
    with open('./snr_0-2.txt', 'w') as snr_0_2, open('./snr_2-5.txt', 'w') as snr_2_5, \
            open('./snr_5_10.txt', 'w') as snr_5_10, open('./snr_10_20.txt', 'w') as snr_10_20, \
            open('./snr_20_50.txt', 'w') as snr_20_50, open('./snr_50_100.txt', 'w') as snr_50_100, \
            open('./snr_100_200.txt', 'w') as snr_100_200, open('./snr_200+.txt', 'w') as snr_200_plus:
        for json_path in json_to_snr:
            snr: float = json_to_snr[json_path]

            if snr < 2:
                snr_0_2.write(json_path + '\n')
            elif snr < 5:
                snr_2_5.write(json_path + '\n')
            elif snr < 10:
                snr_5_10.write(json_path + '\n')
            elif snr < 20:
                snr_10_20.write(json_path + '\n')
            elif snr < 50:
                snr_20_50.write(json_path + '\n')
            elif snr < 100:
                snr_50_100.write(json_path + '\n')
            elif snr < 200:
                snr_100_200.write(json_path + '\n')
            else:
                snr_200_plus.write(json_path + '\n')


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)
