# Copyright (C) 2020  Daniel Magro
# Full License at: https://github.com/DanielMagro97/LEXACTUM/blob/main/LICENSE

from typing import List             # for type annotation

import os                           # for working with paths
import json                         # for reading JSON files
from tqdm import tqdm               # for progress bars
import numpy as np                  # for numpy arrays
from astropy.io import fits         # for opening .fits files
import astropy.visualization        # for ZScaleInterval (data normalisation)


# Function which loads a single .fits file as an image_sizeximage_sizex1 np.ndarray, given its path
def load_fits_image(fits_file_path: str, fits_file_normalisation: str = 'none') -> np.ndarray:
    image_data: np.ndarray = fits.getdata(fits_file_path)

    # image normalisation, according to Hyper Parameters (ZScale etc)
    if fits_file_normalisation == 'none':
        pass
    elif fits_file_normalisation == 'ZScale':
        # normalise the data using astropy's ZScaleInterval
        interval = astropy.visualization.ZScaleInterval()
        image_data = interval(image_data)

    return image_data


# Function which parses the trainset.dat file to traverse all the JSONs and collect the paths of each image and corresponding masks
def read_samples(trainset_path):
    image_mask_paths: List = []
    with open(trainset_path, 'r') as json_paths_file:
        for json_path in tqdm(json_paths_file, desc='Colleting JSON paths'):
            # replace original path with current dir path
            json_path = json_path.replace('/home/riggi/Data/MLData', os.path.abspath(os.pardir))
            json_path = os.path.normpath(json_path).strip()
            with open(json_path, 'r') as label_json:
                label = json.load(label_json)
                # replacing relative path with the absolute one
                label['img'] = label['img'].replace('..', os.sep.join(json_path.split(os.sep)[:-2]))
                label['img'] = os.path.normpath(label['img'])
                image_mask_paths.append(label)

    return image_mask_paths
