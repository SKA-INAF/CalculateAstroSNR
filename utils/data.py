# Copyright (C) 2021  Daniel Magro and Renato Sortino
# Full License at: https://github.com/DanielMagro97/CalculateAstroSNR/blob/main/LICENSE

from typing import Dict, List             # for type annotation

import os                           # for working with paths
import json
from astropy.extern.configobj.validate import numToDottedQuad                         # for reading JSON files
from tqdm import tqdm               # for progress bars
import numpy as np                  # for numpy arrays
from astropy.io import fits         # for opening .fits files


# Function which loads a single .fits file as an image_sizeximage_size np.ndarray, given its path
def load_fits_image(fits_file_path: str, just_data: bool = False):

    if just_data:
        # Load only data array
        image_data: np.ndarray = fits.getdata(fits_file_path)
    else:
        # Load all data from FITS file
        image_data = fits.open(fits_file_path)

    return image_data


# Function which parses the trainset.dat file to traverse all the JSONs and collect the paths of each image and corresponding masks
def read_samples(trainset_path) -> List:
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


class DataEncoder(json.JSONEncoder):
    '''
    Class to encode numpy arrays and floats into JSON
    '''
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return str(obj)
        return json.JSONEncoder.default(self, obj)

def get_output_path(sample: Dict) -> str:
    '''
    Get the absolute image path
    '''
    img_path_list = sample['img'].split(os.sep)
    start_dir = img_path_list.index('MLDataset_cleaned')
    img_path_list = img_path_list[start_dir:]
    return os.path.join(*img_path_list)

def save_to_json(data: Dict, filename: str):
    '''
    Saves the dictionary data to JSON
    '''
    with open(filename, 'w') as out:
        json.dump(data, out, cls=DataEncoder)