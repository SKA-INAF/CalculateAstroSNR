# CalculateAstroSNR

Calculate Astro SNR is a simple tool which quickly and efficiently calculates the Signal to Noise Ratio (SNR) of each image in any given dataset.

## Dependencies

CalculateAstroSNR has been tested on both Windows and Linux (Ubuntu), with Python 3.6 and 3.7.
All the required Python packages can be installed by running this command:

```
pip install astropy numpy tqdm
```

## How to Run CalculateAstroSNR

To run the CalculateAstroSNR script on a dataset, run the following command, substituting `<>`s with your own values.
```
python main.py --json_list_path "<path_to_json_list>" [--3sigma_clip|--mad]
```

* `<path_to_json_list>` - The path (Windows or Ubuntu formats are both accepted) to a text file that lists the paths of images' JSON file. Must be in this format:
```
D:\dataset\...\mask_source2701.json
D:\dataset\...\mask_source1648.json
D:\dataset\...\mask_galaxy0539.json
```
or on Ubuntu:
```
/home/user/dataset/.../mask_source2701.json
/home/user/dataset/.../mask_source1648.json
/home/user/dataset/.../mask_galaxy0539.json
```
Each JSON file must then be in this format:
```
{
  "img": "../imgs/source2701.fits",
  "objs": [
    {
      "class": "source",
      "mask": "mask_source2701_obj1.fits",
      "name": "S1"
    },
    {
      "class": "source",
      "mask": "mask_source2701_obj2.fits",
      "name": "S2"
    },
    {
      "class": "source",
      "mask": "mask_source2701_obj3.fits",
      "name": "S3"
    }
  ]
}
```
Note: the only obligatory fields for the script to work are `img` and `mask` for each entry in `objs`. 


* `[--3sigma_clip|--mad]` - To choose which Background Noise Estimator to use in the SNR calculation, `3 Sigma Clipping` or `Median Absolute Deviation (MAD)`.
This is an optional argument, only 1 of the 2 arguments can be passed, and the default is `3sigma_clip` 

## Outputs

* The first output of CalculateAstroSNR is a JSON file of image paths and their respective SNR. 
```
{".../source2701.fits": "7.454104", ".../source1648.fits": "15.33992", ...}
```
* The second output is a list of JSON paths, similar to the one inputted in `<path_to_json_list>`, with the SNR value for that file.
  This also outputs the **number** of images with an `SNR<2`, `SNR<5`, `SNR<10`, and `SNR>=10`.
  This is to help the user determine whether there are enough instances of an SNR value to accurately calculate performances for those values.
```
.../mask_source2701.json 7.454104
.../mask_source1648.json 15.33992
```
```
Images with an SNR<2: ...
Images with an SNR<5: ...
Images with an SNR<10: ...
Images with an SNR>=10: ...
```
* The third output is 3 separate lists of JSON paths, again in the same format as `<path_to_json_list>`.
  The entries in the 3 separate files are split by their SNR (SNR<5, SNR<10, SNR>=10).
```
.../mask_source2701.json
.../mask_source2061.json
```
* The fourth and final output is 8 separate lists of JSON paths, in the same format as the previous (third) output.
  The entries here are divided into separate bins according to their SNR value, with bins for SNR `0-2`, `2-5`, `5-10`, `10-20`, `20-50`, `50-100`, `100-200`, `200+`
  This format allows for the user to create a graph of performance against SNR.

Note that in each case, the `...`s in the output paths represent the full absolute path of the file.
