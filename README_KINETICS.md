# Kinetics dataset loader
## Description
Script for loading Kinetics video from Youtube

## Prerequisites
### Packages needed:
 - [youtube-dl](https://github.com/ytdl-org/youtube-dl) 
 - [ffmpeg](https://ffmpeg.org/download.html)
 - [pandas](https://pandas.pydata.org/getpandas.html)

## Datasets
### [Kinetics-600](https://deepmind.com/research/open-source/open-source-datasets/kinetics)
#### Usage
```
python load_kinetics.py [-h] --data-path DATA_PATH
                        [--split {train,val,test,holdout_test}]
                        [--num-jobs NUM_JOBS]
```
#### Options
```
  -h, --help            show this help message and exit
  --data-path DATA_PATH
                        Data directory path. All new data will be put in
                        subdirectory DATA_PATH/Kinetics_600
  --split {train,val,test,holdout_test}
                        Dataset split type. If not chosen, "train", then
                        "val", then "test" sets will be loaded. Downloaded
                        files will be put into subdirectories "train", "val",
                        "test", "holdout_test" respectively
  --num-jobs NUM_JOBS   Number of simultaneous downloading processes
```
#### Example
Download `train` video files into directory `D:\Data\Kinetics_600\train` with 12 worker processes
```
python load_kinetics.py --data-path "D:\Data" --split train --num-jobs 12
```

