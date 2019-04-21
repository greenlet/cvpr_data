# AVA dataset loader
## Description
Scripts for loading AVA video from Youtube and [CVDF](https://github.com/cvdfoundation/ava-dataset) Amazon storage. Key moments:
 - Video downloaded from Youtube in `mp4` format if available
 - If Youtube video is unavailable, [CVDF Amazon source](https://s3.amazonaws.com/ava-dataset) is used. Those videos formats are not limited with `mp4` and could be `mkv`, `webm` as well
 - When shortening is enabled (`--shorten` flag), `ffmpeg` is used. It does just cropping without transcoding, so the process is quite fast (few seconds for each video). Also, due to relative timestamps change after shortening videos, all timestamp values in AVA dataset `txt` and `csv` files are changed and saved to the new ones with the `shorten_` prefix

## Prerequisites
### Packages needed:
 - [youtube-dl](https://github.com/ytdl-org/youtube-dl) 
 - [ffmpeg](https://ffmpeg.org/download.html) - only needed when flag `--shorten` is used

## Datasets
### [AVA Actions](https://research.google.com/ava/download.html)
#### Usage
```
python load_ava_actions.py [-h] [--version VERSION] --data-path DATA_PATH
                           [--type {train,validation,test}] [--shorten]
```
#### Options
```
  --version VERSION     AVA Actions dataset version (default: 2.2)
  --data-path DATA_PATH
                        Data directory path. All new data will be put in
                        subdirectory DATA_PATH/AVA_Actions_<version-suffix>
  --type {train,validation,test}
                        Dataset type. If not chosen, all videos will be
                        loaded. Downloaded files will be put into
                        subdirectories src_train, src_val, src_test
                        respectively
  --shorten             Boolean flag. If present, indicates whether to shorten
                        source video files. New video files will be generated
                        into subdirectories short_train, short_val, short_test
                        respectively. All CSV, TXT data containing timestamps
                        will be adjusted accordingly and put into files with
                        prefix "short_"
```
#### Example
Download and shorten all video files into directory `D:\Data\AVA_Actions_v2.2`
```
python load_ava_actions.py --data-path "D:\Data" --shorten
```

### [AVA Active Speaker](https://research.google.com/ava/download.html)
Not implemented yet

