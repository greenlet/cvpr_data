# Video datasets loader for ActiveNet challenge
## Description
Convenience scripts for loading and transforming youtube video for datasets within ActiveNet challenge

## Prerequisites
### Packages needed:
 - [youtube-dl](https://github.com/ytdl-org/youtube-dl) 
 - [ffmpeg](https://ffmpeg.org/download.html) - only needed when flag `--shorten` is used (which is trimming source video to fit included timestamps)

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
Will be available soon

### [ActivityNet](http://activity-net.org/download.html)
Will be available soon

### [ActivityNet Captions](https://cs.stanford.edu/people/ranjaykrishna/densevid)
Will be available soon

