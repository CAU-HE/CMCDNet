# CMCDNet:  Cross-modal change detection flood extraction based on convolutional neural network

This repo contains official implementation of the paper [Cross-modal change detection flood extraction based on convolutional neural network](https://www.sciencedirect.com/science/article/pii/S1569843223000195). 

The code is based on MMSegmentation (version 0.20.0). 

## Install

1. Install mmcv and other dependencies following the MMSegmentation instructions, [mmsegmentation/get_started.md at v0.20.0 · open-mmlab/mmsegmentation (github.com)](https://github.com/open-mmlab/mmsegmentation/blob/v0.20.0/docs/get_started.md#installation).

2. Clone this repo.

   ```
   git clone https://github.com/CAU-HE/CMCDNet.git
   ```

3. Create the data directory to hold the CHU-Flood dataset.

   ```
   mkdir data
   ```

4. Download CAU-Flood from https://pan.baidu.com/s/1i5yxdfwjP-oTyiRmq6FZHQ (rnx6), extract the train.tar.gz and test.tar.gz to the data folder.

5. The code and data should be organized like this:
```
|- data
| |- train
| | |- flood_vv # the ground gruth flood map
| | |- vv       # the post-event SAR images
| | |- opt      # the pre-event optical images
| |- test
| | |- flood_vv 
| | |- vv       
| | |- opt      
|- cmcdnet
```

## Train

CMCDNet was implemented in `mmseg/models/backbone/cmcd.py`. We also created a new dataset named `WCDataset` to read samples from the CHU-Flood dataset. 

The configuration files are in `my_scripts/water_change`, alter batch size, normalization type and other parameters as you need. 

Change to the code directory:

```
cd cmcdnet
```

Single GPU train:

```
python tools/train.py my_scripts/water_change/opt_sar_cmcd_r50-effb2_30e.py
```

Multi-GPU train:

```
bash tools/dist_train.sh my_scripts/water_change/opt_sar_cmcd_r50-effb2_30e.py {num_gpus}
```

## Citation

If you find this repo useful for your research, please consider citing our paper:

```
@article{HE2023103197,
title = {Cross-modal change detection flood extraction based on convolutional neural network},
journal = {International Journal of Applied Earth Observation and Geoinformation},
volume = {117},
pages = {103197},
year = {2023},
issn = {1569-8432},
doi = {https://doi.org/10.1016/j.jag.2023.103197},
url = {https://www.sciencedirect.com/science/article/pii/S1569843223000195},
author = {Xiaoning He and Shuangcheng Zhang and Bowei Xue and Tong Zhao and Tong Wu},
}
```

