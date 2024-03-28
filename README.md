# Spatio-Temporal Aggregation Transformer for Object Detection With Neuromorphic Vision Sensors

This is the official Pytorch implementation of our paper [**Spatio-Temporal Aggregation Transformer for Object Detection With Neuromorphic Vision Sensors**]

## Notes on downloading this code
Since this project contains a large pre-trained model, you need to download it through:
```
git clone 
```
'git clone'. If the download fails, please try the following command: 'git config'.

## Conda Installation
The code heavily depends on PyTorch, which is optimized only for GPUs supporting CUDA. For our implementation the CUDA version 11.7 is used. The Python packages required for the experiment are detailed in: 
```
environment.yml
```
Install the project requirements with:
```
conda env create --file=environment.yml
```

## Required Data
We evaluated our approach on two datasets:
[NCaltech101](https://www.garrickorchard.com/datasets/n-caltech101) and 
[Prophesee Gen1 Automotive](https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-detection-dataset/).
Download them and extract them. By default, they are assumed to be in `/data/`, this can be changed by setting the `dataset_path` environment variable. 

## Training
The model training file is 'detection_training.py', where you can configure the training by adjusting environment variables such as 'batch_size', 'temporal_aggregation_size', 'epochs', and 'dataset_type'.

If you set the 'last_ckpt_name' variable to the name of an existing model, the training will continue based on this model; otherwise, the training will start from scratch.

You may commence the training process by executing the following code:
```
cd object_detection/
python detection_training.py
```

## Testing
The model training file is 'detection_test.py', where you can configure the testing by adjusting environment variables such as 'batch_size', 'temporal_aggregation_size', 'dataset_type' and 'ckpt_name'.

You may commence the testing process by executing the following code:
```
cd object_detection/
python detection_test.py
```

We have provided trained models, 'gen1-best.pth' for the Gen1 dataset and 'n-caltech101-best.pth' for the n-caltech101 dataset. These models are stored in the '/objec_detection/ckpt_gen1/' and '/objec_detection/ckpt_ncaltech101/' directories, respectively. If you directly run the training models we have provided, you will get the following results:
# On n-caltech101 dataset
```
P         R         map0.5     map0.75    map0.5:0.95:0.05
0.884     0.868     86.9%      76.3%      68.2%
```

# On Gen1 dataset
```
P         R         map0.5     map0.75    map0.5:0.95:0.05
0.810     0.762     78.8%      51.9%      49.9%
```

## Code Acknowledgments
* We trained our model based on the ImageNet pre-trained [MaxViT_Tiny](https://github.com/huggingface/pytorch-image-models) model
* This project has used code from the following projects:
- [timm](https://github.com/huggingface/pytorch-image-models) for the MaxViT layer implementation in Pytorch
- [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) for the detection PAFPN/head
- [Gen1 datasets reading tools](https://github.com/prophesee-ai/prophesee-automotive-dataset-toolbox) for tools on how to read the Gen1 datasets
