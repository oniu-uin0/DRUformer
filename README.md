## DRUformer: Enhancing driving scene important object detection with driving scene relationship understanding
![DRUformer](https://github.com/oniu-uin0/DRUformer/blob/main/druformer.png)

## Usage - Important Object detection
There are no extra compiled components in DRUformer and package dependencies are minimal,
so the code is very simple to use. We provide instructions how to install dependencies via conda.
First, clone the repository locally:
```
git clone https://github.com/oniu-uin0/DRUformer.git
```
Then, install PyTorch 1.5+ and torchvision 0.6+:
```
conda install -c pytorch pytorch torchvision
```
Install pycocotools (for evaluation on COCO) and scipy (for training):
```
conda install cython scipy
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```
That's it, should be good to train and evaluate detection models.

## Data preparation

Download and extract drama dataset with annotations from
[https://usa.honda-ri.com/drama](https://usa.honda-ri.com/drama).
We expect the directory structure to be the following:
```
path/to/drama/
  annotations/  # annotation json files
  combined/    # org clips
  processed/   # processed data
    train/
    test/
    val/
```
## Training
To train baseline DRUformer on a single node with 8 gpus for 300 epochs run:
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --img_folder /data/drama/ --ann_path /data/drama/annotations/
```
## Citation
If you find this repository useful, please consider giving a star ⭐ and citation 🦖:
```

```

