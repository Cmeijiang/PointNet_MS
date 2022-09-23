# Pointnet_MindSpore
This repo is implementation for PointNet in MindSpore and on ModelNet40 and ShapeNetPart datasets.
# PointNet Description
PointNet was proposed in 2017, it is a hierarchical neural network that applies PointNet recursively on a nested partitioning of the input point set. The author of this paper proposes a method of applying deep learning model directly to point cloud data, which is called pointnet.

Paper: Qi, Charles R., et al. "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation" arXiv preprint arXiv:1612.00593 (2017).
# Model Architecture
For each n × 3 N\times 3N × 3 point cloud input, The network first aligns it spatially through a t-net (rotate to the front), map it to the 64 dimensional space through MLP, align it, and finally map it to the 1024 dimensional space. At this time, there is a 1024 dimensional vector representation for each point, and such vector representation is obviously redundant for a 3-dimensional point cloud. Therefore, the maximum pool operation is introduced at this time to keep only the maximum on all 1024 dimensional channels The one that gets 1 × 1024 1\times 1024 × 1 The vector of 1024 is the global feature of n nn point clouds.
# Requirements
```
Ubuntu 20.04
python=3.8
mindspore=1.6
cuda=11.1
```

# Pretrained Models

# Classification（ModelNet40）
***
## Run
```
python pointnet_modelnet40_train.py --data_url ./ModelNet40
python pointnet_modelnet40_eval.py --data_url ./ModelNet40 --ckpt_file ./best.ckpt
```
## Performance
|Model  | Acc |
|--|--|
| PointNet (official) | 89.2 |
| PointNet (mindspore) | **89.7** |

# Part Segmentation (ShapeNetPart)
