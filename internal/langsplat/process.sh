#!/bin/bash

dataset_path=$1
dataset_name=$2

# get the language feature of the scene (included in OpenGaussian lerf_ovs dataset)
#python preprocess.py --dataset_name $dataset_path

# ATTENTION: Before you train the LangSplat, please follow https://github.com/graphdeco-inria/gaussian-splatting
# to train the RGB 3D Gaussian Splatting model.
# put the path of your RGB model after '--start_checkpoint'

# in the gs repo and env run training (make sure to include flags to get chkpnt30000.pth file)
# e.g. python train.py -s $dataset_path -m $dataset_path/pretrain --checkpoint_iteration 30000 --eval

# train the autoencoder
cd autoencoder
python train.py --dataset_path $dataset_path --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 --lr 0.0007 --dataset_name $dataset_name
# e.g. python train.py --dataset_path ../data/sofa --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 --lr 0.0007 --dataset_name sofa

# get the 3-dims language feature of the scene
python test.py --dataset_path $dataset_path --dataset_name $dataset_name
# e.g. python test.py --dataset_path ../data/sofa --dataset_name sofa

cd ..
for level in 1 2 3
do
    python train.py -s $dataset_path -m $dataset_path/finetune --start_checkpoint $dataset_path/pretrain/chkpnt30000.pth --feature_level ${level}
    # e.g. python train.py -s data/sofa -m output/sofa --start_checkpoint data/sofa/sofa/chkpnt30000.pth --feature_level 3
done

for level in 1 2 3
do
    # render rgb
    #python render.py -m $dataset_path/finetune_${level}
    # render language features
    python render.py -m $dataset_path/finetune_${level} --include_feature
    # e.g. python render.py -m output/sofa_3 --include_feature
done