#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

# supply your working directory here
PATH_ROOT=${PWD}/
SEMI_SIGN=True


for i in 1
do

   TRAIN_DIR=${PATH_ROOT}results/cifar100/semi_split$i
   DATASET_DIR_L=${PATH_ROOT}tfrecords/cifar100/train/split$i/labeled/
   DATASET_DIR_U=${PATH_ROOT}tfrecords/cifar100/train/split$i/unlabeled/

   python train.py \
          --train_dir=${TRAIN_DIR} \
          --dataset_name=cifar100 \
          --dataset_dir_l=${DATASET_DIR_L} \
          --dataset_dir_u=${DATASET_DIR_U} \
          --preprocessing='cifar' \
          --batch_size=100 \
          --num_epochs=500 \
          --num_classes=100 \
          --num_train_l=10000 \
          --num_train_u=40000 \
          --semi=${SEMI_SIGN} \
          --num_gpus=1


   CHECKPOINT=${PATH_ROOT}results/cifar100/semi_split$i
   EVALUATE=${PATH_ROOT}results/
   DATASET_DIR=${PATH_ROOT}tfrecords/cifar100/test/
   SAVE_FILE=cifar100.txt

   python test.py \
          --dataset_name=cifar100 \
          --preprocessing='cifar' \
          --checkpoint_path=${CHECKPOINT} \
          --dataset_dir=${DATASET_DIR} \
          --eval_dir=${EVALUATE} \
          --save_txt=${SAVE_FILE} \
          --batch_size=100 \
          --num_classes=100

done
