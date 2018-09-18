#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

# supply your working directory here
PATH_ROOT=
SEMI_SIGN=True


for i in 1
do

   TRAIN_DIR=${PATH_ROOT}model/semi/results/svhn/semi_split$i
   DATASET_DIR_L=${PATH_ROOT}model/semi/tfrecords/svhn/train/split$i/labeled/
   DATASET_DIR_U=${PATH_ROOT}model/semi/tfrecords/svhn/train/split$i/unlabeled/

   python train.py \
          --train_dir=${TRAIN_DIR} \
          --dataset_name=svhn \
          --dataset_dir_l=${DATASET_DIR_L} \
          --dataset_dir_u=${DATASET_DIR_U} \
          --preprocessing='svhn' \
          --batch_size=100 \
          --num_epochs=200 \
          --num_classes=10 \
          --num_train_l=1000 \
          --num_train_u=72257 \
          --semi=${SEMI_SIGN} \
          --num_gpus=1


   CHECKPOINT=${PATH_ROOT}model/semi/results/svhn/semi_split$i/
   EVALUATE=${PATH_ROOT}model/semi/results/
   DATASET_DIR=${PATH_ROOT}model/semi/tfrecords/svhn/test/
   SAVE_FILE=svhn.txt

   python test.py \
          --dataset_name=svhn \
          --preprocessing='svhn' \
          --checkpoint_path=${CHECKPOINT} \
          --dataset_dir=${DATASET_DIR} \
          --eval_dir=${EVALUATE} \
          --save_txt=${SAVE_FILE} \
          --batch_size=16 \
          --num_classes=10 \
          --num_examples=26032

done
