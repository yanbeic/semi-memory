#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=

for SPLIT_NAME in {train,test}
do
	DATA_NAME=svhn
	INPUT_DIR=rawdata/${DATA_NAME}/${SPLIT_NAME}
	OUTPUT_DIR=tfrecords/${DATA_NAME}/${SPLIT_NAME}

	python convert_tfrecords.py \
		--dataset_dir=${INPUT_DIR} \
		--split_name=${SPLIT_NAME} \
		--dataset_name=${DATA_NAME} \
		--output_dir=${OUTPUT_DIR} \
		--num_splits=1 \
		--labels_per_class=100
done


for SPLIT_NAME in {train,test}
do
	DATA_NAME=cifar10
	INPUT_DIR=rawdata/${DATA_NAME}/${SPLIT_NAME}
	OUTPUT_DIR=tfrecords/${DATA_NAME}/${SPLIT_NAME}

	python convert_tfrecords.py \
		--dataset_dir=${INPUT_DIR} \
		--split_name=${SPLIT_NAME} \
		--dataset_name=${DATA_NAME} \
		--output_dir=${OUTPUT_DIR} \
		--num_splits=1 \
		--labels_per_class=400
done


for SPLIT_NAME in {train,test}
do
	DATA_NAME=cifar100
	INPUT_DIR=rawdata/${DATA_NAME}/${SPLIT_NAME}
	OUTPUT_DIR=tfrecords/${DATA_NAME}/${SPLIT_NAME}

	python convert_tfrecords.py \
		--dataset_dir=${INPUT_DIR} \
		--split_name=${SPLIT_NAME} \
		--dataset_name=${DATA_NAME} \
		--output_dir=${OUTPUT_DIR} \
		--num_splits=1 \
		--labels_per_class=100
done
