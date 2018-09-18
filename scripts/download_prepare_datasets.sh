#!/usr/bin/env bash

mkdir rawdata
mkdir rawdata/svhn
mkdir rawdata/cifar10
mkdir rawdata/cifar100

mkdir results
mkdir results/svhn
mkdir results/cifar10
mkdir results/cifar100

mkdir tfrecords
mkdir tfrecords/svhn
mkdir tfrecords/cifar10
mkdir tfrecords/cifar100

cd rawdata/svhn
wget http://ufldl.stanford.edu/housenumbers/train_32x32.mat
wget http://ufldl.stanford.edu/housenumbers/test_32x32.mat
cd ..
cd ..

cd rawdata/cifar10
wget https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz
tar xvzf cifar-10-matlab.tar.gz
rm cifar-10-matlab.tar.gz
cd ..
cd ..

cd rawdata/cifar100
wget https://www.cs.toronto.edu/~kriz/cifar-100-matlab.tar.gz
tar xvzf cifar-100-matlab.tar.gz
rm cifar-100-matlab.tar.gz
cd ..
cd ..


for DATA_NAME in {svhn,cifar10,cifar100}
do
	for DATA_TYPE in {test,train}
	do
		python convert_data.py \
			--dataset_name=${DATA_NAME} \
			--dataset_type=${DATA_TYPE}
	done
done
