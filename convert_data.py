from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import scipy.io as sio
import os
from PIL import Image

tf.app.flags.DEFINE_string('dataset_name', None, 'The name of the dataset.')
tf.app.flags.DEFINE_string('dataset_type', None, 'The type of the dataset: "train" or "test".')

FLAGS = tf.app.flags.FLAGS


def make_subfolder(writepath, num_classes):
    for i in range(num_classes):
        classpath = os.path.join(writepath, 'class'+str(i))
        if not os.path.exists(classpath):
            os.makedirs(classpath)


def mat_to_image_cifar(data, labels, num_classes, writepath):
    img_count = np.zeros([num_classes])
    for i in range(labels.shape[0]):
        R = data[i, :1024]
        G = data[i, 1024:2048]
        B = data[i, 2048:]
        img = np.zeros([32, 32, 3])
        k = 0
        for p in range(32):
            for q in range(32):
                img[p, q, 0] = R[k]
                img[p, q, 1] = G[k]
                img[p, q, 2] = B[k]
                k += 1
        img = img.astype(np.uint8)
        classpath = 'class' + str(labels[i])
        imgpath = '{0:05}.png'.format(int(img_count[labels[i]]))
        imgpath = os.path.join(writepath, classpath, imgpath)
        print(imgpath)
        img = Image.fromarray(img)
        img.save(imgpath)
        img_count[labels[i]] += 1


def mat_to_image_svhn(data, labels, num_classes, writepath):
    img_count = np.zeros([num_classes])
    labels = labels-1
    for i in range(labels.shape[0]):
        img = data[:,:,:,i].astype(np.uint8)
        classpath = 'class' + str(labels[i])
        imgpath = '{0:05}.png'.format(int(img_count[labels[i]]))
        imgpath = os.path.join(writepath, classpath, imgpath)
        print(imgpath)
        img = Image.fromarray(img)
        img.save(imgpath)
        img_count[labels[i]] += 1


def convert_cifar100(readpath, writepath):
    make_subfolder(writepath, num_classes=100)
    filename = readpath + FLAGS.dataset_type + '.mat'
    rawdata = sio.loadmat(filename)
    data = rawdata['data']
    labels = np.squeeze(rawdata['fine_labels'])
    mat_to_image_cifar(data, labels, num_classes=100, writepath=writepath)


def convert_cifar10(readpath, writepath):
    make_subfolder(writepath, num_classes=10)
    if FLAGS.dataset_type=='test':
        filename = readpath + 'test_batch.mat'
        rawdata = sio.loadmat(filename)
        data = rawdata['data']
        labels = np.squeeze(rawdata['labels'])
        mat_to_image_cifar(data, labels, num_classes=10, writepath=writepath)
    else:
        data = []
        labels = []
        for i in range(5):
            filename = readpath + 'data_batch_' + str(i+1) + '.mat'
            rawdata = sio.loadmat(filename)
            data.append(rawdata['data'])
            labels.append(np.squeeze(rawdata['labels']))
        data = np.concatenate(data, 0)
        labels = np.concatenate(labels, 0)
        mat_to_image_cifar(data, labels, num_classes=10, writepath=writepath)


def convert_svhn(readpath, writepath):
    make_subfolder(writepath, num_classes=10)
    filename = readpath + FLAGS.dataset_type + '_32x32.mat'
    rawdata = sio.loadmat(filename)
    data = rawdata['X']
    labels = np.squeeze(rawdata['y'])
    mat_to_image_svhn(data, labels, num_classes=10, writepath=writepath)


def main():
    if FLAGS.dataset_name == 'svhn':
        readpath = 'rawdata/svhn/'
        writepath = os.path.join('rawdata/svhn/', FLAGS.dataset_type)
        convert_svhn(readpath, writepath)
    elif FLAGS.dataset_name == 'cifar100':
        readpath = 'rawdata/cifar100/cifar-100-matlab/'
        writepath = os.path.join('rawdata/cifar100/', FLAGS.dataset_type)
        convert_cifar100(readpath, writepath)
    elif FLAGS.dataset_name == 'cifar10':
        readpath = 'rawdata/cifar10/cifar-10-batches-mat/'
        writepath = os.path.join('rawdata/cifar10/', FLAGS.dataset_type)
        convert_cifar10(readpath, writepath)
    else:
        raise ValueError('You must supply the dataset name as -- svhn, cifar100, or cifar10')


if __name__ == "__main__":
    main()
