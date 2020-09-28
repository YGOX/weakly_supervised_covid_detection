import scipy.io as sio
import numpy as np
from PIL import Image
import random
import tensorflow as tf
import pandas as pd
import os

def file_names(files_path, time_window, ind):
    mat_contents = sio.loadmat(files_path)
    file_list = []
    file_names = []
    str = ['train_files', 'val_files', 'test_files']

    [file_list.append(mat_contents[str[ind]][0][i][0]) for i in range(len(mat_contents[str[ind]][0]))]

    list(file_list[0][0])

    for i in range(len(file_list)):
        for j in range(len(file_list[i])):
            file_names.extend(list(file_list[i][j]))

    file_names = [file_names[i:i + time_window] for i in range(0, len(file_names), time_window)]

    return file_names

def get_image_names(file_path, label):

    filelist = os.listdir(file_path)
    filelist = [label+item for item in filelist]

    return filelist

def get_seq(file_path):

    filelist = os.listdir(file_path)
    filelist = [item for item in filelist if 'frame' or 'f' in item]

    return filelist

def load_labels(file_path, ind):
    mat_contents= sio.loadmat(file_path)
    str= ['train_labels', 'val_labels', 'test_labels']
    labels= mat_contents[str[ind]][0]

    return labels


def get_minibatches_ind(n, minibatch_size, shuffle="Flase"):
    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0

    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start: minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)


def load_data(f_names, image_size, data_path):

    img = Image.open(data_path + f_names)
    tf = np.array([0.5, 0.5])
    dx, dy = np.floor((img.size[0] - image_size[1]) * tf[0]) + 1, np.floor(
        (img.size[1] - image_size[0]) * tf[1]) + 1
    sx, sy = np.around(np.linspace(dx, image_size[1] + dx - 1, image_size[1] )), np.around(
        np.linspace(dy, image_size[0] + dy - 1, image_size[0]))
    img = img.crop((sx[0] - 1, sy[0] - 1, sx[len(sx) - 1], sy[len(sy) - 1])).resize((image_size[1], image_size[0]),
                                                                                    Image.BILINEAR)

    x = np.asarray(img, dtype='float32')
    if len(np.shape(x)) != 2:
        x = x[:, :, 0]
    else:
        x = x

    x = (x - x.mean()) / x.std()

    return np.expand_dims(x, 2)

def load_images(f_names, image_size, data_path, time_window):

    seq = np.zeros((image_size[0], image_size[1], time_window, 1), dtype="uint8")
    for t in range(time_window):
        img_path = data_path + f_names[t].replace("\\", "/") + '.png'
        img = Image.open(img_path)
        tf = np.array([0.5, 0.5])
        dx, dy = np.floor((img.size[0] - image_size[0]) * tf[0]) + 1, np.floor(
            (img.size[1] - image_size[1]) * tf[[1]]) + 1
        sx, sy = np.around(np.linspace(dx, image_size[0] + dx - 1, image_size[0])), np.around(
            np.linspace(dy, image_size[1] + dy - 1, image_size[1]))
        img = img.crop((sx[0] - 1, sy[0] - 1, sx[len(sx) - 1], sy[len(sy) - 1]))

        x = np.asarray(img, dtype='uint8')

        if len(np.shape(x)) != 2:
            x= x[:,:,0]
        else:
            x= x

        seq[:, :, t] = x[:, :, None]

    return seq


def augment_data(f_names, image_size, data_path):
    #scaling_f = np.arange(0.8, 1.1, 0.1)
    #rotate_f = np.arange(-25, 26, 1)
    #flip = np.random.rand() > 0.5
    #sf = random.choice(scaling_f)
    #rf = random.choice(rotate_f)

    img = Image.open(data_path + f_names)
    #tf = np.array([0.5, 0.5])
    #dx, dy = np.floor((img.size[0] - image_size[1] * sf) * tf[0]) + 1, np.floor(
    #    (img.size[1] - image_size[0] * sf) * tf[1]) + 1
    #sx, sy = np.around(np.linspace(dx, image_size[1] * sf + dx - 1, image_size[1] * sf)), np.around(
    #    np.linspace(dy, image_size[0] * sf + dy - 1, image_size[0] * sf))
    #img = img.crop((sx[0] - 1, sy[0] - 1, sx[len(sx) - 1], sy[len(sy) - 1])).resize((image_size[1], image_size[0]),
    img = img.resize((image_size[1], image_size[0]), Image.BILINEAR)

    #if flip == True:
    #    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    #img = img.rotate(rf, Image.BILINEAR)

    x = np.asarray(img, dtype='float32')
    if len(np.shape(x)) != 2:
        x = x[:, :, 0]
    else:
        x = x

    x = (x - x.mean()) / x.std()

    return np.expand_dims(x, 2)

def get_2labels_batch(labels):
    batch_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
    return batch_labels

def get_3labels_batch(labels):
    batch_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=3)
    return batch_labels


def class_accuracy(target, logits, num_class, batch_size):

    class_accuracy = []

    label = tf.eye(num_class, num_class)
    for c in range(num_class):
        ones = tf.ones([batch_size, 1])

        class_ind = tf.equal(tf.argmax(target, 1), tf.argmax(tf.matmul(ones, tf.reshape(label[c, :], [1, num_class])), 1))

        class_ind = tf.reshape(tf.boolean_mask(np.arange(batch_size), class_ind), [-1, 1])

        class_acc = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(tf.gather_nd(target, class_ind), 1),
                             tf.argmax(tf.gather_nd(logits, class_ind), 1)), tf.float32))

        class_accuracy.append(class_acc)

    return class_accuracy

def array_to_img(x):
    """
    Util function for converting 4D numpy array to numpy array.
    Returns PIL RGB image.
    References
    ----------
    - adapted from keras preprocessing/image.py
    """
    x = np.asarray(x)
    x += max(-np.min(x), 0)
    x_max = np.max(x)
    if x_max != 0:
        x /= x_max
    x *= 255
    #return Image.fromarray(x.astype('uint8'),'L')
    return x


def gt_box(csv_path, num_test):
    df = pd.read_csv(csv_path, usecols=['region_shape_attributes'])
    gt = []
    W, H = 503, 376
    norm = np.tile(np.array([W, H, W, H]), (num_test, 1))

    for n in range(0, num_test, 1):

        region_shape = df['region_shape_attributes'][n]

        if region_shape == '{}':

            x, y, w, h = 0, 0, 0, 0
            bbox = [[x, y, w, h]]

        else:
            region_shape = region_shape.split(',')
            x = region_shape[1]
            y = region_shape[2]
            w = region_shape[3]
            h = region_shape[4]

            x = int(x.split(':')[1])
            y = int(y.split(':')[1])
            w = int(w.split(':')[1])
            h = h.split(':')[1]
            h = int(h.split('}')[0])
            bbox = [[x, y, w, h]]

        gt.append(bbox)

    gt_bbox = np.concatenate(gt, axis=0)
    norm_gt_bbox = np.divide(gt_bbox, norm)
    scale = np.tile(np.array([224, 224, 224, 224]), (num_test,1))
    gt = np.multiply(norm_gt_bbox, scale)
    gt = gt.astype(int)

    return gt


def bbox_iou(pred_bbox, gt_bbox):

    pred_bbox[:, 2, :] = pred_bbox[:, 0, :] + pred_bbox[:, 2, :]
    pred_bbox[:, 3, :] = pred_bbox[:, 1, :] + pred_bbox[:, 3, :]

    gt_bbox[:, 2, :] = gt_bbox[:, 0, :] + gt_bbox[:, 2, :]
    gt_bbox[:, 3, :] = gt_bbox[:, 1, :] + gt_bbox[:, 3, :]

    XA = np.maximum(pred_bbox[:, 0, :], gt_bbox[:, 0, :]).astype('float32')
    YA = np.maximum(pred_bbox[:, 1, :], gt_bbox[:, 1, :]).astype('float32')
    XB = np.minimum(pred_bbox[:, 2, :], gt_bbox[:, 2, :]).astype('float32')
    YB = np.minimum(pred_bbox[:, 3, :], gt_bbox[:, 3, :]).astype('float32')

    Intersection = (XB - XA) * (YB - YA)

    BoxA_Area = (pred_bbox[:, 2, :] - pred_bbox[:, 0, :]) * (pred_bbox[:, 3, :] - pred_bbox[:, 1, :])
    BoxB_Area = (gt_bbox[:, 2, :] - gt_bbox[:, 0, :]) * (gt_bbox[:, 3, :] - gt_bbox[:, 1, :])

    iou = Intersection / (BoxA_Area + BoxB_Area - Intersection)

    best_iou = np.amax(iou, axis=1)

    return iou, best_iou
