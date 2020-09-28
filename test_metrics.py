from utility import *
from model_cls3 import DAN
from utility import *
from argparse import ArgumentParser
from joblib import Parallel, delayed
import os
from os.path import exists
from scipy import special
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from imblearn.metrics import sensitivity_specificity_support
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import confusion_matrix
import re

import seaborn as sns
sns.set()

import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np


def test(batch_size, image_size, model_id):

    data_path = os.path.join(os.path.dirname(os.getcwd()), "v7/fold3/valid/")
    test_normal = get_image_names(os.path.join(data_path, "normal/"), label='normal/')
    test_covid = get_image_names(os.path.join(data_path, "covid/"), label='covid/')
    test_pneumonia = get_image_names(os.path.join(data_path, "pneumonia/"), label='pneumonia/')

    test_files = test_normal + test_covid +test_pneumonia
    test_labels = np.concatenate((np.zeros(len(test_normal)), np.ones(len(test_covid)), 2*np.ones(len(test_pneumonia))))
    normal_id = [re.split(r'[_]', test_normal[i])[-5] for i in range(len(test_normal))]
    covid_id = [re.split(r'[_]', test_covid[i])[-5] for i in range(len(test_covid))]
    pneumonia_id = [re.split(r'[_]', test_pneumonia[i])[-5] for i in range(len(test_pneumonia))]


    #num_norm, ind = np.unique(np.array(pneumo_id), return_counts=True)
    num_normal= len(set(normal_id))
    num_covid= len(set(covid_id))
    num_neumo= len(set(pneumonia_id))
    print(num_normal)
    print(num_covid)
    print(num_neumo)

    target_names = ['normal', 'neumo', 'pneumonia']

    checkpoint_dir = "../models" + '/' + model_id
    if not exists(checkpoint_dir):
        raise IOError("model path, {}, could not be resolved".format(str(checkpoint_dir)))

    num_test_seqs = len(test_files)
    print("Number of test slices={}".format(num_test_seqs))

    with tf.Graph().as_default() as graph:

        model = DAN(image_size=image_size, batch_size=batch_size, is_train=False)
        C5_logits = graph.get_tensor_by_name('Classifier5GMP:0')
        C4_logits = graph.get_tensor_by_name('Classifier4GMP:0')
        C3_logits = graph.get_tensor_by_name('Classifier3GMP:0')

        saver = tf.train.Saver()
        init = tf.global_variables_initializer()

        # Create a session for running Ops on the Graph.
        sess = tf.Session(
            config=tf.ConfigProto(allow_soft_placement=True)
        )
        sess.run(init)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print("loading checkpoint %s,waiting......" % ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("load complete!")

    labels = []
    C_prob = []
    logits= []
    c5logits=[]
    c4logits=[]
    c3logits= []

    test_batches = get_minibatches_ind(len(test_files), batch_size, shuffle=True)

    with Parallel(n_jobs=batch_size) as parallel:
        shapes = np.repeat(np.array([image_size]), batch_size, axis=0)
        paths = np.repeat(data_path, batch_size, axis=0)
        images = []
        for _, batch_idx in test_batches:
            if len(batch_idx) == batch_size:
                labels.append(test_labels[batch_idx])
                test_seq = np.zeros((batch_size, image_size[0], image_size[1], 1), dtype="float32")
                test_names = np.array(test_files)[batch_idx]

                test_output = parallel(delayed(augment_data)(f, s, p)
                                       for f, s, p in zip(test_names, shapes, paths))

                for i in range(batch_size):
                    test_seq[i] = test_output[i]
                images.append(test_seq)

                output = sess.run(
                    [model.prob_out, model.logit_out,C5_logits, C4_logits, C3_logits],
                    feed_dict={model.x: test_seq, model.target: test_labels[batch_idx], model.dr_rate: 1.0})

                C_prob.append(output[0])
                logits.append(output[1])
                c5logits.append(output[2])
                c4logits.append(output[3])
                c3logits.append(output[4])

        labels = np.concatenate(labels, axis=0).astype('int32')
        scores = np.concatenate(C_prob, axis=0)
        c5prob = special.softmax(np.squeeze(np.concatenate(c5logits, axis=0), axis=(1, 2)), -1)
        c4prob = special.softmax(np.squeeze(np.concatenate(c4logits, axis=0), axis=(1, 2)), -1)
        c3prob = special.softmax(np.squeeze(np.concatenate(c3logits, axis=0), axis=(1, 2)), -1)

    c5logit= np.argmax(c5prob,1)
    c4logit= np.argmax(c4prob,1)
    c3logit= np.argmax(c3prob,1)

    results= [np.argmax(scores,1), c5logit, c4logit, c3logit]

    precision=[]
    recall=[]
    accuracy=[]
    f1score=[]
    for i in range(4):
        precision.append(precision_score(labels, results[i], average='weighted'))
        recall.append(recall_score(labels, results[i], average='weighted'))
        accuracy.append(accuracy_score(labels, results[i]))
        f1score.append(f1_score(labels, results[i], average='weighted'))
    rep= sensitivity_specificity_support(labels, results[0], average='weighted')
    cm = confusion_matrix(labels, results[0])
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm.diagonal())
    print(classification_report_imbalanced(labels,results[0], target_names=target_names))
    print(num_normal)
    print(num_covid)
    print(rep)
    print(precision)
    print(recall)
    print(accuracy)
    print(f1score)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, dest="batch_size",
                          default=12, help="Mini-batch size")
    parser.add_argument("--image_size", type=int, dest="image_size",
                          default=[224,224], help="frame size")
    parser.add_argument("--model_id", type=str, dest="model_id",
                        default='Multi_deepv73clsfold3_image_size=224_batch_size=24_lr=0.0001_r=1', help="model")
    args = parser.parse_args()

    test(**vars(args))