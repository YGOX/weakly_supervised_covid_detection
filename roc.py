from model_cls3 import DAN
from utility import *
from argparse import ArgumentParser
from joblib import Parallel, delayed
import os
from os.path import exists
import seaborn as sns
sns.set()
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES']='1'
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def test(batch_size, image_size, model_id, num_class):

    data_path = os.path.join(os.path.dirname(os.getcwd()), "v7/fold4/valid/")
    test_normal = get_image_names(os.path.join(data_path, "normal/"), label='normal/')
    test_covid = get_image_names(os.path.join(data_path, "covid/"), label='covid/')
    test_pneumonia = get_image_names(os.path.join(data_path, "pneumonia/"), label='pneumonia/')

    test_files = test_normal + test_covid  +test_pneumonia
    test_labels = np.concatenate((np.zeros(len(test_normal)), np.ones(len(test_covid), 2*np.ones(len(test_pneumonia)))))
    target_names = ['normal', 'Covid-19', 'CAP']

    checkpoint_dir = "../models" + '/' + model_id
    if not exists(checkpoint_dir):
        raise IOError("model path, {}, could not be resolved".format(str(checkpoint_dir)))

    num_test_seqs = len(test_files)
    print("Number of test slices={}".format(num_test_seqs))

    with tf.Graph().as_default():

        model = DAN(image_size=image_size, batch_size=batch_size, is_train=False)

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
                    [model.prob_out],
                    feed_dict={model.x: test_seq, model.target: test_labels[batch_idx], model.dr_rate: 1.0})

                C_prob.append(output[0])

        labels = np.concatenate(labels, axis=0).astype('int32')
        scores = np.concatenate(C_prob, axis=0)

    one_hot_gt = np.zeros((labels.size, labels.max() + 1))
    one_hot_gt[np.arange(labels.size), labels] = 1

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_class):
        fpr[i], tpr[i], _ = roc_curve(one_hot_gt[:, i], scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(one_hot_gt.ravel(), scores.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure()
    lw = 2
    plt.plot(fpr[0], tpr[0], color='red',
             lw=lw, label=target_names[0]+'(area = %0.2f)' % roc_auc[0])
    plt.plot(fpr[1], tpr[1], color='green',
             lw=lw, label=target_names[1]+'(area = %0.2f)' % roc_auc[1])
    plt.plot(fpr[2], tpr[2], color='blue',
             lw=lw, label=target_names[2]+'(area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.title('normal/Covid-19/CAP')
    plt.legend(loc="lower right")
    plt.show()

    print("Done.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, dest="batch_size",
                          default=12, help="Mini-batch size")
    parser.add_argument("--num_class", type=int, dest="num_class", default =3, help = "num classes")
    parser.add_argument("--image_size", type=int, dest="image_size",
                          default=[224,224], help="frame size")
    parser.add_argument("--model_id", type=str, dest="model_id",
                        default='Multi_deepv73clsfold3_image_size=224_batch_size=24_lr=0.0001_r=1', help="model")
    args = parser.parse_args()

    test(**vars(args))
