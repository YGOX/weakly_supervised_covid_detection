from utility import *
from argparse import ArgumentParser
from joblib import Parallel, delayed
import os
from os.path import exists
import tensorflow as tf
import numpy as np
from os import makedirs
import matplotlib
matplotlib.use('agg')
from model_cls3 import DAN
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES']= '0'


def test(batch_size, image_size, model_id, fold_id):

    data_path = os.path.join(os.path.dirname(os.getcwd()),"test/")

    test_normal = get_image_names(os.path.join(data_path, "normal/"), label='normal/')
    test_covid = get_image_names(os.path.join(data_path, "covid/"), label='covid/')
    test_neumo = get_image_names(os.path.join(data_path, "pneumonia/"), label='pneumonia/')
    test_files = test_normal + test_covid + test_neumo
    test_labels = np.concatenate((np.zeros(len(test_normal)), np.ones(len(test_covid)), 2*np.ones(len(test_neumo))))
    target_names = ['normal', 'covid', 'cap']

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
    C5_maps = []
    C4_maps =[]
    C3_maps =[]

    test_batches = get_minibatches_ind(len(test_files), batch_size, shuffle=False)

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
                    [model.C5_maps, model.C4_maps, model.C3_maps],
                    feed_dict={model.x: test_seq, model.target: test_labels[batch_idx], model.dr_rate: 1.0})

                C5_maps.append(output[0])
                C4_maps.append(output[1])
                C3_maps.append(output[2])

        images = np.concatenate(images, axis=0)
        labels = np.concatenate(labels, axis=0)

        C5_maps = np.concatenate(C5_maps, axis=0)
        C4_maps= np.concatenate(C4_maps, axis=0)
        C3_maps = np.concatenate(C3_maps, axis=0)


        if not exists("../results/C5/"):
            makedirs("../results/C5/")
        if not exists("../results/C4/"):
            makedirs("../results/C4/")
        if not exists("../results/C3/"):
            makedirs("../results/C3/")
        if not exists("../results/origin/"):
            makedirs("../results/origin/")
        for j in range((len(test_files) // batch_size) * batch_size):
            im = np.asarray(np.reshape(images[j], (image_size[0], image_size[1])))

            extent = 0, 224, 0, 224
            fig, ax = plt.subplots()
            ax.set_axis_off()
            # plt.figure()
            # plt.axis('off')
            # plt.imshow(norm_CAM, cmap=plt.cm.jet, alpha=.5, interpolation='bilinear', extent=extent)
            plt.imshow(im, cmap='gray', interpolation='bilinear', extent=extent)
            plt.imshow(C5_maps[j, :, :, int(labels[j])], cmap=plt.cm.jet, alpha=0.5, interpolation='bilinear',
                       extent=extent)
            extent1 = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            # plt.savefig(results_dir + "out{:d}".format(j), bbox_inches=extent)
            plt.savefig("../results/C5/" + target_names[int(labels[j])] + "out{:d}".format(j), bbox_inches=extent1)
            plt.cla()
            plt.clf()

            fig, ax = plt.subplots()
            ax.set_axis_off()
            # plt.figure()
            # plt.axis('off')
            # plt.imshow(norm_CAM, cmap=plt.cm.jet, alpha=.5, interpolation='bilinear', extent=extent)
            plt.imshow(im, cmap='gray', interpolation='bilinear', extent=extent)
            plt.imshow(C4_maps[j, :, :, int(labels[j])], cmap=plt.cm.jet, alpha=0.5, interpolation='bilinear',
                       extent=extent)
            extent1 = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            # plt.savefig(results_dir + "out{:d}".format(j), bbox_inches=extent)
            plt.savefig("../results/C4/" + target_names[int(labels[j])] + "out{:d}".format(j), bbox_inches=extent1)
            plt.cla()
            plt.clf()

            fig, ax = plt.subplots()
            ax.set_axis_off()
            # plt.figure()
            # plt.axis('off')
            # plt.imshow(norm_CAM, cmap=plt.cm.jet, alpha=.5, interpolation='bilinear', extent=extent)
            plt.imshow(im, cmap='gray', interpolation='bilinear', extent=extent)
            plt.imshow(C3_maps[j, :, :, int(labels[j])], cmap=plt.cm.jet, alpha=0.5, interpolation='bilinear',
                       extent=extent)
            extent1 = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            # plt.savefig(results_dir + "out{:d}".format(j), bbox_inches=extent)
            plt.savefig("../results/C3/" + target_names[int(labels[j])] + "out{:d}".format(j), bbox_inches=extent1)
            plt.cla()
            plt.clf()

            fig, ax = plt.subplots()
            ax.set_axis_off()
            # plt.figure()
            # plt.axis('off')
            # plt.imshow(norm_CAM, cmap=plt.cm.jet, alpha=.5, interpolation='bilinear', extent=extent)
            plt.imshow(im, cmap='gray', interpolation='bilinear', extent=extent)
            extent1 = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            # plt.savefig(results_dir + "out{:d}".format(j), bbox_inches=extent)
            plt.savefig("../results/origin/" +target_names[int(labels[j])]+ "out{:d}".format(j), bbox_inches=extent1)
            plt.cla()
            plt.clf()

        print("Results saved")
        print("Done.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, dest="batch_size",
                          default=12, help="Mini-batch size")
    parser.add_argument("--image_size", type=int, dest="image_size",
                          default=[224,224], help="frame size")
    parser.add_argument("--model_id", type=str, dest="model_id",
                        default='Multi_deepee_image_size=224_batch_size=24_lr=0.0001_r=1/', help="model")
    parser.add_argument("--fold_id", type=int, dest="fold_id",
                        default=0, help="fold id_cross validation")
    args = parser.parse_args()

    test(**vars(args))