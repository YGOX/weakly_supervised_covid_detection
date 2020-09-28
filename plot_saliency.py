from model_cls3 import DAN
from utility import *
from argparse import ArgumentParser
from joblib import Parallel, delayed
import os
from os.path import exists
import tensorflow as tf
from os import makedirs
import saliency.saliency as sal
import matplotlib.pyplot as plt
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES']= '0'


def test(batch_size, image_size, model_id):

    data_path = os.path.join(os.path.dirname(os.getcwd()),"val/")
    test_normal = get_image_names(os.path.join(data_path, "normal/"), label='normal/')
    test_covid = get_image_names(os.path.join(data_path, "covid/"), label='covid/')
    test_neumo = get_image_names(os.path.join(data_path, "pneumonia/"), label='pneumonia/')
    test_files = test_normal + test_covid + test_neumo
    test_labels = np.concatenate((np.zeros(len(test_normal)), np.ones(len(test_covid)), 2*np.ones(len(test_neumo))))
    target_names= ['normal', 'covid', 'neumo']

    checkpoint_dir = "../models" + '/' + model_id
    if not exists(checkpoint_dir):
        raise IOError("model path, {}, could not be resolved".format(str(checkpoint_dir)))

    num_test_seqs = len(test_files)
    print("Number of test slices={}".format(num_test_seqs))

    with tf.Graph().as_default() as graph:

        model = DAN(image_size=image_size, batch_size=batch_size, is_train=False)

        sess = tf.Session(graph=graph)
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        saver.restore(sess, ckpt.model_checkpoint_path)

        logitsc5 = graph.get_tensor_by_name("Classifier5GMP:0")
        logitsc4 = graph.get_tensor_by_name("Classifier4GMP:0")
        logitsc3 = graph.get_tensor_by_name("Classifier3GMP:0")
        logitsC5 = tf.squeeze(logitsc5, [1, 2])
        logitsC4 = tf.squeeze(logitsc4, [1, 2])
        logitsC3 = tf.squeeze(logitsc3, [1, 2])
        neuron_selector = tf.placeholder(tf.int32)
        y5 = logitsC5[0][neuron_selector]
        y4 = logitsC4[0][neuron_selector]
        y3 = logitsC3[0][neuron_selector]
        integrated_gradients_C5 = sal.IntegratedGradients(graph, sess, y5, model.x)
        integrated_gradients_C4 = sal.IntegratedGradients(graph, sess, y4, model.x)
        integrated_gradients_C3 = sal.IntegratedGradients(graph, sess, y3, model.x)

    labels = []
    C5_maps = []
    C4_maps = []
    C3_maps = []

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

                baseline = np.zeros(test_seq.shape[1:])
                baseline.fill(-1)
                #test_labels[batch_idx][0].astype('int32')

                vanilla_integrated_gradients_mask_c5 = integrated_gradients_C5.GetMask(
                    test_seq[0], feed_dict={neuron_selector: test_labels[batch_idx][0].astype('int32'), model.target: test_labels[batch_idx], model.dr_rate: 1.0},
                    x_steps=25, x_baseline=baseline)
                vanilla_mask_grayscale_c5 = sal.VisualizeImageGrayscale(vanilla_integrated_gradients_mask_c5)

                vanilla_integrated_gradients_mask_c4 = integrated_gradients_C4.GetMask(
                    test_seq[0], feed_dict={neuron_selector: test_labels[batch_idx][0].astype('int32'), model.target: test_labels[batch_idx],
                               model.dr_rate: 1.0},
                    x_steps=25, x_baseline=baseline)
                vanilla_mask_grayscale_c4 = sal.VisualizeImageGrayscale(vanilla_integrated_gradients_mask_c4)
                vanilla_integrated_gradients_mask_c3 = integrated_gradients_C3.GetMask(
                    test_seq[0], feed_dict={neuron_selector: test_labels[batch_idx][0].astype('int32'), model.target: test_labels[batch_idx],
                               model.dr_rate: 1.0},
                    x_steps=25, x_baseline=baseline)
                vanilla_mask_grayscale_c3 = sal.VisualizeImageGrayscale(vanilla_integrated_gradients_mask_c3)

                C5_maps.append(np.expand_dims(vanilla_mask_grayscale_c5, axis=0))
                C4_maps.append(np.expand_dims(vanilla_mask_grayscale_c4, axis=0))
                C3_maps.append(np.expand_dims(vanilla_mask_grayscale_c3, axis=0))

        images = np.concatenate(images, axis=0)
        labels = np.concatenate(labels, axis=0)

        C5_maps = np.concatenate(C5_maps, axis=0)
        C4_maps = np.concatenate(C4_maps, axis=0)
        C3_maps = np.concatenate(C3_maps, axis=0)
        #Sal_map= np.multiply(C3_maps, C4_maps)
        #Sal_map= np.multiply(C5_maps, Sal_map)

        result_path = "../results/sal/"
        if not exists(result_path):
            makedirs(result_path)
        for j in range((len(test_files) // batch_size) * batch_size):
            im = np.asarray(np.reshape(images[j], (image_size[0], image_size[1])))
            im = (im - im.min()) / (im.max() - im.min())
            normalised_sal= (C5_maps[j,:,:]+2)*(C4_maps[j,:,:]+2)*(C3_maps[j,:,:]+2)
            normalised_sal = (normalised_sal - normalised_sal.min()) / (normalised_sal.max() - normalised_sal.min())

            extent = 0, 224, 0, 224
            fig, ax = plt.subplots()
            ax.set_axis_off()
            # plt.figure()
            # plt.axis('off')
            # plt.imshow(norm_CAM, cmap=plt.cm.jet, alpha=.5, interpolation='bilinear', extent=extent)
            plt.imshow(im, cmap='gray', interpolation='bilinear', vmin=0, vmax=1, extent=extent)
            plt.imshow(normalised_sal, cmap=plt.cm.hot, alpha=0.5, vmin=0, vmax=1, interpolation='bilinear',
                       extent=extent)
            extent1 = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            # plt.savefig(results_dir + "out{:d}".format(j), bbox_inches=extent)
            plt.savefig(result_path + target_names[int(labels[j])]+"out_all{:d}".format(j), bbox_inches=extent1)
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
            plt.savefig(result_path + "out_or{:d}".format(j), bbox_inches=extent1)
            plt.cla()
            plt.clf()

            fig, ax = plt.subplots()
            ax.set_axis_off()
            # plt.figure()
            # plt.axis('off')
            # plt.imshow(norm_CAM, cmap=plt.cm.jet, alpha=.5, interpolation='bilinear', extent=extent)
            plt.imshow(C5_maps[j, :, :], cmap=plt.cm.hot, alpha=1, vmin=0, vmax=1, interpolation='bilinear',
                       extent=extent)
            extent1 = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            # plt.savefig(results_dir + "out{:d}".format(j), bbox_inches=extent)
            plt.savefig(result_path + "out_c5_{:d}".format(j), bbox_inches=extent1)
            plt.cla()
            plt.clf()

            fig, ax = plt.subplots()
            ax.set_axis_off()
            # plt.figure()
            # plt.axis('off')
            # plt.imshow(norm_CAM, cmap=plt.cm.jet, alpha=.5, interpolation='bilinear', extent=extent)
            plt.imshow(C4_maps[j, :, :], cmap=plt.cm.hot, alpha=1, vmin=0, vmax=1, interpolation='bilinear',
                       extent=extent)
            extent1 = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            # plt.savefig(results_dir + "out{:d}".format(j), bbox_inches=extent)
            plt.savefig(result_path + "out_c4_{:d}".format(j), bbox_inches=extent1)
            plt.cla()
            plt.clf()

            fig, ax = plt.subplots()
            ax.set_axis_off()
            # plt.figure()
            # plt.axis('off')
            # plt.imshow(norm_CAM, cmap=plt.cm.jet, alpha=.5, interpolation='bilinear', extent=extent)
            plt.imshow(C3_maps[j, :, :], cmap=plt.cm.hot, alpha=1, vmin=0, vmax=1, interpolation='bilinear',
                       extent=extent)
            extent1 = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            # plt.savefig(results_dir + "out{:d}".format(j), bbox_inches=extent)
            plt.savefig(result_path + "out_c3_{:d}".format(j), bbox_inches=extent1)
            plt.cla()
            plt.clf()

        print("Results saved")
        print("Done.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, dest="batch_size",
                          default=1, help="Mini-batch size")
    parser.add_argument("--image_size", type=int, dest="image_size",
                          default=[224,224], help="frame size")
    parser.add_argument("--model_id", type=str, dest="model_id",
                        default='Multi_deepv73clsfold1_image_size=224_batch_size=24_lr=0.0001_r=1', help="model")
    args = parser.parse_args()

    test(**vars(args))