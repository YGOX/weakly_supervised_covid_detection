import time
from model_cls2 import DAN
from utility import *
from os import makedirs
from os.path import exists
from argparse import ArgumentParser
from joblib import Parallel, delayed
import tensorflow as tf
import numpy as np
import os
import h5py
os.environ['CUDA_VISIBLE_DEVICES']= '1'


def train_and_predict(lr, batch_size, image_size, num_iter, r, fold_id):

    data_path = os.path.join(os.path.dirname(os.getcwd()), "v3/")
    train_normal = np.load('nptrain_lists.npy', allow_pickle=True)[fold_id]
    train_covid = np.load('covidtrain_lists.npy', allow_pickle=True)[fold_id]

    valid_normal = np.load('npvalid_lists.npy', allow_pickle=True)[fold_id]
    valid_covid = np.load('covidvalid_lists.npy', allow_pickle=True)[fold_id]
    trainfiles = train_normal + train_covid
    train_labels= np.concatenate((np.zeros(len(train_normal)), np.ones(len(train_covid))))
    val_files = valid_normal + valid_covid
    val_labels = np.concatenate((np.zeros(len(valid_normal)), np.ones(len(valid_covid))))

    normal_weights = len(train_labels) / np.sum(train_labels == np.zeros(np.shape(train_labels)))
    covid_weights = len(train_labels) / np.sum(train_labels == np.ones(np.shape(train_labels)))

    iters=0

    weights = np.array([[normal_weights, covid_weights]])
    best_validation_accuracy = 0.0
    last_improvement = 0
    require_improvement = 1000
    prefix = ("Multi_deep_normalcovidv3"
              + "_fold_id=" + str(fold_id)
              + "_image_size=" + str(image_size[0])
              + "_batch_size=" + str(batch_size)
              + "_lr=" + str(lr)
              + "_r=" + str(r))
    print("\n" + prefix + "\n")
    checkpoint_dir = "../models/" + prefix + "/"
    summary_dir = "../logs/" + prefix + "/"
    results_dir = "../results/" + prefix + "/"

    if not exists(checkpoint_dir):
        makedirs(checkpoint_dir)

    if not exists(summary_dir):
        makedirs(summary_dir)

    if not exists(results_dir):
        makedirs(results_dir)
    out_val = open(results_dir + 'val_results.txt', 'w')

    with tf.Graph().as_default():

        global_step = tf.get_variable(
            'global_step',
            [],
            initializer=tf.constant_initializer(0),
            trainable=False
        )

        learning_rate = tf.train.exponential_decay(lr, global_step, decay_steps=3000, decay_rate=0.1,
                                                   staircase=True)
        optim = tf.train.AdamOptimizer(learning_rate)

        model = DAN(image_size= image_size, batch_size=batch_size, r=r, a=weights, is_train=True)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            grads = optim.compute_gradients(model.loss)
            apply_gradient = optim.apply_gradients(grads, global_step=global_step)
            train_op = tf.group(apply_gradient)
            null_op = tf.no_op()

        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        # Create a session for running Ops on the Graph.
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(init)

        Loss_sum = tf.summary.merge([model.loss_sum, model.loss3, model.loss4, model.loss5], name='Loss')
        Accuracy_sum = tf.summary.merge([model.accuracy_sum, model.class1_acc_sum, model.class2_acc_sum], name='Accuracy')

    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print("loading checkpoint %s,waiting......" % ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("load complete!")

    writer = tf.summary.FileWriter(summary_dir, sess.graph)

    start_time = time.time()
    with Parallel(n_jobs=batch_size) as parallel:
        shapes = np.repeat(np.array([image_size]), batch_size, axis=0)
        paths = np.repeat(data_path, batch_size, axis=0)
        while iters < num_iter:
            mini_batches = get_minibatches_ind(len(trainfiles), batch_size, shuffle=True)
            for _, batchidx in mini_batches:
                if len(batchidx) == batch_size:
                    input_seq = np.zeros((batch_size, image_size[0], image_size[1], 1), dtype="float32")
                    names = np.array(trainfiles)[batchidx]
                    output = parallel(delayed(augment_data)(f, s, p)
                                      for f, s, p in zip(names, shapes, paths))

                    for i in range(batch_size):
                        input_seq[i] = output[i]

                    train_out = sess.run([train_op, Loss_sum, Accuracy_sum, model.loss, model.avg_acc,
                                          model.avg_acc_class1, model.avg_acc_class2],
                                         feed_dict = {model.x: input_seq,
                                                      model.target: train_labels[batchidx],
                                                      model.dr_rate:0.5})
                    writer.add_summary(train_out[1], iters)
                    writer.add_summary(train_out[2], iters)

                    print("Iter:{}/{} time:{:4.4f} - loss: {:.4f} - acc:{:.4f} - neumo_acc:{:.4f} "
                          "- covid_acc:{:.4f} "
                          .format(iters, num_iter, time.time() - start_time,
                                  train_out[3], train_out[4], train_out[5], train_out[6]))
                    iters += 1

                    if np.mod(iters, 500) == 5:
                        avg_acc = 0.0
                        avg_class1 = 0.0
                        avg_class2 = 0.0

                        val_batches = get_minibatches_ind(len(val_files), batch_size, shuffle=True)

                        for _, batch_idx in val_batches:
                            if len(batch_idx) == batch_size:
                                val_seq = np.zeros((batch_size, image_size[0], image_size[1], 1),
                                                   dtype="float32")
                                val_names = np.array(val_files)[batch_idx]
                                val_paths = np.repeat(data_path, batch_size, axis=0)
                                val_output = parallel(delayed(augment_data)(f, s, p)
                                                      for f, s, p in zip(val_names, shapes, val_paths))

                                for i in range(batch_size):
                                    val_seq[i] = val_output[i]

                                val_out = sess.run([model.avg_acc, model.avg_acc_class1, model.avg_acc_class2],
                                                   feed_dict={model.x: val_seq,
                                                              model.target: val_labels[batch_idx],
                                                              model.dr_rate: 1.0})

                            avg_acc += val_out[0] / (len(val_files) // batch_size)
                            avg_class1 += val_out[1] / (len(val_files) // batch_size)
                            avg_class2 += val_out[2] / (len(val_files) // batch_size)

                            print("validating...- acc:{:.4f} - neumo_acc:{:.4f} - covid_acc:{:.4f}  "
                                  .format(val_out[0], val_out[1], val_out[2]))

                        out_val.write("val_iter{} acc:{:.4f} neumo_acc:{:.4f}  covid_acc:{:.4f} "
                                      .format(iters, avg_acc, avg_class1, avg_class2))
                        out_val.write("\n")

                        improved_str = 'validation'

                        if avg_acc > best_validation_accuracy:

                            best_validation_accuracy = avg_acc
                            last_improvement = iters
                            saver.save(sess, os.path.join(checkpoint_dir, 'bl__model'), global_step=iters)
                            improved_str = improved_str+'*'
                            print("Val_Iter:{} time:{:4.4f} - acc:{:.4f} - normal_acc:{:.4f} - covid_acc:{:.4f}  -{}"
                                  .format(iters, time.time() - start_time,
                                          avg_acc, avg_class1, avg_class2, improved_str))

            if iters - last_improvement > require_improvement:
                break


if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--lr", type=float, dest="lr",
                      default=0.0001, help="Base Learning Rate")
  parser.add_argument("--batch_size", type=int, dest="batch_size",
                      default=24, help="Mini-batch size")
  parser.add_argument("--r", type=int, dest="r",
                      default=1, help="focal factor")
  parser.add_argument("--image_size", type=int, dest="image_size",
                      default=[224,224], help="frame size")
  parser.add_argument("--num_iter", type=int, dest="num_iter",
                      default=10000, help="Number of iterations")
  parser.add_argument("--fold_id", type=int, dest="fold_id",
                      default=0, help="fold id_cross validation")
  args = parser.parse_args()

  train_and_predict(**vars(args))

