import os
import tensorflow as tf
from layers import *
from utility import *
import numpy as np

class DAN(object):
    def __init__(self, image_size, batch_size=32, c_dim=1, f_dim=32, d=64, num_class=2, is_train=True, r=1, a=None):

        self.batch_size = batch_size
        self.image_size = image_size
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.is_train= is_train
        self.d = d
        self.r = r
        self.a = a
        self.num_class = num_class

        self.x_shape = [batch_size, self.image_size[0], self.image_size[1], self.c_dim]
        self.target_shape = [batch_size,]
        self.build_model()

    def build_model(self):

        self.x = tf.placeholder(tf.float32, self.x_shape, name='x')
        self.target = tf.placeholder(tf.float32, self.target_shape, name='y')
        self.dr_rate= tf.placeholder(tf.float32)

        enc_h = self.forward(self.x, reuse=tf.AUTO_REUSE)

        Pyramid_features = {}
        #Pyramid_features['P1']= self.h_dyn_conv1[:,:,:,:,4]
        #Pyramid_features['P2'] = self.h_dyn_conv2[:, :, :, :, 4]
        #Pyramid_features['P3'] = self.h_dyn_conv3[:, :, :, :, 4]
        #Pyramid_features['P4'] = self.h_dyn_conv4[:, :, :, :, 4]
        #Pyramid_features['P5'] = self.h_dyn_conv5[:, :, :, :, 4]
        #Pyramid_features['P1']= self.h_dyn_conv1[:,:,:,:,4]
        #Pyramid_features['P2']= self.h_dyn_conv2[:,:,:,:,4]
        Pyramid_features['P3']= enc_h[0]
        Pyramid_features['P4']= enc_h[1]
        Pyramid_features['P5']= enc_h[2]

        maps = []
        self.Loss = {}
        self.avg_loss = {}
        self.acc = {}
        self.acc_class1 = {}
        self.acc_class2 = {}
        self.acc_class3 = {}
        self.acc_class4 = {}

        self.avg_acc = {}
        self.avg_acc_class1 = {}
        self.avg_acc_class2 = {}
        self.avg_acc_class3 = {}
        self.avg_acc_class4 = {}

        avg_acc= 0.0
        avg_loss=0.0
        avg_acc_class1= 0.0
        avg_acc_class2= 0.0
        #self.C5_maps= {}
        #self.C4_maps= {}
        #self.C3_maps= {}
        #self.C2_maps= {}
        #self.C1_maps= {}
        #self.C5_logits= {}
        #self.C4_logits= {}
        #self.C3_logits= {}
        #self.C2_logits= {}
        #self.C1_logits= {}
        logit_mean= tf.zeros((self.batch_size, self.num_class))

        for c in range(len(Pyramid_features)+2, 2, -1):
            cmap_temp = []
            logit_temp = []
            features = Pyramid_features['P%d'%c]

            Loss, class_activation_maps, accuracy, class_acc, logits = \
                self.classifiers(features,
                                 target=self.target,
                                 name='Classifier%d'%c,
                                 train=self.is_train)

            if not self.is_train:
                maps.append(class_activation_maps)
                logit_mean += logits / len(Pyramid_features)
            else:
                self.Loss['loss_classifier%d'%c] = Loss
                self.acc['acc_classifier%d'%c] = accuracy
                self.acc_class1['class1_acc_classifier%d'%c] = class_acc[0]
                self.acc_class2['class2_acc_classifier%d'%c] = class_acc[1]

                avg_loss += Loss / len(Pyramid_features)
                avg_acc += accuracy / len(Pyramid_features)
                avg_acc_class1 += class_acc[0] / len(Pyramid_features)
                avg_acc_class2 += class_acc[1] / len(Pyramid_features)

        if self.is_train:
            self.avg_acc = avg_acc
            self.avg_acc_class1 = avg_acc_class1
            self.avg_acc_class2 = avg_acc_class2

            self.loss = avg_loss

            self.loss_sum = tf.summary.scalar("total_loss", self.loss)
            # self.loss1= tf.summary.scalar("loss_classifier1", self.Loss['loss_classifier1'])
            # self.loss2 = tf.summary.scalar("loss_classifier2", self.Loss['loss_classifier2'])
            self.loss3 = tf.summary.scalar("loss_classifier3", self.Loss['loss_classifier3'])
            self.loss4 = tf.summary.scalar("loss_classifier4", self.Loss['loss_classifier4'])
            self.loss5 = tf.summary.scalar("loss_classifier5", self.Loss['loss_classifier5'])

            self.accuracy_sum = tf.summary.scalar("overall_accuracy", self.avg_acc)
            self.class1_acc_sum = tf.summary.scalar("class1_acc", self.avg_acc_class1)
            self.class2_acc_sum = tf.summary.scalar("class2_acc", self.avg_acc_class2)
        else:
            self.C5_maps = maps[0]
            self.C4_maps = maps[1]
            self.C3_maps = maps[2]
            #self.C2_maps = maps[3]
            #self.C1_maps = maps[4]
            self.prob_out= tf.nn.softmax(logit_mean, 1)
            self.logit_out= tf.argmax(logit_mean, axis=1)
            #self.C2_logits = tf.argmax(logits_container[3], axis=1)
            #self.C1_logits = tf.argmax(logits_container[4], axis=1)

    def forward(self, x, reuse):
    #def forward(self, x, stn, reuse):
        enc_h = []

        conv1_1 = relu(batch_norm(conv2d(x, output_dim=self.f_dim, k_h=3, k_w=3,
                              d_h=1, d_w=1, name='conv1_1', reuse=reuse),
                                  name='bn_conv1_1', train=self.is_train, reuse=reuse))
        conv1_2 = relu(batch_norm(conv2d(conv1_1, output_dim=self.f_dim, k_h=3, k_w=3,
                              d_h=1, d_w=1, name='conv1_2', reuse=reuse),
                                  name='bn_conv1_2', train=self.is_train, reuse=reuse))
        pool1 = MaxPooling(conv1_2, 2)

        #conv1_drop = tf.nn.dropout(pool1, keep_prob=self.dr_rate, name='conv1_dr')

        #attention_conv1, conv1_features = A_conv1(pool1, state_conv1, scope='A1', reuse=reuse)
        #attention_maps.append(attention_conv1)
        #enc_h.append(conv1_features)

        conv2_1 = relu(batch_norm(conv2d(pool1, output_dim=self.f_dim * 2, k_h=3, k_w=3,
                              d_h=1, d_w=1, name='conv2_1', reuse=reuse),
                                  name='bn_conv2_1', train=self.is_train, reuse=reuse))
        conv2_2 = relu(batch_norm(conv2d(conv2_1, output_dim=self.f_dim * 2, k_h=3, k_w=3,
                              d_h=1, d_w=1, name='conv2_2', reuse=reuse),
                                  name='bn_conv2_2', train=self.is_train, reuse=reuse))
        pool2 = MaxPooling(conv2_2, 2)

        #conv2_drop = tf.nn.dropout(pool2, keep_prob=self.dr_rate, name='conv2_dr')

        #attention_conv2, conv2_features = A_conv2(pool2, state_conv2, scope='A2', reuse=reuse)
        #attention_maps.append(attention_conv2)
        #enc_h.append(conv2_features)

        conv3_1 = relu(batch_norm(conv2d(pool2, output_dim=self.f_dim * 4, k_h=3, k_w=3,
                              d_h=1, d_w=1, name='conv3_1', reuse=reuse),
                                  name='bn_conv3_1', train=self.is_train, reuse=reuse))
        conv3_2 = relu(batch_norm(conv2d(conv3_1, output_dim=self.f_dim * 4, k_h=3, k_w=3,
                              d_h=1, d_w=1, name='conv3_2', reuse=reuse),
                                  name='bn_conv3_2', train=self.is_train, reuse=reuse))
        conv3_3 = relu(batch_norm(conv2d(conv3_2, output_dim=self.f_dim * 4, k_h=3, k_w=3,
                              d_h=1, d_w=1, name='conv3_3', reuse=reuse),
                                  name='bn_conv3_3', train=self.is_train, reuse=reuse))

        pool3 = MaxPooling(conv3_3, 2)

        #conv3_drop = tf.nn.dropout(pool3, keep_prob=self.dr_rate, name='conv3_dr')

        enc_h.append(pool3)


        conv4_1 = relu(batch_norm(conv2d(pool3, output_dim=self.f_dim * 8, k_h=3, k_w=3,
                              d_h=1, d_w=1, name='conv4_1', reuse=reuse),
                                  name='bn_conv4_1', train=self.is_train, reuse=reuse))
        conv4_2 = relu(batch_norm(conv2d(conv4_1, output_dim=self.f_dim * 8, k_h=3, k_w=3,
                              d_h=1, d_w=1, name='conv4_2', reuse=reuse),
                                  name='bn_conv4_2', train=self.is_train, reuse=reuse))
        conv4_3 = relu(batch_norm(conv2d(conv4_2, output_dim=self.f_dim * 8, k_h=3, k_w=3,
                              d_h=1, d_w=1, name='conv4_3', reuse=reuse),
                                  name='bn_conv4_3', train=self.is_train, reuse=reuse))

        pool4 = MaxPooling(conv4_3, 2)

         #conv4_drop = tf.nn.dropout(pool4, keep_prob=self.dr_rate, name='conv4_dr')
        enc_h.append(pool4)

        conv5_1 = relu(batch_norm(conv2d(pool4, output_dim=self.f_dim * 8, k_h=3, k_w=3,
                              d_h=1, d_w=1, name='conv5_1', reuse=reuse),
                                  name='bn_conv5_1', train=self.is_train, reuse=reuse))
        conv5_2 = relu(batch_norm(conv2d(conv5_1, output_dim=self.f_dim * 8, k_h=3, k_w=3,
                              d_h=1, d_w=1, name='conv5_2', reuse=reuse),
                                  name='bn_conv5_2', train=self.is_train, reuse=reuse))
        conv5_3 = relu(batch_norm(conv2d(conv5_2, output_dim=self.f_dim * 8, k_h=3, k_w=3,
                              d_h=1, d_w=1, name='conv5_3', reuse=reuse),
                                  name='bn_conv5_3', train=self.is_train, reuse=reuse))


        pool5 = MaxPooling(conv5_3, 2)

        #conv5_drop = tf.nn.dropout(pool5, keep_prob=self.dr_rate, name='conv5_dr')

        #theta = self.LocNet(pool5, name='stn_5', reuse=reuse)

        #pool5_trans = stn(pool5, theta, scope='stn5')

        #enc_h.append(pool5_trans)
        enc_h.append(pool5)

        #return enc_h, theta
        return enc_h

    def classifiers(self, features, target, name='Classifier', train=True, reuse=tf.AUTO_REUSE):
        class_activation_maps = relu(batch_norm(conv2d(features, output_dim=self.num_class,
                                                       k_h=1, k_w=1, d_h=1, d_w=1, name=name + 'adaption', reuse=reuse),
                                                name=name + 'bn_adaption', train=train, reuse=reuse))
        CAM= tf.nn.dropout(class_activation_maps, keep_prob=self.dr_rate, name='CAM_dr')

        logits = Flatten(Global_max_pool(CAM, padding='SAME', name=name + 'GMP'))

        #logits = Flatten(Global_max_pool(CAM, padding='SAME', name=name + 'GMP'))

        #GAP, GAP_size= Flatten(Global_avg_pool(features, padding='SAME', name=name + 'GAP'))
        #GAP, GAP_size = Global_avg_pool(features)

        #logits = Dense(GAP, GAP_size, self.num_class, name=name+'weights', reuse=reuse)

        #w = tf.trainable_variables(scope=name+'weights')
        #feature_shape= features.get_shape()
        #class_activation_maps= tf.matmul(tf.reshape(features,[feature_shape[:3].num_elements(), feature_shape[3]]), w[0])
        #class_activation_maps= tf.reshape(class_activation_maps,[feature_shape[0], feature_shape[1], feature_shape[2], self.num_class])

        prob = tf.nn.softmax(logits[0], 1)
        if train:
            target = get_2labels_batch(target)

            focal_factor = tf.multiply(tf.cast(tf.boolean_mask(
                np.matmul(np.ones([self.batch_size, 1]), self.a), tf.cast(target, tf.bool)), tf.float32),
                                   tf.pow(1-tf.boolean_mask(prob, tf.cast(target, tf.bool)), self.r))

            Loss = tf.reduce_mean(tf.multiply(focal_factor, tf.nn.softmax_cross_entropy_with_logits(
                logits=logits[0], labels=target)))

            correct_pred = tf.equal(tf.argmax(logits[0], 1), tf.argmax(target, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            class_acc = class_accuracy(target, logits[0], self.num_class, self.batch_size)

            o1, o2, o3, o4, o5 = Loss, class_activation_maps, accuracy, class_acc, logits[0]
        else:
            o1, o2, o3, o4, o5 = [], class_activation_maps, [], [], logits[0]

        return o1, o2, o3, o4, o5
