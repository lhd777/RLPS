import copy
import math
import os
import random

import numpy as np

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical
import tensorflow.compat.v1 as tf

from model import *


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class RLPre:

    def __init__(self, init_alpha, data_dir):
        self.data_dir = data_dir

        # load training data // clean means there is no extra noise.
        clean_data = np.load(os.path.join(data_dir, 'data/clean_train_data.npz'))
        train_data, train_label = clean_data['data'], clean_data['label']
        # noise_data = np.load(os.path.join(data_dir, 'data/noise_train_data.npz'))
        # train_data, train_label, train_face_label = noise_data['data'], noise_data['label'], noise_data['face_label']

        del clean_data

        # label is 1-7.
        train_label -= 1
        train_label = to_categorical(train_label)
        self.num_labels = train_label.shape[1]
        self.num_each_label = np.sum(train_label, axis=0).astype(int)
        print("Distribution over labels: {}".format(self.num_each_label))

        # Sort the original data according to tags
        self.data_shape = train_data[0].shape
        self.train_bags = []
        cnt_each_label = [0] * self.num_labels
        for i in range(self.num_labels):
            self.train_bags.append(np.zeros((self.num_each_label[i], *self.data_shape)))
        for i in range(train_data.shape[0]):
            label = np.argmax(train_label[i])
            self.train_bags[label][cnt_each_label[label]] = train_data[i]
            cnt_each_label[label] += 1
        del train_data

        # Choose whether the model should be trained from scratch or after pre-training
        self.classifier = load_model(os.path.join(data_dir, "shallow_cnn_filter.h5"))
        # self.classifier = baseModel(self.data_shape)

        # Select the last layer of the classifier as the input of the selector
        dense = self.classifier.layers[14]
        self.img_emb_dim = dense.output_shape[-1]
        print("The dimension of the last layer in the classifier is {}.".format(self.img_emb_dim))

        # convert the data into embed
        self.get_img_emb = tf.keras.backend.function([self.classifier.input], [dense.output])
        self.train_emb_bags = []
        batch_size = 128
        for i in range(self.num_labels):
            size = self.train_bags[i].shape[0]
            lo, hi = 0, 0
            self.train_emb_bags.append(np.zeros((size, self.img_emb_dim)))
            while lo < size:
                if lo + batch_size < size:
                    hi = lo + batch_size
                else:
                    hi = size
                label_emb = self.get_img_emb([self.train_bags[i][lo: hi]])[0]
                self.train_emb_bags[-1][lo: hi] = label_emb
                lo += batch_size

        # Calculate the score for each data
        self.score = []
        tot_score = 0
        for i in range(self.num_labels):
            pred = np.log(self.classifier.predict(self.train_bags[i])[:, i])
            tot_score += np.sum(pred)
            self.score.append(pred)
        self.tot_avg_score = tot_score / np.sum(self.num_each_label)
        self.sample_time = 3

        # Initialize state feat, w, and b
        self.state_feat = np.zeros((2 * self.img_emb_dim,))
        # self.W = np.random.random((2 * self.img_emb_dim, )) * 0.5 - 0.25
        self.W = np.zeros((2 * self.img_emb_dim,))
        self.W_ = np.zeros((2 * self.img_emb_dim,))
        self.best_W = np.zeros((2 * self.img_emb_dim,))
        self.b = 0.0  # todo
        self.b_ = 0.0
        self.best_ratio = 0.0

        self.final_selected_img = []
        self.selected_img_set = []
        self.selected_img = []
        self.rewards = []
        self.best_score = -100

        self.max_avg_score = self.tot_avg_score
        self.alpha = init_alpha / 16

    # Monte Carlo simulation
    def get_action(self):
        prob = sigmoid(np.sum(self.W_ * self.state_feat))
        rand_num = np.random.random()
        if rand_num < prob:
            return 1
        else:
            return 0

    # The selector decides whether to keep
    def decide_action(self):
        prob = sigmoid(np.sum(self.W_ * self.state_feat))
        if prob >= 0.5:
            return 1
        else:
            return 0

    # Initialize state feat
    def init_state_feat(self):
        self.state_feat.fill(0)

    # Update state feat
    def update_state_feat(self, label, k, before_selection):
        if before_selection:
            self.state_feat[:self.img_emb_dim] = self.train_emb_bags[label][k]
        else:
            num_selected = len(self.selected_img)
            tmp_feat = self.state_feat[self.img_emb_dim:]
            tmp_feat = (num_selected * tmp_feat + self.train_emb_bags[label][k]) / (num_selected + 1)
            self.state_feat[self.img_emb_dim:] = tmp_feat

    # Training
    def selection(self, epochs=500):
        self.W_ = self.W
        last_update_epoch = 0
        for epoch in range(epochs):
            rl_score = 0
            shuffle_labels = list(range(self.num_labels))
            random.shuffle(shuffle_labels)
            self.final_selected_img.clear()
            for label in shuffle_labels:
                shuffle_idx = list(range(self.num_each_label[label]))
                random.shuffle(shuffle_idx)
                self.selected_img_set.clear()
                self.rewards.clear()
                for j in range(self.sample_time):
                    tmp_reward = 0
                    self.selected_img.clear()
                    self.init_state_feat()
                    for k in shuffle_idx:
                        self.update_state_feat(label, k, True)
                        action = self.get_action()
                        if action == 1:
                            self.selected_img.append(k)
                            tmp_reward += self.score[label][k]
                            self.update_state_feat(label, k, False)
                    if len(self.selected_img) == 0:
                        tmp_reward = self.tot_avg_score
                    else:
                        tmp_reward /= len(self.selected_img)
                    tmp_set = copy.deepcopy(self.selected_img)
                    self.selected_img_set.append(tmp_set)
                    self.rewards.append(tmp_reward)
                avg_reward = np.mean(self.rewards)

                for j in range(self.sample_time):
                    self.selected_img.clear()
                    self.init_state_feat()
                    selected_img = self.selected_img_set[j]
                    l = 0
                    for k in shuffle_idx:
                        self.update_state_feat(label, k, True)
                        tmp_f = np.sum(self.state_feat * self.W_)
                        tmp_f = sigmoid(tmp_f)
                        num_selected_img = len(selected_img)

                        # Parameter update
                        if len(selected_img) != 0 and l < num_selected_img and k == selected_img[l]:
                            self.W += self.alpha * (self.rewards[j] - avg_reward) * (1 - tmp_f) * self.state_feat
                            self.update_state_feat(label, k, False)
                            l += 1
                        else:
                            self.W -= self.alpha * (self.rewards[j] - avg_reward) * tmp_f * self.state_feat

                self.init_state_feat()

                for k in shuffle_idx:
                    self.update_state_feat(label, k, True)
                    action = self.decide_action()
                    if action == 1:
                        self.final_selected_img.append((label, k))
                        rl_score += self.score[label][k]
                        self.update_state_feat(label, k, False)
            # todo alpha
            if self.final_selected_img:
                rl_score /= len(self.final_selected_img)
                ratio = len(self.final_selected_img) / sum(self.num_each_label)
                if rl_score > self.best_score and ratio > 0.8:
                    print("Weight updated!")
                    self.best_score = rl_score
                    self.best_W = self.W_
                    self.best_ratio = ratio
                    np.save(os.path.join(self.data_dir, 'pretrained_selector.npy'), self.best_W)
                    last_update_epoch = epoch
            else:
                print("No images selected!")
                rl_score = 1
                ratio = 0
            print(
                "Epoch {}\nscore: {:.4f}\tratio:{:.4f}\tbest score:{:.4f}\tbest ratio:{:.4f}\t".format(epoch, rl_score,
                                                                                                       ratio,
                                                                                                       self.best_score,
                                                                                                       self.best_ratio))
            self.W_ = self.W

            if epoch - last_update_epoch > 32:
                break
        print("best score:{:.4f}\tbest ratio:{:.4f}\t".format(self.best_score, self.best_ratio))
        print("best_total_ratio:{:.4f}\tbest_half_ratio:{:.4f}\t best_none_ratio\t:{:.4f}".format(self.best_total_ratio,
                                                                                                  self.best_half_ratio,
                                                                                                  self.best_none_ratio))


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    tf.keras.backend.get_session(session)

    data_dir = '/home/lihuadong/workspace/RL/'
    alpha = 0.003
    model = RLPre(alpha, data_dir)
    model.selection()
