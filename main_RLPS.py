import copy
import math
import os
import random

import numpy as np
from tqdm import tqdm

import tensorflow.compat.v1 as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import *
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from model import baseModel

eps = 1e-5


def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class RLe2e:

    def __init__(self, init_alpha, data_dir):
        self.data_dir = data_dir
        self.policy = self.build_model()

        # test_action is used to check whether the effect of the selector is consistent with the ideal effect
        noise_data = np.load(os.path.join(data_dir, 'data/noise_train_data.npz'))
        clean_data = np.load(os.path.join(data_dir, 'data/clean_test_data.npz'))
        self.train_data, self.train_label_, self.train_face_label = noise_data['data'], noise_data['label'], noise_data[
            'face_label']
        self.test_data, self.test_label = clean_data['data'], clean_data['label']
        self.test_action_data, self.test_action_label = noise_data['data'], noise_data['face_label']
        self.test_action_label[self.test_action_label <= 1] = 1
        self.test_action_label[self.test_action_label > 1] = 0
        # del noise_data
        del clean_data
        self.train_label_ -= 1
        self.test_label -= 1
        self.train_label = to_categorical(self.train_label_)
        self.test_label = to_categorical(self.test_label)
        self.test_action_label = to_categorical(self.test_action_label)

        self.num_data = self.train_label.shape[0]
        self.num_test_action_data = self.test_action_label.shape[0]
        self.num_labels = self.train_label.shape[1]
        self.num_each_label = np.sum(self.train_label, axis=0).astype(int)
        print("Distribution over labels: {}".format(self.num_each_label))

        self.classifier = load_model(os.path.join(data_dir, "shallow_cnn_noise.h5"))
        #         self.classifier = baseModel(self.data_shape)

        dense = self.classifier.layers[12]
        self.img_emb_dim = dense.output_shape[-1]

        self.data_shape = self.train_data[0].shape
        self.get_img_emb = tf.keras.backend.function([self.classifier.input], [dense.output])
        self.state_feat = np.zeros((self.num_data, self.img_emb_dim))
        self.test_emb = np.zeros((self.num_test_action_data, self.img_emb_dim))
        batch_size = 128
        lo, hi = 0, 0
        while lo < self.num_data:
            if lo + batch_size < self.num_data:
                hi = lo + batch_size
            else:
                hi = self.num_data
            label_emb = self.get_img_emb([self.train_data[lo: hi]])[0]
            self.state_feat[lo: hi] = label_emb
            lo += batch_size

        lo, hi = 0, 0
        while lo < self.num_test_action_data:
            if lo + batch_size < self.num_test_action_data:
                hi = lo + batch_size
            else:
                hi = self.num_test_action_data
            label_emb = self.get_img_emb([self.test_action_data[lo: hi]])[0]
            self.test_emb[lo: hi] = label_emb
            lo += batch_size

        self.score = np.zeros(self.num_data, dtype=np.float32)
        score = self.classifier.predict(self.train_data)

        self.label = np.argmax(self.train_label, axis=-1)
        self.pred_label = np.argmax(score, axis=-1)
        f1 = f1_score(self.label, self.pred_label, average='macro')
        for i in range(self.num_data):
            label = self.label[i]
            self.score[i] = score[i][label] - np.mean(score[i])

        self.tot_avg_score = np.mean(self.score)
        total_score = np.sum(self.score)
        self.total_score = []
        self.total_f1 = []
        self.total_score.append(total_score)
        self.total_f1.append(f1)
        print('avg_score:{} total_score:{}'.format(self.tot_avg_score, total_score))
        self.sample_time = 1

        self.best_ratio = 0.0
        self.best_f1 = 0.0

        self.final_selected_img = []
        self.selected_img_set = []
        self.selected_img = []
        self.rewards = []
        self.max_avg_score = self.tot_avg_score
        self.alpha = init_alpha / 16

        self.best_acc = 0

    def build_model(self):
        '''
        policy model
        '''
        inputs = Input(shape=(128,))
        dense1 = Dense(64, activation='relu')(inputs)
        dense2 = Dense(32, activation='relu')(dense1)
        dense3 = Dense(16, activation='relu')(dense2)
        dense4 = Dense(8, activation='relu')(dense3)
        output = Dense(2, activation='sigmoid')(dense4)

        model = Model(inputs=inputs, outputs=output)
        return model

    def update_score(self):
        best_path = os.path.join(self.data_dir, "shallow_cnn_noise_e2e.h5")
        self.classifier = load_model(best_path)
        score = self.classifier.predict(self.train_data)
        self.pred_label = np.argmax(score, axis=-1)
        dense = self.classifier.layers[12]
        self.get_img_emb = tf.keras.backend.function([self.classifier.input], [dense.output])
        batch_size = 128
        lo, hi = 0, 0
        while lo < self.num_data:
            if lo + batch_size < self.num_data:
                hi = lo + batch_size
            else:
                hi = self.num_data
            label_emb = self.get_img_emb([self.train_data[lo: hi]])[0]
            self.state_feat[lo: hi] = label_emb
            lo += batch_size

        lo, hi = 0, 0
        while lo < self.num_test_action_data:
            if lo + batch_size < self.num_test_action_data:
                hi = lo + batch_size
            else:
                hi = self.num_test_action_data
            label_emb = self.get_img_emb([self.test_action_data[lo: hi]])[0]
            self.test_emb[lo: hi] = label_emb
            lo += batch_size

    def get_action(self):
        action = np.zeros(self.num_data)
        for i in range(self.num_data):
            if self.pred_label[i] == self.label[i]:
                action[i] = 1
            else:
                action[i] = 0
        return action

    def decide_action(self):
        prob = self.policy.predict(self.state_feat)[:, 1]
        action = np.zeros(self.num_data)
        for i in range(self.num_data):
            if prob[i] > 0.49:
                action[i] = 1
            else:
                action[i] = 0
        return action

    def update_state_feat(self, label, k, before_selection):
        if before_selection:
            self.state_feat[:self.img_emb_dim] = self.train_emb_bags[label][k]
        else:
            num_selected = len(self.selected_img)
            tmp_feat = self.state_feat[self.img_emb_dim:]
            tmp_feat = (num_selected * tmp_feat + self.train_emb_bags[label][k]) / (num_selected + 1)
            self.state_feat[self.img_emb_dim:] = tmp_feat

    def selection(self, epochs=100):
        last_update_epoch = 0

        self.action_label = np.zeros(self.num_data)
        self.policy.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.01), metrics=['accuracy'])
        for epoch in range(epochs):
            print("Epoch {}".format(epoch))
            if epoch > 0:
                self.update_score()
            rl_score = 0
            total_face = 0
            half_face = 0
            none_face = 0
            shuffle_idx = list(range(self.num_data))
            random.shuffle(shuffle_idx)
            self.final_selected_img.clear()
            self.rewards.clear()
            for j in range(self.sample_time):
                self.selected_img.clear()
                self.action_label = self.get_action()
                self.action_label = to_categorical(self.action_label, 2)
                distribute = np.sum(self.action_label, axis=0).astype(int)
                print(f"distribute: {distribute}")

                policy_name = 'pre-train-policy-new.h5'
                callbacks = [
                    EarlyStopping(monitor='val_loss', patience=5, verbose=0),
                    ModelCheckpoint(os.path.join(self.data_dir, policy_name), monitor='val_accuracy', mode='max',
                                    save_best_only=True)]
                print('policy_train...')
                self.policy.fit(x=self.state_feat, y=self.action_label, batch_size=16, epochs=10,
                                validation_data=(self.test_emb, self.test_action_label), verbose=2, callbacks=callbacks,
                                class_weight='auto')

            self.policy = load_model(os.path.join(self.data_dir, policy_name))
            self.final_action = self.decide_action()
            for k in shuffle_idx:
                face_label = self.train_face_label[k]
                if self.final_action[k] == 1:
                    self.final_selected_img.append(k)
                    if face_label == 0:
                        total_face += 1
                    elif face_label == 1:
                        half_face += 1
                    else:
                        none_face += 1
                    rl_score += self.score[k]

            if self.final_selected_img:
                rl_score /= len(self.final_selected_img)
                num_selected_img = len(self.final_selected_img)
                ratio = num_selected_img / sum(self.num_each_label)
                print("Selected: {} images ratio:{}".format(num_selected_img, ratio))

                train_data = np.zeros((num_selected_img, *self.data_shape))
                train_label = np.zeros(num_selected_img)
                for k, id_ in enumerate(self.final_selected_img):
                    train_data[k] = self.train_data[id_]
                    train_label[k] = self.train_label_[id_]
                opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999)
                model_path = os.path.join(self.data_dir, "../shallow_cnn_noise_e2e.h5")
                # model_path = os.path.join(self.data_dir, "shallow_cnn_noise_e2e_new.h5")
                callbacks = [
                    EarlyStopping(monitor='val_loss', patience=5, verbose=0),
                    ModelCheckpoint(model_path, monitor='val_accuracy', mode='max', save_best_only=True)]
                self.classifier.compile(optimizer=opt,
                                        loss='categorical_crossentropy',
                                        metrics=['accuracy'])

                train_label = to_categorical(train_label, 7)
                print('classifier training')
                self.classifier.fit(x=train_data, y=train_label,
                                    batch_size=16,
                                    validation_data=(self.test_data, self.test_label),
                                    epochs=100,
                                    verbose=2,
                                    callbacks=callbacks)
            else:
                rl_score = 1
                ratio = 0
            print("score: {:.4f}\tratio:{:.4f}".format(rl_score, ratio))
            score = self.classifier.predict(self.train_data)
            self.pred_label = np.argmax(score, axis=-1)
            for i in range(self.num_data):
                label = self.label[i]
                self.score[i] = score[i][label] - np.mean(score[i])

            self.tot_avg_score = np.mean(self.score)
            total_score = np.sum(self.score)
            self.total_score.append(total_score)
            f1 = f1_score(self.label, self.pred_label, average='macro')
            print(f'total_reward:{total_score}')
            self.total_f1.append(f1)

            if epoch - last_update_epoch > 16:
                break
            self.update_score()
            self.classifier = load_model(model_path)
            pred = self.classifier.predict(self.test_data, batch_size=32, verbose=1)
            y_true = np.argmax(self.test_label, axis=-1)
            y_pred = np.argmax(pred, axis=-1)
            val_acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='macro')

            if self.best_acc <= val_acc:
                print("Weight updated!")
                last_update_epoch = epoch
                self.best_acc = val_acc
                self.best_f1 = f1
                self.best_ratio = ratio
                self.classifier.save('shallow_cnn_nise_best_e2e.h5')
                self.policy.save('policy_noise_e2e_best.h5')
            print("val_acc: {:.4f}\tmax val_acc: {:.4f}\t f1: {:.4f}\t ratio:{:.4f}".format(val_acc, self.best_acc,
                                                                                            self.best_f1,
                                                                                            self.best_ratio))


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    tf.keras.backend.get_session(session)

    data_dir = '/home/lihuadong/workspace/RL/'
    alpha = 0.003
    model = RLe2e(alpha, data_dir)
    model.selection()
