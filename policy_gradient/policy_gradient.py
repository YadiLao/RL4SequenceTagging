"""
Created on 2018-06-13
class: RL4SRD
@author: fengyue
"""

# !/usr/bin/python
# -*- coding:utf-8 -*-

import os
import math
from datetime import date
import time
import random
import numpy as np
from sklearn.metrics import f1_score

from util import utils
from pg.pg_args import *
from pg.pg_net import PGNet
from util.feature_extractor import FeatureExtractor


def load_data():
    data = utils.load_data()
    raw_sen_list = data[0]
    sen_list, pos_list, chunk_list, ner_list = data[1], data[2], data[3], data[4]
    word_embed, word_idx_map = data[5], data[6]

    x_train, y_train = sen_list[0], ner_list[0]
    x_dev, y_dev = sen_list[1], ner_list[1]
    x_test, y_test = sen_list[2], ner_list[2]
    print('Size: train={},valid={}, test={}'.format(len(x_train), len(x_dev), len(x_test)))

    return x_train, y_train, x_dev, y_dev, x_test, y_test, word_embed, word_idx_map


class RL4NER(object):
    """docstring for RL4SRD"""
    def __init__(self,  word_embedding, is_train=True):
        super(RL4NER, self).__init__()
        self.config = Parameter().config
        self.learning_rate = self.config['lr']
        self.gamma = 0.2
        self.sen_list = None
        self.label_list = None
        self.id_list = None
        self.pg_model = PGNet(pretrain_word_embed=word_embed.astype(np.float32))
        self.feature_extractor = FeatureExtractor()
        self.n_gram = self.config['n_gram']
        self.label_num = self.config['label_num']
        print(self.config)

        # self.fileResult = open('result_' + sys.argv[1] + '.txt', 'w')

    def init_data(self, sen_list, label_list, id_list):
        self.sen_list = sen_list
        self.label_list = label_list
        self.id_list = id_list

    def label_lookup(self, label):
        """
        lookup label to matrix. e.g.
         [1,3] will become [0,1,0,1,0,0,0,0,0]
        """
        if len(label) == 0:
            label_matrix = np.array([0] * self.config['label_num'])
        else:
            label_matrix = np.zeros([len(label), self.config['label_num']])
            for i in range(len(label)):
                label_matrix[i][label[i]] = 1
            label_matrix = np.sum(label_matrix, axis=0)
        return label_matrix

    def get_discount_reward(self, index, pred_tag):
        """
        get ground truth tagging reward
        :param index: sentence index
        :param pred_tag:
        :return: flag indicates which tagged label is correct or not
                and reward list: when use gain: reward[0] is the largest v at end status
                                the same when not use gain
        """
        # assert len(pred_tag) == len(self.sen_list[index])
        # temp_tag = [int(p) + 1 for p in pred_tag]
        size = len(pred_tag)
        flag = [int(pred_tag[i] == self.label_list[index][i]) for i in range(size)]
        acc = sum(flag)*1.0/size

        if self.config['reward_type'] == 'acc':
            # reward = flag.count(1) / (len(pred_tag) * 1.0)
            r = [1.0 if flag[i] == 1 else 0.0 for i in range(size)]
            reward = [0] * size
            running_add = 0
            for t in reversed(range(size)):
                running_add = running_add * self.gamma + r[t]
                reward[t] = running_add

        elif self.config['reward_type'] == 'f1':
            """f1 reward are the same for all time step, not define clearly jet"""
            reward = [0] * size
            f1 = [0] * size
            running_add = 0
            for i in range(size):
                s = f1_score(self.label_list[index][:i + 1], pred_tag[:i + 1], average=None)
                f1[i] = sum(s) / len(s)
            for t in reversed(range(size)):
                running_add = running_add * self.gamma + flag[t]
                reward[t] = running_add
        else:
            print('Error: No such reward type!')
            return None

        return acc, reward, flag

    def sample_action(self, sen_id, tag_label, cur_word_index, mode='train',dirichlet_param=0.1):
        """
        :return:
        """
        sen = self.sen_list[sen_id]
        keep_prob = self.config['keep_prob'] if mode == 'train' else 1.0
        if cur_word_index == 0:
            fw = [0] * self.n_gram + [sen[cur_word_index]]
            bw = sen[:self.n_gram + 1] + [0] * (self.n_gram - len(sen) + 1)
        else:
            fw = [0] * (self.n_gram - cur_word_index) + sen[:cur_word_index + 1][-(1 + self.n_gram):]
            bw = sen[cur_word_index:cur_word_index + self.n_gram + 1] + [0] * (self.n_gram - len(sen[cur_word_index:]) + 1)

        label = self.label_lookup(tag_label)
        extra_fea = self.feature_extractor.extract_raw_feature(
            sen_id=sen_id, cur_word_id=0, mode=mode, n_gram=self.n_gram)
        # print('fw={}, bw={}, label={}, fea={}'.format(fw, bw, label, np.array(extra_fea).shape))
        pred_prob = self.pg_model.get_policy([fw], [bw], [label], [extra_fea], keep_prob)

        # tmp = random.random()
        # sum_p = 0
        # action = None
        # for id_p, p in enumerate(pred_prob[0]):
        #     sum_p += p
        #     if tmp < sum_p:
        #         action = id_p
        #         break

        tmp = random.random()
        if tmp < 0.8:
            r = random.random()
            sum_p = 0
            action = None
            for id_p, p in enumerate(pred_prob[0]):
                sum_p += p
                if r < sum_p:
                    action = id_p
                    break
        else:
            action = random.randint(0, self.label_num-1)

        return action, pred_prob[0]

    def take_action(self, sen_id, tag_label, cur_word_index, mode='train'):
        """
        :return:
        """
        sen = self.sen_list[sen_id]
        keep_prob = self.config['keep_prob'] if mode == 'train' else 1.0
        if cur_word_index == 0:
            fw = [0] * self.n_gram + [sen[cur_word_index]]
            bw = sen[:self.n_gram + 1] + [0] * (self.n_gram - len(sen) + 1)
        else:
            fw = [0] * (self.n_gram - cur_word_index) + sen[:cur_word_index + 1][-(1 + self.n_gram):]
            bw = sen[cur_word_index:cur_word_index + self.n_gram + 1] + [0] * (
            self.n_gram - len(sen[cur_word_index:]) + 1)

        label = self.label_lookup(tag_label)
        extra_fea = self.feature_extractor.extract_raw_feature(
            sen_id=sen_id, cur_word_id=0, mode=mode, n_gram=self.n_gram)
        pred_prob = self.pg_model.get_policy([fw], [bw], [label], [extra_fea], keep_prob)

        action = np.argmax(pred_prob[0])

        return action

    def update_model(self, sen_id, selected_label, score, mode='train'):
        """
        update pgnet for several episode of single sentence
        """
        sen = self.sen_list[sen_id]
        fw_batch = []
        bw_batch = []
        tag_label_batch = []
        extra_fea_batch = []
        action_batch = []
        Gt_batch = []

        for e_i in range(len(selected_label)):
            for id_i in range(len(sen)):

                if id_i == 0:
                    fw = [0] * self.n_gram + [sen[id_i]]
                    bw = sen[:self.n_gram + 1] + [0] * (self.n_gram - len(sen) + 1)

                else:
                    fw = [0] * (self.n_gram - id_i) + sen[:id_i + 1][-(1 + self.n_gram):]
                    bw = sen[id_i:id_i + self.n_gram + 1] + [0] * (
                        self.n_gram - len(sen[id_i:]) + 1)

                tag_label = self.label_lookup(selected_label[e_i][:id_i])
                extra_fea = self.feature_extractor.extract_raw_feature(
                    sen_id=sen_id, cur_word_id=0, mode=mode, n_gram=self.n_gram)

                action = [0]*self.config['label_num']
                action[selected_label[e_i][id_i]] = 1
                Gt = [score[e_i][id_i]]

                fw_batch.append(fw)
                bw_batch.append(bw)
                tag_label_batch.append(tag_label)
                extra_fea_batch.append(extra_fea)
                action_batch.append(action)
                Gt_batch.append(Gt)

        # print(
        #     sen_id, len(selected_label), np.array(fw_batch).shape, np.array(action_batch).shape,
        #     np.array(Gt_batch).shape)
        Gt_batch = np.array(Gt_batch).reshape([1, len(Gt_batch)])
        feed_dict = [
            fw_batch, bw_batch, tag_label_batch, extra_fea_batch, action_batch, Gt_batch]
        loss, prob, p_loss = self.pg_model.update_model(feed_dict)

        return loss

def run():
    sample_episode = 50
    x_train, y_train, x_dev, y_dev, x_test, y_test, word_embed, word_idx_map = load_data()
    rl4ner = RL4NER(word_embed.astype(np.float32), is_train=False)


    x_train_small = []
    y_train_small =[]
    x_test_small = []
    y_test_small = []
    num = 0
    for k, x in enumerate(x_train):
        if len(x) <8:
            x_train_small.append(x)
            y_train_small.append(y_train[k])
            num += 1
        if num > 600:
            break

    num = 0
    for k, x in enumerate(x_test):
        if len(x) <8:
            x_test_small.append(x)
            y_test_small.append(y_test[k])
        if num > 200:
            break

    x_train = x_train_small
    y_train = y_train_small
    x_test = x_test_small
    y_test = y_test_small
    print('SMALL: train={}, test={}'.format(len(x_train), len(x_test)))

    epoch = 0
    num_wait = rl4ner.config['wait_epoch']
    best_f1 = 0

    is_develop = True
    if is_develop:
        path = os.path.curdir
    else:
        path = '/home/seagate/deadline_log/'

    time_stamp = str(date.today())
    model_save_dir = os.path.abspath(os.path.join(path, 'test-runs', time_stamp))
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    print('>>> save model to {}'.format(model_save_dir))
    valid_file = open(model_save_dir + '/valid_result.txt', 'w')
    train_file = open(model_save_dir + '/train_result.txt', 'w')

    while num_wait > 0:
        train_acc = []
        train_f1 = []
        train_loss  = []
        t1 = time.time()
        # train
        rl4ner.init_data(x_train, y_train, range(len(x_train)))
        for sen_id in range(len(x_train)):
            batch_selected_label = []
            batch_gain = []
            score = 0
            for i in range(sample_episode):
                selected_label = []
                while len(selected_label) < len(x_train[sen_id]):
                    action, prob = rl4ner.sample_action(sen_id, selected_label, len(selected_label), mode='train')
                    selected_label.append(action)

                if selected_label not in batch_selected_label:
                    acc, discount_reward, flag = rl4ner.get_discount_reward(sen_id, selected_label)
                    score += acc

                    if i < 5 and sen_id<5:
                        print('sen={}, e={}, score={}, action={}, y={}, flag={}, r ={}'.format(
                            sen_id, i, acc, selected_label, y_train[sen_id],
                            flag, discount_reward))

                    batch_selected_label.append(selected_label)
                    batch_gain.append(discount_reward)

            train_acc.append(score/len(batch_selected_label))
            train_f1.append(score/len(batch_selected_label))
            # if sen_id < 5:
            #     print('score={}'.format(score/len(batch_selected_label)))

            # optimizer
            loss = rl4ner.update_model(
                sen_id=sen_id, selected_label=batch_selected_label, score=batch_gain, mode='train')
            train_loss.append(loss)

        avg_train_f1 = sum(train_f1)/len(train_f1)
        avg_train_acc = sum(train_acc)/len(train_acc)
        avg_train_loss = sum(train_loss)/len(train_loss)
                
        # test
        if epoch>40:
            rl4ner.init_data(x_test, y_test, range(len(x_test)))
            pred_test = []
            test_acc = []
            for sen_id in range(len(x_test)):
                selected_label = []
                while len(selected_label) < len(x_test[sen_id]):
                    action = rl4ner.take_action(sen_id, selected_label, len(selected_label), mode='test')
                    selected_label.append(action)

                acc, score, flag = rl4ner.get_discount_reward(sen_id, selected_label)
                pred_test.append(selected_label)
                test_acc.append(acc)

                if sen_id<5:
                    print('sen_id={}, acc={},action={}, y={}, flag={}'.format(
                        sen_id, acc, selected_label, y_test[sen_id], flag))

            avg_test_acc = sum(test_acc)/len(test_acc)
            # print('>>> Test: epoch={}, acc={}'.format(epoch, avg_test_f1))

            if avg_test_acc > best_f1:
                best_f1 = avg_test_acc
                num_wait = rl4ner.config['wait_epoch']
                rl4ner.pg_model._save(model_dir=model_save_dir+'/model')
                print('Save model at EPOCH {}, best={}'.format(epoch, best_f1))
            else:
                num_wait -= 1

            print('epoch {}: train acc={}, loss={}, test acc={}, best_acc={}, time={}, '
                  'wait={}'.format(epoch, round(avg_train_acc,4), round(avg_train_loss, 4),
                                   round(avg_test_acc,4), round(best_f1,4), time.time()-t1,
                                   num_wait))
            valid_file.write(' '.join([str(epoch), str(round(avg_test_acc, 4))]) + '\n')
            valid_file.flush()
        else:
            print('epoch {}: train acc={}, loss={}, time={}'.format(
                epoch, round(avg_train_acc,4), round(avg_train_loss, 4), time.time()-t1))

        epoch += 1
        rl4ner.pg_model.current_epoch = epoch

        train_file.write(' '.join([str(epoch), str(round(avg_train_acc, 4)),
                                   str(round(avg_train_loss, 4))]) + '\n')
        train_file.flush()


if __name__ == '__main__':
    run()
    






