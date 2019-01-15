# !/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 2018/12/23
class: test for mmtag
@author: Lao Yadi
"""
import sys
import os
sys.path.append('../')
import gc
import math
import json
from datetime import date
import time
import logging
from treelib import Tree
from collections import defaultdict
from sklearn.metrics import f1_score

import torch.optim as optim
from cnn_lstm.torch_pvnet import CNN_LSTM_PVnet
from cnn_lstm.eval import *
from cnn_lstm.args import *

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

np.set_printoptions(precision=4, suppress=True, linewidth=300, threshold=np.inf)


class Node(object):
    """A node in the MCTS tree.

    Each node keeps track of its own value Q, prior probability P, value evaluated by value network
    and its sentence index
    """

    def __init__(self):
        self.num = 0
        self.Q = 0.0
        self.p = 0.0
        self.label = []  # record the tagging labels
        self.value = None
        self.index = None


class SearchTree(object):
    """An implementation of Monte Carlo Tree Search for batch training"""

    def __init__(self, config, pretrain_char_embedding=None,
                 pretrain_word_embedding=None, is_train=True, seed=0, path=None):
        """
        pv_model: policy_value net to guild the tree search
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        search_depth: num of step to look ahead
        n_playout: num of search time
        raw_sen: use to access raw feature with respect to specific n-gram
        is_train: change mode between train, dev and test
        self.counts: record the n_playout for current node in search tree
        """
        self.config = config
        self.char_alphabet = dict(
            json.load(open(BASE_DIR + DATA_SET + '/mapping/chars_vocab', 'r')))
        self.word_alphabet = dict(
            json.load(open(BASE_DIR + DATA_SET + '/mapping/word_vocab', 'r')))
        if path is None:
            self.pv_model = CNN_LSTM_PVnet(
                self.config, self.char_alphabet, self.word_alphabet,
                pretrain_char_embedding, pretrain_word_embedding)

            self.pv_model_param = []
            for param in self.pv_model.parameters():
                print(type(param.data), param.size())
                self.pv_model_param.append(param)
            print(self.pv_model)
            self.optimizer = optim.Adagrad(
                self.pv_model.parameters(), lr=self.config.tree_args['lr'])
        else:
            print('restore model from {}'.format(path + '/model'))
            checkpoint = torch.load(path + '/model')
            self.pv_model = CNN_LSTM_PVnet(
                self.config, self.char_alphabet, self.word_alphabet,
                pretrain_char_embedding, pretrain_word_embedding)

            self.optimizer = optim.Adagrad(
                self.pv_model.parameters(), lr=self.config.tree_args['lr'])
            self.pv_model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.search_depth = config.tree_args['look_ahead_depth']
        self.n_playout = config.tree_args['n_playout']
        self.c_puct = config.tree_args['c_puct']

        # self.feature_extractor = FeatureExtractor()
        # self.n_gram = config.tree_args['n_gram']

        self.is_train = is_train
        self.learning_rate = config.tree_args['lr']
        # self.lr_multiplier = 1.0
        self.temp = 1.0
        self.kl_targ = 0.02

        self.trees = None
        self.sen_list = []
        self.label_list = []
        self.char_list = []
        self.pos_list = []
        self.batch_size = 0
        self.sen_id_map = None

        self.word_seq_tensor, self.word_seq_lengths, self.word_seq_recover = None, None, None
        self.char_seq_tensor, self.char_seq_lengths, self.char_seq_recover = None, None, None
        self.label_seq_tensor, self.mask = None, None
        self.feature_seq_tensor = None

        self.counts = None
        self.reward_list = []
        self.seq_data = None

        self.label2id = dict(json.load(open(BASE_DIR + DATA_SET + '/mapping/ner_vocab', 'r')))

        # self.prefix2id = dict(json.load(open(BASE_DIR+DATA_SET+'/mapping/char_vocab', 'r')))

    def label_lookup(self, label):
        """
        lookup label to matrix. e.g.
         [1,3] will become [[0,1,0,0,0,0,0,0,0], [0,0,0,1,0,0,0,0,0]]
        """
        if len(label) == 0:
            label_matrix = [0.0] * self.config.data_args['label_num']
        else:
            label_matrix = [0.0] * self.config.data_args['label_num']
            label_matrix[label[-1]] = 1.0
        return list(label_matrix)

    def init_root_list(self, examples, id_list=None, mode='train'):
        """
        init batch tree list
        :param sen: training data
        :param label: ground truth label
        :param id_list: id list is used to map the raw feature extracted by raw_sen
        """
        if_train = True if mode == 'train' else True
        self.batch_size = len(examples)
        self.trees = [Tree() for _ in range(self.batch_size)]

        self.sen_list = [e.sen for e in examples]
        self.label_list = [e.label for e in examples]
        self.char_list = [e.char for e in examples]
        self.pos_list = [e.pos for e in examples]
        self.sen_id_map = id_list

        data = batchify_with_label(
            self.sen_list, self.char_list, self.label_list,
            gpu=self.config.hyper_args['use_gpu'], if_train=if_train)
        self.word_seq_tensor, self.word_seq_lengths, self.word_seq_recover = data[0], data[1], data[2]
        self.char_seq_tensor, self.char_seq_lengths, self.char_seq_recover = data[3], data[4], data[5]
        self.label_seq_tensor, self.mask = data[6], data[7]

        #  if mode == 'valid' or mode == 'train':
        #      print('word_seq_len={}'.format(self.word_seq_lengths))
        #      print('word_seq_recover={}'.format(self.word_seq_recover))
        #      print('word_seq_tensor={}'.format(self.word_seq_tensor))
        #      print('char_seq_tensor={}'.format(self.char_seq_tensor))
        #      print('char_seq_lengths={}'.format(self.char_seq_lengths))
        #      print('char_seq_recover={}'.format(self.char_seq_recover))

        root_node_list = []
        root_init_data = [[], [], []]  # word feature, raw_feature, label

        # get hidden state
        # Attention!!!! seq_data are sorted !!!!!
        self.seq_data = self.pv_model.forward(
            word_inputs=self.word_seq_tensor, word_seq_lengths=self.word_seq_lengths,
            char_inputs=self.char_seq_tensor, char_seq_lengths=self.char_seq_lengths,
            char_seq_recover=self.char_seq_recover, word_state=None)

        # feature_tensor are sorted !!!!!
        self.feature_seq_tensor = batchify_feature(
            char_seq_tensor=self.char_seq_tensor, char_seq_recover=self.char_seq_recover,
            word_seq_recover=self.word_seq_recover, word_seq_lengths=self.word_seq_lengths,
            pos_list=self.pos_list, feature_dim=self.config.tree_args['raw_feature_dim'])

        # print('seq_data={}'.format(self.seq_data.size()))

        for i in range(self.batch_size):
            root_node = self.trees[i].create_node(identifier='root', data=Node())
            # TODO: check visit_num of init node is 0 or 1 ?????
            root_node.data.num = 0
            root_node.data.index = i
            root_node_list.append(root_node)
            idx = self.word_seq_recover[i]
            temp = self.seq_data[idx][0].data.cpu().numpy()
            root_init_data[0].append(temp)  # first raw feature
            root_init_data[1].append(self.label_lookup(root_node.data.label))
            root_init_data[2].append(self.feature_seq_tensor[idx][0].data.cpu().numpy())


        # print(np.array(root_init_data[0]).shape)

        prob_list, _ = self.pv_model.forward(
            word_state=torch.from_numpy(np.array(root_init_data[0])),
            word_feature=torch.from_numpy(np.array(root_init_data[2])),
            cur_label=torch.from_numpy(np.array(root_init_data[1]))
        )

        prob_list = prob_list.detach().numpy()
        # print(prob_list.shape)

        for i in range(self.batch_size):
            self.expand(root_node_list[i], prob_list[i])

        del root_init_data, data
        gc.collect()

        return root_node_list

    def expand(self, leaf_node, prob):
        """
        expand leaf_node in one tree each time
        """
        index = leaf_node.data.index
        depth = self.trees[index].depth(leaf_node)
        cur_word = self.sen_list[index][:depth + 1]
        tag_label = leaf_node.data.label
        for label in range(self.config.data_args['label_num']):
            """
            check len of fw doesn't matter in this version for they do the same thing
            but it matters in previous version due to the change of fw and bw
            """

            if len(cur_word) < len(self.sen_list[index]):
                new_node = Node()
                new_node.label = tag_label + [label]
                new_node.p = float(prob[label])
                new_node.index = index

                # print('new T{} node,l={}, p={}'.format(new_node.index,new_node.label,
                #                                        new_node.p))
                self.trees[index].create_node(
                    identifier=len(self.trees[index].nodes), data=new_node,
                    parent=leaf_node.identifier)
            else:
                new_node = Node()
                # new_node.fw_word = self.sen_list[index]
                # new_node.bw_word = [self.sen_list[index][-1]]
                # new_node.depth = len(fw_word) - 1
                new_node.label = tag_label + [label]
                new_node.p = float(prob[label])
                new_node.index = index
                # if label == 0:
                #    print('equal: new T{} node,l={}'.format(new_node.index,  new_node.label))
                assert len(new_node.label) == len(self.sen_list[index])
                self.trees[index].create_node(
                    identifier=len(self.trees[index].nodes), data=new_node,
                    parent=leaf_node.identifier)

    def update(self, node_list, value):
        """
        update the search node list along one playout trajectory
        :param node_list: the selected node lists
        :param value: value evaluated by value net or the final ground truth
        """
        for tmp_node in node_list:
            tmp_node.data.Q = (tmp_node.data.Q * tmp_node.data.num + value) / (tmp_node.data.num + 1)
            tmp_node.data.num += 1
            # TODO: check !!! the input value is the simulation value result ?? value in node is computed by value net!!
            # tmp_node.data.value = value

    def search(self, start_nodes, mode):
        """
        search all trees in one batch
        :param start_nodes: the current root node of mcts
        """
        # No need to set keep_prob. It can be done by call model.train() or eval()
        # keep_prob = self.config['keep_prob'] if mode == 'train' else 1.0
        # nums = [node.data.num - 1 for node in start_nodes]
        nums = [node.data.num for node in start_nodes]
        keys = [node.data.index for node in start_nodes]
        # counts store the visited time of current node
        self.counts = dict(zip(keys, nums))

        # self.counts = [node.data.num - 1 for node in start_nodes]
        # start_node_search_times = [(node, max(self.n_playout - (node.data.num - 1), 0)) for node in start_nodes]
        start_node_search_times = [(node, max(self.n_playout - node.data.num, 0)) for node in start_nodes]
        # print('counts={}'.format(self.counts))

        cur_time = 0
        """find unfinished search trees in this batch"""
        while len(start_node_search_times) > 0:
            next_search_times = []

            value_compute_data = [[], [], [], []]  # store nodes to be evaluated by value net
            expand_nodes_data = [[], [], [], []]  # store nodes to be expanded

            """loop for every search node for sentences in one batch"""
            for (node, node_search_time) in start_node_search_times:
                if cur_time + 1 < node_search_time:
                    next_search_times.append((node, node_search_time))

                search_list = [node]
                cur_node = node
                index = node.data.index

                """
                whether search to the end of sentence or simply look ahead several step ??
                """
                # while not cur_node.is_leaf() and len(search_list) < self.FLAGS.look_ahead_depth+1:
                while not cur_node.is_leaf():
                    max_score = float("-inf")
                    max_node = None
                    for child_id in cur_node.fpointer:
                        child_node = self.trees[index].get_node(child_id)
                        """ use uct to perform in-tree search"""
                        score = self.c_puct * child_node.data.p * (
                            (cur_node.data.num * 1.0) ** 0.5 / (1 + child_node.data.num))
                        # print(f'p={child_node.data.p}, score={score}, Q={child_node.data.Q}')
                        score += child_node.data.Q

                        if score > max_score:
                            max_node = child_node
                            max_score = score
                    search_list.append(max_node)
                    cur_node = max_node

                """if arrive at the last word, game end"""
                if self.trees[index].depth(cur_node) == len(self.sen_list[index]):
                    if cur_node.data.value:
                        v = cur_node.data.value
                    else:
                        """ test/valid mode : use value net to guide the search"""
                        if mode != 'train':
                            # last seq state must be indexed by seq_len
                            idx = self.word_seq_recover[index]
                            lens = self.word_seq_lengths[index] - 1
                            last_word_state = self.seq_data[idx][lens].data.cpu().numpy()
                            label_state = self.label_lookup(cur_node.data.label)
                            word_feature = self.feature_seq_tensor[idx][lens].data.cpu().numpy()
                            # single sentence
                            # batch size = 1
                            _, v = self.pv_model.forward(
                                word_state=torch.from_numpy(np.array([last_word_state])),
                                word_feature=torch.from_numpy(np.array([word_feature])),
                                cur_label=torch.from_numpy(np.array([label_state]))
                            )
                            v = v.detach().numpy()[0]

                        else:
                            """ train mode : return the ground acc"""
                            _, v = self.get_tag_acc(
                                cur_node.data.index, cur_node.data.label)
                            v = v[0]

                    self.update(search_list, v)
                else:
                    # compute network predicted acc
                    if cur_node.data.value:
                        self.update(search_list, cur_node.data.value)
                    else:
                        depth = self.trees[index].depth(cur_node)
                        index = cur_node.data.index
                        idx = self.word_seq_recover[index]
                        value_compute_data[0].append(search_list)
                        value_compute_data[1].append(
                            self.seq_data[idx][depth].data.cpu().numpy())
                        value_compute_data[2].append(
                            self.label_lookup(cur_node.data.label))
                        value_compute_data[3].append(
                            self.feature_seq_tensor[idx][depth].data.cpu().numpy())

                """expand leaf node for those are not the last word"""
                if cur_node.is_leaf() and self.trees[index].depth(cur_node) < len(self.sen_list[index]):
                    depth = self.trees[index].depth(cur_node)
                    idx = self.word_seq_recover[index]
                    expand_nodes_data[0].append(cur_node)
                    expand_nodes_data[1].append(
                        self.seq_data[idx][depth].data.cpu().numpy())
                    expand_nodes_data[2].append(
                        self.label_lookup(cur_node.data.label))
                    expand_nodes_data[3].append(
                        self.feature_seq_tensor[idx][depth].data.cpu().numpy())

                self.counts[node.data.index] += 1

            """update those first time evaluated nodes"""
            if len(value_compute_data[1]) != 0:
                _, vs = self.pv_model.forward(
                    word_state=torch.from_numpy(np.array(value_compute_data[1])),
                    cur_label=torch.from_numpy(np.array(value_compute_data[2])),
                    word_feature=torch.from_numpy(np.array(value_compute_data[3])),
                )
                vs = vs.detach().numpy()
                # print('update v.....', vs.shape)
                for i in range(len(vs)):
                    self.update(value_compute_data[0][i], float(vs[i]))

            """expand those leaf nodes"""
            if len(expand_nodes_data[1]) != 0:
                probs, _ = self.pv_model.forward(
                    word_state=torch.from_numpy(np.array(expand_nodes_data[1])),
                    cur_label=torch.from_numpy(np.array(expand_nodes_data[2])),
                    word_feature=torch.from_numpy(np.array(value_compute_data[3])),
                )
                # print('expand prob', probs.size())
                probs = probs.detach().numpy()
                for i in range(len(probs)):
                    self.expand(expand_nodes_data[0][i], probs[i])

            start_node_search_times = next_search_times
            cur_time += 1

    def take_action(self, start_node, is_random=False, temp=1e-3, dirichlet_param=0.1):
        """
        take action after n_playout
        :param start_node: current tag node
        :param is_random: set to true when training otherwise false
        :param temp: default temperature=1e-3, it is almost equivalent to choosing the move with the highest prob
        :param dirichlet_param: add Dirichlet Noise for exploration (needed for self-play training)
        :return:
        """
        max_time = -1
        prob = []
        # choose_node = None
        # select_label = None
        act_visits = []
        acts = []

        for child_id in start_node.fpointer:
            child_node = self.trees[start_node.data.index].get_node(child_id)
            visit_time = child_node.data.num
            act_visits.append((child_node, visit_time))
            acts, visits = zip(*act_visits)
            prob = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))

        if is_random:
            choose_node = np.random.choice(
                acts, p=1.0 * prob + 0 * np.random.dirichlet(dirichlet_param * np.ones(len(acts))))
        else:
            # No random
            # choose_node = np.random.choice(acts, p=prob)
            max_prob_index = np.argmax(prob)
            choose_node = acts[max_prob_index]

            # choose the max visit_time node
            # for child_id in start_node.fpointer:
            #     child_node = self.trees[start_node.data.index].get_node(child_id)
            #     prob.append(child_node.data.num * 1.0 / self.counts[child_node.data.index])
            #     if child_node.data.num > max_time:
            #         max_time = child_node.data.num
            #         select_label = child_node.data.label[-1]
            #         choose_node = child_node

        # print('take action: self.counts={}'.format(self.counts))
        # print('select label={}, prob={}, choose_node={}'.format(
        # select_label, prob, choose_node))
        self.counts[start_node.data.index] = 0
        select_label = choose_node.data.label[-1]

        return list(prob), select_label, choose_node

    def get_tag_acc(self, index, pred_tag):
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
        # print('y={}, pred_y={}'.format(self.label_list[index], pred_tag))

        if self.config.tree_args['reward_type'] == 'acc':
            # reward = flag.count(1) / (len(pred_tag) * 1.0)
            reward = [0] * size
            reward[-1] = flag[-1]
            for i in range(size - 2, -1, -1):
                reward[i] = reward[i + 1] + flag[i]
            reward = [r * 1.0 / size for r in reward]
        elif self.config.tree_args['reward_type'] == 'f1':
            """f1 reward are the same for all time step, not define clearly jet"""
            f1 = f1_score(self.label_list[index], pred_tag, average=None)
            reward = [sum(f1) / len(f1)] * size
        else:
            print('Error: No such reward type!')
            return None

        # if self.FLAGS.activate_fun == 'relu':
        #     reward = [r*1 for r in reward]

        return flag, reward

    def train(self, sen_examples, step, mode='train'):
        """
        train step for batch sentence
        :param sen: training data
        :param label: ground truth label
        :param id_list: sentence index used to map in train/dev/test file
        :param mode: train/dev/test
        """
        is_random = True if mode == 'train' else False

        """init root list"""
        search_nodes = self.init_root_list(sen_examples, mode=mode)

        # key is length of fw_word and bw_word

        batch_acc = []
        batch_reward = []
        batch_pred_label = []
        print_count = 0
        # print('----------- start search ------------')
        # word_state, label_state, p, r
        feed_dict = defaultdict(lambda: [[] for _ in range(4)])

        while len(search_nodes) > 0:
            next_search_nodes = []  # next search tree node list
            self.search(search_nodes, mode)

            # loop for all sentences
            for node in search_nodes:
                prob, select_label, choose_node = self.take_action(
                    node, is_random=is_random, temp=self.temp)
                # print('tree {},prob={}, select_label={}, node={}'.format(node.data.index,
                #  prob,select_label,choose_node))
                index = node.data.index
                cur_tag_list = node.data.label + [select_label]

                if len(cur_tag_list) < len(self.sen_list[index]):
                    next_search_nodes.append(choose_node)

                    if mode == 'train' or mode == 'valid':
                        feed = feed_dict[index]
                        depth = self.trees[index].depth(node)
                        # feed[0].append(self.seq_data[index][depth])
                        feed[1].append(self.label_lookup(cur_tag_list))
                        feed[2].append(prob)
                        feed_dict[index] = feed
                else:
                    flag, reward = self.get_tag_acc(index, cur_tag_list)
                    acc = flag.count(1) / (len(cur_tag_list) * 1.0)
                    batch_pred_label.append(cur_tag_list)
                    batch_acc.append(acc)
                    batch_reward.append(reward[0])  # reward[0] here
                    # print('T{}, flag={}, r={}'.format(index, flag, reward))

                    if step % 20 == 0 or (mode == 'valid' and step % 20 == 0):
                        if print_count < 10:
                            """raw policy prediction"""
                            idx = self.word_seq_recover[index]
                            lens = self.word_seq_lengths[idx]
                            temp = self.seq_data[idx][:lens]
                            feature = self.feature_seq_tensor[idx][:lens]
                            raw_pred = self.raw_policy_single_predict(
                                temp, feature_data=feature, mode=mode)

                            print(
                                '{}:T{} acc={}, r={}, mcts={}, raw_p={},ref={}, '
                                'flag={}'.format(mode, index, round(acc, 4),
                                                 round(reward[0], 4), cur_tag_list,
                                                 raw_pred, self.label_list[index], flag))
                            print_count += 1

                    # arrive at the end
                    if mode == 'train' or mode == 'valid':
                        # feed_dict[index][0].append(
                        #     self.seq_data[index][self.word_seq_lengths[index]-1])
                        feed_dict[index][1].append(self.label_lookup(cur_tag_list))
                        feed_dict[index][2].append(prob)

                        if self.config.tree_args['use_gain']:
                            feed_dict[index][3] = [[r] for r in reward]
                        else:
                            """use the final reward at all time step"""
                            feed_dict[index][3] = [[reward[0]]] * len(reward)

            search_nodes = next_search_nodes

        avg_batch_acc = sum(batch_acc) / len(batch_acc)
        avg_batch_f1 = sum(batch_reward) / len(batch_reward)

        loss_list = []
        p_loss_list = []
        v_loss_list = []
        avg_loss = 0
        avg_ploss, avg_vloss = 0, 0

        """update model """
        if mode == 'train' or mode == 'valid':

            if mode == 'train':
                self.pv_model.train()

                cur_label_tensor, pi_tensor, reward_tensor = batchify_pi_reward(
                    feed_dict, self.word_seq_lengths, self.word_seq_recover,
                    self.config.data_args['label_num']
                )
                """
                print(cur_label_tensor.size())
                print(pi_tensor.size())
                print(reward_tensor.size())
                """

                loss, p_loss, v_loss = self.pv_model.forward(
                    word_inputs=self.word_seq_tensor,
                    word_seq_lengths=self.word_seq_lengths,
                    char_inputs=self.char_seq_tensor,
                    char_seq_lengths=self.char_seq_lengths,
                    char_seq_recover=self.char_seq_recover,
                    cur_label=cur_label_tensor,
                    pi=pi_tensor,
                    real_v=reward_tensor,
                    word_feature=self.feature_seq_tensor
                )

                # print('step={}, loss={:.5}, p_loss={:.5}, v_loss={:.5}'.format(
                #     step,loss, p_loss, v_loss))
                loss.backward()
                self.optimizer.step()
                self.pv_model.zero_grad()

                loss_list.append(loss.detach().numpy())
                p_loss_list.append(p_loss.detach().numpy())
                v_loss_list.append(v_loss.detach().numpy())


            else:
                self.pv_model.eval()
                cur_label_tensor, pi_tensor, reward_tensor = batchify_pi_reward(
                    feed_dict, self.word_seq_lengths, self.word_seq_recover,
                    self.config.data_args['label_num']
                )

                #                 print('cur_label_tensor', cur_label_tensor.size())
                #                 print('pi_tensor', pi_tensor.size())
                #                 print('reward_tensor', reward_tensor.size())

                loss, p_loss, v_loss = self.pv_model.forward(
                    word_inputs=self.word_seq_tensor,
                    word_seq_lengths=self.word_seq_lengths,
                    char_inputs=self.char_seq_tensor,
                    char_seq_lengths=self.char_seq_lengths,
                    char_seq_recover=self.char_seq_recover,
                    cur_label=cur_label_tensor,
                    pi=pi_tensor,
                    real_v=reward_tensor,
                    word_feature=self.feature_seq_tensor
                )

                loss_list.append(loss.detach().numpy())
                p_loss_list.append(p_loss.detach().numpy())
                v_loss_list.append(v_loss.detach().numpy())

            avg_loss = sum(loss_list) / len(loss_list)
            avg_ploss = sum(p_loss_list) / len(p_loss_list)
            avg_vloss = sum(v_loss_list) / len(v_loss_list)

        del feed_dict
        gc.collect()

        return avg_batch_acc, avg_batch_f1, avg_loss, avg_ploss, avg_vloss, batch_pred_label

    def save_model(self, path, epoch, loss):
        """
        save model
        """
        torch.save({
            'epoch': epoch, 'model_state_dict': self.pv_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss
        }, path)

    def raw_policy_single_predict(self, seq_data, feature_data, mode):
        """
        single sentence predict
        """
        probs, _ = self.pv_model.forward(
            word_state=torch.from_numpy(seq_data.data.cpu().numpy()),
            cur_label=torch.from_numpy(np.zeros([len(seq_data), self.config.data_args['label_num']])),
            word_feature=torch.from_numpy(feature_data.data.cpu().numpy()),
        )

        #         print('raw_policy', probs)
        action = [np.argmax(p) for p in probs.detach().numpy()]

        return action

    def test_with_raw_policy(self, examples, mode='test'):
        print('test with raw_policy....')
        self.batch_size = len(examples)

        self.sen_list = [e.sen for e in examples]
        self.label_list = [e.label for e in examples]
        self.char_list = [e.char for e in examples]
        self.pos_list = [e.pos for e in examples]

        if_train = True if mode == 'train' else False
        data = batchify_with_label(
            self.sen_list, self.char_list, self.label_list, gpu=False, if_train=if_train)
        self.word_seq_tensor, self.word_seq_lengths, self.word_seq_recover = data[0], data[1], data[2]
        self.char_seq_tensor, self.char_seq_lengths, self.char_seq_recover = data[3], data[4], data[5]
        self.label_seq_tensor, self.mask = data[6], data[7]

        # get hidden state
        # Attention!!!! seq_data are sorted !!!!!
        self.seq_data = self.pv_model.forward(
            word_inputs=self.word_seq_tensor, word_seq_lengths=self.word_seq_lengths,
            char_inputs=self.char_seq_tensor, char_seq_lengths=self.char_seq_lengths,
            char_seq_recover=self.char_seq_recover, word_state=None)

        self.feature_seq_tensor = batchify_feature(
            char_seq_tensor=self.char_seq_tensor, char_seq_recover=self.char_seq_recover,
            word_seq_recover=self.word_seq_recover, word_seq_lengths=self.word_seq_lengths,
            pos_list=self.pos_list, feature_dim=self.config.tree_args['raw_feature_dim'])

        batch_acc = {}
        batch_pred_label = {}
        batch_reward = {}
        print('length={}, recover={}'.format(
            self.word_seq_lengths, self.word_seq_recover))

        for i in range(self.batch_size):
            print('len={},{}'.format(
                len(self.label_list[i]), len(self.sen_list[i])))

            idx = self.word_seq_recover[i]
            lens = self.word_seq_lengths[idx]
            temp = self.seq_data[idx][:lens].data.cpu().numpy()
            feature = self.feature_seq_tensor[idx][:lens].data.cpu().numpy()
            probs, _ = self.pv_model.forward(
                word_state=torch.from_numpy(np.array(temp)),
                cur_label=torch.from_numpy(np.zeros([self.word_seq_lengths[idx], 9])),
                word_feature=torch.from_numpy(np.array(feature)),
            )

            # print(probs)

            action = [np.argmax(p) for p in probs.detach().numpy()]
            print('action', action)
            assert self.word_seq_lengths[idx] == len(self.label_list[i])

            # Attention !!!!  recover before
            flag, reward = self.get_tag_acc(i, action)
            acc = flag.count(1) / (len(action) * 1.0)
            print('T{}: r={},pred={}, ref={}, flag={}'.format(
                i, reward[0], action, self.label_list[i], flag))

            batch_reward[i] = reward[0]
            batch_acc[i] = acc
            batch_pred_label[i] = action

        avg_acc = sum(batch_acc.values()) / self.batch_size
        avg_reward = sum(batch_reward.values()) / self.batch_size

        return avg_acc, avg_reward, batch_pred_label

    def test_with_value(self, examples, mode='test'):
        print('test with value....')
        self.batch_size = len(examples)

        self.sen_list = [e.sen for e in examples]
        self.label_list = [e.label for e in examples]
        self.char_list = [e.char for e in examples]
        self.pos_list = [e.pos for e in examples]

        if_train = True if mode == 'train' else False
        data = batchify_with_label(
            self.sen_list, self.char_list, self.label_list, gpu=False, if_train=if_train)
        self.word_seq_tensor, self.word_seq_lengths, self.word_seq_recover = data[0], data[1], data[2]
        self.char_seq_tensor, self.char_seq_lengths, self.char_seq_recover = data[3], data[4], data[5]
        self.label_seq_tensor, self.mask = data[6], data[7]

        # get hidden state
        # Attention!!!! seq_data are sorted !!!!!
        self.seq_data = self.pv_model.forward(
            word_inputs=self.word_seq_tensor, word_seq_lengths=self.word_seq_lengths,
            char_inputs=self.char_seq_tensor, char_seq_lengths=self.char_seq_lengths,
            char_seq_recover=self.char_seq_recover, word_state=None)

        self.feature_seq_tensor = batchify_feature(
            char_seq_tensor=self.char_seq_tensor, char_seq_recover=self.char_seq_recover,
            word_seq_recover=self.word_seq_recover, word_seq_lengths=self.word_seq_lengths,
            pos_list=self.pos_list, feature_dim=self.config.tree_args['raw_feature_dim'])

        batch_acc = {}
        batch_pred_label = {}
        batch_reward = {}
        print('length={}, recover={}'.format(
            self.word_seq_lengths, self.word_seq_recover))

        label_num = self.config.data_args['label_num']

        for i in range(self.batch_size):
            print('len={},{}'.format(
                len(self.label_list[i]), len(self.sen_list[i])))

            idx = self.word_seq_recover[i]
            lens = self.word_seq_lengths[idx]
            temp = self.seq_data[idx][:lens].data.cpu().numpy()
            feature = self.feature_seq_tensor[idx][:lens].data.cpu().numpy()

            # TODO: check value
            value_list = np.zeros([lens, label_num])
            for label_id in range(label_num):
                temp_label = [0] * label_num
                temp_label[label_id] = 1
                cur_label = [temp_label] * int(lens)
                # print(cur_label)
                _, value = self.pv_model.forward(
                    word_state=torch.from_numpy(np.array(temp)),
                    cur_label=torch.from_numpy(np.array(cur_label)),
                    word_feature=torch.from_numpy(np.array(feature)),
                )
                value_list[:,label_id] = np.reshape(value.detach().numpy(),[1,-1])
            # print(value_list)

            action = [np.argmax(p) for p in value_list]
            # print('action', action)
            assert self.word_seq_lengths[idx] == len(self.label_list[i])

            # Attention !!!!  recover before
            flag, reward = self.get_tag_acc(i, action)
            acc = flag.count(1) / (len(action) * 1.0)
            print('T{}: r={},pred={}, ref={}, flag={}'.format(
                i, reward[0], action, self.label_list[i], flag))

            batch_reward[i] = reward[0]
            batch_acc[i] = acc
            batch_pred_label[i] = action

        avg_acc = sum(batch_acc.values()) / self.batch_size
        avg_reward = sum(batch_reward.values()) / self.batch_size

        return avg_acc, avg_reward, batch_pred_label

    def test_with_mcts(self, examples, mode='test'):
        """
        inference stage for mcts. same function with test_with_raw_policy
        :return: batch avg acc, f1 and tagged results
        batch_pred_label --> dict, use to do mapping to original label due to mixture in predition
        """
        search_nodes = self.init_root_list(examples, mode=mode)
        # key is length of fw_word and bw_word
        batch_acc = {}
        batch_pred_label = {}
        batch_reward = {}
        while len(search_nodes) > 0:
            next_search_nodes = []  # next search tree node list
            self.search(search_nodes, mode)

            for node in search_nodes:
                prob, select_label, choose_node = self.take_action(
                    node, temp=self.temp, is_random=False)
                index = node.data.index
                cur_tag_list = node.data.label + [select_label]

                if len(cur_tag_list) < len(self.sen_list[index]):
                    next_search_nodes.append(choose_node)

                else:
                    flag, reward = self.get_tag_acc(index, cur_tag_list)
                    acc = flag.count(1) / (len(cur_tag_list) * 1.0)

                    batch_reward[node.data.index] = reward[0]
                    batch_acc[node.data.index] = acc
                    batch_pred_label[node.data.index] = cur_tag_list

                    print('{}:T{} acc={},r={}, pre_label={}, ref_label={}, flag={}'.format(
                        mode, index, round(acc, 4), round(reward[0], 4), cur_tag_list,
                        self.label_list[index], flag))

            search_nodes = next_search_nodes

        avg_batch_acc = sum(batch_acc.values()) / self.batch_size
        avg_batch_f1 = sum(batch_reward.values()) / self.batch_size
        batch_pred_label = dict(sorted(batch_pred_label.items(), key=lambda item: item[0]))

        return avg_batch_acc, avg_batch_f1, batch_pred_label


def run(is_restore):
    # 1. init data processor and load data
    config = Parameter()
    print(config.tree_args)

    data = load_data()
    raw_sen_list = data[0]
    sen_list, pos_list, chunk_list, ner_list, char_list = data[1], data[2], data[3], data[4], data[5]
    word_embed, word_idx_map = data[6], data[7]
    print('word_embed={}, dim={}'.format(len(word_embed), len(word_embed[0])))

    x_train, x_dev, x_test = sen_list[0], sen_list[1], sen_list[2]
    y_train, y_dev, y_test = ner_list[0], ner_list[1], ner_list[2]
    char_train, char_dev, char_test = char_list[0], char_list[1], char_list[2]
    pos_train, pos_dev, pos_test = pos_list[0], pos_list[1], pos_list[2]
    chunk_train, chunk_dev, chunk_test = chunk_list[0], chunk_list[1], chunk_list[2]

    print('Size: train={},valid={}, test={}'.format(
        len(x_train), len(x_dev), len(x_test)))

    assert len(x_train) == len(y_train) == len(char_train)
    assert len(x_test) == len(y_test) == len(char_test)
    assert len(x_dev) == len(y_dev) == len(char_dev)

    # create example
    train_examples = create_examples(
        x_train, y_train, char_train, pos_train, chunk_train, 'train')
    valid_examples = create_examples(x_dev, y_dev, char_dev, pos_dev, chunk_dev, 'valid')


    is_develop = True
    if is_develop:
        path = os.path.curdir
    else:
        path = '/home/seagate/deadline_log/'

    # 2. init train batch
    time_stamp = str(date.today())
    model_save_dir = os.path.abspath(os.path.join(path, 'debug-runs', time_stamp))
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    train_batch = batch_iter(
        train_examples, config.tree_args['batch_size'], num_epochs=config.tree_args['epoch'])
    # train_batch = batch_iter(train_examples, 10, num_epochs=5)

    # 3. init search tree
    if is_restore:
        restore_model_path = './debug-runs/2018-12-29'
        config.tree_args['is_restore'] = True
        print('>>> restore from {}'.format(restore_model_path))
        search_tree = SearchTree(
            config, pretrain_word_embedding=word_embed.astype(np.float32),
            is_train=True, seed=0, path=restore_model_path)
        json.dump(config.tree_args, open(model_save_dir + '/config', 'w'))
    else:
        search_tree = SearchTree(
            config, pretrain_word_embedding=word_embed.astype(np.float32),
            is_train=True, seed=0, path=None)
        json.dump(config.tree_args, open(model_save_dir + '/config', 'w'))

    # search_tree = SearchTree(config, processor, is_train=True, seed=0, path=None)

    current_epoch = 0
    best_f1 = 0
    best_loss = 100
    step = 0
    step_per_epoch = int(math.ceil(len(train_examples) * 1.0 / config.tree_args['batch_size']))
    valid_file = open(model_save_dir + '/valid_result.txt', 'w')
    train_file = open(model_save_dir + '/train_result.txt', 'w')
    print('step_per_epoch ={}'.format(step_per_epoch))

    train_acc_list = []
    train_f1_list = []
    train_loss_list = []
    train_ploss_list = []
    train_vloss_list = []
    plot_loss = []
    plot_acc = []
    plot_x = []
    for batch_example in train_batch:
        t1 = time.time()
        train_acc, train_f1, train_loss, ploss, vloss, _ = search_tree.train(
            list(batch_example), mode='train', step=step)

        step += 1
        train_acc_list.append(train_acc)
        train_f1_list.append(train_f1)
        train_loss_list.append(train_loss)
        train_ploss_list.append(ploss)
        train_vloss_list.append(vloss)
        print(
            'step={}, train_acc={:.5}, f1={:.5}, loss={:.5}, ploss={:.5}, v_loss={:.5}, time={}'.format(
                step, train_acc, train_f1, train_loss, ploss, vloss, time.time() - t1))

        # *******************  plot train loss ***************
        if step % 40 == 0:
            plot_loss.append(train_loss)
            plot_acc.append(train_f1)
            plot_x.append(step)
            plot_fig(plot_acc, plot_loss, plot_x, model_save_dir)

        # finish one training epoch
        if step % step_per_epoch == 0 and step > 0:
            size = len(train_acc_list)
            current_epoch += 1
            print('>>>epoch={}, train_acc={}'.format(
                step / step_per_epoch, sum(train_acc_list) / size))
            train_file.write(
                ' '.join([str(step / step_per_epoch),
                          str(round(sum(train_acc_list) / size, 4)),
                          str(round(sum(train_f1_list) / size, 4)),
                          str(round(sum(train_loss_list) / size, 4)),
                          str(round(sum(train_ploss_list) / size, 4)),
                          str(round(sum(train_vloss_list) / size, 4))]) + '\n')

            train_file.flush()
            train_acc_list = []
            train_f1_list = []
            train_loss_list = []
            train_ploss_list = []
            train_vloss_list = []

        # start validation
        if step % step_per_epoch == 0 and step > 1:
            valid_acc_list = []
            valid_f1_list = []
            valid_loss_list = []
            valid_ploss_list = []
            valid_vloss_list = []
            valid_pred_label = []
            # only 1 epoch
            valid_batch = batch_iter(
                valid_examples, config.tree_args['batch_size'], num_epochs=1)

            for batch_example in valid_batch:
                valid_acc, valid_f1, valid_loss, ploss, vloss, label = search_tree.train(
                    list(batch_example), mode='valid', step=step)
                valid_acc_list.append(valid_acc)
                valid_f1_list.append(valid_f1)
                valid_loss_list.append(valid_loss)
                valid_ploss_list.append(ploss)
                valid_vloss_list.append(vloss)
                valid_pred_label.extend(label)

            size = len(valid_acc_list)
            valid_acc = sum(valid_acc_list) / size
            valid_f1 = sum(valid_f1_list) / size
            valid_loss = sum(valid_loss_list) / size
            valid_ploss = sum(valid_ploss_list) / size
            valid_vloss = sum(valid_vloss_list) / size
            print('>>> Valid: epoch={}, acc={}, f1={}, loss={}, vloss={}, '
                  'p_loss={}'.format(step / step_per_epoch, valid_acc, valid_f1,
                                        valid_loss, valid_vloss, valid_ploss))

            valid_file.write(
                ' '.join([str(step), str(round(valid_acc, 5)), str(round(valid_f1, 5)),
                          str(round(valid_loss, 4)), str(round(valid_ploss, 5)),
                          str(round(valid_vloss, 5))]) + '\n')
            valid_file.flush()

            if valid_f1 > best_f1 or valid_loss < best_loss:
                logger.info('SAVE model!!! best = {}'.format(valid_f1))
                search_tree.save_model(
                    path=model_save_dir + '/model', epoch=step % step_per_epoch,
                    loss=valid_loss)
                best_f1 = valid_f1
                best_loss = valid_loss
                pickle.dump(valid_pred_label, open(model_save_dir + '/test_result.pkl', 'wb'))
                print('>>>test: epoch={},acc={:.5},f1={:.5},best_f1={:.5}, time={:.5}'.format(
                    step / step_per_epoch, valid_acc, valid_f1, best_f1, time.time() - t1))


is_train = False
is_restore = True
test_mcts = False
test_value = True

if is_train:
    run(is_restore)
else:
    # step 1: load data and model
    checkpoint_dir = './debug-runs/2018-12-29'
    test_search = 800
    config = Parameter()
    print(config.tree_args)

    data = load_data()
    raw_sen_list = data[0]
    sen_list, pos_list, chunk_list, ner_list, char_list = data[1], data[2], data[3], data[4], data[5]
    word_embed, word_idx_map = data[6], data[7]
    print('word_embed={}, dim={}'.format(len(word_embed), len(word_embed[0])))

    x_train, x_dev, x_test = sen_list[0], sen_list[1], sen_list[2]
    y_train, y_dev, y_test = ner_list[0], ner_list[1], ner_list[2]
    char_train, char_dev, char_test = char_list[0], char_list[1], char_list[2]
    pos_train, pos_dev, pos_test = pos_list[0], pos_list[1], pos_list[2]
    chunk_train, chunk_dev, chunk_test = chunk_list[0], chunk_list[1], chunk_list[2]

    print('Size: train={},valid={}, test={}'.format(
        len(x_train), len(x_dev), len(x_test)))

    assert len(x_train) == len(y_train) == len(char_train)
    assert len(x_test) == len(y_test) == len(char_test)
    assert len(x_dev) == len(y_dev) == len(char_dev)

    # create example
    test_examples = create_examples(x_test, y_test, char_test, pos_test, chunk_test, 'test')

    # step 2: load model and prepare data batch
    search_tree = SearchTree(
        config, pretrain_word_embedding=word_embed.astype(np.float32),
        is_train=False, seed=0, path=checkpoint_dir)
    search_tree.pv_model.eval()

    ner2id = search_tree.label2id
    id2ner = dict(zip(ner2id.values(), ner2id.keys()))

    print(ner2id)
    print(id2ner)

    test_batches = batch_iter(
        test_examples, config.tree_args['batch_size'], num_epochs=1, shuffle=False)

    # step 3: Inference
    i = 0
    acc_list = []
    f1_list = []
    pred_label_list = []

    # step 4: test with MCTS
    if test_mcts:
        for batch in test_batches:
            acc, f1, pred_label = search_tree.test_with_mcts(batch)
            print('epoch {}: f1={}, acc={}'.format(i, f1, acc))
            i += 1
            f1_list.append(f1)
            acc_list.append(acc)
            pred_label_list.append(pred_label)

        pickle.dump(pred_label_list, open(checkpoint_dir + '/mcts_pred.pkl', 'wb'))
        print('END MCTS test, avg f1={}, avg acc ={}'.format(
            sum(f1_list) / len(f1_list), sum(acc_list) / len(acc_list)))

        label_list = pickle.load(open(checkpoint_dir + '/mcts_pred.pkl', 'rb'))

    # step 4: test with Raw policy
    else:
        for batch in test_batches:
            if test_value:
                acc, f1, pred_label = search_tree.test_with_value(batch)
            else:
                acc, f1, pred_label = search_tree.test_with_raw_policy(batch)
            print('epoch {}: f1={}, acc={}'.format(i, f1, acc))
            i += 1
            f1_list.append(f1)
            acc_list.append(acc)
            pred_label_list.append(pred_label)
        pickle.dump(pred_label_list, open(checkpoint_dir + '/raw_policy_pred.pkl', 'wb'))
        print('END raw policy test, avg f1={}, avg acc ={}'.format(
            sum(f1_list) / len(f1_list), sum(acc_list) / len(acc_list)))

        label_list = pickle.load(open(checkpoint_dir + '/raw_policy_pred.pkl', 'rb'))

    # step 4: start evaluation
    y_pred = []
    i = 0
    batch_id = 0
    for batch in label_list:
        # print('>>>>> batch_id ={}'.format(batch_id))
        for k, v in dict(batch).items():
            # print('k={}, v={}'.format(k, v))
            assert len(v) == len(test_examples[i].label)
            y_pred.append(v)
            i += 1
        batch_id += 1

    # label in example has become raw IOB or BIO format
    x_test = [example.sen for example in test_examples]
    y_test = [example.label for example in test_examples]
    print(x_test[0], y_test[0])

    if config.tree_args['tag_format'] == 'IOB':
        print('convert to bio')
        y_pred = [id2rawlabel(ner2id, iob_bio(id2rawlabel(id2ner, tags))) for tags in y_pred]
        y_test = [id2rawlabel(ner2id, iob_bio(id2rawlabel(id2ner, tags))) for tags in y_test]

    eval = eval_batch(
        x_test, y_test, pred_label_list=y_pred, reverse_ner_vocab=id2ner)
    f, precision, recall, accuracy = eval.calc_score(raw_sen=x_test,
                                                     y_true=y_test, y_pred=y_pred)

    print('Normal Eval: f={}, p={}, r={}, acc={}'.format(f, precision, recall, accuracy))
