# !/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 2018-03-22
class: RL4SL
@author: Lao Yadi
@ main update: reformat and label state
"""

import tensorflow as tf
import numpy as np
import os
from pg.pg_args import Parameter

tf.set_random_seed(1)
config = Parameter().config


def fc_layer(inputs, in_dim, out_dim):
    """
    fully connected layer
    """
    weight = tf.Variable(tf.random_uniform([in_dim, out_dim], -0.1, 0.1), name='fc_u')
    bias = tf.Variable(tf.zeros([1, out_dim]) + 0.1, name='fc_b')
    wx_plus_b1 = tf.matmul(inputs, weight) + bias
    output = tf.nn.relu(wx_plus_b1)
    return output


def init_embed_layer(pretrain_word_embed):
    """
    init word embedding (pretrain or random static) and label embedding (one hot)
    """
    if config['use_pretrain_word_embed']:
        word_embedding = tf.Variable(pretrain_word_embed, name='emb_W',
            trainable=config['trainable_embed'])
    else:
        word_embedding = tf.Variable(tf.random_uniform(
            [config['vocab_num'], config['word_embed_dim']], -1.0/100, 1.0/100),
            name='emb_W', trainable=config['trainable_embed'])

    label_embedding = np.eye(config['label_num'], dtype='float32')
    # print('word_embed={}, label_embed={}'.format(word_embedding, label_embedding))
    return word_embedding, label_embedding


def init_policy_value_param():
    """
    init policy value network param
    """
    w_policy = tf.Variable(tf.random_uniform(
        [config['fc_dim'], config['label_num']], -1./100, 1./100))
    w_value = tf.Variable(tf.random_uniform(
        [config['fc_dim'], 1], -1./100, 1./100))

    return w_policy, w_value


def rnn_layer(input, dim):
    """
    RNN or LSTM with dropout
    """
    if config['rnn_type'] == 'lstm':
        rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=dim, reuse=False)
        outputs, states = tf.nn.dynamic_rnn(rnn_cell, input,  dtype=tf.float32)
        states = tf.concat([states[0], states[1]], axis=1)

    elif config['rnn_type'] == 'rnn':
        rnn_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=dim, reuse=False)
        _, states = tf.nn.dynamic_rnn(rnn_cell, input,  dtype=tf.float32)

    elif config['rnn_type'] == 'gru':
        print('use gru ...')
        gru_cell = tf.nn.rnn_cell.GRUCell(num_units=dim, reuse=False)
        _, states = tf.nn.dynamic_rnn(cell=gru_cell, inputs=input, dtype=tf.float32)

    elif config['rnn_type'] == 'bilstm':
        fw_cell = tf.contrib.rnn.BasicLSTMCell(num_units=dim, reuse=False)
        bw_cell = tf.contrib.rnn.BasicLSTMCell(num_units=dim, reuse=False)
        outputs, states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, input,  dtype=tf.float32)

        fw_state = tf.concat(states[0], 1)
        bw_state = tf.concat(states[1], 1)
        states = tf.concat([fw_state, bw_state], axis=1)

    else:
        states = None
    return states


def create_train_op(lr):
    """
    Selects the training algorithm and creates a train operation with it
    TODO: lr decay mode
    """
    if config['optim_type'] == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(lr)
    elif config['optim_type'] == 'adam':
        optimizer = tf.train.AdamOptimizer(lr)
    elif config['optim_type'] == 'rprop':
        optimizer = tf.train.RMSPropOptimizer(lr)
    elif config['optim_type'] == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(lr)
    else:
        raise NotImplementedError('Unsupported optimizer: {}'.format(config['optim_type']))

    return optimizer


class PGNet(object):
    """
    policy value network to guide mcts
    """
    def __init__(self, pretrain_word_embed=None, seed=0, path=None):
        print(">>>>>>>>>>>> init model >>>>>>>>>>>")
        self.config = config
        self.seed = seed
        self.pretrain_word_embed = pretrain_word_embed
        self.path = path
        self.current_epoch = tf.Variable(0, trainable=False)
        self.model = self.build_net()
        self.session = tf.Session()
        self.saver = tf.train.Saver()

        if self.path is None:
            self.session.run(tf.global_variables_initializer())
            self.session.run(tf.local_variables_initializer())
        else:
            print('restore model from {}'.format(self.path))
            self.saver.restore(self.session, tf.train.latest_checkpoint(self.path))

    def build_net(self):
        """
        PVnet Graph
        state = [fw_word, bw_word, tag_label, extra_fea]
        """
        # tf Graph input for state
        fw_word = tf.placeholder(tf.int32, [None, None], name='fw_word')
        bw_word = tf.placeholder(tf.int32, [None, None], name='bw_word')
        tag_label = tf.placeholder(tf.float32, [None, self.config['label_num']], name='tag_label')
        extra_fea = tf.placeholder(tf.float32, [None, None], name='extra_fea')

        action = tf.placeholder(tf.float32, [None, self.config['label_num']], name='action')
        G_t = tf.placeholder(tf.float32, [1, None], name='G_t')
        print('*** placeholder ***')
        print(fw_word, bw_word, tag_label, extra_fea)
        print(action, G_t)

        fw_shape = tf.shape(fw_word)
        bw_shape = tf.shape(bw_word)
        label_shape = tf.shape(tag_label)

        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        learning_rate = tf.Variable(self.config['lr'], dtype=tf.float32)

        # embedding layer
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            embed_W, embed_L = init_embed_layer(self.pretrain_word_embed)
            fw_embed_word = tf.nn.embedding_lookup(embed_W, fw_word)
            bw_embed_word = tf.nn.embedding_lookup(embed_W, bw_word)

        if self.config['use_rnn']:
            with tf.variable_scope('fw'):
                fw_word_state = rnn_layer(fw_embed_word, dim=config['hidden_dim'])
            with tf.variable_scope('bw'):
                bw_word_state = rnn_layer(bw_embed_word, dim=config['hidden_dim'])
            print('lstm, fw={}, bw={}'.format(fw_word_state, bw_word_state))
        else:
            if self.config['concate']:
                fw_word_state = tf.reshape(embed_W, [-1, fw_shape[1] * self.config['word_embed_dim']])
                bw_word_state = tf.reshape(embed_W, [-1, bw_shape[1] * self.config['word_embed_dim']])
            else:
                fw_word_state = tf.reduce_sum(fw_embed_word, axis=1)
                bw_word_state = tf.reduce_sum(bw_embed_word, axis=1)

        with tf.name_scope('pv_net'):
            final_state = tf.concat(
                [fw_word_state, bw_word_state, extra_fea], axis=1)
            print('fw_state={}\nbw_state={}\nlabel_state={}\nextra_fea={}'.format(
                fw_word_state, bw_word_state, tag_label, extra_fea))

            fc_output = fc_layer(final_state, self.config['feature_dim'], self.config['fc_dim'])
            # fc_output = fc_layer(states, self.config['fc_dim'], self.config['fc_dim'])

            w_policy, w_value = init_policy_value_param()
            print('w_policy={}, w_value={}'.format(w_policy, w_value))

            logits = tf.matmul(fc_output, w_policy)
            prob = tf.nn.softmax(logits)
            max_prob_id = tf.argmax(prob, axis=1)
            print('prob={}'.format(prob))

            with tf.variable_scope('loss_policy'):
                prob_loss = - tf.reduce_mean(
                    tf.matmul(G_t, tf.matmul(action, tf.transpose(tf.log(tf.clip_by_value(prob, 1e-30, 1.0)), [1,0]))))

            # L2 penalty (regularization)
            if self.config['use_l2']:
                l2_penalty_beta = 1e-4
                vars = tf.trainable_variables()
                l2_penalty = l2_penalty_beta * tf.add_n(
                    [tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name.lower()])
                loss = prob_loss + l2_penalty
            else:
                loss = prob_loss

            # optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss)
            optimizer = create_train_op(learning_rate).minimize(loss, global_step=self.current_epoch)

            print('final state = {}'.format(final_state))
            print('logit={}, '.format(logits))
            print('prob_loss ={}'.format(prob_loss))

        print('trainable variables are ...')
        ws = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES)
        for v in ws:
            print(v.name)

        return {'fw_word': fw_word, 'bw_word': bw_word, 'tag_label': tag_label,
                'extra_fea': extra_fea, 'keep_prob': keep_prob,
                'learning_rate': learning_rate, 'final_state': final_state,
                'prob': prob, 'logits':logits,
                'max_prob_id': max_prob_id,'Gt': G_t, 'action': action,
                'optimizer': optimizer, 'prob_loss': prob_loss, 'loss': loss}

    def update_model(self, feed_dict):
        """
        update model
        :param feed_dict:
        :return:
        """
        pad_feed_dict = {
            self.model['fw_word']: feed_dict[0], self.model['bw_word']: feed_dict[1],
            self.model['tag_label']: feed_dict[2], self.model['extra_fea']: feed_dict[3],
            self.model['action']: feed_dict[4], self.model['Gt']: feed_dict[5],
            self.model['keep_prob']: self.config['keep_prob']}

        opt, loss, p_loss, prob, = self.session.run(
            [self.model['optimizer'], self.model['loss'], self.model['prob_loss'],
             self.model['prob']], feed_dict=pad_feed_dict)
        return loss, prob, p_loss

    def get_loss(self, feed_dict, keep_prob):
        '''
        print('fw shape={}'.format(np.array(feed_dict[0]).shape))
        print('bw shape={}'.format(np.array(feed_dict[1]).shape))
        print('label shape={}'.format(np.array(feed_dict[2]).shape))
        print('V shape={}'.format(np.array(feed_dict[3]).shape))
        print('P shape={}'.format(np.array(feed_dict[4]).shape))
        '''

        pad_feed_dict = {
            self.model['fw_word']: feed_dict[0], self.model['bw_word']: feed_dict[1],
            self.model['tag_label']: feed_dict[2], self.model['extra_fea']: feed_dict[3],
            self.model['action']: feed_dict[4], self.model['Gt']: feed_dict[5],
            self.model['keep_prob']: self.config['keep_prob']}

        loss, p_loss, prob, = self.session.run(
            [self.model['loss'], self.model['prob_loss'],
             self.model['prob']], feed_dict=pad_feed_dict)
        return loss, prob, p_loss

    def get_policy(self, fw_word, bw_word, label, extra_fea, keep_prob):
        # perform padding first
        pad_feed_dict = {
            self.model['fw_word']: fw_word, self.model['bw_word']: bw_word,
            self.model['tag_label']: label, self.model['extra_fea']: extra_fea,
            self.model['keep_prob']: keep_prob}

        prob = self.session.run(
            self.model['prob'],feed_dict=pad_feed_dict)

        return prob

    def get_lr(self):
        """
        get learning rate
        """
        lr = self.session.run(self.model['learning_rate'])
        return lr

    def _save(self, model_dir):
        """
        Saves the model into model_dir with model_prefix as the model indicator
        """
        self.saver.save(self.session, os.path.join(model_dir))

    def _restore(self):
        """
        Restores the model into model_dir from model_prefix as the model indicator
        """
        self.saver.restore(self.session, tf.train.latest_checkpoint(self.path))


