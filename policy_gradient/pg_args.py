# !/usr/bin/python
# -*- coding:utf-8 -*-


class Parameter(object):
    def __init__(self):
        self.config = {
            'label_num': 9,
            'vocab_num': 402596,
            'word_embed_dim': 100,
            'hidden_dim': 100,
            'fc_dim': 128,
            'fc1_dim': 64,
            'feature_dim': 400 + 370+84,  # 'pos 45*5, chunk 21*5 word 4*5'
            'n_gram': 2,
            'context_only': True,

            'lr': 0.01,
            'optim_type': 'adagrad',
            'activate_fun': 'sigmoid',  # sigmoid or relu
            'use_rnn': True,
            'rnn_type': 'lstm',
            'concate': False,  # mean_sum
            'use_pretrain_word_embed': True,
            'trainable_embed': True,
            'use_l2': False,
            'keep_prob': 0.5,

            'look_ahead_depth': None,
            'epoch': 600,
            'n_playout': 1000,
            'c_puct': 1.0,
            'reward_type': 'acc',  # f1 or acc
            'use_gain': False,
            'discount_factor': 1.0,

            'batch_size': 64,
            'tag_format': 'IOB',
            'wait_epoch': 20

        }
