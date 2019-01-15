# !/usr/bin/python
# -*- coding:utf-8 -*-


class Parameter(object):
    def __init__(self):
        self.data_args = {
            'bert_model': '../torch_checkpoint/bert-base-cased.tar.gz',
            'bert_vocab': '../torch_checkpoint/bert-base-cased-vocab.txt',
            'output_dir': '../output',
            'data_dir': '/home/lyd/go_master/data/conll03',
            'label_num': 9
        }

        self.hyper_args = {
            'use_gpu': False,
            'seed': 42,
            'word_feature_extractor': 'GRU',
        }

        self.tree_args = {
            'use_char': True,
            'use_pretrain_word_embed': True,
            'trainable_embed': True,
            'use_l2': False,
            'bilstm': True,

            'word_emb_dim': 100,
            'char_emb_dim': 30,
            'char_hidden_dim': 50,
            'hidden_dim': 400,
            'feature_dim': 400+192+225,
            'fc_dim': 128,
            'raw_feature_dim':192+225,
            'lstm_layer': 1,

            'lr': 0.001,
            'optim_type': 'adagrad',
            'activate_fun': 'sigmoid',  # sigmoid or relu
            'drop_prob': 0.0,

            'look_ahead_depth': None,
            'n_playout': 100,
            'c_puct': 2.0,
            'reward_type': 'f1',  # f1 or acc
            'use_gain': False,
            'discount_factor': 1.0,
            'epoch': 600,
            'batch_size': 10,
            'tag_format': 'IOB'
        }
