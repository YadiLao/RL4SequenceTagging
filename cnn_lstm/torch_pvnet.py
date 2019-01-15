import numpy as np
import torch
from torch import nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F


class CharCNN(nn.Module):
    def __init__(self, alphabet_size, pretrain_char_embedding, embedding_dim,
                 hidden_dim, dropout, gpu):
        super(CharCNN, self).__init__()
        print("build char sequence feature extractor: CNN ...")
        self.gpu = gpu
        self.hidden_dim = hidden_dim
        self.char_drop = nn.Dropout(dropout)
        self.char_embeddings = nn.Embedding(alphabet_size, embedding_dim)
        if pretrain_char_embedding is not None:
            self.char_embeddings.weight.data.copy_(torch.from_numpy(pretrain_char_embedding))
        else:
            self.char_embeddings.weight.data.copy_(
                torch.from_numpy(self.random_embedding(alphabet_size, embedding_dim)))
        self.char_cnn = nn.Conv1d(embedding_dim, self.hidden_dim, kernel_size=3, padding=1)
        if self.gpu:
            self.char_drop = self.char_drop.cuda()
            self.char_embeddings = self.char_embeddings.cuda()
            self.char_cnn = self.char_cnn.cuda()

    def random_embedding(self, vocab_size, embedding_dim):
        """
        generate random embedding
        """
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def get_last_hiddens(self, input, seq_lengths):
        """
            input:
                input: Variable(batch_size, word_length)
                seq_lengths: numpy array (batch_size,  1)
            output:
                Variable(batch_size, char_hidden_dim)
            Note it only accepts ordered (length) variable, length size is recorded in
            seq_lengths
        """
        batch_size = input.size(0)
        char_embeds = self.char_drop(self.char_embeddings(input))
        char_embeds = char_embeds.transpose(2, 1).contiguous()
        char_cnn_out = self.char_cnn(char_embeds)
        char_cnn_out = F.max_pool1d(char_cnn_out, char_cnn_out.size(2)).view(batch_size, -1)
        return char_cnn_out

    def get_all_hiddens(self, input, seq_lengths):
        """
            input:
                input: Variable(batch_size,  word_length)
                seq_lengths: numpy array (batch_size,  1)
            output:
                Variable(batch_size, word_length, char_hidden_dim)
            Note it only accepts ordered (length) variable, length size is recorded in
            seq_lengths
        """
        batch_size = input.size(0)
        char_embeds = self.char_drop(self.char_embeddings(input))
        char_embeds = char_embeds.transpose(2, 1).contiguous()
        char_cnn_out = self.char_cnn(char_embeds).transpose(2, 1).contiguous()
        return char_cnn_out

    def forward(self, input, seq_lengths):
        return self.get_all_hiddens(input, seq_lengths)


class WordRep(nn.Module):
    def __init__(self, config, char_alphabet, word_alphabet, pretrain_char_embedding=None,
                 pretrain_word_embedding=None):
        super(WordRep, self).__init__()
        print("build word representation...")
        self.gpu = config.hyper_args['use_gpu']
        self.use_char = config.tree_args['use_char']
        self.char_hidden_dim = 0
        self.char_all_feature = False
        if self.use_char:
            self.char_hidden_dim = config.tree_args['char_hidden_dim']
            self.char_embedding_dim = config.tree_args['char_emb_dim']
            self.char_feature = CharCNN(
                len(char_alphabet), pretrain_char_embedding,
                self.char_embedding_dim, self.char_hidden_dim, config.tree_args['drop_prob'], self.gpu)

        self.embedding_dim = config.tree_args['word_emb_dim']
        self.drop = nn.Dropout(config.tree_args['drop_prob'])
        self.word_embedding = nn.Embedding(len(word_alphabet), self.embedding_dim)
        if pretrain_word_embedding is not None:
            self.word_embedding.weight.data.copy_(torch.from_numpy(pretrain_word_embedding))
        else:
            self.word_embedding.weight.data.copy_(
                torch.from_numpy(self.random_embedding(len(word_alphabet),
                                                       self.embedding_dim)))

        if self.gpu:
            self.drop = self.drop.cuda()
            self.word_embedding = self.word_embedding.cuda()

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def forward(self, word_inputs, word_seq_lengths, char_inputs,
                char_seq_lengths, char_seq_recover):
        """
            input:
                word_inputs: (batch_size, sent_len)
                features: list [(batch_size, sent_len), (batch_len, sent_len),...]
                word_seq_lengths: list of batch_size, (batch_size,1)
                char_inputs: (batch_size*sent_len, word_length)
                char_seq_lengths: list of whole batch_size for char, (batch_size*sent_len, 1)
                char_seq_recover: variable which records the char order information, used to
                recover char order
            output:
                Variable(batch_size, sent_len, hidden_dim)
        """
        batch_size = word_inputs.size(0)
        sent_len = word_inputs.size(1)
        word_embs = self.word_embedding(word_inputs)
        word_list = [word_embs]

        if self.use_char:
            # calculate char lstm last hidden
            char_features = self.char_feature.get_last_hiddens(
                char_inputs, char_seq_lengths.cpu().numpy())
            # print('********* char_feature', char_features.size())
            char_features = char_features[char_seq_recover]
            char_features = char_features.view(batch_size, sent_len, -1)
            # concat word and char together
            word_list.append(char_features)
            word_embs = torch.cat([word_embs, char_features], 2)
            if self.char_all_feature:
                char_features_extra = self.char_feature_extra.get_last_hiddens(
                    char_inputs, char_seq_lengths.cpu().numpy())
                char_features_extra = char_features_extra[char_seq_recover]
                char_features_extra = char_features_extra.view(batch_size, sent_len, -1)
                # concat word and char together
                word_list.append(char_features_extra)
        word_embs = torch.cat(word_list, 2)
        word_represent = self.drop(word_embs)
        return word_represent


class WordStateExtractor(nn.Module):
    def __init__(self, config, char_alphabet, word_alphabet, pretrain_char_embedding=None,
                 pretrain_word_embedding=None):
        super(WordStateExtractor, self).__init__()
        print("build word feature extractor: %s..." % (config.hyper_args['word_feature_extractor']))
        self.gpu = config.hyper_args['use_gpu']
        self.use_char = config.tree_args['use_char']

        self.droplstm = nn.Dropout(config.tree_args['drop_prob'])
        self.bilstm_flag = config.tree_args['bilstm']
        self.lstm_layer = config.tree_args['lstm_layer']
        self.wordrep = WordRep(
            config, char_alphabet, word_alphabet, pretrain_char_embedding=pretrain_char_embedding,
            pretrain_word_embedding=pretrain_word_embedding)

        self.input_size = config.tree_args['word_emb_dim']
        if self.use_char:
            self.input_size += config.tree_args['char_hidden_dim']

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        if self.bilstm_flag:
            lstm_hidden = config.tree_args['hidden_dim'] // 2
        else:
            lstm_hidden = config.tree_args['hidden_dim']

        self.word_feature_extractor = config.hyper_args['word_feature_extractor']
        if self.word_feature_extractor == "GRU":
            self.lstm = nn.GRU(
                self.input_size, lstm_hidden, num_layers=self.lstm_layer,
                batch_first=True, bidirectional=self.bilstm_flag)
        elif self.word_feature_extractor == "LSTM":
            self.lstm = nn.LSTM(
                self.input_size, lstm_hidden, num_layers=self.lstm_layer,
                batch_first=True, bidirectional=self.bilstm_flag)

        if self.gpu:
            self.droplstm = self.droplstm.cuda()
            if self.word_feature_extractor == "CNN":
                self.word2cnn = self.word2cnn.cuda()
                for idx in range(self.cnn_layer):
                    self.cnn_list[idx] = self.cnn_list[idx].cuda()
                    self.cnn_drop_list[idx] = self.cnn_drop_list[idx].cuda()
                    self.cnn_batchnorm_list[idx] = self.cnn_batchnorm_list[idx].cuda()
            else:
                self.lstm = self.lstm.cuda()

    def forward(self, word_inputs, word_seq_lengths, char_inputs,
                char_seq_lengths, char_seq_recover):
        """
            input:
                word_inputs: (batch_size, sent_len)
                word_seq_lengths: list of batch_size, (batch_size,1)
                char_inputs: (batch_size*sent_len, word_length)
                char_seq_lengths: list of whole batch_size for char, (batch_size*sent_len, 1)
                char_seq_recover: variable which records the char order information, used to recover char order
            output:
                Variable(batch_size, sent_len, hidden_dim)
        """
        word_represent = self.wordrep.forward(
            word_inputs, word_seq_lengths,
            char_inputs, char_seq_lengths, char_seq_recover)

        packed_words = pack_padded_sequence(
            word_represent, word_seq_lengths.cpu().numpy(), True)
        hidden = None
        lstm_out, hidden = self.lstm(packed_words, hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out)
        # lstm_out (seq_len, seq_len, hidden_size)
        feature_out = self.droplstm(lstm_out.transpose(1, 0))
        # feature_out (batch_size, seq_len, hidden_size)

        return feature_out


class CNN_LSTM_PVnet(nn.Module):
    """
    policy and value network for MCTS
    """

    def __init__(self, config, char_alphabet, word_alphabet,
                 pretrain_char_embedding=None, pretrain_word_embedding=None):
        super(CNN_LSTM_PVnet, self).__init__()
        self.config = config
        self.word_state_extractor = WordStateExtractor(
            config, char_alphabet, word_alphabet,
            pretrain_char_embedding=pretrain_char_embedding,
            pretrain_word_embedding=pretrain_word_embedding,
        )
        self.fc = nn.Linear(
            self.config.tree_args['feature_dim'], self.config.tree_args['fc_dim'])
        self.classifier = nn.Linear(
            self.config.tree_args['fc_dim'], self.config.data_args['label_num'])
        self.softmax = nn.Softmax()
        self.evaluator = nn.Linear(self.config.tree_args['fc_dim'], 9)

    def forward(self, word_inputs=None, word_seq_lengths=None, char_inputs=None,
                char_seq_lengths=None, char_seq_recover=None, cur_label=None,
                word_state=None, pi=None, real_v=None, word_feature=None):

        # word state = [b, t, d]
        if word_state is None:
            word_state = self.word_state_extractor.forward(
                word_inputs, word_seq_lengths, char_inputs,
                char_seq_lengths, char_seq_recover)

            if pi is None:
                return word_state

        # sequence_output = torch.cat([word_state, feature.float()], 1)

        # input: [?, feature_dim] --> output_dim: [?, label_num]
        # 1. simple predict: need to collect input, shape=[b, feature_dim]
        # 2. update_net: run lstm to get input, shape=[b,t,feature_dim], need to view
        if pi is None:
            assert word_state.size()[0] == word_feature.size()[0]
            word_state = torch.cat([word_state, word_feature.float()], 1)
            word_state = self.fc(word_state)
            probs = F.relu(self.classifier(word_state))
            logits = self.softmax(probs)

            value_state = self.evaluator(word_state).mul(cur_label.float())

            # print(value_state.size(), cur_label.size())
            assert value_state.size()[0] == cur_label.size()[0]
            assert value_state.size()[1] == cur_label.size()[1]
            value = torch.sigmoid(value_state.sum(1).unsqueeze(1))
            # print('probs={}, value_state={}'.format(probs.size(), value_state.size()))
            return logits, value
        else:
            # [b*t, feature_dim]
            # print('*****word_state', word_state.size())
            word_state = torch.cat([word_state, word_feature.float()], 2)
            batch_size = word_state.size()[0]
            word_state = word_state.contiguous().view(
                batch_size * word_state.size()[1], -1).data.cpu()
            word_state = self.fc(word_state)
            cur_label = cur_label.view(batch_size * cur_label.size()[1], -1)
            pi = pi.view(batch_size * pi.size()[1], -1)
            real_v = real_v.view(batch_size * real_v.size()[1], -1)
            probs = F.relu(self.classifier(word_state))

            #             print('word state={}, probs={}, cur_label={}'.format(
            #                 word_state.size(), probs.size(), cur_label.size()))

            # TODO: test tensor mul
            value_state = self.evaluator(word_state).mul(cur_label.float())
            # print('update net', value_state.size(), cur_label.size())
            assert value_state.size()[0] == cur_label.size()[0]
            assert value_state.size()[1] == cur_label.size()[1]
            value = torch.sigmoid(value_state.sum(1).unsqueeze(1))
            # print('probs={}, value_state={}'.format(probs.size(), value_state.size()))

            # implement SGD
            log_probs = F.log_softmax(probs)
            prob_loss = pi.mul(log_probs.float()).sum(1).sum(0) / batch_size
            # prob_loss = prob_loss_f(logits, pi.long())
            # print('log_prob', log_probs, 'pi', pi, 'prob_loss:', prob_loss)
            value_loss_f = torch.nn.MSELoss()
            value_loss = value_loss_f(value, real_v.float()).float()
            # print('value', value, 'real_v',real_v, 'vloss',value_loss)
            loss = -prob_loss + value_loss
            # print('final loss ={}'.format(loss))
            return loss, -prob_loss, value_loss