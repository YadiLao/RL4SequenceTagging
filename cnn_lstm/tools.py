import pickle
import json
import numpy as np
import operator
import matplotlib.pyplot as plt
import torch
from torch_process_data import setting

BASE_DIR = setting['data_path']
DATA_SET = setting['dataset']
word2id = dict(json.load(open(BASE_DIR + DATA_SET + '/mapping/word_vocab', 'r')))
char2id = dict(json.load(open(BASE_DIR + DATA_SET + '/mapping/chars_vocab', 'r')))
pos2id = dict(json.load(open(BASE_DIR + DATA_SET + '/mapping/pos_vocab', 'r')))
chunk2id = dict(json.load(open(BASE_DIR + DATA_SET + '/mapping/chunk_vocab', 'r')))
id2word = dict(zip(word2id.values(), word2id.keys()))
id2char = dict(zip(char2id.values(), char2id.keys()))
id2pos = dict(zip(pos2id.values(), pos2id.keys()))
id2chunk = dict(zip(chunk2id.values(), chunk2id.keys()))


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, sen, char, pos, chunk, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.sen = sen
        self.label = label
        self.char = char
        self.pos = pos
        self.chunk = chunk
        self.sen_len = len(self.sen)
        self.char_len = [len(c) for c in self.char]

    def __str__(self):
        return self.__repr__()

    def __repr__(self):

        s = ""
        s += "qas_id: %s" % (self.guid)
        s += ", text-%s:" % (str(self.sen_len), ' '.join([str(s) for s in self.sen]))
        return s

    def _print(self):
        print('******** print example ***********')
        print('id: %s' % self.guid)
        print("text-%s: %s" % (str(self.sen_len), ' '.join([str(s) for s in self.sen])))
        print("char-%s:" % (' '.join([str(c) for c in self.char_len])))


def create_examples(sen_list, ner_list, char_list, pos_list, chunk_list, set_type):
    examples = []
    for i, sen in enumerate(sen_list):
        guid = "%s-%s" % (set_type, i)
        sen = sen_list[i]
        label = ner_list[i]
        char = char_list[i]
        pos = pos_list[i]
        chunk = chunk_list[i]
        examples.append(InputExample(
            guid=guid, sen=sen, char=char, pos=pos, chunk=chunk, label=label))

    return examples


def load_data():
    print('Loading Data...')
    data_file = open(BASE_DIR + DATA_SET + '/' + DATA_SET +'_IOB_100d.p', 'rb')
    x = pickle.load(data_file)
    data_file.close()
    return x


def bio_iob(tags):
    """
    BIO -> IOB
    :param tags:
    :return:
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if 'B' in tag:
            if i == 0:
                new_tags.append('I-' + tag[2:])
            else:
                pre_type = tags[i - 1][2:]
                if pre_type == tag[2:]:
                    new_tags.append(tag)
                else:
                    new_tags.append('I-' + tag[2:])
        else:
            new_tags.append(tag)
    return new_tags


def iob_bio(tags):
    """
    IOB -> BIO
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if 'I' in tag:
            if i == 0:
                new_tags.append('B-' + tag[2:])
            else:
                pre_type = tags[i - 1][2:]
                if pre_type == tag[2:]:
                    new_tags.append(tag)
                else:
                    new_tags.append('B-' + tag[2:])
        else:
            new_tags.append(tag)
    return new_tags


def bio_bioes(tags):
    """
    BIO -> BIOES
    """
    new_tags = []
    length = len(tags)
    for i, tag in enumerate(tags):
        if "-" not in tags[i]:
            new_tags.append(tag)
        else:
            label_type = tag[2:]
            if "B-" in tag:
                if (i == length - 1) or ("I-" not in tags[i + 1]):
                    new_tags.append(" S-" + label_type)
                else:
                    new_tags.append(" B-" + label_type)
            elif "I-" in tag:
                if (i == length - 1) or ("I-" not in tags[i + 1]):
                    new_tags.append(" E-" + label_type)
                else:
                    new_tags.append(" I-" + label_type)

    return new_tags


def bioes_bio(tags):
    """
    BIOES -> BIO
    """
    new_tags = []
    length = len(tags)
    for i, tag in enumerate(tags):
        if "-" not in tags[i]:
            new_tags.append(tag)
        else:
            label_type = tag[2:]
            if "E-" in tag:
                new_tags.append("I-" + label_type)
            elif "S-" in tag:
                new_tags.append("B-" + label_type)
            else:
                new_tags.append(tag)

    return new_tags


def bio_to_spans(sequence, vocab, strict_iob2=False):
    """
    convert to iob to span
    """
    iobtype = 2 if strict_iob2 else 1
    chunks = []
    current = None

    for i, y in enumerate(sequence):
        label = vocab[y]

        if label.startswith('B-'):
            if current is not None:
                chunks.append('@'.join(current))
            current = [label.replace('B-', ''), '%d' % i]

        elif label.startswith('I-'):

            if current is not None:
                base = label.replace('I-', '')
                if base == current[0]:
                    current.append('%d' % i)
                else:
                    chunks.append('@'.join(current))
                    if iobtype == 2:
                        print('Warning, type=IOB2, unexpected format ([%s] follows other tag type [%s] @ %d)' % (label, current[0], i))

                    current = [label, '%d' % i]

            else:
                # current = [label.replace('I-', ''), '%d' % i]
                chunks.append(label+'@'+str(i))
                if iobtype == 2:
                    print('Warning, unexpected format (I before B @ %d) %s' % (i, label))
        else:
            if current is not None:
                chunks.append('@'.join(current))
            current = None

    if current is not None:
        chunks.append('@'.join(current))

    return set(chunks)


def batchify_with_label(words, chars, labels, gpu, if_train=True):
    """
        input: list of words, chars and labels, various length. [[words,chars, labels],[words,chars,labels],...]
            words: word ids for one sentence. (batch_size, sent_len)
            chars: char ids for on sentences, various length. (batch_size, sent_len, each_word_length)
        output:
            zero padding for word and char, with their batch length
            word_seq_tensor: (batch_size, max_sent_len) Variable
            word_seq_lengths: (batch_size,1) Tensor
            char_seq_tensor: (batch_size*max_sent_len, max_word_len) Variable
            char_seq_lengths: (batch_size*max_sent_len,1) Tensor
            char_seq_recover: (batch_size*max_sent_len,1)  recover char sequence order
            label_seq_tensor: (batch_size, max_sent_len)
            mask: (batch_size, max_sent_len)
    """
    batch_size = len(words)

    word_seq_lengths = torch.LongTensor(list(map(len, words)))
    max_seq_len = word_seq_lengths.max().item()
    word_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long()
    label_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long()

    mask = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).byte()
    for idx, (seq, label, seqlen) in enumerate(zip(words, labels, word_seq_lengths)):
        seqlen = seqlen.item()
        word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        label_seq_tensor[idx, :seqlen] = torch.LongTensor(label)
        mask[idx, :seqlen] = torch.Tensor([1] * seqlen)

    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    word_seq_tensor = word_seq_tensor[word_perm_idx]

    label_seq_tensor = label_seq_tensor[word_perm_idx]
    mask = mask[word_perm_idx]

    # deal with char
    # pad_chars (batch_size, max_seq_len)
    pad_chars = [chars[idx] + [[0]] * (max_seq_len - len(chars[idx])) for idx in range(len(chars))]
    length_list = [list(map(len, pad_char)) for pad_char in pad_chars]
    max_word_len = max(map(max, length_list))
    char_seq_tensor = torch.zeros(
        (batch_size, max_seq_len, max_word_len), requires_grad=if_train).long()
    char_seq_lengths = torch.LongTensor(length_list)
    for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
        for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
            # print len(word), wordlen
            char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)

    char_seq_tensor = char_seq_tensor[word_perm_idx].view(batch_size * max_seq_len, -1)
    char_seq_lengths = char_seq_lengths[word_perm_idx].view(batch_size * max_seq_len, )
    char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
    char_seq_tensor = char_seq_tensor[char_perm_idx]
    _, char_seq_recover = char_perm_idx.sort(0, descending=False)
    _, word_seq_recover = word_perm_idx.sort(0, descending=False)
    if gpu:
        word_seq_tensor = word_seq_tensor.cuda()

        word_seq_lengths = word_seq_lengths.cuda()
        word_seq_recover = word_seq_recover.cuda()
        label_seq_tensor = label_seq_tensor.cuda()
        char_seq_tensor = char_seq_tensor.cuda()
        char_seq_recover = char_seq_recover.cuda()
        mask = mask.cuda()
    return word_seq_tensor, word_seq_lengths, word_seq_recover, \
           char_seq_tensor, char_seq_lengths, char_seq_recover, label_seq_tensor, mask


def batchify_pi_reward(feed_data, word_seq_lengths, word_seq_recover, label_num, if_train=False):
    """
        input:
            feed_data: dict, k=index, v=[[cur_label,..], [pi, ..], [reward,..]]
            word_seq_lengths:
            word_seq_recover:
        output:
            zero padding for selected label and pi and reward with their batch length
            cur_label_seq_tensor: (batch_size, max_sent_len) Variable
            pi_seq_tensor: (batch_size, max_sent_len) Tensor
            reward_seq_tensor: (batch_size, max_sent_len) Variable

            mask: (batch_size, max_sent_len)
    """
    batch_size = len(feed_data)
    max_seq_len = word_seq_lengths.max().item()

    cur_label_seq_tensor = torch.zeros(
        (batch_size, max_seq_len, label_num), requires_grad=if_train).float()
    pi_seq_tensor = torch.zeros(
        (batch_size, max_seq_len, label_num), requires_grad=if_train).float()
    reward_seq_tensor = torch.zeros(
        (batch_size, max_seq_len, 1), requires_grad=if_train).float()

    # already sort
    # print('recover', word_seq_recover, 'len', word_seq_lengths)
    for i in range(batch_size):
        #         print('label ', np.array(feed_data[i][1]).shape)
        #         print('pi ', np.array(feed_data[i][2]).shape)
        #         print('reward ', np.array(feed_data[i][3]).shape)
        idx = word_seq_recover[i]

        cur_label_seq_tensor[idx, :word_seq_lengths[idx]] = torch.FloatTensor(
            np.array(feed_data[i][1]))
        pi_seq_tensor[idx, :word_seq_lengths[idx]] = torch.FloatTensor(
            np.array(feed_data[i][2]))
        reward_seq_tensor[idx, :word_seq_lengths[idx]] = torch.FloatTensor(
            np.array(feed_data[i][3]))

    return cur_label_seq_tensor, pi_seq_tensor, reward_seq_tensor


def batchify_feature(
        char_seq_tensor, char_seq_recover, word_seq_recover, word_seq_lengths,
        pos_list, gpu=False, feature_dim=192+225, if_train=True):
    """
    output batch feature [b, t, d]. d is the feature num.
    word_seq_lengths already sorted !!! but char_tensor need to be recovered
    :return
       feature_seq_tensor is also sorted !!!
    """
    batch_size = len(word_seq_lengths)
    # print(char_seq_tensor.size(), 'batch size', batch_size)

    max_seq_len = word_seq_lengths[0]
    feature_seq_tensor = torch.zeros(
        (batch_size, max_seq_len, feature_dim), requires_grad=False).long()
    # loop for sententce in a batch
    for idx in range(batch_size):
        sen_id = word_seq_recover[idx]
        seq_len = word_seq_lengths[sen_id]
        # print('********* check  sen {}, len={} ************************'.format(idx, seq_len))
        feature = extract_raw_feature(
            sen_id=sen_id, char_seq_tensor=char_seq_tensor,
            char_seq_recover=char_seq_recover, seq_len=seq_len, max_seq_len=max_seq_len,
            pos_list=pos_list[idx])

        feature_seq_tensor[sen_id, :seq_len] = torch.LongTensor(feature)

    if gpu:
        feature_seq_tensor = feature_seq_tensor.cuda()

    return feature_seq_tensor


def extract_raw_feature(sen_id, char_seq_tensor, char_seq_recover,
                        seq_len, max_seq_len, pos_list, n_gram=2):
    """
    extract feature for single sentence.
    Attention pos, chunk need to be sorted !!!!
    """
    def contain_digit(chars):
        return any([id2char[e].isdigit() for e in chars])

    def is_all_upper(chars):
        return all([id2char[e].isupper() for e in chars])

    def is_title(chars):
        return id2char[chars[0]].isupper()

    def contain_punctuation(chars):
        chars = [id2char[e] for e in chars]
        return len(set(chars) & set('.,\"\'+/-&(]%@?*\[')) != 0

    # print('extract {}, seq_len={}'.format(chars_list.shape, seq_len))

    feature = [[] for _ in range(seq_len)]
    prefix_feature = [0] * len(id2char)*2    # two prefix feature

    # recover char list for this sentence
    start_char_id = sen_id*max_seq_len
    char_list = []
    for i in range(seq_len):
        char = [int(c) for c in list(char_seq_tensor[char_seq_recover[start_char_id+i]]) if c != char2id['<PAD>']]
        char_list.append(char)

    # print('sen_id={}'.format(sen_id))
    # print(char_list)

    # loop for token in a sentence
    for idx in range(seq_len):
        temp = []

        if len(char_list[idx]) >= 2:
            first_char_id = char_list[idx][0]
            second_char_id = char_list[idx][1]
            prefix_feature[first_char_id] = 1
            prefix_feature[second_char_id] = 1

        for i in range(idx - n_gram, idx + n_gram + 1):
            if i < 0 or i >= seq_len:
                temp.append([[0] * 4, [0]*len(id2pos)])
                continue

            word_feature = list(
                map(int, [contain_digit(char_list[i]), is_all_upper(char_list[i]),
                          is_title(char_list[i]), contain_punctuation(char_list[i])]))
            # print('char_list', ''.join(id2char[c] for c in char_list[i]), word_feature)
            pos_feature = [0] * len(id2pos)
            pos_feature[pos_list[i]] = 1
            # chunk_feature = [0] * len(id2chunk)
            # chunk_feature[chunk_list[i]] = 1
            temp.append([word_feature, pos_feature])

        # temp.append(prefix_feature)
        temp = reduce(operator.add, reduce(operator.add, list(map(list, list(zip(*temp))))))
        temp.extend(prefix_feature)
        feature[idx] = temp

        # print('word', idx, temp[:20], len(temp))

    return feature


def test_feature():
    data = load_data()
    import json
    char2id = dict(json.load(open(BASE_DIR + DATA_SET + '/mapping/chars_vocab', 'r')))
    raw_sen_list = data[0]
    sen_list, pos_list, chunk_list, ner_list, char_list = data[1], data[2], data[3], data[4], data[5]
    word_embed, word_idx_map = data[6], data[7]
    print('word_embed={}, dim={}'.format(len(word_embed), len(word_embed[0])))

    x_train, x_dev, x_test = sen_list[0][:4], sen_list[1], sen_list[2]
    y_train, y_dev, y_test = ner_list[0][:4], ner_list[1], ner_list[2]
    char_train, char_dev, char_test = char_list[0][:4], char_list[1], char_list[2]
    pos_train, pos_dev, pos_test = pos_list[0][:4], pos_list[1], pos_list[2]
    chunk_train, chunk_dev, chunk_test = chunk_list[0][:4], chunk_list[1], chunk_list[2]

    examples = create_examples(x_train, y_train, char_train, pos_train, chunk_train, 'train')
    for token in examples[0].char:
        print(''.join(id2char[c] for c in token))

    examples[0]._print()

    sen_list = [e.sen for e in examples]
    label_list = [e.label for e in examples]
    char_list = [e.char for e in examples]
    pos_list = [e.pos for e in examples]
    chunk_list = [e.chunk for e in examples]

    print(sen_list)

    data = batchify_with_label(
        sen_list, char_list, label_list, gpu=False, if_train=False)

    word_seq_tensor, word_seq_lengths, word_seq_recover = data[0], data[1], data[2]
    char_seq_tensor, char_seq_lengths, char_seq_recover = data[3], data[4], data[5]

    print(word_seq_tensor, word_seq_tensor.size())
    print (word_seq_lengths)
    print(word_seq_recover)
    print(char_seq_tensor, char_seq_tensor.size())
    print(char_seq_lengths)
    print(char_seq_recover)

    feature_seq_tensor = batchify_feature(
        char_seq_tensor, char_seq_recover, word_seq_recover, word_seq_lengths,
        pos_list)
    print(feature_seq_tensor.size())

    for i in range(len(sen_list)):
        idx = word_seq_recover[i]
        print(feature_seq_tensor[idx][0][:20])


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    When used: batches = batch_iter(list(zip(train_x, train_y)), batch_size, epoch)
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1

    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            # If x is an integer, randomly permute np.arange(x). If x is an array, make a copy and shuffle the elements randomly.
            # np.random.permutation(10)
            # array([1, 7, 4, 3, 0, 9, 2, 5, 8, 6])
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def id2rawlabel(ner_vocab, tags):
    """map between id and label"""
    return [ner_vocab[t] for t in tags]


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


def plot_fig(acc, loss, step_data, model_save_dir):

    if len(step_data) != len(loss) or len(step_data) != len(acc):
        raise ValueError('Plot: length of x and y are not the same')

    fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    ax1.plot(step_data, loss, linewidth=3)
    ax1.set_title('train_loss')

    ax2.plot(step_data, acc, linewidth=3)
    ax2.set_title('train_acc')
    plt.savefig(model_save_dir+'/train_fig.png', bbox_inches='tight')


def adjust_lr(epoch):
    lr = 0.01
    lr *= (0.5 ** (epoch // 10))
    print(lr)


if __name__ == '__main__':
    # test_feature()
    # loss = [2.3, 2.0, 1.9, 1.8]
    # step = 200
    # plot_loss(loss, '', step)
    # epoch = range(0, 40, 5)
    # print(map(adjust_lr, epoch))
    acc = [84.1, 86.2, 89.3]
    loss = [2.1, 2.0, 1.7]
    step = [35, 70, 105]
    plot_fig(acc, loss, step, model_save_dir='')
