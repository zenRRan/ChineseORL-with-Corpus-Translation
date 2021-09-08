# encoding:utf-8
from data.Vocab import *
import numpy as np
import torch
from torch.autograd import Variable
import argparse

def judge_language_type(file_name):
    lang = None
    if 'ch' in file_name:
        lang = 'zh'
    elif 'en' in file_name:
        lang = 'en'
    elif 'pt' in file_name:
        lang = 'pt'
    else:
        print(file_name)
        raise RuntimeError

    # in_or_oov = 'in'
    # if 'dev' in file_name or 'test' in file_name:
    #     in_or_oov = 'oov'
    return lang


def read_corpus(file_path, bert_token, lang_dic):
    data = []
    language = judge_language_type(file_path)
    lang_id = -1
    in_or_oov = 'in' if language in lang_dic['in'] else 'oov'
    for idx, lang in enumerate(lang_dic[in_or_oov]):
        if language == lang:
            lang_id = idx
            break
    if lang_id == -1:
        print(language)
        raise RuntimeError
    with open(file_path, 'r', encoding='utf8') as infile:
        for sentence in readSRL(infile, bert_token, lang_id):
            data.append(sentence)
    return data

def sentences_numberize(sentences, vocab):
    for sentence in sentences:
        yield sentence2id(sentence, vocab)

def sentence2id(sentence, vocab):
    tokens = []
    index = 0
    for token in sentence.words:
        wordid = vocab.word2id(token.form)
        extwordid = vocab.extword2id(token.form)
        if index < sentence.key_start or index > sentence.key_end:
            labelid = vocab.label2id(token.label)
        else:
            labelid = vocab.PAD
        tokens.append([wordid, extwordid, labelid])
        index = index + 1

    return tokens,sentence.key_head,sentence.key_start,sentence.key_end, sentence.list_bert_indice, sentence.list_segments_id, sentence.list_piece_id, sentence.lang_id

def batch_slice(data, batch_size, bsorted=True):
    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in range(batch_num):
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        sentences = [data[i * batch_size + b] for b in range(cur_batch_size)]

        batch_size = len(sentences)
        src_ids = list(range(batch_size))
        if bsorted:
            src_ids = sorted(range(batch_size), key=lambda src_id: sentences[src_id].length, reverse=True)

        sorted_sentences = [sentences[src_id] for src_id in src_ids]

        yield sorted_sentences


def data_iter(data, batch_size, shuffle=True, bsorted=True):
    """
    randomly permute data, then sort by source length, and partition into batches
    ensure that the length of  sentences in each batch
    """

    batched_data = []
    if shuffle: np.random.shuffle(data)
    batched_data.extend(list(batch_slice(data, batch_size, bsorted)))

    if shuffle: np.random.shuffle(batched_data)
    for batch in batched_data:
        yield batch


def batch_data_variable(batch, vocab):
    length = batch[0].length
    batch_size = len(batch)
    word_length_list = [length]
    for b in range(1, batch_size):
        if batch[b].length > length: length = batch[b].length
        word_length_list.append(batch[b].length)

    words = Variable(torch.LongTensor(batch_size, length).fill_(vocab.PAD), requires_grad=False)
    extwords = Variable(torch.LongTensor(batch_size, length).fill_(vocab.PAD), requires_grad=False)
    predicts = Variable(torch.LongTensor(batch_size, length).fill_(vocab.PAD), requires_grad=False)
    inmasks = Variable(torch.Tensor(batch_size, length).zero_(), requires_grad=False)
    labels = Variable(torch.LongTensor(batch_size, length).fill_(vocab.PAD), requires_grad=False)
    outmasks = Variable(torch.LongTensor(batch_size, length).zero_(), requires_grad=False)
    lang_ids = -1

    # bert
    bert_length_list = [len(sentence.list_bert_indice) for sentence in batch]
    max_bert_length = max(bert_length_list)
    bert_indices_tensor = torch.LongTensor(batch_size, max_bert_length).zero_()
    bert_segments_tensor = torch.LongTensor(batch_size, max_bert_length).zero_()
    bert_pieces_tensor = torch.Tensor(batch_size, length, max_bert_length).zero_()

    ###

    b = 0
    for tokens, key_head, key_start, key_end, list_bert_indice, list_segments_id, list_piece_id, lang_id in sentences_numberize(batch, vocab):
        index = 0
        lang_ids = lang_id
        for word in tokens:
            words[b, index] = word[0]
            extwords[b, index] = word[1]
            labels[b, index] = word[2]
            inmasks[b, index] = 1
            outmasks[b, index] = 1
            predicts[b, index] = 2
            if index >= key_start and index <= key_end:
                predicts[b, index] = 1
                #outmasks[b, index] = 0
            index += 1

        # bert
        for index in range(bert_length_list[b]):
            bert_indices_tensor[b, index] = list_bert_indice[index]
            bert_segments_tensor[b, index] = list_segments_id[index]
        shift_pos = 1  # remove the first token
        for sindex in range(word_length_list[b]):
            avg_score = 1.0 / len(list_piece_id[sindex + shift_pos])
            for tindex in list_piece_id[sindex + shift_pos]:
                bert_pieces_tensor[b, sindex, tindex] = avg_score
        ###
        b += 1

    return words, extwords, predicts, inmasks, labels, outmasks.byte(), bert_indices_tensor, bert_segments_tensor, bert_pieces_tensor, lang_ids

def batch_variable_srl(inputs, labels, vocab):
    for input, label in zip(inputs, labels):
        predicted_labels = []
        for idx in range(input.length):
            if idx < input.key_start or idx > input.key_end:
                predicted_labels.append(vocab.id2label(label[idx]))
            else:
                predicted_labels.append(input.words[idx].label)
        normed_labels, modifies = normalize_labels(predicted_labels)
        tokens = []
        for idx in range(input.length):
            tokens.append(Word(idx, input.words[idx].org_form, normed_labels[idx]))
        yield Sentence(tokens)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--train', default='expdata/aaai19srl.train.conll')
    argparser.add_argument('--dev', default='expdata/aaai19srl.dev.conll')
    argparser.add_argument('--test', default='expdata/aaai19srl.test.conll')
    argparser.add_argument('--use-cuda', action='store_true', default=True)

    args, extra_args = argparser.parse_known_args()

    vocab = creat_vocab(args.train, 1)

    train_data = read_corpus(args.train)
    dev_data = read_corpus(args.dev)
    test_data = read_corpus(args.test)

    for onebatch in data_iter(train_data, 100, False):
        words, extwords, predicts, labels, lengths, masks = batch_data_variable(onebatch, vocab)
        #print("one batch")

