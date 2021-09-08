# encoding:utf-8

import sys
sys.path.extend(["../../","../","./"])
import time
import torch.optim.lr_scheduler
import torch.nn as nn
import random
import argparse
from driver.Config import *
from driver.Model import *
from driver.Labeler import *
from data.Dataloader import *
import pickle
import os
import re
from driver.BertTokenHelper import BertTokenHelper
from driver.BertModel import BertExtractor

from driver.language_mlp import LanguageMLP

from driver.modeling import BertModel as AdapterBERTModel
from driver.modeling import BertConfig
from transformers import AdamW, get_linear_schedule_with_warmup
from driver.adapterPGNBERT import AdapterPGNBertModel
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def train(data, dev_data, test_data, labeler, vocab, config, bert, language_embedder):
    optimizer = Optimizer(filter(lambda p: p.requires_grad, labeler.model.parameters()), config)
    optimizer_lang = Optimizer(filter(lambda p: p.requires_grad, language_embedder.parameters()), config)
    optimizer_bert = AdamW(filter(lambda p: p.requires_grad, bert.parameters()), lr=5e-6, eps=1e-8)
    batch_num = int(np.ceil(len(data) / float(config.train_batch_size)))
    # scheduler_bert = WarmupLinearSchedule(optimizer_bert, warmup_steps=0, t_total=config.train_iters * batch_num)
    scheduler_bert = get_linear_schedule_with_warmup(optimizer_bert, num_warmup_steps=0,
                                                     num_training_steps=config.train_iters * batch_num)


    global_step = 0
    best_score = -1
    batch_num = int(np.ceil(len(data) / float(config.train_batch_size)))
    for iter in range(config.train_iters):
        total_stats = Statistics()
        print('Iteration: ' + str(iter))
        batch_iter = 0

        for onebatch in data_iter(data, config.train_batch_size, True):
            words, extwords, predicts, inmasks, labels, outmasks, \
            bert_indices_tensor, bert_segments_tensor, bert_pieces_tensor, lang_ids = \
                batch_data_variable(onebatch, vocab)
            labeler.model.train()
            language_embedder.train()
            bert.train()
            if config.use_cuda:
                bert_indices_tensor = bert_indices_tensor.cuda()
                bert_segments_tensor = bert_segments_tensor.cuda()
                bert_pieces_tensor = bert_pieces_tensor.cuda()
            lang_embedding = language_embedder(lang_ids)
            bert_hidden = bert(input_ids=bert_indices_tensor, token_type_ids=bert_segments_tensor, bert_pieces=bert_pieces_tensor, lang_embedding=lang_embedding)

            labeler.forward(words, extwords, predicts, inmasks, bert_hidden)
            loss, stat = labeler.compute_loss(labels, outmasks)
            loss = loss / config.update_every
            loss.backward()

            total_stats.update(stat)
            total_stats.print_out(global_step, iter, batch_iter, batch_num)
            batch_iter += 1
            if batch_iter % config.update_every == 0 or batch_iter == batch_num:
                nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, labeler.model.parameters()), \
                                        max_norm=config.clip)
                optimizer.step()
                optimizer_lang.step()
                optimizer_bert.step()
                labeler.model.zero_grad()
                optimizer_lang.zero_grad()
                optimizer_bert.zero_grad()
                global_step += 1

            if batch_iter % config.validate_every == 0 or batch_iter == batch_num:
                gold_num, predict_num, correct_num, \
                gold_agent_num, predict_agent_num, correct_agent_num, \
                gold_target_num, predict_target_num, correct_target_num, \
                binary_gold_num, binary_predict_num, binary_gold_correct_num, binary_predict_correct_num, \
                binary_gold_agent_num, binary_predict_agent_num, binary_gold_correct_agent_num, binary_predict_correct_agent_num, \
                binary_gold_target_num, binary_predict_target_num, binary_gold_correct_target_num, binary_predict_correct_target_num, \
                prop_gold_num, prop_predict_num, prop_gold_correct_num, prop_predict_correct_num, \
                prop_gold_agent_num, prop_predict_agent_num, prop_gold_correct_agent_num, prop_predict_correct_agent_num, \
                prop_gold_target_num, prop_predict_target_num, prop_gold_correct_target_num, prop_predict_correct_target_num \
                    = evaluate(dev_data, labeler, vocab, config.target_dev_file + '.' + str(global_step))

                dev_score = 200.0 * correct_num / (gold_num + predict_num) if correct_num > 0 else 0.0
                print("Exact Dev: Recall = %d/%d = %.2f, Precision = %d/%d =%.2f, F-measure = %.2f" % \
                      (correct_num, gold_num, 100.0 * correct_num / gold_num if correct_num > 0 else 0.0, \
                       correct_num, predict_num, 100.0 * correct_num / predict_num if correct_num > 0 else 0.0, \
                       dev_score))

                dev_agent_score = 200.0 * correct_agent_num / (
                        gold_agent_num + predict_agent_num) if correct_agent_num > 0 else 0.0
                print("Exact Dev Agent: Recall = %d/%d = %.2f, Precision = %d/%d =%.2f, F-measure = %.2f" % \
                      (correct_agent_num, gold_agent_num,
                       100.0 * correct_agent_num / gold_agent_num if correct_agent_num > 0 else 0.0, \
                       correct_agent_num, predict_agent_num,
                       100.0 * correct_agent_num / predict_agent_num if correct_agent_num > 0 else 0.0, \
                       dev_agent_score))

                dev_target_score = 200.0 * correct_target_num / (
                        gold_target_num + predict_target_num) if correct_target_num > 0 else 0.0
                print("Exact Dev Target: Recall = %d/%d = %.2f, Precision = %d/%d =%.2f, F-measure = %.2f" % \
                      (correct_target_num, gold_target_num,
                       100.0 * correct_target_num / gold_target_num if correct_target_num > 0 else 0.0, \
                       correct_target_num, predict_target_num,
                       100.0 * correct_target_num / predict_target_num if correct_target_num > 0 else 0.0, \
                       dev_target_score))
                print()

                binary_dev_P = binary_predict_correct_num / binary_predict_num if binary_predict_num > 0 else 0.0
                binary_dev_R = binary_gold_correct_num / binary_gold_num if binary_gold_num > 0 else 0.0
                dev_binary_score = 200 * binary_dev_P * binary_dev_R / (
                        binary_dev_P + binary_dev_R) if binary_dev_P + binary_dev_R > 0 else 0.0
                print("Binary Dev: Recall = %d/%d = %.2f, Precision = %d/%d =%.2f, F-measure = %.2f" % \
                      (binary_gold_correct_num, binary_gold_num, 100.0 * binary_dev_R, \
                       binary_predict_correct_num, binary_predict_num, 100.0 * binary_dev_P, \
                       dev_binary_score))

                binary_dev_agent_P = binary_predict_correct_agent_num / binary_predict_agent_num if binary_predict_agent_num > 0 else 0.0
                binary_dev_agent_R = binary_gold_correct_agent_num / binary_gold_agent_num if binary_gold_agent_num > 0 else 0.0
                dev_binary_agent_score = 200 * binary_dev_agent_P * binary_dev_agent_R / (
                        binary_dev_agent_P + binary_dev_agent_R) if binary_dev_agent_P + binary_dev_agent_R > 0 else 0.0
                print("Binary Dev Agent: Recall = %d/%d = %.2f, Precision = %d/%d =%.2f, F-measure = %.2f" % \
                      (binary_gold_correct_agent_num, binary_gold_agent_num, 100.0 * binary_dev_agent_R, \
                       binary_predict_correct_agent_num, binary_predict_agent_num, 100.0 * binary_dev_agent_P, \
                       dev_binary_agent_score))

                binary_dev_target_P = binary_predict_correct_target_num / binary_predict_target_num if binary_predict_target_num > 0 else 0.0
                binary_dev_target_R = binary_gold_correct_target_num / binary_gold_target_num if binary_gold_target_num > 0 else 0.0
                dev_binary_target_score = 200 * binary_dev_target_P * binary_dev_target_R / (
                        binary_dev_target_P + binary_dev_target_R) if binary_dev_target_P + binary_dev_target_R > 0 else 0.0
                print("Binary Dev Target: Recall = %d/%d = %.2f, Precision = %d/%d =%.2f, F-measure = %.2f" % \
                      (binary_gold_correct_target_num, binary_gold_target_num, 100.0 * binary_dev_target_R, \
                       binary_predict_correct_target_num, binary_predict_target_num, 100.0 * binary_dev_target_P, \
                       dev_binary_target_score))
                print()

                prop_dev_P = prop_predict_correct_num / prop_predict_num if prop_predict_num > 0 else 0.0
                prop_dev_R = prop_gold_correct_num / prop_gold_num if prop_gold_num > 0 else 0.0
                dev_prop_score = 200 * prop_dev_P * prop_dev_R / (
                        prop_dev_P + prop_dev_R) if prop_dev_P + prop_dev_R > 0 else 0.0
                print("Prop Dev: Recall = %.2f/%d = %.2f, Precision = %.2f/%d =%.2f, F-measure = %.2f" % \
                      (prop_gold_correct_num, prop_gold_num, 100.0 * prop_dev_R, \
                       prop_predict_correct_num, prop_predict_num, 100.0 * prop_dev_P, \
                       dev_prop_score))

                prop_dev_agent_P = prop_predict_correct_agent_num / prop_predict_agent_num if prop_predict_agent_num > 0 else 0.0
                prop_dev_agent_R = prop_gold_correct_agent_num / prop_gold_agent_num if prop_gold_agent_num > 0 else 0.0
                dev_prop_agent_score = 200 * prop_dev_agent_P * prop_dev_agent_R / (
                        prop_dev_agent_P + prop_dev_agent_R) if prop_dev_agent_P + prop_dev_agent_R > 0 else 0.0
                print("Prop Dev Agent: Recall = %.2f/%d = %.2f, Precision = %.2f/%d =%.2f, F-measure = %.2f" % \
                      (prop_gold_correct_agent_num, prop_gold_agent_num, 100.0 * prop_dev_agent_R, \
                       prop_predict_correct_agent_num, prop_predict_agent_num, 100.0 * prop_dev_agent_P, \
                       dev_prop_agent_score))

                prop_dev_target_P = prop_predict_correct_target_num / prop_predict_target_num if prop_predict_target_num > 0 else 0.0
                prop_dev_target_R = prop_gold_correct_target_num / prop_gold_target_num if prop_gold_target_num > 0 else 0.0
                dev_prop_target_score = 200 * prop_dev_target_P * prop_dev_target_R / (
                        prop_dev_target_P + prop_dev_target_R) if prop_dev_target_P + prop_dev_target_R > 0 else 0.0
                print("Prop Dev Target: Recall = %.2f/%d = %.2f, Precision = %.2f/%d =%.2f, F-measure = %.2f" % \
                      (prop_gold_correct_target_num, prop_gold_target_num, 100.0 * prop_dev_target_R, \
                       prop_predict_correct_target_num, prop_predict_target_num, 100.0 * prop_dev_target_P, \
                       dev_prop_target_score))
                print()
                '''
                    Test
                '''
                test_gold_num, test_predict_num, test_correct_num, \
                test_gold_agent_num, test_predict_agent_num, test_correct_agent_num, \
                test_gold_target_num, test_predict_target_num, test_correct_target_num, \
                test_binary_gold_num, test_binary_predict_num, test_binary_gold_correct_num, test_binary_predict_correct_num, \
                test_binary_gold_agent_num, test_binary_predict_agent_num, test_binary_gold_correct_agent_num, test_binary_predict_correct_agent_num, \
                test_binary_gold_target_num, test_binary_predict_target_num, test_binary_gold_correct_target_num, test_binary_predict_correct_target_num, \
                test_prop_gold_num, test_prop_predict_num, test_prop_gold_correct_num, test_prop_predict_correct_num, \
                test_prop_gold_agent_num, test_prop_predict_agent_num, test_prop_gold_correct_agent_num, test_prop_predict_correct_agent_num, \
                test_prop_gold_target_num, test_prop_predict_target_num, test_prop_gold_correct_target_num, test_prop_predict_correct_target_num \
                    = evaluate(test_data, labeler, vocab, config.target_test_file + '.' + str(global_step))

                test_score = 200.0 * test_correct_num / (test_gold_num + test_predict_num) \
                    if test_correct_num > 0 else 0.0
                print("Exact Test: Recall = %d/%d = %.2f, Precision = %d/%d =%.2f, F-measure = %.2f" % \
                      (test_correct_num, test_gold_num, \
                       100.0 * test_correct_num / test_gold_num if test_correct_num > 0 else 0.0, \
                       test_correct_num, test_predict_num, \
                       100.0 * test_correct_num / test_predict_num if test_correct_num > 0 else 0.0, \
                       test_score))

                test_agent_score = 200.0 * test_correct_agent_num / (
                        test_gold_agent_num + test_predict_agent_num) if test_correct_agent_num > 0 else 0.0
                print("Exact Test Agent: Recall = %d/%d = %.2f, Precision = %d/%d =%.2f, F-measure = %.2f" % \
                      (test_correct_agent_num, test_gold_agent_num,
                       100.0 * test_correct_agent_num / test_gold_agent_num if test_correct_agent_num > 0 else 0.0, \
                       test_correct_agent_num, test_predict_agent_num,
                       100.0 * test_correct_agent_num / test_predict_agent_num if test_correct_agent_num > 0 else 0.0, \
                       test_agent_score))

                test_target_score = 200.0 * test_correct_target_num / (
                        test_gold_target_num + test_predict_target_num) if test_correct_target_num > 0 else 0.0
                print("Exact Test Target: Recall = %d/%d = %.2f, Precision = %d/%d =%.2f, F-measure = %.2f" % \
                      (test_correct_target_num, test_gold_target_num,
                       100.0 * test_correct_target_num / test_gold_target_num if test_correct_target_num > 0 else 0.0, \
                       test_correct_target_num, test_predict_target_num,
                       100.0 * test_correct_target_num / test_predict_target_num if test_correct_target_num > 0 else 0.0, \
                       test_target_score))
                print()

                binary_test_P = test_binary_predict_correct_num / test_binary_predict_num if test_binary_predict_num > 0 else 0.0
                binary_test_R = test_binary_gold_correct_num / test_binary_gold_num if test_binary_gold_num > 0 else 0.0
                binary_test_score = 200 * binary_test_P * binary_test_R / (
                        binary_test_P + binary_test_R) if binary_test_P + binary_test_R > 0 else 0.0
                print("Binary Test: Recall = %d/%d = %.2f, Precision = %d/%d =%.2f, F-measure = %.2f" % \
                      (test_binary_gold_correct_num, test_binary_gold_num, 100.0 * binary_test_R, \
                       test_binary_predict_correct_num, test_binary_predict_num, 100.0 * binary_test_P, \
                       binary_test_score))

                binary_test_agent_P = test_binary_predict_correct_agent_num / test_binary_predict_agent_num if test_binary_predict_agent_num > 0 else 0.0
                binary_test_agent_R = test_binary_gold_correct_agent_num / test_binary_gold_agent_num if test_binary_gold_agent_num > 0 else 0.0
                binary_test_agent_score = 200 * binary_test_agent_P * binary_test_agent_R / (
                        binary_test_agent_P + binary_test_agent_R) if binary_test_agent_P + binary_test_agent_R > 0 else 0.0
                print("Binary Test Agent: Recall = %d/%d = %.2f, Precision = %d/%d =%.2f, F-measure = %.2f" % \
                      (test_binary_gold_correct_agent_num, test_binary_gold_agent_num, 100.0 * binary_test_agent_R, \
                       test_binary_predict_correct_agent_num, test_binary_predict_agent_num,
                       100.0 * binary_test_agent_P, \
                       binary_test_agent_score))

                binary_test_target_P = test_binary_predict_correct_target_num / test_binary_predict_target_num if test_binary_predict_target_num > 0 else 0.0
                binary_test_target_R = test_binary_gold_correct_target_num / test_binary_gold_target_num if test_binary_gold_target_num > 0 else 0.0
                binary_test_target_score = 200 * binary_test_target_P * binary_test_target_R / (
                        binary_test_target_P + binary_test_target_R) if binary_test_target_P + binary_test_target_R > 0 else 0.0
                print("Binary Test Target: Recall = %d/%d = %.2f, Precision = %d/%d =%.2f, F-measure = %.2f" % \
                      (test_binary_gold_correct_target_num, test_binary_gold_target_num, 100.0 * binary_test_target_R, \
                       test_binary_predict_correct_target_num, test_binary_predict_target_num,
                       100.0 * binary_test_target_P, \
                       binary_test_target_score))
                print()

                prop_test_P = test_prop_predict_correct_num / test_prop_predict_num if test_prop_predict_num > 0 else 0.0
                prop_test_R = test_prop_gold_correct_num / test_prop_gold_num if test_prop_gold_num > 0 else 0.0
                prop_test_score = 200 * prop_test_P * prop_test_R / (
                        prop_test_P + prop_test_R) if prop_test_P + prop_test_R > 0 else 0.0
                print("Prop Test: Recall = %.2f/%d = %.2f, Precision = %.2f/%d =%.2f, F-measure = %.2f" % \
                      (test_prop_gold_correct_num, test_prop_gold_num, 100.0 * prop_test_R, \
                       test_prop_predict_correct_num, test_prop_predict_num, 100.0 * prop_test_P, \
                       prop_test_score))

                prop_test_agent_P = test_prop_predict_correct_agent_num / test_prop_predict_agent_num if test_prop_predict_agent_num > 0 else 0.0
                prop_test_agent_R = test_prop_gold_correct_agent_num / test_prop_gold_agent_num if test_prop_gold_agent_num > 0 else 0.0
                prop_test_agent_score = 200 * prop_test_agent_P * prop_test_agent_R / (
                        prop_test_agent_P + prop_test_agent_R) if prop_test_agent_P + prop_test_agent_R > 0 else 0.0
                print("prop Test Agent: Recall = %.2f/%d = %.2f, Precision = %.2f/%d =%.2f, F-measure = %.2f" % \
                      (test_prop_gold_correct_agent_num, test_prop_gold_agent_num, 100.0 * prop_test_agent_R, \
                       test_prop_predict_correct_agent_num, test_prop_predict_agent_num,
                       100.0 * prop_test_agent_P, \
                       prop_test_agent_score))

                prop_test_target_P = test_prop_predict_correct_target_num / test_prop_predict_target_num if test_prop_predict_target_num > 0 else 0.0
                prop_test_target_R = test_prop_gold_correct_target_num / test_prop_gold_target_num if test_prop_gold_target_num > 0 else 0.0
                prop_test_target_score = 200 * prop_test_target_P * prop_test_target_R / (
                        prop_test_target_P + prop_test_target_R) if prop_test_target_P + prop_test_target_R > 0 else 0.0
                print("Prop Test Target: Recall = %.2f/%d = %.2f, Precision = %.2f/%d =%.2f, F-measure = %.2f" % \
                      (test_prop_gold_correct_target_num, test_prop_gold_target_num,
                       100.0 * prop_test_target_R, \
                       test_prop_predict_correct_target_num, test_prop_predict_target_num,
                       100.0 * prop_test_target_P, \
                       prop_test_target_score))

                if dev_score > best_score:
                    print("Exceed best score: history = %.2f, current = %.2f" %(best_score, dev_score))
                    best_score = dev_score
                    if config.save_after > 0 and iter > config.save_after:
                        torch.save(labeler.model.state_dict(), config.save_model_path)


def evaluate(data, labeler, vocab, outputFile):
    start = time.time()
    labeler.model.eval()
    language_embedder.eval()
    bert.eval()
    output = open(outputFile, 'w', encoding='utf-8')
    total_gold_entity_num, total_predict_entity_num, total_correct_entity_num = 0, 0, 0
    total_gold_agent_entity_num, total_predict_agent_entity_num, total_correct_agent_entity_num = 0, 0, 0
    total_gold_target_entity_num, total_predict_target_entity_num, total_correct_target_entity_num = 0, 0, 0

    binary_total_gold_entity_num, binary_total_predict_entity_num, binary_gold_total_correct_entity_num, binary_predict_total_correct_entity_num = 0, 0, 0, 0
    binary_total_gold_agent_entity_num, binary_total_predict_agent_entity_num, binary_gold_total_correct_agent_entity_num, binary_predict_total_correct_agent_entity_num = 0, 0, 0, 0
    binary_total_gold_target_entity_num, binary_total_predict_target_entity_num, binary_gold_total_correct_target_entity_num, binary_predict_total_correct_target_entity_num = 0, 0, 0, 0

    prop_total_gold_entity_num, prop_total_predict_entity_num, prop_gold_total_correct_entity_num, prop_predict_total_correct_entity_num = 0, 0, 0, 0
    prop_total_gold_agent_entity_num, prop_total_predict_agent_entity_num, prop_gold_total_correct_agent_entity_num, prop_predict_total_correct_agent_entity_num = 0, 0, 0, 0
    prop_total_gold_target_entity_num, prop_total_predict_target_entity_num, prop_gold_total_correct_target_entity_num, prop_predict_total_correct_target_entity_num = 0, 0, 0, 0

    for onebatch in data_iter(data, config.test_batch_size, False, False):
        words, extwords, predicts, inmasks, labels, outmasks, \
        bert_indices_tensor, bert_segments_tensor, bert_pieces_tensor, lang_ids  = \
            batch_data_variable(onebatch, vocab)
        if config.use_cuda:
            bert_indices_tensor = bert_indices_tensor.cuda()
            bert_segments_tensor = bert_segments_tensor.cuda()
            bert_pieces_tensor = bert_pieces_tensor.cuda()
        count = 0
        lang_embedding = language_embedder(lang_ids)
        bert_hidden = bert(input_ids=bert_indices_tensor, token_type_ids=bert_segments_tensor, bert_pieces=bert_pieces_tensor, lang_embedding=lang_embedding)
        predict_labels = labeler.label(words, extwords, predicts, inmasks, bert_hidden)
        for result in batch_variable_srl(onebatch, predict_labels, vocab):
            printSRL(output, result)
            gold_entity_num, predict_entity_num, correct_entity_num, \
            gold_agent_entity_num, predict_agent_entity_num, correct_agent_entity_num, \
            gold_target_entity_num, predict_target_entity_num, correct_target_entity_num = evalSRLExact(onebatch[count],
                                                                                                        result)

            total_gold_entity_num += gold_entity_num
            total_predict_entity_num += predict_entity_num
            total_correct_entity_num += correct_entity_num

            total_gold_agent_entity_num += gold_agent_entity_num
            total_predict_agent_entity_num += predict_agent_entity_num
            total_correct_agent_entity_num += correct_agent_entity_num

            total_gold_target_entity_num += gold_target_entity_num
            total_predict_target_entity_num += predict_target_entity_num
            total_correct_target_entity_num += correct_target_entity_num

            binary_gold_entity_num, binary_predict_entity_num, binary_gold_correct_entity_num, binary_predict_correct_entity_num, \
            binary_gold_agent_entity_num, binary_predict_agent_entity_num, binary_gold_correct_agent_entity_num, binary_predict_correct_agent_entity_num, \
            binary_gold_target_entity_num, binary_predict_target_entity_num, binary_gold_correct_target_entity_num, binary_predict_correct_target_entity_num = evalSRLBinary(
                onebatch[count], result)

            binary_total_gold_entity_num += binary_gold_entity_num
            binary_total_predict_entity_num += binary_predict_entity_num
            binary_gold_total_correct_entity_num += binary_gold_correct_entity_num
            binary_predict_total_correct_entity_num += binary_predict_correct_entity_num

            binary_total_gold_agent_entity_num += binary_gold_agent_entity_num
            binary_total_predict_agent_entity_num += binary_predict_agent_entity_num
            binary_gold_total_correct_agent_entity_num += binary_gold_correct_agent_entity_num
            binary_predict_total_correct_agent_entity_num += binary_predict_correct_agent_entity_num

            binary_total_gold_target_entity_num += binary_gold_target_entity_num
            binary_total_predict_target_entity_num += binary_predict_target_entity_num
            binary_gold_total_correct_target_entity_num += binary_gold_correct_target_entity_num
            binary_predict_total_correct_target_entity_num += binary_predict_correct_target_entity_num

            prop_gold_entity_num, prop_predict_entity_num, prop_gold_correct_entity_num, prop_predict_correct_entity_num, \
            prop_gold_agent_entity_num, prop_predict_agent_entity_num, prop_gold_correct_agent_entity_num, prop_predict_correct_agent_entity_num, \
            prop_gold_target_entity_num, prop_predict_target_entity_num, prop_gold_correct_target_entity_num, prop_predict_correct_target_entity_num = evalSRLProportional(
                onebatch[count], result)

            prop_total_gold_entity_num += prop_gold_entity_num
            prop_total_predict_entity_num += prop_predict_entity_num
            prop_gold_total_correct_entity_num += prop_gold_correct_entity_num
            prop_predict_total_correct_entity_num += prop_predict_correct_entity_num

            prop_total_gold_agent_entity_num += prop_gold_agent_entity_num
            prop_total_predict_agent_entity_num += prop_predict_agent_entity_num
            prop_gold_total_correct_agent_entity_num += prop_gold_correct_agent_entity_num
            prop_predict_total_correct_agent_entity_num += prop_predict_correct_agent_entity_num

            prop_total_gold_target_entity_num += prop_gold_target_entity_num
            prop_total_predict_target_entity_num += prop_predict_target_entity_num
            prop_gold_total_correct_target_entity_num += prop_gold_correct_target_entity_num
            prop_predict_total_correct_target_entity_num += prop_predict_correct_target_entity_num
            count += 1

    output.close()

    #R = np.float64(total_correct_entity_num) * 100.0 / np.float64(total_gold_entity_num)
    #P = np.float64(total_correct_entity_num) * 100.0 / np.float64(total_predict_entity_num)
    #F = np.float64(total_correct_entity_num) * 200.0 / np.float64(total_gold_entity_num + total_predict_entity_num)


    end = time.time()
    during_time = float(end - start)
    print("sentence num: %d,  parser time = %.2f " % (len(data), during_time))

    return total_gold_entity_num, total_predict_entity_num, total_correct_entity_num, \
           total_gold_agent_entity_num, total_predict_agent_entity_num, total_correct_agent_entity_num, \
           total_gold_target_entity_num, total_predict_target_entity_num, total_correct_target_entity_num, \
           binary_total_gold_entity_num, binary_total_predict_entity_num, binary_gold_total_correct_entity_num, binary_predict_total_correct_entity_num, \
           binary_total_gold_agent_entity_num, binary_total_predict_agent_entity_num, binary_gold_total_correct_agent_entity_num, binary_predict_total_correct_agent_entity_num, \
           binary_total_gold_target_entity_num, binary_total_predict_target_entity_num, binary_gold_total_correct_target_entity_num, binary_predict_total_correct_target_entity_num, \
           prop_total_gold_entity_num, prop_total_predict_entity_num, prop_gold_total_correct_entity_num, prop_predict_total_correct_entity_num, \
           prop_total_gold_agent_entity_num, prop_total_predict_agent_entity_num, prop_gold_total_correct_agent_entity_num, prop_predict_total_correct_agent_entity_num, \
           prop_total_gold_target_entity_num, prop_total_predict_target_entity_num, prop_gold_total_correct_target_entity_num, prop_predict_total_correct_target_entity_num


class Optimizer:
    def __init__(self, parameter, config):
        self.optim = torch.optim.Adam(parameter, lr=config.learning_rate, betas=(config.beta_1, config.beta_2),
                                      eps=config.epsilon)
        #self.optim = torch.optim.Adadelta(parameter, lr=1.0, rho=0.95)
        decay, decay_step = config.decay, config.decay_steps
        l = lambda epoch: decay ** (epoch // decay_step)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=l)

    def step(self):
        self.optim.step()
        self.schedule()
        self.optim.zero_grad()

    def schedule(self):
        self.scheduler.step()

    def zero_grad(self):
        self.optim.zero_grad()

    @property
    def lr(self):
        return self.scheduler.get_lr()


if __name__ == '__main__':
    random.seed(666)
    np.random.seed(666)
    torch.cuda.manual_seed(666)
    torch.manual_seed(666)

    ### gpu
    gpu = torch.cuda.is_available()
    print("GPU available: ", gpu)
    print("CuDNN: \n", torch.backends.cudnn.enabled)

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', default='expdata/opinion.cfg')
    argparser.add_argument('--thread', default=4, type=int, help='thread num')
    argparser.add_argument('--use-cuda', action='store_true', default=False)

    args, extra_args = argparser.parse_known_args()
    config = Configurable(args.config_file, extra_args)

    vocab = creat_vocab(config.source_train_file, config.target_train_file, config.min_occur_count)
    vec = vocab.load_pretrained_embs(config.pretrained_embeddings_file)

    pickle.dump(vocab, open(config.save_vocab_path, 'wb'))

    args, extra_args = argparser.parse_known_args()
    config = Configurable(args.config_file, extra_args)
    torch.set_num_threads(args.thread)

    config.use_cuda = False
    if gpu and args.use_cuda: config.use_cuda = True
    print("\nGPU using status: ", config.use_cuda)

    language_embedder = LanguageMLP(config=config)

    model = eval(config.model)(vocab, config, vec)
    # bert = BertExtractor(config)
    bert_config = BertConfig.from_json_file(config.bert_config_path)
    bert_config.use_adapter = config.use_adapter
    bert_config.use_language_emb = config.use_language_emb
    bert_config.num_adapters = config.num_adapters
    bert_config.adapter_size = config.adapter_size
    bert_config.language_emb_size = config.language_emb_size
    bert_config.num_language_features = config.language_features
    bert_config.nl_project = config.nl_project
    bert = AdapterBERTModel.from_pretrained(config.bert_path, config=bert_config)

    if config.use_cuda:
        torch.backends.cudnn.enabled = True
        model = model.cuda()
        bert = bert.cuda()
        language_embedder = language_embedder.cuda()

    labeler = SRLLabeler(model)

    bert_token = BertTokenHelper(config.bert_path)

    in_language_list = config.in_langs
    out_language_list = config.out_langs

    lang_dic = {}
    lang_dic['in'] = in_language_list
    lang_dic['out'] = out_language_list

    source_data = read_corpus(config.source_train_file, bert_token, lang_dic)
    target_data = read_corpus(config.target_train_file, bert_token, lang_dic)
    data = source_data + target_data
    dev_data = read_corpus(config.target_dev_file, bert_token, lang_dic)
    test_data = read_corpus(config.target_test_file, bert_token, lang_dic)

    train(data, dev_data, test_data, labeler, vocab, config, bert, language_embedder)
