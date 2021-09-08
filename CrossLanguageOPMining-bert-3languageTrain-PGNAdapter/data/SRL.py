# encoding:utf-8

class Word:
    def __init__(self, id, form, label):
        self.id = id
        self.org_form = form
        self.form = form.lower()
        self.label = label

    def __str__(self):
        values = [str(self.id), self.org_form, self.label]
        return '\t'.join(values)


class Sentence:
    def __init__(self, words, bert_token=None, lang_id=None):
        self.words = list(words)
        self.length = len(self.words)
        self.key_head = -1
        self.key_start = -1
        self.key_end = -1
        self.key_label = ""
        self.lang_id = lang_id

        if bert_token is not None:
            sentence_list = [word.org_form for word in self.words]
            self.list_bert_indice, self.list_segments_id, self.list_piece_id = bert_token.bert_ids(sentence_list)

        for idx in range(self.length):
            if words[idx].label.endswith("-*"):
                self.key_head = idx
                self.key_label = words[idx].label[2:-2]
                break

        for idx in range(self.length):
            cur_label = words[idx].label
            if cur_label.startswith("B-"+self.key_label) or cur_label.startswith("S-"+self.key_label):
                self.key_start = idx
            if cur_label.startswith("E-"+self.key_label) or cur_label.startswith("S-"+self.key_label):
                self.key_end = idx


def label_to_entity(labels):
    length = len(labels)
    entities = set()
    idx = 0
    while idx < length:
        if labels[idx] == "O":
            idx = idx + 1
        elif labels[idx].startswith("B-"):
            label = labels[idx][2:]
            predict = False
            if label.endswith("-*"):
                label = label[0:-2]
                predict = True
            next_idx = idx + 1
            end_idx = idx
            while next_idx < length:
                if labels[next_idx] == "O" or labels[next_idx].startswith("B-") \
                        or labels[next_idx].startswith("S-"):
                    break
                next_label = labels[next_idx][2:]
                if next_label.endswith("-*"):
                    next_label = next_label[0:-2]
                    predict = True
                if next_label != label:
                    break
                end_idx = next_idx
                next_idx = next_idx + 1
            if end_idx == idx:
                new_label = "S-" + labels[idx][2:]
                print("Change %s to %s" % (labels[idx], new_label))
                labels[idx] = new_label
            if not predict:
                entities.add("[%d,%d]%s"%(idx, end_idx, label))
            idx = end_idx + 1
        elif labels[idx].startswith("S-"):
            label = labels[idx][2:]
            predict = False
            if label.endswith("-*"):
                label = label[0:-2]
                predict = True
            if not predict:
                entities.add("[%d,%d]%s"%(idx, idx, label))
            idx = idx + 1
        elif labels[idx].startswith("M-"):
            new_label = "B-" + labels[idx][2:]
            print("Change %s to %s" % (labels[idx], new_label))
            labels[idx] = new_label
        else:
            new_label = "S-" + labels[idx][2:]
            print("Change %s to %s" % (labels[idx], new_label))
            labels[idx] = new_label

    return entities

def normalize_labels(labels):
    length = len(labels)
    change = 0
    normed_labels = []
    for idx in range(length):
        normed_labels.append(labels[idx])
    idx = 0
    while idx < length:
        if labels[idx] == "O":
            idx = idx + 1
        elif labels[idx].startswith("B-"):
            label = labels[idx][2:]
            if label.endswith("-*"):
                label = label[0:-2]
            next_idx = idx + 1
            end_idx = idx
            while next_idx < length:
                if labels[next_idx] == "O" or labels[next_idx].startswith("B-") \
                        or labels[next_idx].startswith("S-"):
                    break
                next_label = labels[next_idx][2:]
                if next_label.endswith("-*"):
                    next_label = next_label[0:-2]
                if next_label != label:
                    break
                end_idx = next_idx
                next_idx = next_idx + 1
            if end_idx == idx:
                new_label = "S-" + labels[idx][2:]
                #print("Change %s to %s" % (labels[idx], new_label))
                labels[idx] = new_label
                normed_labels[idx] = new_label
                change = change + 1
            idx = end_idx + 1
        elif labels[idx].startswith("S-"):
            idx = idx + 1
        elif labels[idx].startswith("M-"):
            new_label = "B-" + labels[idx][2:]
            #print("Change %s to %s" % (labels[idx], new_label))
            normed_labels[idx] = new_label
            labels[idx] = new_label
            change = change + 1
        else:
            new_label = "S-" + labels[idx][2:]
            #print("Change %s to %s" % (labels[idx], new_label))
            normed_labels[idx] = new_label
            labels[idx] = new_label
            change = change + 1

    return normed_labels, change

def getListFromStr(entity):
    entity_del_start = ''.join(list(entity)[1:])
    # entity_del_start: '2,3]TARGET'
    new_entity = entity_del_start.split(']')
    start, end = new_entity[0].split(',')
    start, end = int(start), int(end)
    # start: 2 end: 3
    label = new_entity[1]
    # label: 'TARGET'
    return [start, end, label]

def evalSRLExact(gold, predict):
    glength, plength = gold.length, predict.length
    if glength != plength:
        raise Exception('gold length does not match predict length.')

    goldlabels, predictlabels = [], []

    for idx in range(glength):
        goldlabels.append(gold.words[idx].label)
        predictlabels.append(predict.words[idx].label)

    # class set{'[2,4]TARGET', '[0,0]AGENT'}
    gold_entities = label_to_entity(goldlabels)
    # class set{'[2,3]TARGET', '[0,1]AGENT', '[2,4]ad>'}
    predict_entities = label_to_entity(predictlabels)
    gold_entity_num, predict_entity_num, correct_entity_num = len(gold_entities), len(predict_entities), 0

    gold_agent_entity_num, predict_agent_entity_num, correct_agent_entity_num = 0, 0, 0
    gold_target_entity_num, predict_target_entity_num, correct_target_entity_num = 0, 0, 0

    for entity in gold_entities:
        if entity.endswith('AGENT'):
            gold_agent_entity_num += 1
        elif entity.endswith('TARGET'):
            gold_target_entity_num += 1
    for entity in predict_entities:
        if entity.endswith('AGENT'):
            predict_agent_entity_num += 1
        elif entity.endswith('TARGET'):
            predict_target_entity_num += 1

    for one_entity in gold_entities:
        if one_entity in predict_entities:
            correct_entity_num = correct_entity_num + 1
            if one_entity.endswith('AGENT'):
                correct_agent_entity_num += 1
            elif one_entity.endswith('TARGET'):
                correct_target_entity_num += 1

    return gold_entity_num, predict_entity_num, correct_entity_num, \
           gold_agent_entity_num, predict_agent_entity_num, correct_agent_entity_num, \
           gold_target_entity_num, predict_target_entity_num, correct_target_entity_num

def jiaoji(a1, a2, b1, b2):
    if a1 == b1 and a2 == b2:
        return True
    else:
        list1 = list(range(a1, a2+1))
        list2 = list(range(b1, b2+1))
        if len(set(list1).intersection(set(list2))) != 0:
            return True
    return False

def contain_len(a1, a2, b1, b2):
    return len(set(list(range(a1, a2 + 1))).intersection(set(list(range(b1, b2 + 1)))))

def evalSRLBinary(gold, predict):
    glength, plength = gold.length, predict.length
    if glength != plength:
        raise Exception('gold length does not match predict length.')

    goldlabels, predictlabels = [], []

    for idx in range(glength):
        goldlabels.append(gold.words[idx].label)
        predictlabels.append(predict.words[idx].label)

    # class set{'[2,4]TARGET', '[0,0]AGENT'}
    gold_entities = label_to_entity(goldlabels)
    # class set{'[2,3]TARGET', '[0,1]AGENT', '[2,4]ad>'}
    predict_entities = label_to_entity(predictlabels)
    gold_entity_num, predict_entity_num, gold_correct_entity_num, predict_correct_entity_num = len(gold_entities), len(
        predict_entities), 0, 0

    gold_agent_entity_num, predict_agent_entity_num, gold_correct_agent_entity_num, predict_correct_agent_entity_num = 0, 0, 0, 0
    gold_target_entity_num, predict_target_entity_num, gold_correct_target_entity_num, predict_correct_target_entity_num = 0, 0, 0, 0

    for entity in gold_entities:
        if entity.endswith('AGENT'):
            gold_agent_entity_num += 1
        elif entity.endswith('TARGET'):
            gold_target_entity_num += 1

    for entity in predict_entities:
        if entity.endswith('AGENT'):
            predict_agent_entity_num += 1
        elif entity.endswith('TARGET'):
            predict_target_entity_num += 1

    for gold_entity in gold_entities:
        for predict_entity in predict_entities:
            gold_start, gold_end, gold_label = getListFromStr(gold_entity)
            predict_start, predict_end, predict_label = getListFromStr(predict_entity)
            if gold_label == predict_label and jiaoji(gold_start, gold_end, predict_start, predict_end):
                gold_correct_entity_num += 1
                if gold_label == 'AGENT':
                    gold_correct_agent_entity_num += 1
                elif gold_label == 'TARGET':
                    gold_correct_target_entity_num += 1
                break

    for predict_entity in predict_entities:
        for gold_entity in gold_entities:
            gold_start, gold_end, gold_label = getListFromStr(gold_entity)
            predict_start, predict_end, predict_label = getListFromStr(predict_entity)
            if gold_label == predict_label and jiaoji(gold_start, gold_end, predict_start, predict_end):
                predict_correct_entity_num += 1
                if gold_label == 'AGENT':
                    predict_correct_agent_entity_num += 1
                elif gold_label == 'TARGET':
                    predict_correct_target_entity_num += 1
                break

    return gold_entity_num, predict_entity_num, gold_correct_entity_num, predict_correct_entity_num, \
           gold_agent_entity_num, predict_agent_entity_num, gold_correct_agent_entity_num, predict_correct_agent_entity_num, \
           gold_target_entity_num, predict_target_entity_num, gold_correct_target_entity_num, predict_correct_target_entity_num

def evalSRLProportional(gold, predict):
    glength, plength = gold.length, predict.length
    if glength != plength:
        raise Exception('gold length does not match predict length.')

    goldlabels, predictlabels = [], []

    for idx in range(glength):
        goldlabels.append(gold.words[idx].label)
        predictlabels.append(predict.words[idx].label)

    # class set{'[2,4]TARGET', '[0,0]AGENT'}
    gold_entities = label_to_entity(goldlabels)
    # class set{'[2,3]TARGET', '[0,1]AGENT', '[2,4]ad>'}
    predict_entities = label_to_entity(predictlabels)
    gold_entity_num, predict_entity_num, gold_correct_entity_num, predict_correct_entity_num = len(gold_entities), len(
        predict_entities), 0, 0

    gold_agent_entity_num, predict_agent_entity_num, gold_correct_agent_entity_num, predict_correct_agent_entity_num = 0, 0, 0, 0
    gold_target_entity_num, predict_target_entity_num, gold_correct_target_entity_num, predict_correct_target_entity_num = 0, 0, 0, 0

    for entity in gold_entities:
        if entity.endswith('AGENT'):
            gold_agent_entity_num += 1
        elif entity.endswith('TARGET'):
            gold_target_entity_num += 1

    for entity in predict_entities:
        if entity.endswith('AGENT'):
            predict_agent_entity_num += 1
        elif entity.endswith('TARGET'):
            predict_target_entity_num += 1

    for gold_entity in gold_entities:
        for predict_entity in predict_entities:
            gold_start, gold_end, gold_label = getListFromStr(gold_entity)
            predict_start, predict_end, predict_label = getListFromStr(predict_entity)
            if gold_label == predict_label and jiaoji(gold_start, gold_end, predict_start, predict_end):
                correct_len = contain_len(gold_start, gold_end, predict_start, predict_end)
                gold_correct_rate = (correct_len / (gold_end - gold_start + 1))
                gold_correct_entity_num += gold_correct_rate
                if gold_label == 'AGENT':
                    gold_correct_agent_entity_num += gold_correct_rate
                elif gold_label == 'TARGET':
                    gold_correct_target_entity_num += gold_correct_rate
                break
    for predict_entity in predict_entities:
        for gold_entity in gold_entities:
            gold_start, gold_end, gold_label = getListFromStr(gold_entity)
            predict_start, predict_end, predict_label = getListFromStr(predict_entity)
            if gold_label == predict_label and jiaoji(gold_start, gold_end, predict_start, predict_end):
                correct_len = contain_len(gold_start, gold_end, predict_start, predict_end)
                predict_correct_rate = (correct_len / (predict_end - predict_start + 1))
                predict_correct_entity_num += predict_correct_rate
                if gold_label == 'AGENT':
                    predict_correct_agent_entity_num += predict_correct_rate
                elif gold_label == 'TARGET':
                    predict_correct_target_entity_num += predict_correct_rate
                break
    return gold_entity_num, predict_entity_num, gold_correct_entity_num, predict_correct_entity_num, \
           gold_agent_entity_num, predict_agent_entity_num, gold_correct_agent_entity_num, predict_correct_agent_entity_num, \
           gold_target_entity_num, predict_target_entity_num, gold_correct_target_entity_num, predict_correct_target_entity_num

def readSRL(file, bert_token=None, lang_id=None):
    min_count = 1
    total = 0
    words = []
    for line in file:
        tok = line.strip().split()
        if not tok or line.strip() == '' or line.strip().startswith('#'):
            if len(words) > min_count:
                total += 1
                yield Sentence(words, bert_token, lang_id)
            words = []
        elif len(tok) == 3:
            try:
                words.append(Word(int(tok[0]), tok[1], tok[2]))
            except Exception:
                pass
        else:
            pass

    if len(words) > min_count:
        total += 1
        yield Sentence(words, bert_token, lang_id)

    print("Total num: ", total)


def writeSRL(filename, sentences):
    with open(filename, 'w') as file:
        for sentence in sentences:
            for entry in sentence.words:
                file.write(str(entry) + '\n')
            file.write('\n')


def printSRL(output, sentence):
    for entry in sentence.words:
        output.write(str(entry) + '\n')
    output.write('\n')