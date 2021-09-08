from transformers import BertTokenizer
import re
zh_pattern = re.compile(u'[\u4e00-\u9fa5]+')


def contain_zh(word):
    '''
    判断传入字符串是否包含中文
    :param word: 待判断字符串
    :return: True:包含中文  False:不包含中文
    '''
    # word = word.decode()
    global zh_pattern
    match = zh_pattern.search(word)

    return match

class BertTokenHelper(object):
    def __init__(self, bert_vocab_file):
        # self.tokenizer = BertTokenizer.from_pretrained(bert_vocab_file, force_download=True)
        self.tokenizer = BertTokenizer.from_pretrained(bert_vocab_file)
        print("Load bert vocabulary finished")

    def word_len_of(self, text_list):
        length = []
        for word in text_list:
            # word = word.lower()
            # if word == '超过':
            #     print()
            if contain_zh(word):
                length.append(len(word))
            else:
                # ('a' <= word[0] and word[0] <= 'z') or word.isdigit()
                length.append(1)
        return length

    def filt(self, text_list):
        new_text_list = []
        if contain_zh(''.join(text_list)):
            for elem in text_list:
                elem = elem.lower()
                if elem in ['.', ',', '。', '，', '!', '！', '？', '?', ':', '：', '（', '）']:
                    new_text_list.append(elem)
                elif contain_zh(elem):
                    new_text_list.append(elem)
                else:
                    new_text_list.append('x')
        else:
            for elem in text_list:
                for char in ["'", ",", ":", "", '/', '//', '\\\\', '\\', "$", "*", '@', '.', '_', '&', '--', '-', '``', '`']:
                    if len(elem) != 1 and char in elem:
                        elem = elem.replace(char, '')
                    if len(elem) == 0:
                        elem = 'x'
                new_text_list.append(elem)
            # new_text_list = text_list
        return new_text_list

    def bert_ids(self, text_list):
        assert type(text_list) is list
        # print(text_list)
        text_list = self.filt(text_list)
        text = ' '.join(text_list)
        word_length_list = self.word_len_of(text_list)
        outputs = self.tokenizer.encode_plus(text, add_special_tokens=True, return_tensors='pt')
        bert_indice = outputs["input_ids"].squeeze()
        segments_id = outputs["token_type_ids"].squeeze()

        list_bert_indice = [idx.item() for idx in bert_indice]
        list_segments_id = [idx.item() for idx in segments_id]

        bert_tokens = self.tokenizer.convert_ids_to_tokens(list_bert_indice)
        tokens = self.tokenizer.convert_tokens_to_string(bert_tokens)

        # print(bert_tokens)
        # print(tokens)
        # print(word_length_list)
        list_piece_id = []
        sub_index = 0
        word_count = 0
        # in_word_flag = False
        for idx, bpe_u in enumerate(bert_tokens):
            if idx == 0:
                list_piece_id.append([idx])
            elif idx == len(bert_tokens) - 1:
                list_piece_id.append([idx])
            else:
                if not bpe_u.startswith('##') and sub_index == word_length_list[word_count]:
                    # print(word_count)
                    word_count += 1
                    sub_index = 0
                if sub_index == 0:
                    list_piece_id.append([idx])
                else:
                    list_piece_id[-1].append(idx)
                # in_word_flag = True
                if bpe_u.startswith('##'):
                    continue
                else:
                    sub_index += 1
                # if bpe_u.startswith('##'):
                #     in_word_flag = False
                # elif bpe_u == '<unk>':
                #     sub_index += 1
                # else:
                #     sub_index += 1

        # list_piece_id = []
        # for idx, bpe_u in enumerate(bert_tokens):
        #     if bpe_u.startswith("##"):
        #         tok_len = len(list_piece_id)
        #         list_piece_id[tok_len-1].append(idx)
        #     else:
        #         list_piece_id.append([idx])

        return list_bert_indice, list_segments_id, list_piece_id


# bert_token = BertTokenHelper('bert-base-multilingual-cased')
# bert_token.bert_ids(['He','likes','singing'])
# bert_token.bert_ids(['我','爱','这个','世界','singing!'])
