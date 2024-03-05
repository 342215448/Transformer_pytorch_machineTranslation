import spacy
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from collections import Counter
import torch
import numpy as np
import os
from torch.autograd import Variable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

lang_map = {
    'zh':'Chinese',
    'en':'English'
}

# C:\Users\Administrator\Documents\GitHub\Transformer_pytorch_machineTranslation\transformer\process.py
def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"文件夹 '{folder_path}' 创建成功")
def nopeak_mask(size):
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    np_mask = Variable(torch.from_numpy(np_mask==0))
    return np_mask


def create_masks(src, trg):
    src_mask = (src != 0).unsqueeze(-2).to(device)

    if trg is not None:
        trg_mask = (trg != 0).unsqueeze(-2).to(device)
        size = trg.size(1)  # get seq_len for matrix
        np_mask = nopeak_mask(size).to(device)
        trg_mask = trg_mask & np_mask

    else:
        trg_mask = None
    return src_mask, trg_mask
def read_file(src_path):
    text_list = []
    with open(src_path, 'r', encoding='utf-8') as reader:
        for line in reader:
            text_list.append(line.strip().replace('\n',''))
    return text_list
class CustomDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        # 在这里对数据进行必要的转换或处理，然后返回
        return sample
class Spacy_tokenizer(object):
    def __init__(self, lang):
        if lang == 'en':
            lang = 'en_core_web_trf'
        elif lang == 'fr':
            lang = 'fr_dep_news_trf'
        elif lang == 'zh':
            lang = 'zh_core_web_trf'
        else:
            print("Please use spacy download to download other language...")
        self.nlp = spacy.load(lang)
    def tokenizer(self, sentence):
        return [tok.text for tok in self.nlp.tokenizer(sentence)]
# 构建词汇表
def build_vocab(raw, tk):
    counter = Counter()
    # with open(filepath, 'r', encoding='utf-8') as f:
    #     for string_ in f:
    #         counter.update(tk.tokenizer(string_.strip()))
    for line in tqdm(raw, desc='building vocab'):
        counter.update(tk.tokenizer(line))
    return counter
# 将词汇表转换为词汇索引
def vocab_to_index(vocab):
    vocab_dict = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}
    for word, _ in tqdm(vocab.items(), desc='building index dict'):
        vocab_dict[word] = len(vocab_dict)
    return vocab_dict
def create_vocab(lang, path, TGT=False):
    print(f'Processing {lang_map[lang]}...')
    print(f'[1/7] loading raw data...')
    raw_data = read_file(path)
    print(f'[2/7] loading {lang_map[lang]}\'s tokenizer...')
    tk = Spacy_tokenizer(lang)
    print(f'[3/7] building vocab...')
    vocab_temp = build_vocab(raw_data, tk)
    print(f'[4/7] building index...')
    vocab_index = vocab_to_index(vocab_temp)
    print(f'[5/7] saving {lang_map[lang]} vocab dict...')
    create_folder_if_not_exists('dict')
    torch.save(vocab_index, 'dict/'+lang_map[lang]+'_dict.fpn')
    vocab_len = len(vocab_index)
    print(f'[6/7] building index...')
    raw_list = [tk.tokenizer(cur_line) for cur_line in tqdm(raw_data, desc='building index')]
    print(f'[7/7] building data...')
    raw_process = []
    if TGT:
        for line in tqdm(raw_list, desc='building data'):
            line.insert(0, '<sos>')
            line.append('<eos>')
            temp = [vocab_index[cur_word] for cur_word in line]
            raw_process.append(torch.tensor(temp, dtype=torch.float32))
    else:
        for line in tqdm(raw_list, desc='building data'):
            temp = [vocab_index[cur_word] for cur_word in line]
            raw_process.append(torch.tensor(temp, dtype=torch.float32))
    return vocab_len, raw_process
# 实现动态padding
def custom_collate(batch):
    max_src_lenth = 0
    max_tgt_lenth = 0
    # 找到该batch中最长的一个
    for example in batch:
        cur_src_len = len(example['src'])
        cur_tgt_len = len(example['tgt'])
        if cur_src_len > max_src_lenth:
            max_src_lenth = cur_src_len
        if cur_tgt_len > max_tgt_lenth:
            max_tgt_lenth = cur_tgt_len
    new_batch = []
    for example in batch:
        src_padding_len = max_src_lenth - len(example['src'])
        tgt_padding_len = max_tgt_lenth - len(example['tgt'])
        new_batch.append([
            torch.cat((example['src'], torch.zeros(src_padding_len, dtype=example['src'].dtype))),
            torch.cat((example['tgt'], torch.zeros(tgt_padding_len, dtype=example['tgt'].dtype)))
        ])
        # print(example)
    return {
        'src':torch.stack([cur_[0].to(torch.int64) for cur_ in new_batch]),
        'tgt':torch.stack([cur_[1].to(torch.int64) for cur_ in new_batch])
    }

def create_dataloader(src_lang, tgt_lang, src_path, tgt_path, batch_size):
    src_vocab_size, src_list = create_vocab(src_lang, src_path, False)
    tgt_vocab_size, tgt_list = create_vocab(tgt_lang, tgt_path, True)
    raw_list = []
    max_seq_len = 0
    if (len(src_list) == len(tgt_list)):
        for i in range(len(src_list)):
            raw_list.append({
                'src': src_list[i],
                'tgt': tgt_list[i]
            })
            max_seq_len = max(len(src_list[i]), len(tgt_list[i]))
    custom_dataset = CustomDataset(raw_list)
    dataloader = DataLoader(custom_dataset, batch_size=batch_size, collate_fn=custom_collate, num_workers=4, shuffle=True)
    return dataloader, src_vocab_size, tgt_vocab_size, max_seq_len
