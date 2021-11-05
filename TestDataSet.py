from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import math
import pandas as pd
import torch

def getTrain(sentences, labels, word2num):
    l = len(sentences)
    t = []
    for s, l in zip(sentences, labels):
        s = [word2num[ch] for ch in s]
        s = Variable(torch.LongTensor(np.array(s)))
        t.append((s, l-1))
    return t[:int(0.7*l)], t[int(0.7*l):]

def getSenLab(root):
    file = pd.read_csv(root)
    df = pd.DataFrame(file)
    data = []
    for i in range(len(df)):
        d = df[i:i + 1]
        data.append(d)
    data = np.array(data)
    sentences = []
    Labels = []
    max_seq_len = 0
    max_seq_num = 0
    for sub in data:
        for d in sub:
            dialogs, labels = d[1], d[2]
            dialogs = dialogs.split('__eou__')
            dialogs = ["".join(dialog.split()) for dialog in dialogs]
            label = []
            while labels > 0:
                label.insert(0, labels % 10)
                labels = labels // 10
            max_seq_num = len(label) if max_seq_num < len(label) else max_seq_num
            for d, l in zip(dialogs, label):
                max_seq_len = len(d) if max_seq_len < len(d) else max_seq_len
                sentences.append(d)
                Labels.append(l)
    return sentences, Labels, max_seq_len, max_seq_num

def getDict(sentences):
    word2num = dict()
    num2word = dict()
    s = "".join(sentences)
    # 字典自动去重 序号有问题，要重新去重
    s = list(set([ch for ch in s]))
    word2num = {w:i for i, w in enumerate(s)}
    num2word = {i:w for i, w in enumerate(s)}
    word2num['_'] = len(word2num)
    num2word[len(num2word)] = '_'
    return word2num, num2word

def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]
class MyData(Dataset):
    def __init__(self, sentences, labels, word2num):
        self.sentences = sentences
        self.labels = labels
        self.word2num = word2num
    def __len__(self):
        return len(self.sentences)
    def __getitem__(self, idx):
        sen = self.sentences[idx]
        sen = torch.LongTensor([self.word2num[ch] for ch in sen])
        lab = self.labels[idx] - 1
        return sen, lab

if __name__ == '__main__':
    root = './train_data.csv'
    sentences, labels, max_seq_len, max_seq_num = getSenLab(root)
    word2num, num2word = getDict(sentences)
    assert len(word2num) == len(num2word)
    vocab_size = len(word2num)

    dataset = MyData(sentences, labels, word2num)
    train_data = DataLoader(dataset, batch_size=6, shuffle=False, collate_fn=my_collate)

    s, l = iter(train_data).next()
    print(s, l)
