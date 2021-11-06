import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class MyData(Dataset):
    def __init__(self, data, word2num):
        self.data = data
        self.word2num = word2num
    def __getitem__(self, idx):
        sen = self.data[idx][0]
        for i in range(len(sen)):
            sen[i] = [self.word2num[ch] for ch in sen[i]]
            sen[i] = torch.LongTensor(sen[i])
        lab = self.data[idx][1]
        lab = [l - 1 for l in lab]
        return sen, lab
    def __len__(self):
        return len(self.data)

def my_collate(batch):
    sen = [item[0] for item in batch]
    label = [item[1] for item in batch]
    return [sen, label]


def isNecessary(ch):
    if '\u4e00' <= ch <= '\u9fff':
        return True
    elif ch.isalpha():
        return True
    elif ch.isdigit():
        return True
    return False

def getData(root):
    file = pd.read_csv(root)
    df = pd.DataFrame(file)
    data = []
    word2num = dict()
    idx = 1
    for i in range(len(df)):
        d = df[i:i + 1]
        data.append(d)
    data = np.array(data)
    res = []
    for i in range(len(data)):
        _, sen, label = data[i][0]
        sen = sen.split('__eou__')
        for j in range(len(sen)):
            s_t = []
            for ch in sen[j]:
                if isNecessary(ch):
                    if ch not in word2num:
                        word2num[ch] = idx
                        idx += 1
                    s_t.append(ch)
            sen[j] = "".join(s_t)
        lab = []
        while label > 0:
            lab.insert(0, label % 10)
            label = label // 10
        res.append((sen, lab))

    return word2num, res

if __name__ == '__main__':
    root = 'train_data.csv'

    word2num, data = getData(root)
    vocab_size = len(word2num)
    batch_size = 6

    train_data = MyData(data, word2num)
    train_data = DataLoader(train_data, batch_size=batch_size, shuffle=False, collate_fn=my_collate)

    sen, lab = iter(train_data).next()
    print(sen, lab)
