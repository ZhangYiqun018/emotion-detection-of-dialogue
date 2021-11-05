from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
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

root = './train_data.csv'
sentences, labels, max_seq_len, max_seq_num = getSenLab(root)
word2num, num2word = getDict(sentences)
assert len(word2num) == len(num2word)
vocab_size = len(word2num)