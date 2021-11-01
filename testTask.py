import csv
import numpy as np
import pandas as pd

def getDict(sentences):
    word2num = dict()
    num2word = dict()

    s = "".join(sentences)
    # 字典自动去重 序号有问题，要重新去重
    s = list(set([ch for ch in s]))
    word2num = {w:i for i, w in enumerate(s)}
    num2word = {i:w for i, w in enumerate(s)}
    return word2num, num2word

if __name__ == '__main__':
    root = './train_data.csv'
    file = pd.read_csv(root)
    df = pd.DataFrame(file)
    data = []
    for i in range(len(df)):
        d = df[i:i + 1]
        data.append(d)
    data = np.array(data)
    res = []
    for sub in data:
        for d in sub:
            dialogs, labels = d[1], d[2]
            # print(dialogs, labels)
            dialogs = dialogs.split('__eou__')
            # 去每句话的空格，现在还没去标点符号，不知道该不该去
            dialogs = ["".join(dialog.split()) for dialog in dialogs]
            label = []
            while labels > 0:
                label.insert(0, labels % 10)
                labels = labels // 10
            labels = label
            res.append([dialogs, labels])
    sentences = []
    labels = []
    max_seq_len = 0
    max_label_len = 0
    for a in res:
        max_label_len = len(a[0]) if len(a[0]) > max_label_len else max_label_len
        for sentence, label in zip(a[0], a[1]):
            max_seq_len = len(sentence) if len(sentence) > max_seq_len else max_seq_len
            sentences.append(sentence)
            labels.append(label)

    word2num, num2word = getDict(sentences)
    vocab_size = len(word2num)
    print(vocab_size)