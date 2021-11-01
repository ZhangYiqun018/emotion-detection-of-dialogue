import csv
import numpy as np
import pandas as pd

def getDict(sentences):
    word2num = dict()
    num2word = dict()


    return word2num, num2word

if __name__ == '__main__':
    root = './train_data.csv'
    file = pd.read_csv(root)
    df = pd.DataFrame(file)
    data = []
    for i in range(len(df)):
        d = df[i:i+1]
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
    data = np.array(res, dtype=object)
    print(data.shape)
    np.save('./train_data.npy', data)