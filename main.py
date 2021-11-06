# 这是汇总的
import random
import numpy as np
import torch
import math
import torch.nn as nn
import pandas as pd
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

"""
embedding               : [batch_size, seq_len, embedding_dim]
positionalEncoding      : [batch_size, seq_len, embedding_dim]
multiHeadAttention      : [batch_size, seq_len, embedding_dim]
add & norm              : [batch_size, seq_len, embedding_dim]
FeedForward             : [batch_size, seq_len, embedding_dim]
Linear                  : [batch_size, seq_len, output_dim]
"""

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.Embedding = Embedding(vocab_size, input_dim, pad)
        self.PositionalEncoding = PositionalEncoding(input_dim)
        encoder_layer = nn.TransformerEncoderLayer(input_dim, heads_num)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_dim = output_dim
    def forward(self, X):
        x1 = self.Embedding(X)
        # print(x1.shape)
        x2 = self.PositionalEncoding(x1[0]).to(device)
        x = x1 + x2
        x = x.to(torch.float32)
        x_trans = self.transformer_encoder(x)
        x_trans = x_trans.reshape(-1, x.shape[1] * x.shape[2])
        # x_t = x_trans.to('cpu')
        layer = nn.Linear(x_trans.shape[1], self.output_dim)
        layer.to(device)
        x_layer = layer(x_trans)
        # print(x_trans.device)
        output = nn.Softmax(dim=-1)(x_layer)
        return output

class Embedding(nn.Module):
    def __init__(self, vocab_size, input_dim, pad):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, input_dim, padding_idx=0)
    def forward(self, X):
        max_len = 0
        for i in range(len(X)):
            max_len = len(X[i]) if len(X[i]) > max_len else max_len
        for i in range(len(X)):
            if len(X[i]) < max_len:
                X[i].extend([0] * (max_len - len(X[i])))
        X = torch.LongTensor(X).to(device)
        return self.embedding(X)

class PositionalEncoding(nn.Module):
    def __init__(self, input_dim):
        super(PositionalEncoding, self).__init__()
        self.input_dim = input_dim
    def forward(self, X):
        seq_len = X.shape[0]
        pe = np.zeros((seq_len, self.input_dim))
        for i in range(seq_len):
            for j in range(self.input_dim):
                if j % 2 == 0:
                    pe[i][j] = math.sin(j / pow(10000, 2*j / self.input_dim))
                else:
                    pe[i][j] = math.cos(j / pow(10000, 2*j / self.input_dim))
        return torch.from_numpy(pe)

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
        sen = [self.word2num[ch] for ch in sen]
        lab = self.labels[idx] - 1
        return sen, lab

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
    word2num = {w:i+1 for i, w in enumerate(s)}
    num2word = {i+1:w for i, w in enumerate(s)}
    word2num['_'] = 0
    num2word[0] = '_'
    return word2num, num2word

def train_transformer(model, train_data, valid_data):
    criteon = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = learn_rate)

    batch_number = len(train_data)
    for epoch in range(epochs):
        for batch_idx, (X, label) in enumerate(train_data):
            label = torch.LongTensor(label).to(device)
            # print(x.shape, label.shape)
            output = model(X)
            # print(output.shape)
            loss = criteon(output, label)
            if (batch_idx+1) % 100 == 0:
                print('epoch', '%04d,' % (epoch+1), 'step', f'{batch_idx+1} / {batch_number}, ', 'loss:', '{:.6f},'.format(loss.item()))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        with torch.no_grad():
            total_correct = 0
            total_num = 0
            for _, (X, label_valid) in enumerate(valid_data):
                x_valid = X
                label_valid = torch.LongTensor(label_valid).to(device)
                valid_output = model(x_valid)
                # print(valid_output)
                valid_loss = criteon(valid_output, label_valid)
                pred = valid_output.argmax(dim=1)
                total_correct += torch.eq(pred, label_valid).float().sum().item()
                total_num += len(x_valid)
            acc = total_correct / total_num
            print(f'\nValidating at epoch', '%04d'% (epoch+1) , 'acc:', '{:.6f},'.format(acc))

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = "cpu"
    root = './train_data.csv'
    sentences, labels, max_seq_len, max_seq_num = getSenLab(root)
    word2num, num2word = getDict(sentences)
    assert len(word2num) == len(num2word)
    vocab_size = len(word2num)

    # 超参数
    batch_size = 32
    # dim_q = 64
    # dim_k = 64
    # dim_v = 128
    heads_num = 8
    input_dim = embedding_dim = 16
    pad = 0
    p_drop = 0.1
    hidden_dim = 500
    output_dim = 6
    learn_rate = 1e-3
    epochs = 1000
    num_layers = 4
    #
    dataset = MyData(sentences, labels, word2num)

    train_size = int(len(dataset) * 0.7)
    valid_size = len(dataset) - train_size
    train_data, valid_data = torch.utils.data.random_split(dataset, [train_size, valid_size])

    train_data = DataLoader(train_data, batch_size=batch_size, shuffle=False, collate_fn=my_collate)
    valid_data = DataLoader(valid_data, batch_size=batch_size, shuffle=False, collate_fn=my_collate)
    model = Model()
    model.to(device)
    # train_transformer(model, train_data, valid_data)

    model = Model(vocab_size)
