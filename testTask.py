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
        x2 = self.PositionalEncoding(X[0]).to(device)
        x = x1 + x2
        x = x.to(torch.float32)
        x_trans = self.transformer_encoder(x)
        x_trans = x_trans.reshape(-1, x.shape[1] * x.shape[2]).to(device)
        # print(x_trans.device)
        layer = nn.Linear(x_trans.shape[1], self.output_dim)
        x_layer = layer(x_trans)
        output = nn.Softmax(dim=-1)(x_layer)
        return output

class Embedding(nn.Module):
    def __init__(self, vocab_size, input_dim, pad):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, input_dim, padding_idx=pad)
    def forward(self, X):
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
        sen = torch.LongTensor([self.word2num[ch] for ch in sen])
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
    word2num = {w:i for i, w in enumerate(s)}
    num2word = {i:w for i, w in enumerate(s)}
    word2num['_'] = len(word2num)
    num2word[len(num2word)] = '_'
    return word2num, num2word

def train_transformer(model, train_data, test_data):
    criteon = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = learn_rate)

    batch_number = len(train_data)
    # print(batch_number)
    for epoch in range(epochs):
        for batch_idx, x, label in enumerate(train_data):
            print(batch_idx, x, label)
            x, label = x.to(device), label.to(device)
            output = model(x)
            print(output.shape)
            loss = criteon(output, label)
            # if (batch_idx+1) % 50 == 0:
            print('epoch', '%04d,' % (epoch+1), 'step', f'{batch_idx+1} / {batch_number}, ', 'loss:', '{:.6f},'.format(loss.item()))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # model.eval()
        # with torch.no_grad():
        #     total_correct = 0
        #     total_num = 0
        #     for _, (x_t, label_t) in enumerate(test_data):
        #         x_t, label_t = x_t.to(device), label_t.to(device)
        #         valid_output = model(x_t)
        #         valid_loss = criteon(valid_output, label_t)
        #         pred = valid_output.argmax(dim=1)
        #         total_correct += torch.eq(pred, label_t).float().sum().item()
        #         total_num += x_t.size(0)
        #     acc = total_correct / total_num
        #     print(epoch, acc)

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = "cpu"
    root = './train_data.csv'
    sentences, labels, max_seq_len, max_seq_num = getSenLab(root)
    word2num, num2word = getDict(sentences)
    assert len(word2num) == len(num2word)
    vocab_size = len(word2num)

    # 超参数
    batch_size = 1
    dim_q = 20
    dim_k = 20
    dim_v = 80
    heads_num = 4
    input_dim = embedding_dim = 80
    pad = 0
    p_drop = 0.1
    hidden_dim = 100
    output_dim = 6
    learn_rate = 1e-3
    epochs = 1000
    num_layers = 1
    #

    dataset = MyData(sentences, labels, word2num)
    train_data = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    s, l = iter(train_data).next()
    print(s, l)

    # train_data, test_data = getTrain(sentences, labels, word2num)
    # model = Model()
    # model.to(device)
    # train_transformer(model, train_data, test_data)