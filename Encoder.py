# 这是汇总的
import numpy as np
import torch
import math
import torch.nn as nn
import pandas as pd
from torch.autograd import Variable

"""
embedding               : [batch_size, seq_len, embedding_dim]
positionalEncoding      : [batch_size, seq_len, embedding_dim]
multiHeadAttention      : [batch_size, seq_len, embedding_dim]
add & norm              : [batch_size, seq_len, embedding_dim]
FeedForward             : [batch_size, seq_len, embedding_dim]
Linear                  : [batch_size, seq_len, output_dim]
"""

class Encoder(nn.Module):
    def __init__(self, vocab_size):
        super(Encoder, self).__init__()
        self.Embedding = Embedding(vocab_size, input_dim, pad)
        self.PositionalEncoding = PositionalEncoding(input_dim)
        self.MultiHeadAttention = MultiHeadAttention(input_dim, dim_q, dim_k, dim_v, heads_num, required_mask=False)
        self.dropout = nn.Dropout(p_drop)
        self.FeedForward = FeedForward(input_dim, hidden_dim)
        self.fc = nn.Linear(input_dim * seq_len, output_dim)
    def forward(self, X):
        x1 = self.Embedding(X)
        x2 = self.PositionalEncoding(X[0])
        # 利用广播机制
        x = x1 + x2
        x = x.to(torch.float32)
        # 多头
        x_multi = self.MultiHeadAttention(x, y=x)
        x = self.dropout(x + x_multi)
        ln = nn.LayerNorm(x.shape[1:])
        x = ln(x)
        x_feedforward = self.FeedForward(x)
        x = self.dropout(x + x_feedforward)
        ln = nn.LayerNorm(x.shape[1:])
        x = ln(x)
        x = x.reshape(-1, x.shape[1] * x.shape[2])
        print(x.shape)
        x = self.fc(x)
        x = nn.Softmax(dim=-1)(x)
        return x


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

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, dim_q, dim_k, dim_v, heads_num, required_mask=False):
        super(MultiHeadAttention, self).__init__()
        assert dim_k % heads_num == 0
        assert dim_v % heads_num == 0
        self.Q = nn.Linear(input_dim, dim_q)
        self.K = nn.Linear(input_dim, dim_k)
        self.V = nn.Linear(input_dim, dim_v)
        self.out = nn.Linear(dim_v, input_dim)

        self.heads_num = heads_num
        self.dim_q = dim_q
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.required_mask = required_mask
    def getMask(self, dim):
        mask = np.ones((dim, dim))
        mask = torch.tensor(np.tril(mask))
        return mask.bool()
    def forward(self, X, y):
        Q = self.Q(X).reshape(-1, X.shape[0], X.shape[1], self.dim_q // self.heads_num)
        K = self.K(X).reshape(-1, X.shape[0], X.shape[1], self.dim_k // self.heads_num)
        V = self.V(y).reshape(-1, y.shape[0], y.shape[1], self.dim_v // self.heads_num)
        output = torch.matmul(Q, K.permute(0, 1, 3, 2)) / math.sqrt(self.dim_k)
        if self.required_mask == True:
            mask = self.getMask(X.shape[1])
            # print(output.shape, mask.shape)
            output = torch.masked_fill(output, mask, value=float("-inf"))
        output = nn.Softmax(-1)(output)
        output = torch.matmul(output, V).reshape(X.shape[0], X.shape[1], -1)
        return self.out(output)

class FeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FeedForward, self).__init__()
        self.Layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    def forward(self, X):
        return self.Layer(X)

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
    return word2num, num2word

if __name__ == '__main__':
    root = './train_data.csv'
    sentences, labels, max_seq_len, max_seq_num = getSenLab(root)
    word2num, num2word = getDict(sentences)
    assert len(word2num) == len(num2word)
    vocab_size = len(word2num)
    print(len(sentences), len(labels), vocab_size)

    # 超参数
    dim_q = 20
    dim_k = 20
    dim_v = 80
    heads_num = 4
    input_dim = embedding_dim = 80
    pad = 0
    p_drop = 0.1
    hidden_dim = 100
    output_dim = 6
    seq_len = 3
    #
    # sen2idx = []
    # for sen in sentences:
    #     ss = []
    #     for s in sen.split():
    #         ss.append(word2num[s])
    #     sen2idx.append(ss)
    # x = Variable(torch.LongTensor(sen2idx))
    # model = Encoder(vocab_size)
    # y = model(x)
    # print(y, y.shape)


