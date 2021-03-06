from singleData import MyData
from singleData import getSenLab
from singleData import getDict
from singleData import my_collate

import torch
import math
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

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
        encoder_layer = nn.TransformerEncoderLayer(input_dim, heads_num, hidden_dim, p_drop)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_dim = output_dim
        # self.fc = nn.Linear(, output_dim)
    def forward(self, X):
        x1 = self.Embedding(X)
        x2 = self.PositionalEncoding(x1[0]).to(device)
        x = x1 + x2
        x = x.to(torch.float32)
        x_trans = self.transformer_encoder(x)
        # print(x_trans.shape)
        x1, x2, x3 = x_trans.shape
        layer = nn.Linear(x2*x3, output_dim).to(device)
        output = x_trans.reshape(x1, x2*x3)
        output = layer(output).to(device)
        output = nn.Softmax(dim=-1)(output)
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
        X = Variable(torch.LongTensor(X).to(device))
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


def train_transformer(model, train_data, valid_data):
    criteon = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = learn_rate)

    batch_number = len(train_data)
    for epoch in range(epochs):
        model.train()
        for batch_idx, (X, label) in enumerate(train_data):
            label = Variable(torch.LongTensor(label).to(device))
            # print(len(X), label.shape)
            output = model(X)
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
                label_valid = Variable(torch.LongTensor(label_valid).to(device))
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

    # ?????????
    batch_size = 32
    heads_num = 8
    input_dim = embedding_dim = 512
    pad = 0
    p_drop = 0.4
    hidden_dim = 256
    output_dim = 6
    learn_rate = 1e-3
    epochs = 20
    num_layers = 1
    #
    dataset = MyData(sentences, labels, word2num)

    train_size = int(len(dataset) * 0.7)
    valid_size = len(dataset) - train_size
    train_data, valid_data = torch.utils.data.random_split(dataset, [train_size, valid_size])

    train_data = DataLoader(train_data, batch_size=batch_size, shuffle=False, collate_fn=my_collate)
    valid_data = DataLoader(valid_data, batch_size=batch_size, shuffle=False, collate_fn=my_collate)

    # x, label = iter(valid_data).next()
    # print(x, label)

    model = Model()
    model.to(device)
    train_transformer(model, train_data, valid_data)