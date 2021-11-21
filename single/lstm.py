from singleData import MyData
from singleData import getSenLab
from singleData import getDict
from singleData import my_collate

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader

"""
TP 预测为1 实际为1 预测正确
FP 预测为1 实际为0 预测错误
FN 预测为0 实际为1 预测错误
TN 预测为0 实际为0 预测正确
"""
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.Embedding = Embedding(vocab_size, input_dim, pad)
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, bidirectional=True, dropout=p_dropout)
        self.dropout = nn.Dropout(p_dropout)
        self.layer = nn.Linear(2*hidden_dim, output_dim)
    def forward(self, X):
        embedding = self.Embedding(X)
        embedding = embedding.to(torch.float32)
        lstm, (hidden, cell) = self.rnn(embedding.transpose(0, 1))
        # print(hidden.shape)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        # print(hidden.shape)
        hidden = self.dropout(hidden)
        out = self.layer(hidden)
        # print(out.shape)
        return out

class Embedding(nn.Module):
    def __init__(self, vocab_size, input_dim, pad):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, input_dim, padding_idx=pad)
    def forward(self, X):
        max_len = 0
        for i in range(len(X)):
            max_len = len(X[i]) if len(X[i]) > max_len else max_len
        for i in range(len(X)):
            if len(X[i]) < max_len:
                X[i].extend([0] * (max_len - len(X[i])))
        X = Variable(torch.LongTensor(X).to(device))
        return self.embedding(X)

def train_lstm(model, train_data, valid_data):
    criteon = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr = learn_rate)

    batch_number = len(train_data)
    for epoch in range(epochs):
        model.train()
        for batch_idx, (X, label) in enumerate(train_data):
            label = Variable(torch.LongTensor(label).to(device))
            # print(label.shape)
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
            confused_matrix = torch.zeros(output_dim, output_dim).to(device)
            for _, (X, label_valid) in enumerate(valid_data):
                x_valid = X
                label_valid = Variable(torch.LongTensor(label_valid).to(device))
                valid_output = model(x_valid)
                # print(valid_output)
                valid_loss = criteon(valid_output, label_valid)
                pred = valid_output.argmax(dim=1)
                # print(pred, label_valid)
                for p, l in zip(pred, label_valid):
                    confused_matrix[p][l] += 1
                total_correct += torch.eq(pred, label_valid).float().sum().item()
                total_num += len(x_valid)
            acc = total_correct / total_num
            TP, FN, FP = getScore(confused_matrix)
            prec = float(TP / (TP + FP))
            recall = float(TP / (TP + FN))
            F1 = float(2 * prec * recall / (prec + recall))
            print(f'\nValidating at epoch', '%04d'% (epoch+1) , 'acc:{:6f},'.format(acc), 'prec:', '{:.6f},'.format(prec), 'recall:', '{:.6f},'.format(recall), 'F1:{:.6f}'.format(F1))

def getScore(confused_matrix):
    TP, FP, FN = 0, 0, 0
    for i in range(output_dim):
        for j in range(output_dim):
            if i == j:
                TP += confused_matrix[i][j]
            if i >= 1:
                FN += confused_matrix[i][j]
            if j >= 1:
                FP += confused_matrix[i][j]
    return TP, FN - TP, FP - TP

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = "cpu"
    root = './train_data.csv'
    sentences, labels, max_seq_len, max_seq_num = getSenLab(root)
    word2num, num2word = getDict(sentences)
    assert len(word2num) == len(num2word)
    vocab_size = len(word2num)

    # 超参数
    batch_size = 64
    input_dim = embedding_dim = 512
    pad = 0
    hidden_dim = 150
    output_dim = 6
    learn_rate = 1e-3 * 0.6
    epochs = 20
    num_layers = 1
    p_dropout = 0.1
    #
    dataset = MyData(sentences, labels, word2num)

    train_size = int(len(dataset) * 0.7)
    valid_size = len(dataset) - train_size
    train_data, valid_data = torch.utils.data.random_split(dataset, [train_size, valid_size])

    train_data = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=my_collate)
    valid_data = DataLoader(valid_data, batch_size=batch_size, shuffle=True, collate_fn=my_collate)

    # x, label = iter(valid_data).next()
    # print(x, label)

    model = Model()
    model.to(device)
    train_lstm(model, train_data, valid_data)
