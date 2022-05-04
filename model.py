from torch.utils.data import Dataset, DataLoader
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PoetryDataset(Dataset):
    def __init__(self, data, word_vec, word_to_index):
        self.word_vec = word_vec
        self.word_to_index = word_to_index
        self.data = data

    def __getitem__(self, index):
        sentence = self.data[index]
        sentence_index = [self.word_to_index[token]
                          for token in sentence.strip()]
        X_index = sentence_index[:-1]
        Y_index = sentence_index[1:]
        X_embedding = self.word_vec[X_index]
        return X_embedding, torch.tensor(Y_index, dtype=torch.int64)

    def __len__(self):
        return len(self.data)


class LSTM(nn.Module):

    def __init__(self, params):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_num = params["embedding_num"]
        self.hidden_num = params["hidden_num"]
        self.word_size = params["word_size"]
        self.word_vec = params["word_vec"]
        self.index_to_key = params["index_to_key"]
        self.key_to_index = params["key_to_index"]
        self.num_layers = params["num_layers"]
        self.lstm = nn.LSTM(input_size=self.embedding_num, hidden_size=self.hidden_num,
                            batch_first=True, num_layers=self.num_layers, bidirectional=False)
        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten(0, 1)
        self.linear1 = nn.Linear(self.hidden_num, 1000)
        self.linear2 = nn.Linear(1000, self.word_size)

    def forward(self, input, state):
        Y, state = self.lstm(input, state)
        Y = self.dropout(Y)
        Y = self.flatten(Y)
        Y = self.linear1(Y)
        Y = F.relu(Y)
        Y = self.dropout(Y)
        return self.linear2(Y), state

    def begin_state(self, batch_size=1):
        return (torch.zeros((self.lstm.num_layers, batch_size, self.hidden_num), device=self.device),
                torch.zeros((self.lstm.num_layers, batch_size, self.hidden_num), device=self.device))

    def generate_poetry(self):
        result = ""
        word_index = np.random.randint(0, self.word_size, 1)[0]
        result += self.index_to_key[word_index]
        state = self.begin_state()
        for i in range(31):
            word_embedding = torch.tensor(
                self.word_vec[word_index].reshape(1, 1, -1)).to(self.device)
            pre, state = self(word_embedding, state)
            word_index = int(torch.argmax(pre))
            result += self.index_to_key[word_index]
        print(result)

    def generate_cangtou_poetry(self):
        heads = input("请输入前四个字(输入q退出):")
        while heads != "q":
            res = ""
            if len(heads) < 4:
                print("长度不够!")
                heads = input("请输入前四个字(输入q退出):")
                continue
            with torch.no_grad():
                for i in range(4):
                    word = heads[i]
                    state = self.begin_state()
                    try:
                        word_index = self.key_to_index[word]
                    except:
                        word_index = np.random.randint(2, 1000)
                        word = self.index_to_key[word_index]
                    res += word
                    for j in range(6):
                        word_index = self.key_to_index[word]
                        word_embedding = torch.tensor(
                            self.word_vec[word_index].reshape(1, 1, -1)).to(self.device)
                        pre, state = self(word_embedding, state)
                        word = self.index_to_key[int(torch.argmax(pre))]
                        res += word
                    if i % 2 != 0:
                        res += "。"
                    else:
                        res += "，"
            print(res)
            heads = input("请输入前四个字(输入q退出):")
