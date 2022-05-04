import os
import torch
import torch.nn as nn
from model import PoetryDataset, LSTM
from utils import data_process, get_wordvec
from torch.utils.data import Dataset, DataLoader

if __name__ == '__main__':
    os.chdir("./LSTM/")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_split, data_naive = data_process(
        path=r"C:\Users\86158\Desktop\python project\LSTM\dataset\poetry_dataset.txt")
    word_vec, key_to_index, index_to_key = get_wordvec(data_split.split("\n"))
    batch_size = 128
    dataset = PoetryDataset(data_naive, word_vec, key_to_index)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, drop_last=True)
    hidden_num = 256
    lr = 0.01
    num_epochs = 1000
    word_size, embedding_num = word_vec.shape
    params = {
        "embedding_num": embedding_num,
        "hidden_num": hidden_num,
        "word_size": word_size,
        "word_vec": word_vec,
        "key_to_index": key_to_index,
        "index_to_key": index_to_key,
        "num_layers": 2
    }

    model = LSTM(params)
    model = model.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        print("Epoch:", epoch, end=" ")
        for batch_index, (X, Y) in enumerate(dataloader):
            state = model.begin_state(batch_size=batch_size)
            for s in state:
                s.detach_()
            X, Y = X.to(device), Y.to(device)
            pred, state = model(X, state)
            loss = loss_function(pred, Y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_index % 100 == 0:
                print(f"loss:{loss:.3f}")
                model.generate_poetry()
    torch.save(model.state_dict(), "./save/model.pkl")
