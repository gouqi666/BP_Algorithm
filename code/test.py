import numpy as np
import torch
from model import MLP,Optimizer,Cross_Entropy,Scheduler
from dataset import Dataset,DataLoader


def Predict(model):
    dataset = Dataset("./data/t10k-images.idx3-ubyte", "./data/t10k-labels.idx1-ubyte")
    test_dataloader = DataLoader(dataset, batch_size=32)
    test_dataloader = test_dataloader
    total_acc = []
    for data,label in test_dataloader:
        data =  data / 255
        logits = model.forward(data)
        ans = np.argmax(logits,1)
        acc = sum(ans == label) / (len(data))
        total_acc.append(acc)
    print("Test:   acc:%.5f" % (sum(total_acc) / len(total_acc)))
