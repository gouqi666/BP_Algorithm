import numpy as np
from utils import open_image,open_label
class Dataset:
    def __init__(self,train_data_path,train_label):
        self.train_data = open_image(train_data_path)
        self.train_label = open_label(train_label)
        self.label_num = [0] * 10
        for label in self.train_label:
            self.label_num[int(label)] += 1
        num_max = max(self.label_num)
        self.weights = num_max / np.array(self.label_num)
        self.index = 0
    def __getitem__(self, item):
        return np.stack(self.train_data[item]),np.stack(self.train_label[item])
    def __len__(self):
        return len(self.train_data)
class DataLoader:
    def __init__(self,dataset,batch_size):
        self.batch_size  = batch_size
        self.dataset = dataset
        self.data_len = len(self.dataset)
        self.index = 0
    def __next__(self):
        if self.index < self.data_len:
            data = self.dataset[self.index:min(self.index+self.batch_size,self.data_len)]
            self.index += self.batch_size
            return data
        else:
            self.index = 0
            raise StopIteration
    def __iter__(self):
        return self


if __name__ == "__main__":
    dataset = Dataset()
    dataloader = DataLoader(dataset,batch_size=16)
    for data,label in dataloader:
        print(data,label)
        break

