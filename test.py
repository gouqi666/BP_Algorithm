import numpy as np
import torch

#  1.
x = [[1,2,3],[3,5,2],[5,8,10],[10,2,1]]
y = [2,1,2,0]
x = x - np.expand_dims(np.max(x, 1), -1)
print(x)
output = x - np.expand_dims(np.log(np.sum(np.exp(x), 1)), -1)
loss = 0
for i, yi in enumerate(y):
    loss += -output[i, yi]  # 要加负号才是损失函数
print( loss / len(x))


#  2.
critierion = torch.nn.CrossEntropyLoss()
print(critierion(torch.tensor(x,dtype = torch.float),torch.tensor(y,dtype=torch.long)))


#  3.
print(np.expand_dims(np.mean(np.exp(x),1),1))
x = np.exp(x) /  np.expand_dims(np.mean(np.exp(x),1),1)
loss = 0
for i, yi in enumerate(y):
    loss += -output[i, yi]  # 要加负号才是损失函数
print(loss / len(x))
