import numpy as np
from MODEL import MLP,Optimizer,Cross_Entropy
from dataset import Dataset,DataLoader
import random
def fix_seed(seed = 43):
    random.seed(seed)
    np.random.seed(seed)

fix_seed()
model = MLP(784)
learning_rate = 1e-3
optimizer = Optimizer(model,learning_rate=learning_rate)
critierion = Cross_Entropy()

epochs = 10
dataset = Dataset()


min_loss = float('inf')
ans = []
## x -> (batch_size,input_size)
## y -> (batch_size,)
step = 0
for epoch in range(epochs):
    total_loss = []
    total_acc = []
    dataloader = DataLoader(dataset, batch_size=16)
    for data,label in dataloader:
        data =  data / 255
        logits = model.forward(data)
        loss = critierion.forward(logits,label)
        total_loss.append(loss)
        grad = critierion.backward()
        ans = np.argmax(logits,1)
        acc = sum(ans == label) / (len(data))
        total_acc.append(acc)
        model.backward(grad)
        optimizer.step()
        step += 1
    print("epoch:%d,step:%d,loss:%.5f,acc:%.5f" % (epoch,step,sum(total_loss) / len(total_loss), sum(total_acc) / len(total_acc)))