import numpy as np
from model import MLP,Optimizer,Cross_Entropy,Scheduler
from dataset import Dataset,DataLoader
from matplotlib import pyplot as plt
import random
def fix_seed(seed = 43):
    random.seed(seed)
    np.random.seed(seed)
def plot(baseline_loss,other_loss,other_label = 'He initialization'):
    fig, ax = plt.subplots()
    ax.set_title('baseline vs ' + other_label)
    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')
    ax.plot(list(range(len(baseline_loss))),baseline_loss,color='red',label='baseline')
    ax.plot(list(range(len(other_loss))),other_loss,color='blue',label= other_label)
    ax.legend()
    fig.show()
    fig.savefig('output/baseline vs ' + other_label + '.png')

class Trainer():
    def __init__(self,initiliaztion = 'rand',adjust_lr = False):
        fix_seed()
        self.model = MLP(784,initilization= initiliaztion)
        self.learning_rate = 3e-3
        self.adjust_lr = adjust_lr
        self.optimizer = Optimizer(self.model,learning_rate=self.learning_rate)
        if adjust_lr:
            self.learning_rate = 3e-3
        self.scheduler = Scheduler(optimizer=self.optimizer,step_size=5000)
        self.critierion = Cross_Entropy()
        dataset = Dataset()
        self.dataloader = DataLoader(dataset, batch_size=32)
    def train(self):
        min_loss = float('inf')
        ret = []
        step = 0
        model = self.model
        dataloader = self.dataloader
        critierion = self.critierion
        optimizer = self.optimizer
        scheduler = self.scheduler
        epochs = 50
        for epoch in range(epochs):
            total_loss = []
            total_acc = []
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
                if self.adjust_lr:
                    scheduler.step(step)
                step += 1
            ret.append(sum(total_loss) / len(total_loss))
            print("epoch:%d,step:%d,loss:%.5f,acc:%.5f" % (epoch,step,sum(total_loss) / len(total_loss), sum(total_acc) / len(total_acc)))
        return ret
trainer = Trainer()
baseline_loss = trainer.train()

# 1. change the init method
# he_init_trainer = Trainer(initiliaztion='He init')
# he_init_loss = he_init_trainer.train()
# plot(baseline_loss,he_init_loss,other_label='He initilization')


# 3. change to  dynamic lr
dynamic_lr_trainer = Trainer(adjust_lr= True)
dynamic_lr_loss = dynamic_lr_trainer.train()
plot(baseline_loss,dynamic_lr_loss,other_label='Dynamic leraning rate')

## 4.  正则项

