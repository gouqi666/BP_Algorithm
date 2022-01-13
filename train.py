import numpy as np
from model import MLP,Optimizer,Cross_Entropy,Scheduler
from dataset import Dataset,DataLoader
from matplotlib import pyplot as plt
from test import Predict
import argparse
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
    def __init__(self,initiliaztion = 'rand',lr = 3e-3,epochs = 100,batch_size = 32,step_size = 10000, adjust_lr = False,is_regular = False,use_weights = False):
        fix_seed()
        self.is_regular = is_regular
        self.dataset = Dataset("./data/train-images.idx3-ubyte","./data/train-labels.idx1-ubyte")
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size)
        self.model = MLP(784,initilization= initiliaztion,is_regular = self.is_regular)
        self.learning_rate = lr
        self.epochs = epochs
        self.adjust_lr = adjust_lr
        self.optimizer = Optimizer(self.model,learning_rate=self.learning_rate)
        self.scheduler = Scheduler(optimizer=self.optimizer,step_size=step_size)
        self.critierion = Cross_Entropy()
        if use_weights:
            self.critierion = Cross_Entropy(self.dataset.weights)

    def train(self):
        min_loss = float('inf')
        ret = []
        step = 0
        model = self.model
        dataloader = self.dataloader
        critierion = self.critierion
        optimizer = self.optimizer
        scheduler = self.scheduler
        epochs = self.epochs
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
    def predict(self):
        Predict(self.model)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr',type=int,default= 3e-3,help="learning rate, notice: don't be more than 5e-3")
    parser.add_argument('--epochs',type=int,default=100,help='epochs')
    parser.add_argument('--step-size',type=int,default=50000,help='step_size')
    parser.add_argument('--batch-size',type=int,default=32,help='batch_size')
    args = parser.parse_args()
    return args
if __name__  == "__main__":
    args = get_args()
    print("baseline  train ")
    trainer = Trainer(lr=args.lr,epochs=args.epochs)
    baseline_loss = trainer.train()

    print("baseline  test")
    trainer.predict()
    # 1. change the init method
    print("He init train")
    he_init_trainer = Trainer(lr=args.lr,epochs=args.epochs,initiliaztion='He init')
    he_init_loss = he_init_trainer.train()
    plot(baseline_loss,he_init_loss,other_label='He initilization')
    print("He init test")
    he_init_trainer.predict()
    # 2. change to weights cross_entropy
    print("weights cross_entropy train")
    weights_trainer = Trainer(lr=args.lr,epochs=args.epochs,use_weights=True)
    weights_loss = weights_trainer.train()
    plot(baseline_loss,weights_loss,other_label='weights cross_entropy')
    print("weights cross_entropy test")
    weights_trainer.predict()
    # 3. change to  dynamic lr
    print("dynamic lr train")
    dynamic_lr_trainer = Trainer(lr=args.lr,epochs=args.epochs,adjust_lr= True)
    dynamic_lr_loss = dynamic_lr_trainer.train()
    plot(baseline_loss,dynamic_lr_loss,other_label='Dynamic leraning rate')
    print("dynamic lr test")
    dynamic_lr_trainer.predict()
    ## 4.  regular term
    print("regular term train")
    regular_trainer = Trainer(lr=args.lr,epochs=args.epochs,is_regular= True)
    regular_loss = regular_trainer.train()
    plot(baseline_loss,regular_loss,other_label='regular term')
    print("regular term tst")
    regular_trainer.predict()

    ## 5.  最优情况，使用He initilization
    print("best train")
    args.epochs = 200 #
    args.step_size = 50000
    best_trainer = Trainer(initiliaztion="He init",lr=args.lr,epochs=args.epochs,batch_size= args.batch_size,step_size= args.step_size,adjust_lr=True,is_regular=True)
    best_loss = best_trainer.train()
    best_trainer.predict()

