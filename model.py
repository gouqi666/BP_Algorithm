import numpy as np
def fix_seed(seed = 43):
    np.random.seed(seed)
class Linear:
    def __init__(self,input_size,hidden_size,is_regular=False,regular_lambda = 1e-5):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.is_regular = is_regular
        self.regular_lambda = regular_lambda
        self.w = np.random.randn(input_size,hidden_size)
        self.b = np.random.randn(hidden_size)
        self.grad_w = np.random.randn(input_size,hidden_size)
        self.grad_b = np.random.randn(hidden_size)
    def forward(self,x):
        y = np.matmul(x,self.w) + self.b
        if self.is_regular:
            y += 0.5 * self.regular_lambda * np.sum(np.square(self.w))  # 正则项，lambda/2 * ||w||
        return y
    def backward(self,x,grad): # x是输入，grad是上一层反向传播回来的梯度
        self.grad_w =  np.matmul(x.T,grad)
        self.grad_b  = np.matmul(grad.T,np.ones(len(x)))
        if self.is_regular:
            self.grad_w += self.regular_lambda * self.w
        return np.matmul(grad,self.w.T) #  传给上一层的梯度，注意其相对位置不能出错
    def update(self,learning_rate):
        self.w -= self.grad_w * learning_rate
        self.b = self.grad_b * learning_rate
class RELU:
    def forward(self,x):
        return np.where(x < 0, 0, x)
    def backward(self,x,grad):
        return np.where(x>0,1,0) * grad # 这里激活函数是对每个元素求的，因此梯度也对每个元素反向传播即可，是对应元素相乘

class Scheduler:
    def __init__(self,optimizer,step_size):
        self.optimizer = optimizer
        self.step_size =  step_size
        self.gamma = 0.9
        self.current_pos = 1
    def step(self,step):
        if step // self.step_size > self.current_pos:
            self.optimizer.learning_rate *= self.gamma
            self.current_pos += 1

class Optimizer:
    def __init__(self,model,learning_rate):
        self.model = model
        self.learning_rate = learning_rate
    def step(self):
        for layer in self.model.layers:
            layer.update(self.learning_rate)

class Cross_Entropy:
    def __init__(self,weigths=[]):
        self.weights = weigths
    def forward(self,x,y):

        # 归一化缩小范围，否则计算softmax的时候要溢出
        assert len(y) == len(x)
        self.y = y # backward的时候用
        x = x - np.expand_dims(np.max(x,1),-1)
        # 进行softmax
        # x = np.exp(x) / np.sum(np.exp(x),1)
        # self.output = [np.log(x[i,yi]) for i,yi in enumerate(y)]  # 这样计算会溢出
        # for i in range(len(y)):
        #     each_sample = []
        #     for j in range(len(x[0])):
        #         each_sample.append((x[i, j] - np.log(np.sum(np.exp(x)[i])))) # 里面的是logp(x)

        ##  将上面的过程 用一行代码 表示出来，其实就是log_sum_exp函数
        self.output = x - np.expand_dims(np.log(np.sum(np.exp(x),1)),-1)  # 注意，这里的output是logp(x)而不是p(x),因此梯度回传的时候要加上exp
        loss = 0
        for i,yi in enumerate(y):
            loss += -self.output[i,yi] # 要加负号才是损失函数
        loss = loss / len(x)
        return loss / len(x)
    def backward(self): # cross_entropy(带softmax)关于MLP最终输出的梯度
        tmp = np.zeros_like(self.output)
        self.output = np.exp(self.output)
        for i,yi in enumerate(self.y): # one-hot
            tmp[i,yi] = 1
            if len(self.weights) != 0:
                tmp[i,yi] = self.weights[yi] # 带权重的交叉熵
                self.output[i] *= self.weights[yi]
        grad = self.output - tmp  # e(logp(x)) = p(X)
        return grad

class MLP:
    '''
    如果改模型结构需要改三个地方
    1.self.layers，这是要更新的层数，激活函数不用更新可以不包括
    2.self.prior,模型的运行顺序（包括激活函数）
    3.self.outputs,从最开始的输入x起到最终输出的前一个，这里需要记录中间输出以便后面算梯度。
    '''
    def __init__(self,input_size,initilization = 'rand',is_regular = False,regular_lambda = 1e-5):
        self.linear1 = Linear(input_size,100,is_regular,regular_lambda)
        self.relu = RELU()
        self.linear2 = Linear(100,100,is_regular,regular_lambda)
        self.linear3 = Linear(100,10,is_regular,regular_lambda)
        ## 记录参数
        self.layers = [self.linear1,self.linear2,self.linear3] #
        if initilization == 'He init':
            last_dim = 2 # 初始化设置2
            for layer in self.layers:
                layer.w = np.random.randn(layer.input_size,layer.hidden_size) * np.sqrt(2/last_dim)
                layer.b = np.random.randn(layer.hidden_size) * np.sqrt(2/last_dim)
                last_dim = layer.hidden_size

    def forward(self,x): ## 这里因为没用计算图，就必须将每步的输出结果保存下来
        o1 = self.linear1.forward(x)
        a1 = self.relu.forward(o1)
        o2 = self.linear2.forward(a1)
        a2 = self.relu.forward(o2)
        o3 = self.linear3.forward(a2)
        ## 记录运行顺序
        self.prior = [self.linear1,self.relu,self.linear2,self.relu,self.linear3] #
        self.outputs = [x,o1,a1,o2,a2] #
        return o3
    def backward(self,grad): # 这里假设loss一定是标量，不是标量可以转换成标量是一样的。
        # assert loss
        for layer,o in zip(self.prior[::-1],self.outputs[::-1]):
            grad = layer.backward(o,grad)
        return grad

