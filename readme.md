## 本次作业全程使用numpy实现

###  data 是训练和测试数据目录， output是对应的结果图像目录。
###  dataset.py保存了dataset类和dataloader类
###  模型的各种实现都在model.py中，包括损失函数，激活函数，MLP,Optimizer，Scheduler
- 开始运行，可以直接运行train.py，在train.py中有baseline和各种优化及其说明。
- test.py仅仅实现predict函数，但是需要在train中调用，因为保存模型很麻烦。