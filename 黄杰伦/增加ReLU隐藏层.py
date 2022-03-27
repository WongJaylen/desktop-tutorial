import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



file=pd.read_csv('data/train.csv')
df=pd.DataFrame(file)

#创建矩阵X存放样本不同特征的数据
X=np.array(df.iloc[:,1:-1]) #2万个样本，784个特征

#创建矩阵Y存放实际的y值
Y=np.array(df.iloc[:,-1],dtype=int)#2w*1



#归一化
X=(X-np.mean(X))/np.std(X)



#sotfmax回归模型
	

def Z(x,weight):#x是一个785*2w矩阵，weight是一个10*785矩阵
	return (weight.dot(x))#返回一个10*2w矩阵表示每个结点的值
		
def softmax(z):#传入一个10*2w矩阵
	#计算机处理数值位数有限，防止指数运算溢出，加上输入信号中的最大值的负数不会改变运算的结果
	c=np.max(z,axis=0)#每个样本中10个输出节点的最大值
	z_exp=np.exp(z-c)

	#print(np.sum(z_exp,axis=0))
	s=z_exp/np.sum(z_exp,axis=0)#是一个10*2w矩阵代表每个标签在每个样本的概率值

	return s
	#return int(argwhere(s==np.max(s))#放回概率最大值所在的索引表示预测的标签
	
def Relu(wx):
	return np.maximum(wx,0)
		
		 
def Loss(y,s,m,lamda):#损失函数
	
	return 1/m*(-1.0*np.sum(y*np.log(s+1e-5),axis=1))#+lamda/(2*m)*np.sum(W*W)




alpha=0.3#学习率
iteration=10000#迭代次数
m=len(X)#数据总数
W1=np.random.random((5,784))
W1=np.insert(W1,0,values=np.ones(5),axis=1)#添加偏置，5*785

X=np.insert(X,0,values=np.ones(20000),axis=1)#添加偏置,X为2w*785
X1=Relu(X.dot(W1.T))#(20000,5)


W2=np.random.random((10,5))#第二层和第三层之间的权重
X1=np.insert(X1,0,values=np.ones(20000),axis=1)#添加偏置,X1为2w*6
W2=np.insert(W2,0,values=np.ones(10),axis=1)#添加偏置，10*6

cost=[]
lamda=0.01
y=np.zeros((10,m))#创建10*2w矩阵存放每个样本值的标签向量
for i in range(m):
	y[Y[i],i]=1#要把Y标签值转换成向量



for i in range(2):#iteration
	z=Z(X1.T,W2)
	s=softmax(z)#10*2w
	loss=Loss(y,s,m,lamda)	#放回一个10*1矩阵存放每个标签单元总共产生的损失值
	cost.append(np.sum(loss))#-1.0*y*np.log(s+1e-5)#每个样本产生的损失值
	

	#求偏导
	#a=lamda*W[:,1:]#除偏置b其他全部用正则化惩罚
	#a=np.insert(a,0,values=W[:,0],axis=1)
	J=(s-y).dot(X1)#+a    #得到10*6矩阵
	
	#更新w值
	W2=W2-1/m*alpha*J
	c=1/m*alpha*np.sum(J[:,1:],axis=0)
	print(c.shape)
	W1=W1-1/m*alpha*np.sum(J[:,1:],axis=0)
	
	X1=Relu(X.dot(W1.T))
	
	

print(cost[-1])



plt.figure()
plt.plot(range(iteration),cost)
plt.show()
