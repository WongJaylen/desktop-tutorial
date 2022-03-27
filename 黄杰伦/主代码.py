import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''  对数据的预处理
      并将处理好的数据保存到新的文件'''

#该处理失败
'''
#对训练样本处理
file=pd.read_csv('train.csv')
df=pd.DataFrame(file)   

X_train=np.array(df.iloc[:,1:])#包含标签值
index=np.argwhere(np.sum(X_train,axis=0)<=0) #找出特征值全为0的索引
print(index.shape)#(91,1)


for i in range(len(index)-1,-1,-1):#按index从大到小删除
	X_train=np.delete(X_train,index[i],axis=1)
print(X_train.shape) 
      
df1=pd.DataFrame(X_train)
df1.to_csv('train1.csv')     
      
#对测试样本处理,将在训练样本删除的特征值也在测试样本中删除
file=pd.read_csv('test.csv')
df=pd.DataFrame(file)     
X_test=np.array(df.iloc[:,1:])    
 
for i in range(len(index)-1,-1,-1):#按index从大到小删除
	X_test=np.delete(X_test,index[i],axis=1)    

df1=pd.DataFrame(X_test)
df1.to_csv('test1.csv')   '''
      
      



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
	
def Loss(y,s,m,lamda):#损失函数
	
	return 1/m*(-1.0*np.sum(y*np.log(s+1e-5),axis=1))#+lamda/(2*m)*np.sum(W*W)


file=pd.read_csv('data/train.csv')
df=pd.DataFrame(file)

#创建矩阵X存放样本不同特征的数据
X=np.array(df.iloc[:,1:-1]) #2万个样本，784个特征

#创建矩阵Y存放实际的y值
Y=np.array(df.iloc[:,-1],dtype=int)#2w*1


#归一化
X=(X-np.mean(X))/np.std(X)

alpha=0.3#学习率
iteration=10000#迭代次数
m=len(X)#数据总数
W=np.random.random((10,784))#权重

X=np.insert(X,0,values=np.ones(20000),axis=1)#添加偏置,X为2w*785
W=np.insert(W,0,values=np.ones(10),axis=1)#添加偏置，10*785

cost=[]#存放损失值

lamda=0.01#正则化参数

y=np.zeros((10,m))#创建10*2w矩阵存放每个样本值的标签向量
for i in range(m):
	y[Y[i],i]=1#要把Y标签值转换成向量


#梯度下降迭代
for i in range(iteration):#iteration
	z=Z(X.T,W)
	s=softmax(z)#10*2w
	loss=Loss(y,s,m,lamda)	#放回一个10*1矩阵存放每个标签单元总共产生的损失值
	cost.append(np.sum(loss))#-1.0*y*np.log(s+1e-5)#每个样本产生的损失值
	
	#a=lamda*W[:,1:]#除偏置b其他全部用正则化惩罚
	#a=np.insert(a,0,values=W[:,0],axis=1)
	J=(s-y).dot(X)#+a    #求偏导，得到10*785矩阵
	
	#更新w值
	W=W-1/m*alpha*J
	
	
print(cost[-1])



plt.figure()
plt.plot(range(iteration),cost)
plt.show()







'''
0.2   0.18909644132174977
0.3 一万次0.15681448019723054
'''



#对测试集训练，并保存结果到文件
'''
file=pd.read_csv('data/test.csv')
df1=pd.DataFrame(file)
X1=np.array(df1.iloc[:,1:])#1w*
X1=(X1-np.mean(X))/np.std(X)

X1=np.insert(X1,0,values=np.ones(10000),axis=1)#添加偏置

z1=Z(X1.T,W)
s1=softmax(z1)#10*2w

Y1=np.zeros((10000,1))
for i in range(len(X1)):
	a=np.where(s1[:,i]==np.max(s1[:,i]))#找出
	Y1[i]=int(a[0])

data=pd.DataFrame(Y1)#,index=range(1,len(Y1)+1)
data.columns=['label']
data.to_csv('outcome.csv',index_label='id')
'''






    
   
    
 
