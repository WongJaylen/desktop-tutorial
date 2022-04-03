import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


'''  对数据的预处理，删去冗杂特征
      并将处理好的数据保存到新的文件'''
      
'''
file=pd.read_csv('data/train.csv')
df=pd.DataFrame(file)         
X_train=np.array(df.iloc[:,1:-1])


index=np.argwhere(np.sum(X_train,axis=0)/len(X_train)<=0.1)#找出平均值小于0.1的特征
#在训练集中进行特征删除
for i in range(len(index)-1,-1,-1):
	X_train=np.delete(X_train,index[i],axis=1)

X_train=np.insert(X_train,0,values=np.array(df.iloc[:,-1]),axis=1)

df1=pd.DataFrame(X_train)
df1.to_csv('data/train1.csv')     
      
#对测试样本处理,将在训练样本删除的特征值也在测试样本中删除
file=pd.read_csv('data/test.csv')
df=pd.DataFrame(file)     

X_test=np.array(df.iloc[:,1:])

for i in range(len(index)-1,-1,-1):#按index从大到小删除
	X_test=np.delete(X_test,index[i],axis=1)    
	
df1=pd.DataFrame(X_test)
df1.to_csv('data/test1.csv')  
'''




#sotfmax回归模型

#输入层与权重做线性运算
def Z(x,weight):#x是一个598*2w矩阵，weight是一个10*598矩阵
	return (weight.dot(x))#返回一个10*2w矩阵表示每个结点的值
	
		
def softmax(z):#传入一个10*2w矩阵
	#计算机处理数值位数有限，防止指数运算溢出，加上输入信号中的最大值的负数不会改变运算的结果
	c=np.max(z,axis=0)#每个样本中10个输出节点的最大值
	z_exp=np.exp(z-c)

	s=z_exp/np.sum(z_exp,axis=0)#是一个10*2w矩阵代表每个标签在每个样本的概率值
	return s
	
	
	
def Loss(y,s,m,lamda):#损失函数
	
	return 1/m*(-1.0*np.sum(y*np.log(s+1e-5),axis=1))#+lamda/(2*m)*np.sum(W*W)


'''            主函数        '''
file=pd.read_csv('data/train1.csv')
df=pd.DataFrame(file)


#创建矩阵X存放样本不同特征的数据
X=np.array(df.iloc[:,2:]) #2万个样本，597个特征

#创建矩阵Y存放实际的y值
Y=np.array(df.iloc[:,1],dtype=int)#2w*1


#归一化
X=X/255.0

alpha=0.3#学习率
iteration=10000#迭代次数
m=len(X)#数据总数
W=np.random.random((10,597))#权重

X=np.insert(X,0,values=np.ones(20000),axis=1)#添加偏置,X为2w*598
W=np.insert(W,0,values=np.ones(10),axis=1)#添加偏置，10*598

cost=[]#存放损失值


y=np.zeros((10,m))#创建10*2w矩阵存放每个样本值的标签向量
for i in range(m):
	y[Y[i],i]=1#要把Y标签值转换成向量


#梯度下降迭代
for i in range(iteration):#iteration
	z=Z(X.T,W)
	
	s=softmax(z)#10*2w
	
	loss=Loss(y,s,m,lamda)	#放回一个10*1矩阵存放每个标签单元总共产生的损失值
	
	cost.append(np.sum(loss))#每个样本产生的损失值
	
	J=(s-y).dot(X)#+a    #求偏导，得到10*785矩阵
	
	#更新w值
	W=W-1/m*alpha*J
	
	
print(cost[-1])


#画出损失值与迭代次数的关系图
plt.figure()
plt.plot(range(iteration),cost)
plt.xlabel('iterations')
plt.ylabel('loss')
plt.show()



#对测试集训练，并保存结果到文件

file=pd.read_csv('data/test1.csv')
df1=pd.DataFrame(file)
X1=np.array(df1.iloc[:,1:])#1w*
#X1=(X1-np.mean(X))/np.std(X)
X1=X1/255
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

