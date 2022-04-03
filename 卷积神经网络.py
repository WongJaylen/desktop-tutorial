import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



#卷积层
def Kernal(W_kernal,x):#卷积层,输入2w*28*28，权重为3*3
	size,length,width=x.shape
	step=1#步长
	
	a=np.zeros((676,9))#创建矩阵存放不同的图像特征
	b=np.zeros((20000,676))#创建矩阵存放所有样本卷积后的结果
	
	t=0#计数
	
	for s in range(size):
		
		for i in range(0,length-W_kernal.shape[0]+step,step):
			for j in range(width-W_kernal.shape[0]+step):
				a[t]=x[s][i:i+W_kernal.shape[0],j:j+W_kernal.shape[0]].reshape((1,9))#a为676*9
				t=t+1
		t=0
		b[s]=a.dot(W_kernal.reshape((9,1))).T#W_kernal为9*1，相乘转置得1*676，加入到b
	
	
	return b.reshape(20000,26,26)#(20000,26,26)
	


#池化层
def maxpool(z):#z为(20000,26,26),权重为2*2
	size,length,width=z.shape
	poolsize=2
	step=2
	a=np.zeros((169,4))
	b=np.zeros((20000,169))
	index=np.zeros((20000,169))
	t=0
	
	for s in range(size):
		for i in range(0,length-poolsize+step,step):
			for j in range(0,width-poolsize+step,step):
				a[t]=z[s][i:i+poolsize,j:j+poolsize].reshape((1,4))#取出输入矩阵的每一个小窗，a为169*4
				t=t+1
		t=0
		b[s]=np.max(a,axis=1).T#取每一个特征组的最大值,np.max(a,axis=1)为169*1,加入到b
		index[s]=np.argmax(a,axis=1)#获得每一个特征组的最大值所在的索引
	
	#print(index.)
	return b.reshape(20000,13,13),index	#b(20000,13,13)
			
			
#隐藏层
def Relu(z):
	return np.maximum(z,0)
#输出层	
def Softmax(z):#传入一个10*2w矩阵
	#计算机处理数值位数有限，防止指数运算溢出，加上输入信号中的最大值的负数不会改变运算的结果
	c=np.max(z,axis=0)#每个样本中10个输出节点的最大值
	z_exp=np.exp(z-c)

	#print(np.sum(z_exp,axis=0))
	s=z_exp/np.sum(z_exp,axis=0)#是一个10*2w矩阵代表每个标签在每个样本的概率值

	return s
	

def Loss(y,s):#损失函数
	m=len(s[1])
	return 1/m*(-1.0*np.sum(y*np.log(s+1e-5),axis=1))#+lamda/(2*m)*np.sum(W*W)
	
	
	

#反向传播梯度下降

#隐藏层
def hiddenlayer_backprop(alpha,s,y,w,z):
	
	#s,y为10，2w
	#z为50,2w
	#w为10,50
	
	m=len(s[1])
	w=w-1/m*alpha*(s-y).dot(z.T)
	return w
	
#Relu层
def Relulayer_backprop(alpha,s,y,W_relu,W_Softmax,z_reluinput,z_reluoutput):#z1为池化层输出
	

	#w_relu为50,169
	#w_softmax为10,50
	#s-y (10,2w)
	#J=np.sum(W_Softmax.T.dot(s-y),axis=1)#W_Softmax.T.dot(s-y)得到50*2w矩阵，进行sum求和得到50,1，代表50个relu层结点的总损失值
	J=W_Softmax.T.dot(s-y)#得到50*2w矩阵,代表50个relu层结点的2w个样本的损失值
	relu_derivative=z_reluoutput.astype(bool).astype(int)#relu函数求导，正值化为1,50*2w
	W_relu=W_relu-1/20000*alpha*np.dot(relu_derivative*J,z_reluinput)#梯度下降更新W_relu值
	#print(W_relu)
	return W_relu
	
	

#池化层
def pool_backprop(s,y,W_relu,W_Softmax,z_poolinput,z_pooloutput,index):
	#z_pooloutput(2w,169)
	J_relu=W_Softmax.T.dot(s-y)#得到50*2w矩阵,代表50个relu层结点的2w个样本的损失值
	J_pool=W_relu.T.dot(J_relu)#得到169*2w矩阵,代表13*13个pool层输出结点的2w个样本的损失值
	
	loss_kernal=np.zeros((20000,676))#创建矩阵，进行上采样存放损失值
	#上采样
	size,length,width=z_poolinput.shape
	poolsize=2
	step=2
	t=0
	#print(z_pooloutput)
	
	for s in range(size):#样本数
		for i in range(length-poolsize+step,step):#图像长
			for j in  range(width-poolsize+step,step):#图像宽
				loss_kernal[s][i:i+poolsize,j:j+poolsize].reshape(1,4)[index[s][t]]=z_pooloutput[s][t]#a为169*4
				t=t+1

		t=0
	return loss_kernal#(2w,676)返回卷积层的损失值

#卷积层
def Kernal_backprop(alpha,loss_kernal,W_kernal,z_kernalinput,z_kernaloutput):
	#z_kernalinput(2w,28,28)
	#输入层与损失值作卷积运算
	size,length,width=z_kernalinput.shape
	loss_kernalsize=26
	
	step=1
	t=0
	#求卷积核参数的偏导
	J=np.zeros((20000,9))#创建矩阵存放卷积核参数的偏导
	for s in range(size):#样本数
		for i in range(length-loss_kernalsize+step,step):#图像长
			for j in  range(width-loss_kernalsize+step,step):#图像宽
				J[s][t]=np.dot(z_kernalinput[s][i:i+loss_kernalsize,j:j+loss_kernalsize].reshape(676,1),loss_kernal[s])
				t=t+1
		t=0
			
	#梯度下降
	W_kernal=W_kernal-1/size*alpha*(np.sum(J,axis=0).reshape(3,3))
	return W_kernal



'''                   主函数                        '''
#载入文件
file=pd.read_csv('data/train.csv')
df=pd.DataFrame(file)

#创建矩阵X存放样本不同特征的数据
X=np.array(df.iloc[:,1:-1]) #2万个样本，784个特征

#创建矩阵Y存放实际的y值
Y=np.array(df.iloc[:,-1],dtype=int)	#2w*1

#创建10*2w矩阵存放每个样本值的标签向量
y=np.zeros((10,len(Y)))
for i in range(len(Y)):
	y[Y[i],i]=1#要把Y标签值转换成向量
	
#归一化
X=X/255.0

X=X.reshape(20000,28,28)
W_kernal=np.random.random((3,3))#卷积层参数
W_relu=np.random.random((50,169))#隐藏层参数
W_Softmax=np.random.random((10,50))#输出层参数	

alpha=0.5#学习率
b1=np.ones((50,1))#池化层到relu层的偏置
b2=np.ones((10,1))#relu层到softmax层偏置

for i in range(500):
	#前向传播
	Z1=Kernal(W_kernal,X)#(20000, 26, 26)

	Z2,index=maxpool(Z1)#(20000, 13, 13)

	Z3=Relu(W_relu.dot(Z2.reshape(20000,169).T)+b1)#(50,2w)

	Z4=W_Softmax.dot(Z3)+b2#(10,2w)

	S=Softmax(Z4)#(10,2w)

	loss=Loss(y,S)#(10，2W)

	print(np.sum(loss))
	
	#反向传播
	W_Softmax=hiddenlayer_backprop(alpha,S,y,W_Softmax,Z3)
		
	W_relu=Relulayer_backprop(0.01,S,y,W_relu,W_Softmax,Z2.reshape(20000,169),Z3)

	kernal_loss=pool_backprop(S,y,W_relu,W_Softmax,Z1,Z2.reshape(20000,169),index)

	W_keral=Kernal_backprop(alpha,kernal_loss,W_kernal,X,Z1)

