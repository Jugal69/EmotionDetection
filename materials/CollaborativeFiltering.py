import pandas as pd
import numpy as np

def normalize(y,r):
    [m,n]=y.shape
    ymean=np.zeros((m,1))
    ynorm=np.zeros((m,n))
    for i in range(1,m+1):
        index=np.where(r.loc[i]==1)
        ymean[i-1]=(r.loc[i][index[0]+np.ones(len(index))]).mean(axis=1)
        ynorm[i-1][index[0]]=y.loc[i][index[0]+np.ones(len(index))]-ymean[i-1]
    return ynorm,ymean

Location='TrainingData.csv'
df = pd.read_csv(Location)
y=df
nrows=y.shape[0]
ncols=y.shape[1]

for i in df.columns:
    y[i]=y[i].astype(int)

r=(y==1).astype(int)
nfeatures=7
niteration=150
lambdaReg=10
alpha=0.004
mu,sigma=0,1
x=np.random.normal(mu,sigma,nrows*nfeatures).reshape((nrows,nfeatures))
theta=np.random.normal(mu,sigma,ncols*nfeatures).reshape((nfeatures,ncols))

#y,ymean=normalize(y,r)
cost=0.5*(((np.square(np.dot(x,theta)-y)*r).sum().sum())+lambdaReg*(np.sum(np.sum(np.square(theta)))+np.sum(np.sum(np.square(x)))))
for i in range(niteration):
    x-=alpha*(lambdaReg*x+np.dot((np.dot(x,theta)-y)*r,theta.T))
    theta-=alpha*(lambdaReg*theta+np.dot(x.T,(np.dot(x,theta)-y)*r))
    cost=0.5*(((np.square(np.dot(x,theta)-y)*r).sum().sum())+lambdaReg*(np.sum(np.sum(np.square(theta)))+np.sum(np.sum(np.square(x)))))
    print('Iteration ',i,':',cost)

np.savetxt('parameter.csv',theta,delimiter=',')
    
