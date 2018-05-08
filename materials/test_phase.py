import pandas as pd
import jenkspy
from data_preprocessing import jenkspy_clustering
import csv
import numpy as np
import xlrd
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings("ignore")

labels={0:'Contempt',1:'Disgust',2:'Sadness',3:'Happy',4:'Surprise',5:'Anger',6:'Fear'}
df=pd.read_csv('TrainingData.csv')
row={attr:0 for attr in df.columns}
temp=pd.read_csv('../AMFED/AMFED/AU Labels/3b8c0000-397e-4d82-83ec-8f5eceaa76cb-label.csv')
columns=temp.columns.tolist()
columns.remove('Time')
for col in columns:
    row[col]=jenkspy_clustering(temp[col])
f=open('testData.csv','w')
writer=csv.writer(f)
line=[row[col] for col in df.columns]
writer.writerow(line)
f.close()

theta=np.loadtxt('parameter.csv',delimiter=',')
test=np.loadtxt('testData.csv',delimiter=',')

# Predicting & displaying the output label(human emotion) for the video 
print(labels[np.argmax(np.matmul(theta,test))])

df_y=pd.read_excel('../AMFED/AMFED/Summary.xlsx',usecols='C:E')
forest=RandomForestClassifier(n_estimators=4)
predicted=MultiOutputClassifier(forest).fit(df,df_y).predict(test.reshape((1,df.shape[1])))

# Classifying & displaying the feedback answers in the form of labels to
# 3 specific questions 
print(predicted)
