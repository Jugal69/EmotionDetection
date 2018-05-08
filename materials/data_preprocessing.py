import numpy as np
import pandas as pd
import csv
import xlrd
import jenkspy

def jenkspy_clustering(column):
    if column.empty:
        column=pd.Series(0,dtype=object)
    nclass=len(column.unique())
    if nclass<3:
        breaks = [max(column.unique())]
    else:
        breaks=jenkspy.jenks_breaks(column, nb_class=nclass)
    return max(set(breaks),key=breaks.count)

file_list=pd.read_excel(r'F:\Masters\ComputerVision\AMFED\AMFED\Summary.xlsx')
location='../AMFED/AMFED/AU Labels/'
attribute_list=set()
for file in file_list['File ID:']:
    temp=pd.read_csv('../AMFED/AMFED/AU Labels/'+file+'-label.csv')
    attribute_list.update(temp.columns.tolist())

attribute_list=list(attribute_list)
attribute_list.remove('Time')

f=open('TrainingData.csv','w')
writer=csv.writer(f)
writer.writerow(attribute_list)

for file in file_list['File ID:']:
    row={attr:0 for attr in attribute_list}
    temp=pd.read_csv('../AMFED/AMFED/AU Labels/'+file+'-label.csv')
    columns=temp.columns.tolist()
    columns.remove('Time')
    for col in columns:
        row[col]=jenkspy_clustering(temp[col])
    line=[row[col] for col in attribute_list]
    writer.writerow(line)
f.close()
