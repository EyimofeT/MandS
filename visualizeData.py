import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import visualizeData as vd

def visualizeData():
    #processing files
    data = pd.read_csv('datafile.tsv', sep='\t')
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 150)
    data.interpolate(method ='linear', limit_direction ='forward')
    pd.DataFrame(data).fillna(data.mean(), inplace=True)
    data.interpolate()
    #pd.reset_option('display.max_rows|display.max_columns|display.width')

    #print(data)
    #print(data.head())
    # data.head()
    # print('here now : \t')
    # print(data.tail())

    # data.shape
    #print(data.describe())
    # print(data.isnull().sum())

    datagraph = pd.read_csv('workdata.tsv', sep='\t')
    sns.set(style="darkgrid")
    #print(sns.relplot(x='Diagnosis Age',y='Overall Survival Status',data=datagraph))
    sns.relplot(y='Diagnosis Age',
            x='Overall Survival Status'
            ,hue="Sex",data=datagraph ,kind='line')
    # sns.relplot(y='Diagnosis Age',
    #             x='Sex'
    #             ,data=datagraph, kind='line')

    #diagram 1
    # show=sns.FacetGrid(data,col='Diagnosis Age')
    # show.map(plt.hist,'Overall Survival Status')

    #diagram 2
    show=sns.FacetGrid(data,col='Sex')
    show.map(plt.hist,'Overall Survival Status')

    #diagram 3
    show=sns.FacetGrid(data,col='Sex')
    show.map(plt.hist,'Diagnosis Age')

    plt.show()