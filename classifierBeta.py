# Load libraries
import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

def train():
# Load dataset
    # print("oner")
    data = pd.read_csv('datafile.tsv', sep='\t')
    data.interpolate(method ='linear', limit_direction ='forward')
    pd.DataFrame(data).fillna(data.mean(), inplace=True)
    data.interpolate()
    
    train=data.drop([
    'Study ID',
    'Patient ID',
    'Sample ID',
    # 'Diagnosis Age',
    'Neoplasm Disease Stage American Joint Committee on Cancer Code',
    'American Joint Committee on Cancer Publication Version Type',
    # 'Aneuploidy Score',
    # 'Buffa Hypoxia Score',
    'Cancer Type',
    'TCGA PanCanAtlas Cancer Type Acronym',
    'Cancer Type Detailed',
    'Last Communication Contact from Initial Pathologic Diagnosis Date',
    #'Birth from Initial Pathologic Diagnosis Date',
    #'Last Alive Less Initial Pathologic Diagnosis Date Calculated Day Value',
    'Disease Free (Months)',
    'Disease Free Status',
    #'Months of disease-specific survival',
    #'Disease-specific Survival status',
    'Ethnicity Category',
    'Form completion date',
    #'Fraction Genome Altered',
    'Neoplasm Histologic Grade',
    'Neoadjuvant Therapy Type Administered Prior To Resection Text',
    'ICD-10 Classification',
    'International Classification of Diseases for Oncology, Third Edition ICD-O-3 Histology Code',
    'International Classification of Diseases for Oncology, Third Edition ICD-O-3 Site Code',
    'Informed consent verified',
    'In PanCan Pathway Analysis',
    #'MSI MANTIS Score',
    #'MSIsensor Score',
    #'Mutation Count',
    'New Neoplasm Event Post Initial Therapy Indicator',
    'Oncotree Code',	
    #'Overall Survival (Months)',
    'Overall Survival Status',
    'Other Patient ID',
    'American Joint Committee on Cancer Metastasis Stage Code',
    'Neoplasm Disease Lymph Node Stage American Joint Committee on Cancer Code',
    'American Joint Committee on Cancer Tumor Stage Code',
    'Person Neoplasm Cancer Status',
    #'Progress Free Survival (Months)',	
    'Progression Free Status',	
    'Primary Lymph Node Presentation Assessment',
    'Prior Diagnosis',
    'Race Category',
    'Radiation Therapy',    
    #'Ragnum Hypoxia Score',
    'Number of Samples Per Patient',
    'Sample Type',
    #'Sex',
    'Somatic Status',
    'Subtype',
    'Tissue Prospective Collection Indicator',
    'Tissue Retrospective Collection Indicator',
    'Tissue Source Site',
    'Tissue Source Site Code',
    #'TMB (nonsynonymous)',	
    'Tumor Disease Anatomic Site',
    'Tumor Type'
    #'Patient Weight',
    #'Winter Hypoxia Score'
    ],axis=1)
    # test=data['Overall Survival Status']
    # Split dataset into train, test and validation sets

    array = train.values
    # X = array[:,0:4]
    # Y = array[:,4]
    X = train
    y = data['Overall Survival Status']
    #
     #making int values for y because classifier models only collect interger values and not continuous values like floats
    from sklearn import preprocessing
    from sklearn import utils

    #convert y values to categorical values
    lab = preprocessing.LabelEncoder()
    y_transformed = lab.fit_transform(y)

    #
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, y_transformed, test_size=0.2, random_state=7)
    # Building and evaluating  classification Algorithms
    models=[]
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC()))
    #evaluate each model in turn
    results=[]
    names=[]
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=7,shuffle=True)
        #kfold = model_selection.KFold(n_splits=10)
        cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
        
        # Compare Algorithms

    print("Comparing Algorithms")
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()
    
train()
