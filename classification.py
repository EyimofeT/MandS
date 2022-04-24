#load libraries
import pandas as pd
import numpy as np
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import ensemble

def train():
    data = pd.read_csv('datafile.tsv', sep='\t')
    data.interpolate(method ='linear', limit_direction ='forward')
    pd.DataFrame(data).fillna(data.mean(), inplace=True)
    data.interpolate()
    
    # data.info()
    # sb.heatmap(data.isnull())
    
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
    
    # Seperate the dataframe into X and y data
    # print("Classification Models")
    #using only male data
    train=train[train.Sex != '1']
    X = train
    y = data['Overall Survival Status']
    
    #making int values for y because classifier models only collect interger values and not continuous values like floats
    from sklearn import preprocessing
    from sklearn import utils

    #convert y values to categorical values
    lab = preprocessing.LabelEncoder()
    y_transformed = lab.fit_transform(y)

   # Split the dataset into 70% Training and 30% Test
    
    #X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)
    X_train,X_test,y_train,y_test=train_test_split(train,y_transformed,test_size=0.3,random_state=2)
    
    # Using simple Decision Tree classifier
    print("Using Simple Decision Tree Classfier")
    dt_clf = tree.DecisionTreeClassifier(max_depth=5)
    dt_clf.fit(X_train, y_train)
    percentage=round((dt_clf.score(X_test,y_test)*100),2)
    print("Accuracy [ + ] :",percentage,"%")

    
    # Using simple Random Forest classifier
    print("Using Simple Random Forest Clasifier")
    rf_clf = ensemble.RandomForestClassifier(n_estimators=100)
    rf_clf.fit(X_train, y_train)
    #print(round(rf_clf.score(X_test, y_test)*100),"%")
    percentage=round((rf_clf.score(X_test,y_test)*100),2)
    print("Accuracy [ + ] :",percentage,"%")
    
    # using Gradient Boosting Classifier
    print("Using Gradient Boosting Clasifier")
    gb_clf = ensemble.GradientBoostingClassifier()
    gb_clf.fit(X_train, y_train)
    #print(round(gb_clf.score(X_test, y_test)*100),"%")
    percentage=round((gb_clf.score(X_test,y_test)*100),2)
    print("Accuracy [ + ] :",percentage,"%")
    

