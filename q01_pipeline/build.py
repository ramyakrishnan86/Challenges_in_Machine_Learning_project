# %load q01_pipeline/build.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score

bank = pd.read_csv('data/Bank_data_to_class.csv', sep=',')

# Write your solution here :

le = LabelEncoder()

model = RandomForestClassifier(random_state=9,class_weight = 'balanced',n_estimators=50)

for column in bank.select_dtypes(include=['object']).columns.values:
    bank[column] = le.fit_transform(bank[column])

X=bank.drop(['y'],axis=1)
y=bank['y']
    
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.3,random_state=9)

def pipeline(X_train,X_test,y_train,y_test,model):
        
    param_grid = {'max_depth':[2,3,4,5,6,10,15,20,30],'max_features':[8,10,12,14],'max_leaf_nodes':[2,5,10,15,20]}
    
    gs = GridSearchCV(estimator=model,param_grid=param_grid)
    
    obj = gs.fit(X_train,y_train)
    
    y_pred = gs.predict(X_test)
    auc = roc_auc_score(y_test,y_pred)
    
    return obj,auc

# pipeline(X_train,X_test,y_train,y_test,model)



