#%%
import warnings
import xgboost as xgb
from sklearn.feature_selection import RFE
from sklearn import datasets
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
# from sklearn.model_selection import train_test_split
def one_hot(df):
    labelencoder = LabelEncoder()
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    df['Sex'] = labelencoder.fit_transform(df['Sex'])
    df['Cabin'] = labelencoder.fit_transform(df['Cabin'])
    df['Embarked'] = labelencoder.fit_transform(df['Embarked'])
    return df


print("datasets is loading...",end='')
train = pd.read_csv("input/train.csv")
train_X = train.drop(['Survived','Name','Ticket'],axis='columns')
train_Y = train['Survived']

test = pd.read_csv("input/Submissionm_Titanic.csv")
test_X = test.drop(['Survived','Name','Ticket'],axis='columns')
test_Y = test['Survived']
print(' done.')

print('data scrubbing ...',end='')
train_X = one_hot(train_X)
test_X = one_hot(test_X)
print(' done.')


#%%
print('xgb Parameter auto optimization')
xgb_model = xgb.XGBClassifier(random_state=42)
param_grid = {'objective':['binary:logistic'],
              'learning_rate': [0.001,0.05,0.1, 10], 
              'max_depth': [2,3,4,5,6],
              'min_child_weight': [11],
              'subsample': [0.8],
              'colsample_bytree': [0.7],
              'n_estimators': [1000],
               'eval_metric':['mlogloss'],
               'use_label_encoder':[False]
              
              }

grid = GridSearchCV(estimator = xgb_model, cv=5, param_grid = param_grid , scoring = 'roc_auc', verbose = 1, n_jobs = -1, refit=True)
grid.fit(train_X,train_Y)

print("Best Score:" + str(grid.best_score_))
print("Best Parameters: " + str(grid.best_params_))

best_parameters = grid.best_params_

# %%
# XGBoost model with RFE and 70 features
xgb_model = xgb.XGBClassifier(**best_parameters)
xgb_model.fit(train_X,train_Y)

selector = RFE(xgb_model,  step=1)
selector.fit(train_X,train_Y)

xgb_preds = selector.predict_proba(test_X)[:,1]

RFE_test_predict = selector.predict(test_X)
XGB_test_predict = xgb_model.predict(test_X)
xg = roc_auc_score(test_Y, XGB_test_predict)
rf = roc_auc_score(test_Y, RFE_test_predict)
print("特徵篩選前",xg)

print("特徵篩選後",rf)

# %%
