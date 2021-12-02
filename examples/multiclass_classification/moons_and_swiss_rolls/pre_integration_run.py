# Data
from get_data import get_data

# Modeling
from sklearn.metrics import confusion_matrix, accuracy_score
import xgboost

dtrain, dtest, y = get_data()

arguments = { 
  "dtrain": dtrain,  
  "num_boost_round": 25,  
  "params": {  
    "learning_rate": 0.05,  
    "gamma": 1,  
    "max_depth": 4, 
    "objective": 'multi:softmax',
    "num_class": max(y) + 1,
  },  
  "evals": [(dtest, "ValidationSet")], 
}

booster = xgboost.train(**arguments) 