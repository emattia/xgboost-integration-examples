from get_data import get_data
from sklearn.metrics import confusion_matrix, accuracy_score
import xgboost
import sigopt.xgboost
dtrain, dtest, y = get_data()
sigopt.set_project("xgboost-integration-test")
my_config = dict( 
  name="My XGBoost Experiment", 
  parameters = [ 
    dict(name="max_depth", type="int", bounds=dict(min=3, max=12)), 
    dict(name="learning_rate", type="double", bounds=dict(min=.05, max =.5)) 
  ], 
  metrics=[ 
    dict(name="accuracy", strategy="optimize", objective="maximize"),
    dict(name="F1", strategy="optimize", objective="maximize"),
    dict(name="precision", strategy="store", objective="maximize") 
  ], 
  budget=5 
) 
experiment = sigopt.xgboost.experiment(
  experiment_config=my_config,
  dtrain=dtrain, 
  params={
    "objective": 'multi:softmax',
    "num_class": max(y) + 1
	},
  evals=[(dtest, "ValidationSet")]
) 