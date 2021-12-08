from get_data import get_data
from sklearn.metrics import confusion_matrix, accuracy_score
import xgboost
import sigopt.xgboost
dtrain, dtest, y = get_data()
sigopt.set_project("xgboost-integration-test")
xgb_model_context = sigopt.xgboost.run(
  dtrain=dtrain, 
  params={
    "objective": 'multi:softmax',
    "num_class": max(y) + 1
  },
  evals=[(dtest, "ValidationSet")]
) 
xgb_model_context.run.end()