from sklearn import datasets
from sklearn.model_selection import train_test_split
import xgboost
import sigopt.xgboost

sigopt.set_project('xgboost-integration-test')

bc = datasets.load_breast_cancer()
(Xtrain, Xtest, ytrain, ytest) = train_test_split(bc.data, bc.target, test_size=0.5, random_state=42)
dtrain = xgboost.DMatrix(data=Xtrain, label=ytrain, feature_names=bc.feature_names)
dtest = xgboost.DMatrix(data=Xtest, label=ytest, feature_names=bc.feature_names)
my_config = dict(
  name="My XGBoost Experiment - Remove evals",
  parameters = [
    dict(name="max_depth", type="int", bounds=dict(min=3, max=12)),
    dict(name="eta", type="double", bounds=dict(min=.05, max =.5), transformation="log"),
  ],
  metrics=[
    dict(name="precision", strategy="optimize", objective="maximize"),
    dict(name="recall", strategy="optimize", objective="maximize"),
  ],
  budget=2
)
experiment = sigopt.xgboost.experiment(
  experiment_config=my_config,
  dtrain=dtrain,
  evals=[(dtest, "ValidationSet")],
  #params = {"objective": "binary:logistic"}
)