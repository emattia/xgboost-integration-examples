import sklearn.datasets, sklearn.metrics, sklearn.model_selection
import xgboost
import sigopt.xgboost
X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
(Xtrain, Xtest, ytrain, ytest) = sklearn.model_selection.train_test_split(X, y, test_size=0.5, random_state=42)
dtrain = xgboost.DMatrix(data=Xtrain, label=ytrain)
dtest = xgboost.DMatrix(data=Xtest, label=ytest)

sigopt.set_project("xgboost-integration-test")
my_config = dict(
  name="My XGBoost Experiment",
  parameters = [
    dict(name="max_depth", type="int", bounds=dict(min=3, max=12)),
    dict(name="learning_rate", type="double", bounds=dict(min=.05, max =.5))
  ],
  metrics = [
    dict(name="recall", strategy="optimize", objective="maximize"),
    dict(name="precision", strategy="optimize", objective="maximize"),
    dict(name="F1", strategy="store", objective="maximize"),
  ],
  budget = 5
)
experiment = sigopt.xgboost.experiment(
  experiment_config=my_config,
  dtrain=dtrain, 
  evals=[(dtest, "ValidationSet")],
  params = {"objective": "binary:logistic"}
)
