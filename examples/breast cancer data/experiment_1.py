import sklearn.datasets, sklearn.metrics, sklearn.model_selection
import xgboost
import sigopt.xgboost
X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
(Xtrain, Xtest, ytrain, ytest) = sklearn.model_selection.train_test_split(X, y, test_size=0.5, random_state=42)
dtrain = xgboost.DMatrix(data=Xtrain, label=ytrain)
dtest = xgboost.DMatrix(data=Xtest, label=ytest)
experiment = sigopt.xgboost.experiment(dtrain=dtrain, evals=[(dtest, "Test")])
