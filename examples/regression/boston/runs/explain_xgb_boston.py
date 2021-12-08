import xgboost
import sklearn
import shap

X,y = shap.datasets.boston()
(Xtrain, Xtest, ytrain, ytest) = sklearn.model_selection.train_test_split(X, y, test_size=0.5, random_state=42) 
dtrain = xgboost.DMatrix(data=Xtrain, label=ytrain)
dtest = xgboost.DMatrix(data=Xtest, label=ytest)
booster = xgboost.train(
  dtrain = dtrain,  
	num_boost_round = 25,  
	params = {"objective": "reg:squarederror"},  
	evals = [(dtest, "Test")]
)
explainer = shap.Explainer(booster)
shap_values = explainer(X)
shap.plots.waterfall(shap_values[0])