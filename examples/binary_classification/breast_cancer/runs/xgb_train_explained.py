import sklearn.datasets, sklearn.metrics, sklearn.model_selection 
import xgboost 
import shap
X, y = sklearn.datasets.load_breast_cancer(return_X_y=True) 
(Xtrain, Xtest, ytrain, ytest) = sklearn.model_selection.train_test_split(X, y, test_size=0.5, random_state=42) 
dtrain = xgboost.DMatrix(data=Xtrain, label=ytrain)
dtest = xgboost.DMatrix(data=Xtest, label=ytest)
arguments = { 
	"dtrain": dtrain,  
	"num_boost_round": 25,  
	"params": {  
		"learning_rate": 0.35,  
		"gamma": 1,  
		"max_depth": 6, 
    "objective": "binary:logistic" 
	},  
	"evals": [(dtest, "Test")]
}  
booster = xgboost.train(**arguments)
explainer = shap.TreeExplainer(booster)
shap_values = explainer.shap_values(Xtest)

'''
Shap values sum to the difference between expected output of the model and the current output for current player.
The units of the SHAP values for this model (since we use Tree SHAP) are log odds ratios.
Large positive values mean a player is likely to win, while large negative values mean they are likely to lose. 
'''
#shap.force_plot(explainer.expected_value, shap_values[0,:], Xtest[0,:])