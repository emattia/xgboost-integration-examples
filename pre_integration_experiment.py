import numpy 
import sklearn.datasets, sklearn.metrics, sklearn.model_selection
import xgboost
import sigopt.xgboost
X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
(Xtrain, Xtest, ytrain, ytest) = sklearn.model_selection.train_test_split(X, y, test_size=0.5, random_state=42)
dtrain = xgboost.DMatrix(data=Xtrain, label=ytrain)
dtest = xgboost.DMatrix(data=Xtest, label=ytest)
parameter_space = [ 
		{
			"name": "learning_rate",  
 			"type": "double",  
 			"bounds": {"min":0.1,"max":0.5}
		},  
		{
			"name": "max_depth",  
			"type": "int", 
			"bounds": {"min":4,"max":12}
		}
]
metric_space = [ 
		{"name": "Test-accuracy", "objective": "maximize", "strategy": "store"}, 
  	{"name": "Test-F1", "objective": "maximize", "strategy": "store"}, 
  	{"name": "Test-recall", "objective": "maximize", "strategy": "optimize"}, 
  	{"name": "Test-precision", "objective": "maximize", "strategy": "optimize"},	
		{"name": "Training Set-accuracy", "objective": "maximize", "strategy": "store"}, 
  	{"name": "Training Set-F1", "objective": "maximize", "strategy": "store"}, 
  	{"name": "Training Set-recall", "objective": "maximize", "strategy": "store"}, 
  	{"name": "Training Set-precision", "objective": "maximize", "strategy": "store"},	
		{"name": "Training time", "objective": "minimize", "strategy": "store"}	
]
experiment_config = { 
	"name": "logloss multimetric", 
	"parameters": parameter_space, 
	"metrics": metric_space,
  "budget": 10
}
experiment = sigopt.create_experiment(**experiment_config)
for run in experiment.loop():
	with run:
		arguments = {
			"dtrain": dtrain, 
			"num_boost_round": 10, 
			"params": { 
				"learning_rate": run.params.learning_rate, 
				"gamma": 1, 
				"max_depth": run.params.max_depth,
				"objective": "binary:logistic"
			}, 
			"evals": [(dtest, "Test")],
			"run_options": {"run": run}
		} 
		xgb_model_context = sigopt.xgboost.run(**arguments)
		xgb_model_context.run.end()