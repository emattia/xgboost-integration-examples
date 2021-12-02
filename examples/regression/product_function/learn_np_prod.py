### EXAMPLE 1 ### 
import numpy
import xgboost
import sigopt.xgboost
n_features = 12
X = numpy.random.random(size=(50000, n_features))
y = numpy.prod(X, axis=1)
dtrain = xgboost.DMatrix(data=X, label=y)
test_X = numpy.random.random(size=(5000, n_features))
test_y = numpy.prod(test_X, axis=1)
dtest0 = xgboost.DMatrix(data=test_X, label=test_y)
test_X = numpy.random.random(size=(5000, n_features))
test_y = numpy.prod(test_X, axis=1)
dtest1 = xgboost.DMatrix(data=test_X, label=test_y)
arguments = {
	"dtrain": dtrain, 
	"num_boost_round": 10, 
	"params": { 
		"learning_rate": 0.3, 
		"gamma": .5, 
		"max_depth": 8,
    "objective": "reg:squarederror"
	}, 
	"evals": [(dtest0, "dtest0"), (dtest1, "dtest1")]   
} 
booster = xgboost.train(**arguments)
xgb_model_context = sigopt.xgboost.run(**arguments)
assert type(xgb_model_context.run) == sigopt.run_context.RunContext
assert type(xgb_model_context.model) == xgboost.core.Booster