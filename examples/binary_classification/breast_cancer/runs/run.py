import numpy 
import sklearn.datasets, sklearn.metrics, sklearn.model_selection
import xgboost
import sigopt.xgboost
X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
(Xtrain, Xtest, ytrain, ytest) = sklearn.model_selection.train_test_split(X, y, test_size=0.5, random_state=42)
dtrain = xgboost.DMatrix(data=Xtrain, label=ytrain)
# one test set
ytest_pos_indices = numpy.where(ytest == 1)
ytest_neg_indices = numpy.where(ytest == 0)
Xtest_pos = Xtest[ytest_pos_indices,:][0]
ytest_pos = ytest[ytest_pos_indices]
Xtest_neg = Xtest[ytest_neg_indices,:][0]
ytest_neg = ytest[ytest_neg_indices]
dtest_pos = xgboost.DMatrix(data=Xtest_pos, label=ytest_pos)
dtest_neg = xgboost.DMatrix(data=Xtest_neg, label=ytest_neg)
arguments = {
	"dtrain": dtrain, 
	"num_boost_round": 25, 
	"params": { 
		"learning_rate": 0.35, 
		"gamma": 1, 
		"max_depth": 6,
    "objective": "binary:logistic"
	}, 
	"evals": [(dtest_pos, "malignant"), (dtest_neg, "benign")],
} 
xgb_model_context = sigopt.xgboost.run(**arguments)
xgb_model_context.run.end()