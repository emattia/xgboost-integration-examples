import xgboost
import sklearn
import shap
import matplotlib.pyplot as plt
X,y = shap.datasets.boston()
(Xtrain, Xtest, ytrain, ytest) = sklearn.model_selection.train_test_split(X, y, test_size=0.5, random_state=42) 
dtrain = xgboost.DMatrix(data=Xtrain, label=ytrain)
dtest = xgboost.DMatrix(data=Xtest, label=ytest)
booster = xgboost.train(
  dtrain = dtrain,  
	num_boost_round = 25,  
	params = {"objective": "reg:squarederror"},  
	evals = [(dtest, "Test")], verbose_eval=False
)
explainer = shap.Explainer(booster)
shap_values = explainer(Xtest) 
assert shap_values.shape == Xtest.shape

### Can't use until bug in waterfall code is fixed:
	### https://github.com/slundberg/shap/pull/2306
# waterfall_plot = shap.plots.waterfall(shap_values[0])

force_plot_single_data_pt = shap.plots.force(shap_values[0])
fig = force_plot_single_data_pt.matplotlib(figsize=(12,6), show=False, text_rotation=0)