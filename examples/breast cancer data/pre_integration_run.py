import numpy
import sklearn.datasets, sklearn.metrics, sklearn.model_selection 
import xgboost 
import time 
import sigopt 

def compute_accuracy(y_true, y_pred):
  accuracy = numpy.count_nonzero(y_true == y_pred) / len(y_true)
  return accuracy

def compute_positives_and_negatives(y_true, y_pred, class_label):
  y_true_equals = y_true == class_label
  y_true_notequals = y_true != class_label
  y_pred_equals = y_pred == class_label
  y_pred_notequals = y_pred != class_label
  tp = numpy.count_nonzero(numpy.logical_and(y_true_equals, y_pred_equals))
  tn = numpy.count_nonzero(numpy.logical_and(y_true_notequals, y_pred_notequals))
  fp = numpy.count_nonzero(numpy.logical_and(y_true_notequals, y_pred_equals))
  fn = numpy.count_nonzero(numpy.logical_and(y_true_equals, y_pred_notequals))
  return tp, tn, fp, fn

def compute_classification_report(y_true, y_pred):
  classes = numpy.unique(y_true)
  classification_report = {}
  classification_report['weighted avg'] = {
    'f1-score': 0,
    'recall': 0,
    'precision': 0
  }
  for class_label in classes:
    tp, _, fp, fn = compute_positives_and_negatives(y_true, y_pred, class_label)
    precision = tp / (tp + fp) if (tp + fp)!=0 else 0
    recall = tp / (tp + fn) if (tp + fn)!=0 else 0
    f1 = tp / (tp + 0.5 * (fp + fn)) if (tp + 0.5 * (fp + fn))!=0 else 0
    support = numpy.count_nonzero(y_true == class_label)
    classification_report[str(class_label)] = {
      'precision': precision,
      'recall': recall,
      'f1-score': f1,
      'support': support
    }
    classification_report['weighted avg']['precision'] += (support / len(y_pred)) * precision
    classification_report['weighted avg']['recall'] += (support / len(y_pred)) * recall
    classification_report['weighted avg']['f1-score'] += (support / len(y_pred)) * f1
  return classification_report

def process_preds(preds):
  if len(preds.shape) == 2:
    preds = numpy.argmax(preds, axis=1)
  else:
    preds = numpy.round(preds)
  return preds

X, y = sklearn.datasets.load_breast_cancer(return_X_y=True) 
(Xtrain, Xtest, ytrain, ytest) = sklearn.model_selection.train_test_split(X, y, test_size=0.5, random_state=42) 
dtrain = xgboost.DMatrix(data=Xtrain, label=ytrain)
ytest_pos_indices = numpy.where(ytest == 1) 
ytest_neg_indices = numpy.where(ytest == 0) 
Xtest_pos = Xtest[ytest_pos_indices,:][0] 
ytest_pos = ytest[ytest_pos_indices] 
Xtest_neg = Xtest[ytest_neg_indices,:][0] 
ytest_neg = ytest[ytest_neg_indices] 
dtest_pos = xgboost.DMatrix(data=Xtest_pos, label=ytest_pos)
dtest_neg = xgboost.DMatrix(data=Xtest_neg, label=ytest_neg) 

evals_result = {}
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
  "evals_result": evals_result 
}  

run = sigopt.create_run(name="My XGBoost Run") 
run_params = {"num_boost_round": 25} 
for param_name, value in arguments["params"].items(): 
    if param_name == "objective": 
        run.log_metadata("objective", value) 
    else: 
        run_params[param_name] = value 
run.params.setdefaults(run_params) 

t0 = time.time() 
booster = xgboost.train(**arguments) 
train_time = time.time() - t0 
train_preds = process_preds(booster.predict(dtrain))
train_accuracy = compute_accuracy(dtrain.get_label(), train_preds) 
train_rep = compute_classification_report(dtrain.get_label(), train_preds) 
train_other_metrics = train_rep['weighted avg'] 
pos_preds = process_preds(booster.predict(dtest_pos))
pos_accuracy = compute_accuracy(dtest_pos.get_label(), pos_preds) 
pos_rep = compute_classification_report(dtest_pos.get_label(), pos_preds) 
pos_other_metrics = pos_rep['weighted avg'] 
neg_preds = process_preds(booster.predict(dtest_neg))
neg_accuracy = compute_accuracy(dtest_neg.get_label(), neg_preds) 
neg_rep = compute_classification_report(dtest_neg.get_label(), neg_preds) 
neg_other_metrics = neg_rep['weighted avg'] 
classification_metrics = { 
    "malignant-accuracy": pos_accuracy, 
    "malignant-F1": pos_other_metrics['f1-score'], 
    "malignant-recall": pos_other_metrics['recall'], 
    "malignant-precision": pos_other_metrics['precision'], 
    "benign-accuracy": neg_accuracy, 
    "benign-F1": neg_other_metrics['f1-score'], 
    "benign-recall": neg_other_metrics['recall'], 
    "benign-precision": neg_other_metrics['precision'], 
    "Training-Set-accuracy": train_accuracy, 
    "Training-Set-F1": train_other_metrics['f1-score'], 
    "Training-Set-recall": train_other_metrics['recall'], 
    "Training-Set-precision": train_other_metrics['precision'], 
    "Training Time": train_time 
} 
for k, v in classification_metrics.items(): 
  run.log_metric(k, v) 
for dataset, metric_dict in evals_result.items():
  for metric_label, metric_record in metric_dict.items():
    run.log_metric(f"{dataset}-{metric_label}", metric_record[-1])
run.end() 