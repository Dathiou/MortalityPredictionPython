import models
from sklearn.cross_validation import KFold, ShuffleSplit
from numpy import mean
from sklearn.metrics import *
import utils
import sklearn



from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import random

#input: training data and corresponding labels
#output: accuracy, auc
def get_acc_auc_kfold(X,Y,k=5):
	#TODO:First get the train indices and test indices for each iteration
	#Then train the classifier accordingly
	#Report the mean accuracy and mean auc of all the folds
	a=KFold(len(Y),k)
	acc=[]
	auc=[]
	for train_index, test_index in a:
		X_train, X_test = X[train_index], X[test_index]
		Y_train, Y_test = Y[train_index], Y[test_index]
		Y_pred=models.logistic_regression_pred(X_train, Y_train, X_test)
		'''
		false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, Y_pred)
		roc_auc = sklearn.metrics.roc_auc_score(Y_test, Y_pred)
		plt.title('Receiver Operating Characteristic')
		plt.plot(false_positive_rate, true_positive_rate, 'b',label='AUC = %0.2f'% roc_auc)
		plt.legend(loc='lower right')
		plt.plot([0,1],[0,1],'r--')
		plt.xlim([-0.1,1.2])
		plt.ylim([-0.1,1.2])
		plt.ylabel('True Positive Rate')
		plt.xlabel('False Positive Rate')
		plt.show()
		'''
		
		acc_1=sklearn.metrics.accuracy_score(Y_test, Y_pred)
		auc_1=sklearn.metrics.roc_auc_score(Y_test, Y_pred)
		acc.append(acc_1)
		auc.append(auc_1)
	acc_mean=mean(acc)
	auc_mean=mean(auc)
	
	return acc_mean,auc_mean


#input: training data and corresponding labels
#output: accuracy, auc
def get_acc_auc_randomisedCV(X,Y,iterNo=5,test_percent=0.2):
	#TODO: First get the train indices and test indices for each iteration
	#Then train the classifier accordingly
	#Report the mean accuracy and mean auc of all the iterations
	a=ShuffleSplit(len(Y),iterNo,test_percent)
	acc=[]
	auc=[]
	for train_index, test_index in a:
		X_train, X_test = X[train_index], X[test_index]
		Y_train, Y_test = Y[train_index], Y[test_index]
		Y_pred=models.logistic_regression_pred(X_train, Y_train, X_test)
		acc_1=sklearn.metrics.accuracy_score(Y_test, Y_pred)
		auc_1=sklearn.metrics.roc_auc_score(Y_test, Y_pred)
		acc.append(acc_1)
		auc.append(auc_1)
	acc_mean=mean(acc)
	auc_mean=mean(auc)
	
	return acc_mean,auc_mean


def main():
	X,Y = utils.get_data_from_svmlight('../deliverables/features_svmlight.train')
	print "Classifier: Logistic Regression__________"
	acc_k,auc_k = get_acc_auc_kfold(X,Y)
	print "Average Accuracy in KFold CV: "+str(acc_k)
	print "Average AUC in KFold CV: "+str(auc_k)
	acc_r,auc_r = get_acc_auc_randomisedCV(X,Y)
	print "Average Accuracy in Randomised CV: "+str(acc_r)
	print "Average AUC in Randomised CV: "+str(auc_r)

if __name__ == "__main__":
	main()

