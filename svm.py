import process
import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

def svm_train_cv():
    x, y = process.read_data("train.csv", 1, 1)
    sscaler = StandardScaler()
    sscaler.fit(x)
    x_train_preprocessed = sscaler.transform(x)
    y_train = y.T
    best_score = 0.0
    num = 0
    for coef0 in [0.1, 0.12, 0.14, 0.16]:
        svm_rbf = svm.SVC(C=0.90, cache_size=500, kernel='sigmoid', gamma='scale', coef0=coef0)
        num = num + 1
        scores = cross_val_score(svm_rbf, x_train_preprocessed, y_train, cv=3)
        score = scores.mean()
        print("Iteration time:{}th".format(num))
        print("Current score on validation set:{:.9f}".format(score))
        print("Current parameters:{:.2f}".format(coef0))
        if score > best_score:
            best_score = score
            best_parameters = {"coef0": coef0}
    print("Best score on validation set:{:.9f}".format(best_score))
    print("Best parameters:{}".format(best_parameters))

    #scores_sigmoid = cross_val_score(sigmoid, x_train_preprocessed, y_train, cv=5)
    #print (scores_sigmoid)
    #print ("sigmoid Accuracy: %0.9f (+/- %0.9f)" % (scores_sigmoid.mean(), scores_sigmoid.std() * 2))
    #clf.fit(x_train_preprocessed, y.T)
    #clf_score = clf.score(x_valid, y_valid)
    #print("The score of rbf is : %f" % clf_score)

def svm_train():
    x, y = process.read_data("train.csv", 1, 1)
    print(x)
    print(y)
    x_test, y_test = process.read_data("test.csv", 1, 0)
    sscaler = StandardScaler()
    sscaler.fit(x)
    x_train_preprocessed = sscaler.transform(x)
    x_test_preprocessed = sscaler.transform(x_test)
    y_train = y.T
    svm_rbf = svm.SVC(C=0.90, cache_size=500, kernel='sigmoid', gamma='auto', coef0=0.1)
    svm_rbf.fit(x_train_preprocessed, y_train)
    return svm_rbf,x_test_preprocessed

def mlp_train():
    x, y = process.read_data("train.csv", 1, 1)
    sscaler = StandardScaler()
    sscaler.fit(x)
    x_train_preprocessed = sscaler.transform(x)
    y_train = y.T
    mlp = MLPClassifier(activation='logistic', solver='sgd', alpha=1e-5, batch_size=200, hidden_layer_sizes=(50, 50),
                        random_state=1, learning_rate='adaptive', max_iter=600)
    scores_mlp = cross_val_score(mlp, x_train_preprocessed, y_train, cv=3)
    print (scores_mlp)
    print ("scores_mlp Accuracy: %0.9f (+/- %0.9f)" % (scores_mlp.mean(), scores_mlp.std() * 2))

def svm_predict(clf, x_test):
    # num_test = x_test.shape[0]
    # y_pred = np.zeros((num_test, 2), dtype=np.int)
    y_pred = clf.predict(x_test)
    index = np.arange(y_pred.shape[0]).reshape(y_pred.shape[0], 1)
    return np.column_stack((index, y_pred))


def main():
    #svm_train_cv()
    #clf, x_test = svm_train()
    #y_pred = svm_predict(clf, x_test)
    #process.write_data(y_pred, "Submission.csv")
    mlp_train()
    '''
    rats = arange(0.1, 0.4, 0.1)
    for rat in rats:
        clf, x_test, y_test = svm_train_2(rat)
        y_pred = svm_predict(clf, x_test)
        # process.write_data(y_pred, "Submission.csv")
        d = np.argwhere(y_pred[:, 1] == np.reshape(y_test, (y_test.shape[0],)))
        print(rat, ": ", len(d)/y_pred.shape[0], len(d), y_pred.shape[0])
    '''


if __name__ == '__main__':
    main()
