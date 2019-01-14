import process
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB

def lr_cv():
    x, y = process.read_data("train.csv", 1, 1)
    x_train_preprocessed = preprocessing.scale(x)
    y_train = y.T
    best_score = 0.0
    num = 0
    for C in [0.03, 0.05, 0.07]:
        num = num + 1
        lr = LogisticRegression(C=C, random_state=1, solver='saga',
                                multi_class='multinomial', max_iter=1000, penalty='l2')
        scores = cross_val_score(lr, x_train_preprocessed, y_train, cv=3)
        score = scores.mean()
        print("Iteration time:{}th".format(num))
        print("Current score on validation set:{:.9f}".format(score))
        print("Current parameters:{:.4f}".format(C))
        if score > best_score:
            best_score = score
            best_parameters = {"C": C}
    print("Best score on validation set:{:.9f}".format(best_score))
    print("Best parameters:{}".format(best_parameters))

def lr():
    x, y = process.read_data("train.csv", 1, 1)
    x_train_preprocessed = preprocessing.scale(x)
    x_test, y_test = process.read_data("test.csv", 1, 0)
    x_test_preprocessed = preprocessing.scale(x_test)
    y_train = y.T
    lr = LogisticRegression(C=0.05, random_state=1, solver='saga',
                            multi_class='multinomial', max_iter=800)
    lr.fit(x_train_preprocessed, y_train)
    return lr, x_test_preprocessed

def mnb_cv():
    x, y = process.read_data("train.csv", 1, 1)
    #x_train_preprocessed = preprocessing.scale(x)
    y_train = y.T
    best_score = 0.0
    num = 0
    for alpha in [19.6, 19.8, 20, 20.2, 20.4]:
        num = num + 1
        mnb = MultinomialNB(alpha=alpha, fit_prior=False)
        scores = cross_val_score(mnb, x, y_train, cv=4)
        score = scores.mean()
        print("Iteration time:{}th".format(num))
        print("Current score on validation set:{:.9f}".format(score))
        print("Current parameters:{:.2f}".format(alpha))
        if score > best_score:
            best_score = score
            best_parameters = {"alpha": alpha}
    print("Best score on validation set:{:.9f}".format(best_score))
    print("Best parameters:{}".format(best_parameters))

def mnb():
    x, y = process.read_data("train.csv", 1, 1)
    x_test, y_test = process.read_data("test.csv", 1, 0)
    y_train = y.T
    mnb = MultinomialNB(alpha=19.6, fit_prior=False)
    mnb.fit(x,y_train)
    return mnb, x_test

def cnb_cv():
    x, y = process.read_data("train.csv", 1, 1)
    #x_train_preprocessed = preprocessing.scale(x)
    y_train = y.T
    best_score = 0.0
    num = 0
    for alpha in [0.00001, 0.0001,0.001,0.01,0.1,1]:
        num = num + 1
        cnb = ComplementNB(alpha=alpha, fit_prior=False)
        scores = cross_val_score(cnb, x, y_train, cv=4)
        score = scores.mean()
        print("Iteration time:{}th".format(num))
        print("Current score on validation set:{:.9f}".format(score))
        print("Current parameters:{:.2f}".format(alpha))
        if score > best_score:
            best_score = score
            best_parameters = {"alpha": alpha}
    print("Best score on validation set:{:.9f}".format(best_score))
    print("Best parameters:{}".format(best_parameters))

def predict(model, x_test):
    y_pred = model.predict(x_test)
    index = np.arange(y_pred.shape[0]).reshape(y_pred.shape[0], 1)
    return np.column_stack((index, y_pred))

def main():
    #model, x_test =mnb()
    #lr_cv()
    model, x_test = lr()
    y_pred = predict(model, x_test)
    process.write_data(y_pred, "Submission.csv")


if __name__ == '__main__':
    main()