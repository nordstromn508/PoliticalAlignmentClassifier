"""
model.py
    Home for different algorithm implementations
    @author Nicholas Nordstrom and Jason Zou
"""

from keras import callbacks
from keras import models, layers
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, RepeatedStratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import evaluation
import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.python.keras import regularizers

def XG_Boost(x_train, y_train,x_test,y_test):
    model = XGBClassifier()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

def random_forests(x,y):
    rf = RandomForestClassifier(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=42)
    rf.fit(X_train,y_train)

    print("Cross Val Score: ",cross_val_score(rf,X_train,y_train,cv=10).mean())
    print("Train Score: ", round(rf.score(X_train,y_train),4))
    print("Test Score: ", round(rf.score(X_test,y_test),4))

def dense_dropout_nn(X_train, Y_train, x_test, y_test, input_dim=1000, num_ouput=1, verbose=True):

    model = models.Sequential([
        layers.Dense(1500,input_dim=input_dim, activation='relu'),
        layers.Dense(2000, activation='relu'),
        layers.Dropout(0.42),
        layers.Dense(2000, activation='relu'),
        layers.Dropout(0.42),
        layers.Dense(2000, activation='relu'),
        layers.Dropout(0.42),
        layers.Dense(2000, activation='relu'),
        layers.Dense(2000, activation='relu'),
        layers.Dense(num_ouput, activation='sigmoid')
    ])
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']

    )
    model.fit(X_train, Y_train, epochs=300, batch_size=50, verbose=False, callbacks=[callbacks.EarlyStopping(monitor='loss', patience=3)])
    # y_pred = model.predict(X_train)
    _, accuracy = model.evaluate(X_train, Y_train, verbose=False)
    print('NN Training Accuracy: %.2f' % (accuracy * 100))
    # print("NN Training Confusion Matrix:\n", evaluation.confusion_matrix(Y_train, y_pred, display=True))
    _, accuracy = model.evaluate(x_test, y_test, verbose=False)
    # y_pred = model.predict(x_test)
    print('NN Testing Accuracy: %.2f' % (accuracy * 100))
    # print("NN Testing Confusion Matrix:\n", evaluation.confusion_matrix(y_test, y_pred, display=True))

    if verbose:
        model.summary()
        print("input_dim = {}".format(input_dim))
        print("num_output = {}".format(num_ouput))
        print("optimizer = {}".format('adam'))
        print("activation = {}".format('sigmoid'))

    return model
def evaluate_model(model, X, y):

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    return scores


def svm(X_train, y_train, X_test, y_test, c=1.0, kernel='linear', degree=3, gamma='auto', verbose=False):
    model = SVC(C=c, kernel=kernel, degree=degree, gamma=gamma)

    if verbose:
        print("Creating SVM Model")
        print("c = {}".format(c))
        print("kernel = {}".format(kernel))
        print("degree = {}".format(degree))
        print("gamma = {}".format(gamma))

    model.fit(X_train, y_train)

    # Training Dataset Tests
    y_pred = model.predict(X_train)
    accuracy = model.score(X_train, y_train)
    print('SVM Training Accuracy: {0:.2f}%'.format(accuracy * 100))
    print("SVM Training Confusion Matrix:\n", evaluation.confusion_matrix(y_train, y_pred, display=True))

    # Testing Dataset Tests
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    print('SVM Testing Accuracy: {0:.2f}%'.format(accuracy * 100))
    print("SVM Testing Confusion Matrix:\n", evaluation.confusion_matrix(y_test, y_pred, display=True))

    return model


def naive_bayes(X_train, y_train, X_test, y_test, verbose=False):
    model = MultinomialNB()

    if verbose:
        print("Creating Naive Bayes Model")

    model.fit(X_train, y_train)

    # Training Dataset Tests
    y_pred = model.predict(X_train)
    accuracy = model.score(X_train, y_train)
    print('Naive Bayes Training Accuracy: {0:.2f}%'.format(accuracy * 100))
    print("Naive Bayes Training Confusion Matrix:\n", evaluation.confusion_matrix(y_train, y_pred, display=True))

    # Testing Dataset Tests
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    print('Naive Bayes Testing Accuracy: {0:.2f}%'.format(accuracy * 100))
    print("Naive Bayes Testing Confusion Matrix:\n", evaluation.confusion_matrix(y_test, y_pred, display=True))

    return model
