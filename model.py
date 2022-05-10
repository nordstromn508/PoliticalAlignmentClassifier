"""
model.py
    Home for different algorithm implementations

    @author Nicholas Nordstrom and Jason Zou
"""
import sklearn
import tensorflow as tf
from tensorflow.keras import models, layers
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score,train_test_split,GridSearchCV
from sklearn import naive_bayes, svm
from sklearn.svm import SVC
import evaluation


def random_forests(x,y):
    rf = RandomForestClassifier(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=42)
    rf.fit(X_train,y_train)
    print("Cross Val Score: ",cross_val_score(rf,X_train,y_train,cv=10).mean())
    print("Train Score: ", round(rf.score(X_train,y_train),4))
    print("Test Score: ", round(rf.score(X_test,y_test),4))


def dense_dropout_nn(X_train,Y_train,x_test, y_test, input_dim=1000, num_ouput=1, verbose=True):
    # y = y.map({'Conservative' : 0,
    #             'Liberal' : 1},
    #             na_action = None)
    model = models.Sequential([
        layers.Dense(12, input_dim=input_dim, activation='relu'),
        layers.Dense(8, activation='relu'),
        layers.Dense(50, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(12, activation='relu'),
        layers.Dense(50, activation='relu'),
        layers.Dense(20,activation = 'relu'),
        layers.Dense(num_ouput, activation='sigmoid')
    ])

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']

    )
    model.fit(X_train, Y_train , epochs=300, batch_size=50)
    _, accuracy = model.evaluate(X_train, Y_train)
    print('Training Accuracy: %.2f' % (accuracy * 100))
    _, accuracy = model.evaluate(x_test, y_test)
    print('Testing Accuracy: %.2f' % (accuracy * 100))

    if verbose:
        model.summary()

    return model


def svm(X_train, y_train,X_test, y_test, c=1.0, kernel='linear', degree=3, gamma='auto', verbose=False):
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
    print('Training Accuracy: %.2f' % (accuracy * 100))
    print("Training Confusion Matrix:\n", evaluation.confusion_matrix(y_train, y_pred, display=True))

    # Testing Dataset Tests
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    print('Testing Accuracy: %.2f' % (accuracy * 100))
    print("Testing Confusion Matrix:\n", evaluation.confusion_matrix(y_test, y_pred, display=True))

    return model


def naive_bayes(X_train, y_train,X_test, y_test, verbose=False):
    model = naive_bayes.MultinomialNB()
    if verbose:
        print("Creating Naive Bayes Model")
    return model
