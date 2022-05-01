"""
model.py
    Home for different algorithm implementations

    @author Nicholas Nordstrom and Jason Zou
"""
import sklearn
import tensorflow
from tensorflow.keras import models, layers
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score,train_test_split,GridSearchCV
from sklearn import naive_bayes, svm


def random_forests(x,y):
    rf = RandomForestClassifier(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=42)


    rf.fit(X_train,y_train)
    print("Cross Val Score: ",cross_val_score(rf,X_train,y_train,cv=5).mean())
    print("Train Score: ", round(rf.score(X_train,y_train),4))
    print("Train Score: ", round(rf.score(X_test,y_test),4))


def dense_dropout_nn(input_dim=1000, num_ouput=2, verbose=False):
    model = models.Sequential([
        layers.Dense(12, input_dim=input_dim, activation='relu'),
        layers.Dense(8, activation='relu'),
        layers.Dense(50, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(12, activation='relu'),
        layers.Dense(50, activation='relu'),
        layers.Dense(num_ouput, activation='sigmoid')
    ])

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    if verbose:
        model.summary()

    return model


def svm(c=1.0, kernel='linear', degree=3, gamma='auto', verbose=False):
    model = svm.SVC(C=c, kernel=kernel, degree=degree, gamma=gamma)

    if verbose:
        print("Creating SVM Model")
        print("c = {}".format(c))
        print("kernel = {}".format(kernel))
        print("degree = {}".format(degree))
        print("gamma = {}".format(gamma))
    return model


def naive_bayes(verbose=False):
    model = naive_bayes.MultinomialNB()
    if verbose:
        print("Creating Naive Bayes Model")
    return model

