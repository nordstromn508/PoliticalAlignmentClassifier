import sklearn
import tensorflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score,train_test_split,GridSearchCV

def random_forests(x,y):
    rf = RandomForestClassifier(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=42)


    rf.fit(X_train,y_train)
    print("Cross Val Score: ",cross_val_score(rf,X_train,y_train,cv=5).mean())
    print("Train Score: ", round(rf.score(X_train,y_train),4))
    print("Train Score: ", round(rf.score(X_test,y_test),4))

