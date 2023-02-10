from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from pandas import read_csv
import logging
import mlflow
import mlflow.sklearn
from sklearn import datasets
import os

#mlflow.set_tracking_uri('http://127.0.0.1:5000')

os.environ['MLFLOW_TRACKING_USERNAME'] = "aegorov"
os.environ['MLFLOW_TRACKING_PASSWORD'] = "Qaz123qwe"
os.environ["AWS_ACCESS_KEY_ID"] = "RHFGTAN4OVNKE99HWAD7"
os.environ["AWS_SECRET_ACCESS_KEY"] = "JFJyW1unoPYBAJOUQtLwGW4W1CTCLx18fd2hpyQq"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"http://ceph.ml.neoflex.ru:7480"

mlflow.set_tracking_uri('http://app.istio.ml.neoflex.ru/project-api/AEGIS/mlflow/')

try:
    # Creating an experiment 
    mlflow.create_experiment('demo_data_process_flow')
except:
    pass
# Setting the environment with the created experiment
mlflow.set_experiment('demo_data_process_flow')



def read_data():
    #names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    dataset = datasets.load_iris()
    print('Data Reading Successfully Done')
    return dataset


def make_train_test():
    array = read_data()
    x = array.data
    y = array.target
    X_train, X_validation, Y_train, Y_validation = train_test_split(x, y, test_size=0.2, random_state=7)
    print('train_shape: %d , validation_shape: %d' % (len(X_train), len(X_validation)))
    return [X_train, X_validation, Y_train, Y_validation]


def run_model():

    data_sets = make_train_test()
    print('Running Model...')
    models = [('LR', LogisticRegression(solver='liblinear', multi_class='ovr')), ('LDA', LinearDiscriminantAnalysis()),
              ('KNN', KNeighborsClassifier()), ('CART', DecisionTreeClassifier()), ('NB', GaussianNB()),
              ('SVM', SVC(gamma='auto'))]
    # evaluate each model in turn
    results = []
    names = []
    for name, model in models:
        with mlflow.start_run(run_name=name):
            kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
            cv_results = cross_val_score(model, data_sets[0], data_sets[2], cv=kfold, scoring='accuracy')
            results.append(cv_results)
            names.append(name)
            # logging different metrics in mlflow
            mlflow.log_metric('mean_cv_score_accuracy', cv_results.mean())
            mlflow.log_metric('std_cv_score_accuracy', cv_results.std())
            print('%s model cross-validation performance: mean - %f , std - (%f)' % (name, cv_results.mean(),
                                                                                            cv_results.std()))
            model.fit(data_sets[0], data_sets[2])
            predictions = model.predict(data_sets[1])
            acc = accuracy_score(data_sets[3], predictions)
            cnfm = precision_score(data_sets[3], predictions,average= 'macro')
            cr = recall_score(data_sets[3], predictions, average= 'macro')
            # logging some more metrics in mlflow
            mlflow.log_metric('test_accuracy', acc)
            mlflow.log_metric('test_precision', cnfm)
            mlflow.log_metric('test_recall', cr)
            # logging the model as well
            mlflow.sklearn.log_model(model, name)
            print('MLFlow has run, please check the mlflow UI')

run_model()