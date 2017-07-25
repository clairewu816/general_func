from .recreate_scores import get_train_test
from constants import features, target
import pandas as pd
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import roc_curve, auc, accuracy_score
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import numpy as np


class ModelHelper(object):
    def __init__(self, clf, features, seed=0, params=None):
        params['random_state'] = seed
        self.params = params
        self.clf = clf(**self.params)
        self.features = features

    def train(self, x_train, y_train, plot=False, xgb_fscore=False, useTrainCV=False):
        """
        Train model, produce train error, and plot feature importance if wanted
        :param plot: if plot feature importance
        :param xgb_fscore: if use F1 score as the value of feature_importance plot, only applicable for XGB
        :param useTrainCV: if use CV to do early stopping, only applicable for XGB
        """
        if useTrainCV:
            xgtrain = xgb.DMatrix(x_train, label=y_train)
            cvresult = xgb.cv(self.params, xgtrain, num_boost_round=3000, nfold=5,
                              metrics='logloss', early_stopping_rounds=100)
            print('set n_estimators as', cvresult.shape[0])
            self.clf.set_params(n_estimators=cvresult.shape[0])

        self.clf.fit(x_train, y_train)
        print("In Sample Accuracy : {0:.4g}".format(accuracy_score(y_train, self.clf.predict(x_train))))
        if plot:
            plot_feature_importance(self.clf, self.features, xgb_fscore)
        return self

    def test(self, x_test, y_test, roc_curve=False):
        """
        # clf.predict_proba() will give you an array of 2n_samples in a binary problem [0 class, 1 class]
        # XGB.Booster.predict() will give you a 1n_samples vector where it is just the probability of the 1 class.
        """
        preds = self.clf.predict(x_test)
        print("Test Sample Accuracy : {0:.4g}".format(accuracy_score(y_test, preds)))
        print('Confusion matrix: \n')
        print(pd.crosstab(y_test, preds, colnames=['pred']))
        if roc_curve:
            ytest_predprob = self.clf.predict_proba(x_test)[:, 1]
            plot_roc_curve(y_test, ytest_predprob)


def plot_roc_curve(y_true, y_score):
    """
    Calculate and then plot ROC curve
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')  # random predictions curve
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve of Test Data')
    plt.legend(loc="lower right")


def plot_feature_importance(clf, features, xgb_fscore=False):
    if xgb_fscore:
        # F-score version
        feat_imp = pd.Series(clf.get_booster().get_fscore()).sort_values(ascending=False)
        feat_imp.plot(kind='barh', title='Feature Importances')
    else:
        importance = clf.feature_importances_
        indices = np.argsort(importance)

        plt.figure(figsize=(20, 10))
        plt.title('Feature Importances')
        plt.barh(range(len(indices)), importance[indices], color='b', align='center')
        plt.yticks(range(len(indices)), np.array(features)[indices])
        plt.xlabel('Relative Importance')


def tune_params(x_train, y_train, ind_params, cv_params):
    """
    Use cross validation to do grid search
    :param ind_params: dict
    :param cv_params: dict with list values
    """
    optimized_GBM = GridSearchCV(XGBClassifier(**ind_params), cv_params, scoring='recall', cv=5, n_jobs=-1)
    optimized_GBM.fit(x_train, y_train)
    return optimized_GBM.best_params_


def plot_learning_curve(train_set, test_set, features, target, params):
    """Plot learning curve based on train and test set
    to find out how much we benefit from 1. adding more training data and
                                         2. whether the estimator suffers more from a variance error or a bias error.
    """
    model = XGBClassifier(**params)
    eval_set = [(train_set[features], train_set[target]), (test_set[features], test_set[target])]
    model.fit(train_set[features], train_set[target], eval_metric=["auc", "logloss"], eval_set=eval_set, verbose=True)
    # retrieve performance metrics
    results = model.evals_result()
    epochs = len(results['validation_0']['auc'])
    x_axis = range(0, epochs)
    # plot log loss
    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
    ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
    ax.legend()
    plt.ylabel('Log Loss')
    plt.title('XGBoost Log Loss')


def use_early_stop(train_set, test_set, features, target, params):
    """similar to useTrainCV"""
    model = XGBClassifier(**params)
    eval_set = [(test_set[features], test_set[target])]

    model.fit(train_set[features], train_set[target], early_stopping_rounds=10, eval_metric="auc", eval_set=eval_set,
              verbose=True)
    return model


def xgb_main():
    x_train, y_train, x_test, y_test = get_train_test()
    # 1st round: first step=2, then smaller near best (e.g [6,7,8]; [0.5,1,2])
    cv_params = {'max_depth': list(range(3, 10, 2)), 'min_child_weight': list(range(1, 6, 2))}
    ind_params = {'learning_rate': 0.1, 'n_estimators': 1000, 'seed': 0, 'subsample': 0.8, 'colsample_bytree': 0.8,
                  'objective': 'binary:logistic'}
    tune_params(x_train, y_train, ind_params, cv_params)
    # 2nd round
    cv_params = {'gamma': [i / 10.0 for i in range(0, 5)]}
    ind_params = {'learning_rate': 0.1, 'n_estimators': 1000, 'seed': 0, 'subsample': 0.8, 'colsample_bytree': 0.8,
                  'objective': 'binary:logistic', 'max_depth': 9, 'min_child_weight': .75}
    tune_params(x_train, y_train, ind_params, cv_params)
    # 3nd round
    cv_params = {'subsample': [i / 10.0 for i in range(3, 10)], 'colsample_bytree': [i / 10.0 for i in range(6, 10)]}
    ind_params = {'learning_rate': 0.1, 'n_estimators': 1000, 'seed': 0,
                  'objective': 'binary:logistic', 'max_depth': 9, 'min_child_weight': .75, 'gamma': .4}
    tune_params(x_train, y_train, ind_params, cv_params)
    # 4th round
    cv_params = {'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100], 'learning_rate': [.1, .01]}
    ind_params = {'learning_rate': 0.1, 'n_estimators': 1000, 'seed': 0, 'colsample_bytree': 0.7, 'subsample': 0.95,
                  'objective': 'binary:logistic', 'max_depth': 9, 'min_child_weight': .75, 'gamma': .4}
    tune_params(x_train, y_train, ind_params, cv_params)

    our_params = {'learning_rate': 0.01, 'reg_alpha': 1, 'n_estimators': 1000, 'seed': 0, 'colsample_bytree': 0.7,
                  'subsample': 0.95, 'objective': 'binary:logistic', 'max_depth': 9, 'min_child_weight': .75,
                  'gamma': .4}
    gbm = ModelHelper(XGBClassifier, features, params=our_params).train(x_train, y_train, plot=True, xgb_fscore=True,
                                                                        useTrainCV=True)
    gbm.test(x_test, y_test, roc_curve=True)


def logistic_main():
    x_train, y_train, x_test, y_test = get_train_test()
    lg = ModelHelper(LogisticRegression, features, params={}).train(x_train, y_train)
    lg.test(x_test, y_test)


def rf_main():
    x_train, y_train, x_test, y_test = get_train_test()
    # Tune params
    param_grid = {"max_depth": [3, None],
                  "max_features": [1, 3, 10],
                  "min_samples_split": [2, 3, 10],
                  "min_samples_leaf": [1, 3, 10],
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]}
    grid_search = GridSearchCV(RandomForestClassifier(n_estimators=20), param_grid=param_grid, scoring='recall', cv=5)
    best_params = grid_search.fit(x_train, y_train).best_params_

    rf = ModelHelper(RandomForestClassifier, features, params=best_params).train(x_train, y_train, plot=True)
    rf.test(x_test, y_test)


def et_main():
    pass


def get_oof(clf, x_train, y_train, x_test):
    """
    Use KFold to get the train_set feature value
    Use average of the KFold prediction as the test_set feature value
    """
    SEED = 0  # for reproducibility
    NFOLDS = 5  # set folds for out-of-fold prediction
    kf = KFold(n_splits=NFOLDS, random_state=SEED)
    # store prediction results
    oof_train = np.zeros((x_train.shape[0],))
    oof_test = np.zeros((x_test.shape[0],))
    oof_test_skf = np.empty((NFOLDS, x_test.shape[0]))

    i = 0
    for train_index, test_index in kf.split(x_train):
        x_tr = x_train.iloc[train_index]
        y_tr = y_train.iloc[train_index]
        x_te = x_train.iloc[test_index]

        clf.fit(x_tr, y_tr)
        # only update test split
        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)
        i += 1

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


def stacking(x_train, y_train, x_test, y_test, gb_params, rf_model, lg_model):
    # Using 1st layer models to generate new features
    gb_model = XGBClassifier(**gb_params)
    rf_oof_train, rf_oof_test = get_oof(rf_model, x_train, y_train, x_test)  # Random Forest
    gb_oof_train, gb_oof_test = get_oof(gb_model, x_train, y_train, x_test)  # Gradient Boost
    lg_oof_train, lg_oof_test = get_oof(lg_model, x_train, y_train, x_test)  # Logistic Regression
    print("Training is complete")

    x_train = np.concatenate((rf_oof_train, gb_oof_train, lg_oof_train), axis=1)
    x_test = np.concatenate((rf_oof_test, gb_oof_test, lg_oof_test), axis=1)

    our_params = {'learning_rate': 1e-5, 'reg_alpha': 0.05, 'n_estimators': 1000, 'seed': 0, 'colsample_bytree': 0.55,
                  'subsample': 0.25, 'objective': 'binary:logistic', 'max_depth': 2, 'min_child_weight': .5,
                  'gamma': 0}
    final_model = ModelHelper(XGBClassifier, features, params=our_params).train(x_train, y_train, useTrainCV=True)
    final_model.test(x_test, y_test, roc_curve=True)
