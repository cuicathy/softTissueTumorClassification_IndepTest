import numpy as np
import pandas as pd
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from numpy import mean
from numpy import std
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import brier_score_loss
from xgboost import XGBClassifier
import warnings


class Classifier:
    # Example: classifier1 = Classifier(data_keys)
    # classifier1.outer_results
    def __init__(self, data_keys):
        self.outer_results = list()
        self.outer_results_prob = list()
        self.outer_gt = list()
        self.featureImportance_allfold = np.zeros(len(data_keys))
        self.featureImportance_fold = np.zeros(len(data_keys))
        self.featureImportance_allfold_classifier = np.zeros(len(data_keys))
        self.featureImportance_fold_classifier = np.zeros(len(data_keys))
        self.tprs = list()
        self.aucs = list()
        self.acc_folds = list()
        self.brier_folds = list()
        self.mean_fpr = np.linspace(0, 1, 100)
        self.train_img_index_list = list()
        self.test_img_index_list = list()

    def reinitialize(self, data_keys):
        self.outer_results = list()
        self.outer_results_prob = list()
        self.outer_gt = list()
        self.featureImportance_allfold = np.zeros(len(data_keys))
        self.featureImportance_fold = np.zeros(len(data_keys))
        self.tprs = list()
        self.aucs = list()
        self.acc_folds = list()
        self.brier_folds = list()
        self.mean_fpr = np.linspace(0, 1, 100)


def scale_features_train_test(data_train, data_test,keys=None):
    """
    Scale features by their mean and std
    Args:
        data (pandas dataframe): dataframe containing only the numeric
            radiomic values
    Returns:
    """
    df = data_train.copy()
    df_test = data_test.copy()

    if keys == None:
        keys = data_train.keys()

    for key in keys:
        d = data_train[key]
        d_test = data_test[key]

        # d = (d - d.mean()) / d.std()
        d_mean = d.mean()
        d_std = d.std()

        d = (d - d_mean) / d_std
        d_test = (d_test - d_mean) / d_std

        df[key] = d
        df_test[key] = d_test
    return df, df_test


def fxn():
    warnings.warn("deprecated", DeprecationWarning)


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()


def scale_features(data):
    """
    Scale features by their mean and std
    Args:
        data (pandas dataframe): dataframe containing only the numeric
            radiomic values
    Returns:
    """
    df = data.copy()
    for key in data.keys():
        d = data[key]

        d = (d - d.mean()) / d.std()

        df[key] = d
    return df


def scale_features_keys(data, keys):
    """
    Scale features by their mean and std
    Only scale the feature in keys
    Args:
        data (pandas dataframe): dataframe containing only the numeric
            radiomic values
    Returns:
    """
    df = data.copy()
    for key in keys:
        d = data[key]

        d = (d - d.mean()) / d.std()

        df[key] = d
    return df


def stability_selection(lasso, alphas, n_bootstrap_iterations, X, y, seed):
    """
    Bootstrap Logistic Regression to select features
    Args:
        lasso: Model
        alphas: regularization coefficient
        n_bootstrap_iterations: number of bootstrap
        X: Features
        y: Labels (Benign-0, Malignant-1)
        seed: random seed
    Returns:
        stability_scores of all features
    source:
    https://thuijskens.github.io/stability-selection/docs/index.html
    https://thuijskens.github.io/2018/07/25/stability-selection/
    """
    n_samples, n_variables = X.shape
    try:
        n_alphas = alphas.shape[0]
    except:
        n_alphas = 1

    if n_alphas > 1:
        rnd = np.random.RandomState(seed)
        selected_variables = np.zeros((n_variables, n_bootstrap_iterations))
        stability_scores = np.zeros((n_variables, n_alphas))

        for idx, alpha, in enumerate(alphas):
            # This is the sampling step, where bootstrap samples are generated
            # and the structure learner is fitted
            for iteration in range(n_bootstrap_iterations):
                bootstrap = rnd.choice(np.arange(n_samples),
                                       size=round(n_samples * 0.8),
                                       replace=True)
                X_train = X[bootstrap, :]
                y_train = y[bootstrap]
                lasso.set_params(C=alpha).fit(X_train,
                                              y_train)  # smaller C specifies stronger regularization and fewer selected features
                selected_variables[:, iteration] = (np.abs(lasso.coef_) > 1e-4)

            # This is the scoring step, where the final stability
            # scores are computed
            stability_scores[:, idx] = selected_variables.mean(
                axis=1)  # use mean value respect to the number of bootstrap. stability_scores: features x alpha
    else:
        rnd = np.random.RandomState(seed)
        selected_variables = np.zeros((n_variables, n_bootstrap_iterations))
        stability_scores = np.zeros((n_variables, 1))

        alpha = alphas
        # This is the sampling step, where bootstrap samples are generated
        # and the structure learner is fitted
        for iteration in range(n_bootstrap_iterations):
            bootstrap = rnd.choice(np.arange(n_samples),
                                   size=round(n_samples * 0.8),
                                   replace=True)
            X_train = X[bootstrap, :]
            y_train = y[bootstrap]
            lasso.set_params(C=alpha).fit(X_train,
                                          y_train)  # smaller C specifies stronger regularization and fewer selected features
            selected_variables[:, iteration] = (np.abs(lasso.coef_) > 1e-4)

        # This is the scoring step, where the final stability
        # scores are computed
        stability_scores[:, 0] = selected_variables.mean(
            axis=1)  # use mean value respect to the number of bootstrap. stability_scores: features x alpha

    return stability_scores


def thres_tuning(X, y, model, params, weights, cv):
    """
        Run a 5-fold cross validation to select the best threshold for feature reduction
        Args:
            X (nparray): Features
            y (nparray): Labels (Benign-0, Malignant-1)
            params (nparray): List of thresholds
            weights: weights from the stability selection
        Returns:
            best_thres: best threshold
    """
    from sklearn.model_selection import cross_validate
    # cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    best_thres = 0
    best_score = 0
    #cv_feaReduce = StratifiedKFold(n_splits=8, shuffle=True, random_state=1)
    for thres in params:
        selected_scores_bin = (weights.max(axis=1) > thres)
        selected_variables_idx = np.asarray(np.where(selected_scores_bin == True))
        X_selected = X[:, selected_variables_idx.ravel()]

        #result = cross_validate(model, X_selected, y, cv=cv_feaReduce, scoring='accuracy', return_train_score=True) # Change0410
        result = cross_validate(model, X_selected, y, cv=cv, scoring='accuracy', return_train_score=True)
        score = np.mean(result['test_score'])
        if best_score < score:
            best_score = score
            best_thres = thres
    return best_thres


def FeatureReduce_stability_selection(X, y, keys, cv, regularization_params=np.linspace(0.001, 0.5, num=100)):
    """
    Run Bootstrap Logestic Regression (L1 penalty) for feature reduction
    Args:
        X (nparray): Features
        y (nparray): Labels (Benign-0, Malignant-1)
        keys (list): Names of Features
        regularization_params (nparray): List of regularization coefficient where to compute the models
    Returns:
        idx of selected features
    Reference:
    https://thuijskens.github.io/2018/07/25/stability-selection/
    """
    lasso = LogisticRegression(penalty='l1', solver='liblinear', max_iter=100000,
                               random_state=3)  # C = 1/alpha Lasso-alpha, logesticReg-C
    n_bootstrap_iterations = 100
    stability_out = stability_selection(lasso, regularization_params, n_bootstrap_iterations, X, y, seed=3)
    selected_scores = stability_out.max(axis=1)
    thres_params = np.linspace(0, 0.7, num=14)
    print(thres_params)
    # tune the threshold for feature selection
    best_thres = thres_tuning(X, y, lasso, thres_params, stability_out, cv)
    print("threshold for feature reduction (by 5-fold cv): %f" % best_thres)
    selected_scores_bin = (stability_out.max(axis=1) > best_thres)
    selected_variables_idx = np.where(selected_scores_bin == True)
    for idx, (variable, score) in enumerate(zip(keys[selected_variables_idx], selected_scores[selected_variables_idx])):
        print('Variable %d: [%s], score %.3f' % (idx + 1, variable, score))
    return selected_variables_idx



def classifier_tuning_training_(classifier_model, classifier_param_grid, X_train_reduced, X_test_reduced, y_train,
                                y_test, cv_inner, selected_fea_idx, i,
                                classifier_property, classifier_name='rf'):
    search = GridSearchCV(classifier_model, classifier_param_grid, scoring='accuracy', cv=cv_inner, refit=True,
                          verbose=0)  # Evaluation metrics - accuracy
    result = search.fit(X_train_reduced, y_train.ravel())  # execute search
    # get the best performing model fit on the whole outer-training set
    best_model = result.best_estimator_
    print('Best Param', search.best_params_)
    # Test on the outer testing set
    classifier = best_model.fit(X_train_reduced, y_train.ravel())
    y_test_pred = classifier.predict(X_test_reduced)
    y_test_pred_prob = classifier.predict_proba(X_test_reduced)

    classifier_property.featureImportance_fold[selected_fea_idx] = 1  # classifier.feature_importances_
    classifier_property.featureImportance_allfold = classifier_property.featureImportance_fold + classifier_property.featureImportance_allfold

    if classifier_name == 'rf' or classifier_name == 'xgboost':
        classifier_property.featureImportance_fold_classifier[selected_fea_idx] = \
        classifier_property.featureImportance_fold_classifier[selected_fea_idx] + classifier.feature_importances_
        classifier_property.featureImportance_allfold_classifier = classifier_property.featureImportance_fold_classifier + classifier_property.featureImportance_allfold_classifier
    elif classifier_name == 'svm' or classifier_name == 'lr':
        classifier_property.featureImportance_fold_classifier[selected_fea_idx] = \
            classifier_property.featureImportance_fold_classifier[selected_fea_idx] + classifier.coef_[0]
        classifier_property.featureImportance_allfold_classifier = classifier_property.featureImportance_fold_classifier + classifier_property.featureImportance_allfold_classifier

    ###########################################################
    # Result Evaluation per outer-fold
    ###########################################################
    # Save the result of each outer testing set to a list for evaluation
    classifier_property.outer_results = [y for x in [classifier_property.outer_results, y_test_pred.ravel()] for y in x]
    classifier_property.outer_results_prob = [y for x in
                                              [classifier_property.outer_results_prob, y_test_pred_prob[:, 1]] for y in
                                              x]
    classifier_property.outer_gt = [y for x in [classifier_property.outer_gt, y_test.ravel()] for y in x]

    # Evaluate each outer-fold
    acc = accuracy_score(y_test, y_test_pred)
    classifier_property.acc_folds.append(acc)
    brier_score = brier_score_loss(y_test, y_test_pred)
    classifier_property.brier_folds.append(brier_score)
    # ROC per fold
    fpr, tpr, thresholds = roc_curve(y_test.ravel(), y_test_pred_prob[:, 1].ravel())
    mean_fpr = np.linspace(0, 1, 100)
    classifier_property.tprs.append(interp(mean_fpr, fpr, tpr))
    classifier_property.tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    classifier_property.aucs.append(roc_auc)

    print('Result of the %d th outer fold: Accuracy(testing set)=%.3f, Accuracy(Estimated from validation sets)=%.3f, '
          'Brier_score(testing set)=%.3f, AUC(testing set)=%.3f, Best Params=%s.  **IndepTest: Acc=%.3f, Brier=%.3f, AUC=%.3f' % (
              i, acc, result.best_score_, brier_score, roc_auc, result.best_params_, acc, brier_score, roc_auc))
    return classifier_property


def Overall_Evaluation_(classifier_property, data_keys, save_dir, plot_name, plot_flag=False):
    print("***************** Overall Evaluation Results *****************")
    all_acc = accuracy_score(classifier_property.outer_results[:], classifier_property.outer_gt[:])
    print(
    "confusion matrix:\n", confusion_matrix(classifier_property.outer_results[:], classifier_property.outer_gt[:]))
    print('Accuracy: %.3f' % all_acc)
    brier_score = brier_score_loss(classifier_property.outer_results[:],
                                   classifier_property.outer_gt[:])
    print('Brier score: %.3f' % brier_score)
    print("------")
    print('Average Accuracy:%.3f (%.3f)' % (
    np.mean(classifier_property.acc_folds, axis=0), np.std(classifier_property.acc_folds, axis=0)))
    print('Average Brier Score:%.3f (%.3f)' % (
    np.mean(classifier_property.brier_folds, axis=0), np.std(classifier_property.brier_folds, axis=0)))

    # Feature Importance: (Weighted_score)
    num_feature = 15
    classifier_property.featureImportance_allfold = classifier_property.featureImportance_allfold
    print("Max importance score:", np.amax(classifier_property.featureImportance_allfold))  # = featureImportance
    sorted_idx = classifier_property.featureImportance_allfold.argsort()[::-1]
    data_keys = np.asarray(data_keys)
    plt.figure(
        figsize=(18, 4))
    plt.barh(data_keys[sorted_idx[0:num_feature]],
             classifier_property.featureImportance_allfold[sorted_idx[0:num_feature]])
    print(data_keys[sorted_idx[0:num_feature]])
    plt.xlabel("Random Forest Feature Importance")
    plt.title(plot_name + ' :Feature Importance')
    plt.savefig(save_dir + '/' + plot_name + '_Feature_Importance.png')

    if plot_flag == True:
        plt.show()
    plt.close()

    # ROC_Curve
    fpr, tpr, thresholds = roc_curve(classifier_property.outer_gt, classifier_property.outer_results_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1, alpha=0.8, label='ROC (AUC = %0.2f)' % (roc_auc))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(plot_name + ' :AUC curve of tumor prediction (Average by image)')
    plt.legend(loc="lower right")
    plt.savefig(save_dir + '/' + plot_name + '_meanAUC_folds.png')
    if plot_flag == True:
        plt.show()
    plt.close()
    print('AUC of all images: %.3f' % (roc_auc))

    # Additional: ROC Curve Per Fold:
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k',
             label='Chance', alpha=.8)

    mean_tpr = np.mean(classifier_property.tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(classifier_property.mean_fpr, mean_tpr)
    mean_auc2 = np.mean(classifier_property.aucs)
    std_auc = np.std(classifier_property.aucs)
    plt.plot(classifier_property.mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f (%0.2f)$\pm$ %0.2f)' % (mean_auc, mean_auc2, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(classifier_property.tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(classifier_property.mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.' % std_tpr)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(plot_name + ':AUC curve of tumor prediction (Average by folds)')
    plt.legend(loc="lower right")
    plt.savefig(save_dir + '/' + plot_name + '_meanAUC_all.png')  # , dpi=300)
    if plot_flag == True:
        plt.show()
    plt.close()

    print('AUC of all folds: %.3f (%.3f)' % (mean(classifier_property.aucs), std(classifier_property.aucs)))
    return roc_auc, all_acc

def combine_two_class_all(classifier1, classifier2):
    classifier1.outer_results = classifier1.outer_results + classifier2.outer_results
    classifier1.outer_results_prob = classifier1.outer_results_prob + classifier2.outer_results_prob
    classifier1.outer_gt = classifier1.outer_gt + classifier2.outer_gt
    classifier1.featureImportance_allfold = classifier1.featureImportance_allfold + classifier2.featureImportance_allfold
    classifier1.featureImportance_fold = classifier1.featureImportance_fold + classifier2.featureImportance_fold
    classifier1.featureImportance_allfold_classifier = classifier1.featureImportance_allfold_classifier + classifier2.featureImportance_allfold_classifier
    classifier1.featureImportance_fold_classifier = classifier1.featureImportance_fold_classifier + classifier2.featureImportance_fold_classifier
    classifier1.tprs = classifier1.tprs + classifier2.tprs
    classifier1.aucs = classifier1.aucs + classifier2.aucs
    classifier1.acc_folds = classifier1.acc_folds + classifier2.acc_folds
    classifier1.brier_folds = classifier1.brier_folds + classifier2.brier_folds
    return classifier1


def LASSO_Classifier_IndepTesting(save_dir, df_features, df_label, df_features_IndepTest,
                                                    df_label_IndepTest, data_keys , outer_fold=1, inner_fold=8,
                                                    reduceFeature_method='LASSO_ALL_ALPHAS', plot_name='',
                                                    feaReduce_flag=True):
    assert(df_features_IndepTest.shape[1] == len(data_keys))
    assert (df_features.shape[1] == len(data_keys))
    assert (df_features_IndepTest.shape[0] == len(df_label_IndepTest))
    assert (df_features.shape[0] == len(df_label))
    """
    Run Bootstrap LASSO for feature reduction and 5 classifiers to diagnose
    Output:
        Diagnosis results
    """
    selection_param_grid = {
        'C': np.logspace(-2, 2, 100)
    }
    classifier_model1 = RandomForestClassifier(random_state=3)
    
    print(reduceFeature_method)
    classifier_param_grid1 = {
        'min_samples_leaf': [1, 2, 3],
        'n_estimators': [100, 300, 500],
        'max_features': ['auto', 'log2'],
        'max_depth': [3, 5],
        'criterion': ['gini', 'entropy']
    }

    classifier_model2 = LogisticRegression(random_state=3, penalty='l2', solver='liblinear', max_iter=100000)
    classifier_param_grid2 = {
        'C': np.logspace(-2, 2, 100)
    }  # logistic regression

    from sklearn.neural_network import MLPClassifier
    classifier_param_grid3 = {'learning_rate_init': [1e-04, 1e-03, 1e-02],
        'alpha': np.logspace(-3, 1, 5),
        'hidden_layer_sizes': [(64, 16), (64, 32, 16)]
    }

    classifier_model3 = MLPClassifier(random_state=1, solver='lbfgs', max_iter=2000, early_stopping=True) # Add 0410 max_iter=2000, early_stopping=True

    classifier_param_grid4 = {
        'n_estimators': [100, 300, 500],
        'max_depth': [3, 5]
    }

    classifier_model4 = XGBClassifier(learning_rate=0.01, gamma=0,
                                      min_child_weight=1, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.005,
                                      random_state=3)
    classifier_param_grid5 = {
        'C': np.logspace(-3, 2, 20),
        'kernel': ['rbf', 'linear']
    }
    from sklearn.svm import SVC
    classifier_model5 = SVC(random_state=3, probability=True, gamma='scale')

    print('classifier_param_grid1: \n', classifier_param_grid1)
    print('classifier_param_grid2: \n', classifier_param_grid2)
    print('classifier_param_grid3: \n', classifier_param_grid3)
    print('classifier_param_grid4: \n', classifier_param_grid4)
    print('classifier_param_grid5: \n', classifier_param_grid5)
    if feaReduce_flag == True:
        print('**********************************************')
        print('selection_param_grid: \n', selection_param_grid)
        print('reduceFeature_method: \n', reduceFeature_method)

    features = df_features.values
    label = df_label.values.ravel()

    label_IndepTest = df_label_IndepTest.values.ravel()
    data_keys = np.asarray(data_keys)

    mean_auc_perCV_list1 = []
    mean_auc_perCV_list2 = []
    mean_auc_perCV_list3 = []
    mean_auc_perCV_list4 = []
    mean_auc_perCV_list5 = []

    mean_acc_perCV_list1 = []
    mean_acc_perCV_list2 = []
    mean_acc_perCV_list3 = []
    mean_acc_perCV_list4 = []
    mean_acc_perCV_list5 = []

    classifier_propertyall_1 = Classifier(data_keys)
    classifier_propertyall_2 = Classifier(data_keys)
    classifier_propertyall_3 = Classifier(data_keys)
    classifier_propertyall_4 = Classifier(data_keys)
    classifier_propertyall_5 = Classifier(data_keys)

    outer_gt_cvs_1 = []
    outer_results_prob_cvs_1 = []
    outer_gt_cvs_2 = []
    outer_results_prob_cvs_2 = []
    outer_gt_cvs_3 = []
    outer_results_prob_cvs_3 = []
    outer_gt_cvs_4 = []
    outer_results_prob_cvs_4 = []
    outer_gt_cvs_5 = []
    outer_results_prob_cvs_5 = []

    ##########################
    # Outer loop
    ##########################
    for cv_i in range(100):
        cv_inner = StratifiedKFold(n_splits=inner_fold, shuffle=True, random_state=cv_i)
        classifier_property1_1 = Classifier(data_keys)
        classifier_property1_2 = Classifier(data_keys)
        classifier_property1_3 = Classifier(data_keys)
        classifier_property1_4 = Classifier(data_keys)
        classifier_property1_5 = Classifier(data_keys)
        print('----------------------------------------------------------------------------------')
        print('-----------------------------CROSS VALIDATION-----------------------------------')
        print('----------------------------------------------------------------------------------')

        train_ix = np.array([r for r in range(len(label))])
        test_ix = np.array([r for r in range(len(label_IndepTest))])
        # Record the testing index
        classifier_property1_1.train_img_index_list.append(train_ix)
        classifier_property1_1.test_img_index_list.append(test_ix)
        classifier_propertyall_1.test_img_index_list.append(
            test_ix.tolist())  # The format of classifier_propertyall_1 is different from others
        X_train = features
        y_train = label

        #X_test = features_IndepTest
        y_test = label_IndepTest

        selected_fea_idx = []
        if feaReduce_flag == True:
            #########################################
            # Feature Selection(Reduction) with LASSO
            #########################################
            if reduceFeature_method == 'LASSO_ALL_ALPHAS':
                #alphas = np.linspace(1, 100, num=100)
                C_values = np.logspace(-2, 2, 100) ### Change
                selected_fea_idx = FeatureReduce_stability_selection(X_train, y_train, data_keys, cv_inner, C_values)
        else:
            selected_fea_idx = np.linspace(0, len(data_keys) - 1, len(data_keys)).astype(int)
        print("Selected Features:", data_keys[selected_fea_idx])

        selected_fea = df_features[data_keys[selected_fea_idx]].copy().values
        X_train_reduced = selected_fea[train_ix, :]
        X_test_reduced = df_features_IndepTest[data_keys[selected_fea_idx]].copy().values  # Training and Testing sets with reduced features
        #######################################################################
        # Tuning and Training the classifier with reduced features
        #######################################################################
        classifier_property1_1 = classifier_tuning_training_(classifier_model1, classifier_param_grid1,
                                                             X_train_reduced, X_test_reduced, y_train,
                                                             y_test, cv_inner, selected_fea_idx, 0,
                                                             classifier_property1_1, 'rf')

        classifier_property1_2 = classifier_tuning_training_(classifier_model2, classifier_param_grid2,
                                                             X_train_reduced, X_test_reduced, y_train,
                                                             y_test, cv_inner, selected_fea_idx, 0,
                                                             classifier_property1_2, 'lr')
        classifier_property1_3 = classifier_tuning_training_(classifier_model3, classifier_param_grid3,
                                                             X_train_reduced, X_test_reduced, y_train,
                                                             y_test, cv_inner, selected_fea_idx, 0,
                                                             classifier_property1_3, 'nn')
        classifier_property1_4 = classifier_tuning_training_(classifier_model4, classifier_param_grid4,
                                                             X_train_reduced,
                                                             X_test_reduced, y_train,
                                                             y_test, cv_inner, selected_fea_idx, 0,
                                                             classifier_property1_4, '_xgboost')
        classifier_property1_5 = classifier_tuning_training_(classifier_model5, classifier_param_grid5,
                                                             X_train_reduced,
                                                             X_test_reduced, y_train,
                                                             y_test, cv_inner, selected_fea_idx, 0,
                                                             classifier_property1_5, 'svm')

        print('**********Random Forest**************')
        plot_name1 = 'CV' + str(cv_i) + '_rf'
        auc1, acc1 = Overall_Evaluation_(classifier_property1_1, data_keys, outer_fold, save_dir, plot_name=plot_name1)
        print('**********Logistic Regression**************')
        plot_name2 = 'CV' + str(cv_i) + '_lr'
        auc2, acc2 = Overall_Evaluation_(classifier_property1_2, data_keys, outer_fold, save_dir, plot_name=plot_name2)
        print('**********Neural Network**************')
        plot_name3 = 'CV' + str(cv_i) + '_nn'
        auc3, acc3 = Overall_Evaluation_(classifier_property1_3, data_keys, outer_fold, save_dir, plot_name=plot_name3)
        print('**********xgboost**************')
        plot_name4 = 'CV' + str(cv_i) + '_xgboost'
        auc4, acc4 = Overall_Evaluation_(classifier_property1_4, data_keys, outer_fold, save_dir, plot_name=plot_name4)
        print('**********SVM**************')
        plot_name5 = 'CV' + str(cv_i) + '_svm'
        auc5, acc5 = Overall_Evaluation_(classifier_property1_5, data_keys, outer_fold, save_dir, plot_name=plot_name5)


        mean_auc_perCV_list1.append(auc1)
        mean_auc_perCV_list2.append(auc2)
        mean_auc_perCV_list3.append(auc3)
        mean_auc_perCV_list4.append(auc4)
        mean_auc_perCV_list5.append(auc5)

        mean_acc_perCV_list1.append(acc1)
        mean_acc_perCV_list2.append(acc2)
        mean_acc_perCV_list3.append(acc3)
        mean_acc_perCV_list4.append(acc4)
        mean_acc_perCV_list5.append(acc5)


        print('CV d%:', cv_i)
        print(classifier_property1_1.train_img_index_list)
        print(classifier_property1_1.test_img_index_list)
        print('gt:', classifier_property1_1.outer_gt)
        print('rf :', classifier_property1_1.outer_results)
        print('lr :', classifier_property1_2.outer_results)
        print('nn :', classifier_property1_3.outer_results)
        print('xgb:', classifier_property1_4.outer_results)
        print('svm:', classifier_property1_5.outer_results)


        classifier_propertyall_1 = combine_two_class_all(classifier_propertyall_1, classifier_property1_1)
        classifier_propertyall_2 = combine_two_class_all(classifier_propertyall_2, classifier_property1_2)
        classifier_propertyall_3 = combine_two_class_all(classifier_propertyall_3, classifier_property1_3)
        classifier_propertyall_4 = combine_two_class_all(classifier_propertyall_4, classifier_property1_4)
        classifier_propertyall_5 = combine_two_class_all(classifier_propertyall_5, classifier_property1_5)

        outer_gt_cvs_1.append(classifier_property1_1.outer_gt)
        outer_results_prob_cvs_1.append(classifier_property1_1.outer_results_prob)
        outer_gt_cvs_2.append(classifier_property1_2.outer_gt)
        outer_results_prob_cvs_2.append(classifier_property1_2.outer_results_prob)
        outer_gt_cvs_3.append(classifier_property1_3.outer_gt)
        outer_results_prob_cvs_3.append(classifier_property1_3.outer_results_prob)
        outer_gt_cvs_4.append(classifier_property1_4.outer_gt)
        outer_results_prob_cvs_4.append(classifier_property1_4.outer_results_prob)
        outer_gt_cvs_5.append(classifier_property1_5.outer_gt)
        outer_results_prob_cvs_5.append(classifier_property1_5.outer_results_prob)

    print('----------ALL-Evaluation--------------')
    print('**********Random Forest**************')
    plot_name1 = 'CV' + '_all' + '_rf'
    Overall_Evaluation_crossCV(classifier_propertyall_1, outer_gt_cvs_1, outer_results_prob_cvs_1, data_keys,
                               outer_fold, save_dir, plot_name=plot_name1)
    print('**********Logistic Regression**************')
    plot_name2 = 'CV' + '_all' + '_lr'
    Overall_Evaluation_crossCV(classifier_propertyall_2, outer_gt_cvs_2, outer_results_prob_cvs_2, data_keys,
                               outer_fold, save_dir, plot_name=plot_name2)
    print('**********Neural Network**************')
    plot_name3 = 'CV' + '_all' + '_nn'
    Overall_Evaluation_crossCV(classifier_propertyall_3, outer_gt_cvs_3, outer_results_prob_cvs_3, data_keys,
                               outer_fold, save_dir, plot_name=plot_name3)
    print('**********xgboost**************')
    plot_name4 = 'CV' + '_all' + '_xgboost'
    Overall_Evaluation_crossCV(classifier_propertyall_4, outer_gt_cvs_4, outer_results_prob_cvs_4, data_keys,
                               outer_fold, save_dir, plot_name=plot_name4)
    print('**********SVM**************')
    plot_name5 = 'CV' + '_all' + '_svm'
    Overall_Evaluation_crossCV(classifier_propertyall_5, outer_gt_cvs_5, outer_results_prob_cvs_5, data_keys,
                               outer_fold, save_dir, plot_name=plot_name5)
    print('----------ALL-AUC--------------')
    print('rf Average AUC (cv): %.3f (%.3f)' % (mean(mean_auc_perCV_list1), std(mean_auc_perCV_list1)))
    print('lr Average AUC (cv): %.3f (%.3f)' % (mean(mean_auc_perCV_list2), std(mean_auc_perCV_list2)))
    print('nn Average AUC (cv): %.3f (%.3f)' % (mean(mean_auc_perCV_list3), std(mean_auc_perCV_list3)))
    print('xgb Average AUC (cv): %.3f (%.3f)' % (mean(mean_auc_perCV_list4), std(mean_auc_perCV_list4)))
    print('svm Average AUC (cv): %.3f (%.3f)' % (mean(mean_auc_perCV_list5), std(mean_auc_perCV_list5)))

    print('----------ALL-ACC-------------')
    print('rf Average ACC (cv): %.3f (%.3f)' % (mean(mean_acc_perCV_list1), std(mean_acc_perCV_list1)))
    print('lr Average ACC (cv): %.3f (%.3f)' % (mean(mean_acc_perCV_list2), std(mean_acc_perCV_list2)))
    print('nn Average ACC (cv): %.3f (%.3f)' % (mean(mean_acc_perCV_list3), std(mean_acc_perCV_list3)))
    print('xgb Average ACC (cv): %.3f (%.3f)' % (mean(mean_acc_perCV_list4), std(mean_acc_perCV_list4)))
    print('svm Average ACC (cv): %.3f (%.3f)' % (mean(mean_acc_perCV_list5), std(mean_acc_perCV_list5)))


def Overall_Evaluation_crossCV(classifier_property, outer_gt_cvs, outer_results_prob_cvs, data_keys,
                               save_dir, plot_name, plot_flag=False):
    print("***************** Overall Evaluation Results *****************")
    number_of_cv = len(outer_gt_cvs)
    tprs_cvs = []
    auc_cvs = []
    mean_fpr = np.linspace(0, 1, 100)
    for i in range(number_of_cv):
        fpr, tpr, thresholds = roc_curve(outer_gt_cvs[i], outer_results_prob_cvs[i])
        tprs_temp = interp(mean_fpr, fpr, tpr)
        tprs_temp[-1] = 0.0
        tprs_cvs.append(tprs_temp)
        roc_auc = auc(fpr, tpr)
        auc_cvs.append(roc_auc)

        plt.plot(fpr, tpr, lw=1, alpha=0.3)  # , label='ROC (AUC = %0.2f)' % (roc_auc))

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k',
             label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs_cvs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    mean_auc2 = np.mean(auc_cvs)
    std_auc = np.std(auc_cvs)
    plt.plot(classifier_property.mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f (%0.2f)$\pm$ %0.2f)' % (mean_auc, mean_auc2, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs_cvs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.' % std_tpr)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(plot_name + ':AUC curve of tumor prediction (Average by folds)')
    plt.legend(loc="lower right")
    plt.savefig(save_dir + '/' + plot_name + '_meanAUC_all.png')  # , dpi=300)
    if plot_flag == True:
        plt.show()
    plt.close()

    # print('Accuracy of folds: %.3f (%.3f)' % (mean(outer_results), std(outer_results)))
    print('AUC of all folds: %.3f (%.3f)' % (mean_auc, std_auc))
    print('AUC(true) of all folds: %.3f (%.3f)' % (mean_auc2, std_auc))
    # return roc_auc, all_acc

    all_acc = accuracy_score(classifier_property.outer_results[:], classifier_property.outer_gt[:])
    print(
    "confusion matrix:\n", confusion_matrix(classifier_property.outer_results[:], classifier_property.outer_gt[:]))
    print('Accuracy: %.3f' % all_acc)
    brier_score = brier_score_loss(classifier_property.outer_results[:], classifier_property.outer_gt[:])
    print('Brier score: %.3f' % brier_score)
    print("------")
    print('Average Accuracy:%.3f (%.3f)' % (
    np.mean(classifier_property.acc_folds, axis=0), np.std(classifier_property.acc_folds, axis=0)))
    print('Average Brier Score:%.3f (%.3f)' % (
    np.mean(classifier_property.brier_folds, axis=0), np.std(classifier_property.brier_folds, axis=0)))

    num_feature = 30
    classifier_property.featureImportance_allfold = classifier_property.featureImportance_allfold
    print("Max importance score:", np.amax(classifier_property.featureImportance_allfold))  # = featureImportance
    sorted_idx = classifier_property.featureImportance_allfold.argsort()[::-1]
    data_keys = np.asarray(data_keys)
    plt.figure(
        figsize=(30, 4))  #########################################################################################
    plt.barh(data_keys[sorted_idx[0:num_feature]],
             classifier_property.featureImportance_allfold[sorted_idx[0:num_feature]])
    print(data_keys[sorted_idx[0:num_feature]])
    plt.xlabel("Feature Importance")
    plt.title(plot_name + ' :Feature Importance')
    plt.savefig(save_dir + '/' + plot_name + '_Feature_Importance.png')
    print(data_keys[sorted_idx[0:num_feature]])
    if plot_flag == True:
        plt.show()
    plt.close()

    # Feature Importance Classifier
    num_feature = 30
    classifier_property.featureImportance_allfold_classifier = classifier_property.featureImportance_allfold_classifier / number_of_cv
    print(
    "Max importance score:", np.amax(classifier_property.featureImportance_allfold_classifier))  # = featureImportance
    sorted_idx = classifier_property.featureImportance_allfold_classifier.argsort()[::-1]
    data_keys = np.asarray(data_keys)
    plt.figure(
        figsize=(30, 4))  #########################################################################################
    plt.barh(data_keys[sorted_idx[0:num_feature]],
             classifier_property.featureImportance_allfold_classifier[sorted_idx[0:num_feature]])
    print(data_keys[sorted_idx[0:num_feature]])
    plt.xlabel("Feature Importance")
    plt.title(plot_name + ' :Feature Importance')
    plt.savefig(save_dir + '/' + plot_name + '_Classifier_Feature_Importance.png')
    print(data_keys[sorted_idx[0:num_feature]])
    if plot_flag == True:
        plt.show()
    plt.close()

    # ROC_Curve
    fpr, tpr, thresholds = roc_curve(classifier_property.outer_gt, classifier_property.outer_results_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1, alpha=0.8, label='ROC (AUC = %0.2f)' % (roc_auc))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(plot_name + ' :AUC curve of tumor prediction (Average by image)')
    plt.legend(loc="lower right")
    plt.savefig(save_dir + '/' + plot_name + '_meanAUC_folds.png')
    if plot_flag == True:
        plt.show()
    plt.close()
    print('AUC of all images: %.3f' % (roc_auc))

    # print('Accuracy of folds: %.3f (%.3f)' % (mean(outer_results), std(outer_results)))
    print('AUC of all folds: %.3f (%.3f)' % (mean(classifier_property.aucs), std(classifier_property.aucs)))
    return roc_auc, all_acc





