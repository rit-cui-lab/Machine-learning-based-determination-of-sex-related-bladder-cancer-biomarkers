from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import os
os.environ['R_HOME'] = "C:\\Program Files\\R\\R-4.4.2"
import rpy2.robjects as ro
from rpy2.robjects import r
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
import random

# set random seeds in addition to random state throughout the code 
random.seed(3)
np.random.seed(3)


r.source("C:/Users/joepi/Code/Thesis/R/biomart.R")


# function to optimize a single random forest with each pipeline before 1000 iterations are run
# uses the validation set to perform hyperparameter optimization
def single_rf_optimizer(x_train, y_train, model, n_iters, know_param=False, n_folds=5):
    if not know_param:
        print("Now executing hyperparameter grid search for random forest prior to 1000 fittings...")
        # set up grid with values to test for each parameter
        rf_param_grid = dict()
        rf_param_grid['max_depth'] = [3, 5, 10, None]
        rf_param_grid['criterion'] = ["gini", "entropy", "log_loss"]
        rf_param_grid['n_estimators'] = [300, 500]
        rf_param_grid['min_samples_split'] = [2, 5, 10]
        rf_param_grid['max_features'] = ["sqrt", "log2"]
        rf_param_grid['min_samples_leaf'] = [1, 4, 8]
        # validation set is created through 10-fold stratified k-fold
        # 10 subsets within training set, each one becomes the test set
        cv = StratifiedKFold(n_splits=n_folds, random_state=3, shuffle=True)
        # search the parameter grid with 10-fold stratified CV over multiple iterations
        rf_hyper = GridSearchCV(model, rf_param_grid, n_jobs=1, verbose=0,
                                      error_score='raise',
                                      cv=cv, scoring=composite_score)
        random_search = rf_hyper.fit(x_train, y_train)
        # extract the best parameters, display them, then output them
        best_grid = random_search.best_params_
        print("The best parameters are:")
        print(best_grid)
    else:
        # include option for defined parameters
        best_grid = dict()
        

    return best_grid


def composite_score(estimator, X, y):

    y_pred = estimator.predict(X)
    y_proba = estimator.predict_proba(X)[:, 1]  

    f1 = f1_score(y, y_pred)
    auroc = roc_auc_score(y, y_proba)
    bal_acc = balanced_accuracy_score(y, y_pred)

    return 0.5*f1 + 0.3*auroc + 0.2*bal_acc



# function to execute optimized random forest technique for feature selection
def optimized_rf_gene_select(df, target, rf_n_iter=1000, cv_n_iter=80, skip_param=False, n_folds=5):
    
    # avoid warning message by converting target Series to list
    target = target.to_list()
    
    # instantiate RF model with defined random state
    rf_model = RandomForestClassifier(random_state=3, class_weight='balanced')

    if not skip_param:
        # if we are validating hyperparameters, make sure internal classifier is not too resource heavy
        rf_model.set_params(n_jobs=1)
        # execute hyperparameter optimization on RF model
        best_grid = single_rf_optimizer(df, target, rf_model, n_iters=cv_n_iter, know_param=False, n_folds=n_folds)
        # set the best parameters from the search and let classifier use all available CPUs afterward
        rf_model.set_params(**best_grid)
        rf_model.set_params(n_jobs=1)
    else:
        rf_model.set_params(n_jobs=1)
        best_acc = None
        knn_n = None

    # initialize series to store the top 100 genes in each RF trial
    genes = pd.Series(dtype=float)

    # repeat the random forest algorithm for the desired number of iterations
    print("Now running", rf_n_iter, "iterations of RandomForest, top 100 genes are stored at each one. ")
    for i in tqdm(range(rf_n_iter)):
        # set random state of random forest classifier with the optimal hyperparameters
        rf_model.set_params(random_state=i)
        rf_model.fit(df, target)

        # extract feature importances from the RF model, sort from highest to lowest
        rf_importances = pd.Series(rf_model.feature_importances_, index=rf_model.feature_names_in_).sort_values(ascending=False)

        # extract the names and RF feature importances of the top 300 genes and put them in a pandas Series, then append it to the working Series
        genes = pd.concat([genes, pd.Series(rf_importances.iloc[:300])])

    # the genes Series will contain all 100,000 genes and importances selected in the 1000 iterations of RF WITH REPEATS!
    # group by index, such that entries with the same gene name have their importances summed
    gene_top100_count = genes.groupby(genes.index).sum().sort_values(ascending=False)
    tally_genes = gene_top100_count.index.to_list()

    # then extract genes with the top 300 importance sums across all 1000 iterations
    top100 = tally_genes[:300]

    return top100

def svm_rfe(X_train, Y_train, n_folds=5):

    # avoid warning message by converting target Series to list
    Y_train = Y_train.to_list()

    # instantiate SVM classifier and cross-validation scheme
    svm = SVC(probability=True, random_state=3, kernel='linear', class_weight='balanced')

    cv = StratifiedKFold(n_splits=n_folds, random_state=3, shuffle=True)

    # hyperparameter optimize the SVM model
    # first set up parameter grid, only C since a linear kernel is used
    param_grid = dict()
    param_grid['svc__C'] = [0.001, 0.01, .1, 1, 5, 10, 100]

    # set up pipeline with both scaler and SVM 
    # we z-score each dataset independently before merging, but we need to do this again to bring all features to mean = 0 and sd = 1. 
    svm_pipe = make_pipeline(StandardScaler(), svm)

    # execute exhaustive search, extract best-performing regularization param value in terms of composite metric
    hyper_search = GridSearchCV(svm_pipe, param_grid, n_jobs=1, cv=cv, verbose=1,
                                      error_score='raise', scoring=composite_score)
    print('Performing exhaustive grid search for SVM pipeline to find optimal C value...')
    hyper_search.fit(X_train, Y_train)
    svm_pipe = hyper_search.best_estimator_

    # now that the best parameters for the SVM have been identified, execute recursive feature elimination
    rfe = RFECV(svm_pipe, step=0.001, min_features_to_select=300, cv=cv, scoring=composite_score, importance_getter='named_steps.svc.coef_', n_jobs=1)
    print('Now executing SVM-RFE to select the top 300 genes...')
    rfe.fit(X_train, Y_train)

    # get selected features from RFE
    selected = X_train.columns[rfe.support_]
    
    # get coefs from final svm model 
    svm_final = rfe.estimator_.named_steps["svc"]
    coefs = np.abs(svm_final.coef_.ravel())

    # sort from most to least important 
    sorted_genes = np.argsort(coefs)[::-1]
    sorted_genes = sorted_genes[:300]

    top300 = pd.DataFrame({"gene":selected[sorted_genes], "abs_coef": coefs[sorted_genes]}).reset_index(drop=True)

    return top300['gene'].to_list()


def logit_cv(X_train, Y_train, n_folds=5): 
    
    # avoid warning message by converting target Series to list
    Y_train = Y_train.to_list()

    # set up cross-validation folds and load composite metric
    folds = StratifiedKFold(n_splits=n_folds, random_state=3, shuffle=True)

    # create logit regression with cross validation, L1 penalty, and increase max iterations
    logit = LogisticRegressionCV(Cs=[0.01, 0.1, 1, 10], scoring=composite_score, cv=folds, random_state=3, penalty='l1', solver='saga', max_iter=5000, n_jobs=1)

    # add logitcv into pipeline with scaling (needed for linear model)
    logit_pipe = make_pipeline(StandardScaler(), logit)

    # fit to training data
    print('Performing cross-validated logistic regression to select the top 300 genes...')
    logit_pipe.fit(X_train, Y_train)

    # extract feature weights from logistic regression model
    logit = logit_pipe.named_steps['logisticregressioncv']
    logit_coefs = logit.coef_.ravel()
    coef_df = pd.DataFrame({'gene': X_train.columns, 'logit_coef': logit_coefs})

    # take absolute value and sort 
    coef_df['abs_coef'] = np.abs(coef_df['logit_coef'])
    coef_df = coef_df.sort_values('abs_coef', ascending=False)

    # get top 300
    top300 = coef_df['gene'].iloc[:300].to_list()

    return top300


def merged_dataset_gene_selection(train_expr_1, train_expr_2, test_expr_1, test_expr_2, train_target_1, train_target_2, test_target_1, test_target_2, mode, training, correction_method, fold_num):

    # perform feature selection for cohort1 (male/tumor)
    top100_svm_1 = svm_rfe(train_expr_1, train_target_1)
    top100_lasso_1 = logit_cv(train_expr_1, train_target_1)
    top100_rf_1  = optimized_rf_gene_select(train_expr_1, train_target_1, rf_n_iter=1000, cv_n_iter=60)

    if mode == "gender_stratified":
        
        # doing feature selection for healthy male vs. female is not useful
        # we only needed to use this cohort for DGEA - remove DEGs for healthy male vs. female from DEGs for tumor male vs. female
         # only need to do cohort2 for gender-stratified analysis (female)  
        top100_lasso_2 = logit_cv(train_expr_2, train_target_2)
        top100_svm_2 = svm_rfe(train_expr_2, train_target_2)
        top100_rf_2 = optimized_rf_gene_select(train_expr_2, train_target_2, rf_n_iter=1000, cv_n_iter=60)

        train_expr_1[top100_lasso_1].to_csv("./post_selection_data/" + training + "_" + correction_method + "_logit_cv_300_male_train_" + str(fold_num) + ".csv")
        test_expr_1[top100_lasso_1].to_csv("./post_selection_data/" + training + "_" + correction_method + "_logit_cv_300_male_test_" + str(fold_num) + ".csv")
        train_expr_1[top100_rf_1].to_csv("./post_selection_data/" + training + "_" + correction_method + "_opt_rf_300_male_train_" + str(fold_num) + ".csv")
        test_expr_1[top100_rf_1].to_csv("./post_selection_data/" + training + "_" + correction_method + "_opt_rf_300_male_test_" + str(fold_num) + ".csv")
        train_expr_1[top100_svm_1].to_csv("./post_selection_data/" + training + "_" + correction_method + "_svm_rfe_300_male_train_" + str(fold_num) + ".csv")
        test_expr_1[top100_svm_1].to_csv("./post_selection_data/" + training + "_" + correction_method + "_svm_rfe_300_male_test_" + str(fold_num) + ".csv")
        train_expr_1.iloc[:, : 300].to_csv("./post_selection_data/" + training + "_" + correction_method + "_adjp_300_male_train_" + str(fold_num) + ".csv")
        test_expr_1.iloc[:, : 300].to_csv("./post_selection_data/" + training + "_" + correction_method + "_adjp_300_male_test_" + str(fold_num) + ".csv")
        train_expr_2[top100_lasso_2].to_csv("./post_selection_data/" + training + "_" + correction_method + "_logit_cv_300_female_train_" + str(fold_num) + ".csv")
        test_expr_2[top100_lasso_2].to_csv("./post_selection_data/" + training + "_" + correction_method + "_logit_cv_300_female_test_" + str(fold_num) + ".csv")
        train_expr_2[top100_rf_2].to_csv("./post_selection_data/" + training + "_" + correction_method + "_opt_rf_300_female_train_" + str(fold_num) + ".csv")
        test_expr_2[top100_rf_2].to_csv("./post_selection_data/" + training + "_" + correction_method + "_opt_rf_300_female_test_" + str(fold_num) + ".csv")
        train_expr_2[top100_svm_2].to_csv("./post_selection_data/" + training + "_" + correction_method + "_svm_rfe_300_female_train_" + str(fold_num) + ".csv")
        test_expr_2[top100_svm_2].to_csv("./post_selection_data/" + training + "_" + correction_method + "_svm_rfe_300_female_test_" + str(fold_num) + ".csv")
        train_expr_2.iloc[:, : 300].to_csv("./post_selection_data/" + training + "_" + correction_method + "_adjp_300_female_train_" + str(fold_num) + ".csv")
        test_expr_2.iloc[:, : 300].to_csv("./post_selection_data/" + training + "_" + correction_method + "_adjp_300_female_test_" + str(fold_num) + ".csv")
        
        # targets
        train_target_1.to_csv("./post_selection_data/" + training + "_" + correction_method + "_male_train_target_" + str(fold_num) + ".csv")
        test_target_1.to_csv("./post_selection_data/" + training + "_" + correction_method + "_male_test_target_" + str(fold_num) + ".csv")
        train_target_2.to_csv("./post_selection_data/" + training + "_" + correction_method + "_female_train_target_" + str(fold_num) + ".csv")
        test_target_2.to_csv("./post_selection_data/" + training + "_" + correction_method + "_female_test_target_" + str(fold_num) + ".csv")

    elif mode == "disease_stratified":

        # only need top 300 panels for tumor cohort 
        train_expr_1[top100_lasso_1].to_csv("./post_selection_data/" + training + "_" + correction_method + "_logit_cv_300_tumor_train_" + str(fold_num) + ".csv")
        test_expr_1[top100_lasso_1].to_csv("./post_selection_data/" + training + "_" + correction_method + "_logit_cv_300_tumor_test_" + str(fold_num) + ".csv")
        train_expr_1[top100_rf_1].to_csv("./post_selection_data/" + training + "_" + correction_method + "_opt_rf_300_tumor_train_" + str(fold_num) + ".csv")
        test_expr_1[top100_rf_1].to_csv("./post_selection_data/" + training + "_" + correction_method + "_opt_rf_300_tumor_test_" + str(fold_num) + ".csv")
        train_expr_1[top100_svm_1].to_csv("./post_selection_data/" + training + "_" + correction_method + "_svm_rfe_300_tumor_train_" + str(fold_num) + ".csv")
        test_expr_1[top100_svm_1].to_csv("./post_selection_data/" + training + "_" + correction_method + "_svm_rfe_300_tumor_test_" + str(fold_num) + ".csv")
        train_expr_1.iloc[:, : 300].to_csv("./post_selection_data/" + training + "_" + correction_method + "_adjp_300_tumor_train_" + str(fold_num) + ".csv")
        test_expr_1.iloc[:, : 300].to_csv("./post_selection_data/" + training + "_" + correction_method + "_adjp_300_tumor_test_" + str(fold_num) + ".csv")

        # targets
        train_target_1.to_csv("./post_selection_data/" + training + "_" + correction_method + "_tumor_train_target_" + str(fold_num) + ".csv")
        test_target_1.to_csv("./post_selection_data/" + training + "_" + correction_method + "_tumor_test_target_" + str(fold_num) + ".csv")

    return
