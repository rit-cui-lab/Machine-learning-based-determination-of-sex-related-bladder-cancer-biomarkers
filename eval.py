from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score, roc_curve
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, GridSearchCV
import pandas as pd
import os
from gene_selection import composite_score
from xgboost import XGBClassifier
from preprocessing import robust_rank_agg, find_pathways, get_external_for_merged
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
import matplotlib 
matplotlib.use("Agg")  # non-GUI backend
import matplotlib.pyplot as plt 
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from joblib import Parallel, delayed 
from tqdm import tqdm
from preprocessing import get_stratified_tpm, robust_rank_agg, run_dgea, get_stratified_counts
import random
from venn import venn
from rpy2.robjects import r

# manually set cores if need be 
#os.environ["LOKY_MAX_CPU_COUNT"] = "16"

# establish source file as the script housing all the Bioconductor-based functions
r.source("./biomart.R")

# set random seeds in addition to random state throughout the code 
random.seed(3)
np.random.seed(3)


def optimize_hyperparameters(model, X_train, Y_train):
    param_grid = dict()
    if isinstance(model, RandomForestClassifier):
        param_grid['max_depth'] = [None, 5, 10, 20]
        param_grid['criterion'] = ["gini", "entropy", "log_loss"]
        param_grid['n_estimators'] = [100, 300, 500]
        param_grid['min_samples_split'] = [2, 5, 10]
        param_grid['max_features'] = ["sqrt", "log2", None]
        param_grid['min_samples_leaf'] = [1, 2, 4]
    elif isinstance(model, Pipeline):
        if any(isinstance(step, SVC) for step in model.named_steps.values()):
            param_grid = [
    # General-purpose kernels (gamma relevant for RBF, sigmoid)
    {'svc__C': [0.01, .1, 1, 10, 100, 1000],
     'svc__kernel': ['linear', 'sigmoid'],
     'svc__gamma': ['scale', 'auto']},
    
    # RBF-specific tuning (often the best choice)
    {'svc__C': [0.01, .1, 1, 10, 100, 1000],
     'svc__kernel': ['rbf'],
     'svc__gamma': [0.001, 0.01, 0.1, 1, 'scale', 'auto']},
     
    # Polynomial-specific tuning (C, gamma, and degree)
    {'svc__C': [0.01, .1, 1, 10, 100, 1000],
     'svc__kernel': ['poly'],
     'svc__gamma': ['scale', 'auto'],
     'svc__degree': [2, 3, 4]}
                ]
        elif any(isinstance(step, KNeighborsClassifier) for step in model.named_steps.values()):
            param_grid['kneighborsclassifier__n_neighbors'] = [3, 5, 7]
            param_grid['kneighborsclassifier__p'] = [1, 2]
            param_grid['kneighborsclassifier__weights'] = ['uniform', 'distance']
        else: 
            raise Exception("Unexpected pipeline!")
    elif isinstance(model, XGBClassifier):
        # bayesian optimization does better with a large array of 
        param_grid = {
            'n_estimators': Integer(100, 500),
            'max_depth': Integer(3, 7),
            'learning_rate': Real(0.01, 0.2, prior='log-uniform'),
            'subsample': Real(0.8, 1.0),
            'colsample_bytree': Real(0.8, 1.0),
            'gamma': Real(0, 1),
            'reg_alpha': Real(0, 1),
            'reg_lambda': Real(0.1, 10, prior='log-uniform'),
            'scale_pos_weight': Integer(1, 5)
        }
    else:
        raise Exception("Choose a valid machine learning model please.")

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)

    if isinstance(model, Pipeline):
        hyper_search = GridSearchCV(model, param_grid, n_jobs=1, cv=kfold, verbose=0, error_score='raise', scoring=composite_score)
    elif isinstance(model, RandomForestClassifier):
        hyper_search = RandomizedSearchCV(model, param_grid, n_iter=200, random_state=3, n_jobs=1, cv=kfold, verbose=0,
                                      error_score='raise', scoring=composite_score)
    elif isinstance(model, XGBClassifier): 
        hyper_search = BayesSearchCV(model, param_grid, n_iter=50, scoring=composite_score, cv=kfold, n_jobs=1, verbose=0, random_state=3)

    hyper_search.fit(X_train, Y_train)
    best_grid = hyper_search.best_params_
    model.set_params(**best_grid)

    return model


def get_internal_data(cohort, training, correction, fold_num): 

    logit_train = pd.read_csv('./post_selection_data//' + training + "_" + correction + '_logit_cv_300_' + cohort + '_train_' + str(fold_num) + '.csv', index_col=0)
    logit_test = pd.read_csv('./post_selection_data//' + training + "_" + correction + '_logit_cv_300_' + cohort + '_test_' + str(fold_num) + '.csv', index_col=0)
    rf_train = pd.read_csv('./post_selection_data//' + training + "_" + correction + '_opt_rf_300_' + cohort + '_train_' + str(fold_num) + '.csv', index_col=0)
    rf_test = pd.read_csv('./post_selection_data//' + training + "_" + correction + '_opt_rf_300_' + cohort + '_test_' + str(fold_num) + '.csv', index_col=0)
    svm_train = pd.read_csv('./post_selection_data//' + training + "_" + correction + '_svm_rfe_300_' + cohort + '_train_' + str(fold_num) + '.csv', index_col=0)
    svm_test = pd.read_csv('./post_selection_data//' + training + "_" + correction + '_svm_rfe_300_' + cohort + '_test_' + str(fold_num) + '.csv', index_col=0)
    dge_train = pd.read_csv('./post_selection_data//' + training + "_" + correction + '_adjp_300_' + cohort + '_train_' + str(fold_num) + '.csv', index_col=0)
    dge_test = pd.read_csv('./post_selection_data//' + training + "_" + correction + '_adjp_300_' + cohort + '_test_' + str(fold_num) + '.csv', index_col=0)
    train_target = pd.read_csv('./post_selection_data//' + training + "_" + correction + '_' + cohort + '_train_target_' + str(fold_num) + '.csv', index_col=0).iloc[:, 0]
    test_target = pd.read_csv('./post_selection_data//' + training + "_" + correction + '_' + cohort + '_test_target_' + str(fold_num) + '.csv', index_col=0).iloc[:, 0]
    
    for train_test in [["logit", logit_train, logit_test], ["opt_rf", rf_train, rf_test], ["svm_rfe", svm_train, svm_test], ["dgea_adjp", dge_train, dge_test]]: 

        train_test.append(train_target)
        train_test.append(test_target)

        yield train_test

    return


def internal_evaluation(cohort, training, correction):

    tasks = []

    for fold in [1, 2, 3, 4, 5]: 

        for train_test_target in get_internal_data(cohort, training, correction, fold): 

            panel_name = train_test_target[0]

            X_train = train_test_target[1]
            X_test = train_test_target[2]
            y_train = train_test_target[3]
            y_test = train_test_target[4]

            for top_n in [300, 200, 100, 50, 25, 10]: 

                
                panel_name_n = panel_name + "_" + str(top_n)

                X_train_sub = X_train.iloc[:, :top_n]
                X_test_sub = X_test.iloc[:, :top_n]

                tasks.append((panel_name_n, X_train_sub, X_test_sub, y_train, y_test))
                
    # Execute evaluation in parallel and unpack results
    result = Parallel(n_jobs=-1, verbose=1)(delayed(run_eval)(*args) for args in tasks)
    all_results, all_roc_curves = zip(*result)

    mega_df = pd.concat(all_results, ignore_index=True)
    mega_df = mega_df.groupby('name').mean().sort_values(by='Composite Score', ascending=False)
    mega_df.to_csv(".\\evaluation\\internal_testing_" + training + "_" + correction + "_" + cohort + ".csv")

    top_curves = [c for group in all_roc_curves for c in group]

    # get the best panel for each method 
    best_by_method = {}
    for curve in top_curves:
        name = curve[0]
        auroc = curve[1]
        
        # break up name so you can interpret method, top_n, and evaluator separately
        parts = name.split('_')
        # Find where the number is (top_n) and take everything before it
        for i, part in enumerate(parts):
            if part.isdigit():
                selection_method = '_'.join(parts[:i])
                break
        
        # Keep only if this is the best AUROC for this method
        if selection_method not in best_by_method or auroc > best_by_method[selection_method][1]:
            best_by_method[selection_method] = curve  # curve still has full name with top_n
    
    # Get the specific methods you want (just the 3 best)
    top_curves = [
        best_by_method.get('svm_rfe'),
        best_by_method.get('opt_rf'), 
        best_by_method.get('logit'), 
        best_by_method.get("dgea_adjp")
    ]

    # pass to plotting function
    plot_best_rocs(top_curves, training, correction, cohort, 0)


    return


def run_eval(name, X_train, X_test, y_train, y_test): 

    accs, f1s, aurocs, comps, roc_curves = get_metrics(X_train, X_test, y_train, y_test)

    # store performance metrics in a dataframe
    result_dict = {}
    result_dict['name'] = [name + "_" + "rf", name + "_" + "svm", name + "_" + "knn", name + "_" + "xgb"]
    result_dict['Accuracy'] = accs
    result_dict['F1 Score'] = f1s
    result_dict['AUROC'] = aurocs
    result_dict['Composite Score'] = comps

    indiv_result = pd.DataFrame(result_dict)

    # put roc_curves in a nested list where each entry is: 
    # [name, auroc, (fpr, tpr, threshold)]

    roc_curves_labeled = [[name + "_" + "rf", aurocs[0], roc_curves[0]], [name + "_" + "svm", aurocs[1], roc_curves[1]], [name + "_" + "xgb", aurocs[3], roc_curves[2]]]

    return indiv_result, roc_curves_labeled
            

def get_metrics(X_train, X_test, y_train, y_test): 

    # Instantiate a random forest classifier
    rf = RandomForestClassifier(n_jobs=1, random_state=3)
    rf = optimize_hyperparameters(rf, X_train, y_train)
    # fit to training data
    rf.fit(X_train, y_train)
    # predict the class labels of the test set expression values, as well as the specific probabilities
    y_pred = rf.predict(X_test)
    y_pred_p = rf.predict_proba(X_test)[:, 1]
    # Determine the accuracy by comparing actual test labels to predictions
    rf_acc = balanced_accuracy_score(y_test, y_pred)
    # Do the same but for F1 score
    rf_f1 = f1_score(y_test, y_pred)
    # Use the predicted class probabilities to determine the AUROC curve score
    rf_auroc = roc_auc_score(y_test, y_pred_p)
    # get score for composite metric
    rf_comp_score = 0.5*rf_f1 + 0.3*rf_auroc + 0.2*rf_acc
    # get info for roc curve 
    rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, y_pred_p)
    # if shap

    

    # Instantiate the model
    svm = make_pipeline(StandardScaler(), SVC(probability=True, random_state=3))
    svm = optimize_hyperparameters(svm, X_train, y_train)
    # Fit the model to the training data
    svm.fit(X_train, y_train)
    # Use the fitted SVM to predict the label of the test training set as well as the class probability
    y_pred = svm.predict(X_test)
    y_pred_p = svm.predict_proba(X_test)[:, 1]
    # Compare actuality vs. predictions to get accuracy and F1
    svm_acc = balanced_accuracy_score(y_test, y_pred)
    svm_f1 = f1_score(y_test, y_pred)
    # Compare labels to probabilities to determine AUROC curve score
    svm_auroc = roc_auc_score(y_test, y_pred_p)
    # get composite metric
    svm_comp_score = 0.5*svm_f1 + 0.3*svm_auroc + 0.2*svm_acc
    # get info for roc curve 
    svm_fpr, svm_tpr, svm_thresholds = roc_curve(y_test, y_pred_p)

    # create the model
    knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_jobs=1))
    knn = optimize_hyperparameters(knn, X_train, y_train)
    # fit to training data
    knn.fit(X_train, y_train)
    # generate predictions and class probability
    y_pred = knn.predict(X_test)
    y_pred_p = knn.predict_proba(X_test)[:, 1]
    # attain performance metrics
    knn_acc = balanced_accuracy_score(y_test, y_pred)
    knn_f1 = f1_score(y_test, y_pred)
    knn_auroc = roc_auc_score(y_test, y_pred_p)
    # get composite metric
    knn_comp_score = 0.5*knn_f1 + 0.3*knn_auroc + 0.2*knn_acc
    # no AUROC plot for KNN, predicted probabilities are based on number of neighbors - too discrete for low
    # neighbor count

    # XGB
    xgb = XGBClassifier(random_state=3, objective="binary:logistic", eval_metric='logloss')
    xgb = optimize_hyperparameters(xgb, X_train, y_train)
    weights = compute_sample_weight(class_weight='balanced', y=y_train)
    xgb.fit(X_train, y_train, sample_weight=weights)
    y_pred = xgb.predict(X_test)
    y_pred_p = xgb.predict_proba(X_test)[:, 1]
    # attain performance metrics
    xgb_acc = balanced_accuracy_score(y_test, y_pred)
    xgb_f1 = f1_score(y_test, y_pred)
    xgb_auroc = roc_auc_score(y_test, y_pred_p)
    # get composite metric
    xgb_comp_score = 0.5*xgb_f1 + 0.3*xgb_auroc + 0.2*xgb_acc
    # get info for roc curve 
    xgb_fpr, xgb_tpr, xgb_thresholds = roc_curve(y_test, y_pred_p)

    
    # get all metrics 
    accs = [rf_acc, svm_acc, knn_acc, xgb_acc]
    f1s = [rf_f1, svm_f1, knn_f1, xgb_f1]
    aurocs = [rf_auroc, svm_auroc, knn_auroc, xgb_auroc]
    comp_scores = [rf_comp_score, svm_comp_score, knn_comp_score, xgb_comp_score]
    roc_curves = [(rf_fpr, rf_tpr, rf_thresholds), (svm_fpr, svm_tpr, svm_thresholds), (xgb_fpr, xgb_tpr, xgb_thresholds)]
    
    return accs, f1s, aurocs, comp_scores, roc_curves


def get_consensus_panels(cohort, training, correction, save=False, cross_check=False): 

    # hard code the panels you want 

    consensus_panels = {}

    # top 100,200,300 for SVM-RFE and opt RF 

    methods = ["svm_rfe", "opt_rf", "logit_cv", "adjp"]
    top_ns = [300, 200, 100, 50]
    
    # for each method
    for method in methods: 
        # and top N using the selection method
        for n in top_ns: 

            cross_fold = []
            # iterate through all folds and append gene ranking to nested list 
            for fold in [1, 2, 3, 4, 5]:

                panel = pd.read_csv(".\\post_selection_data\\" + training + "_" + correction + "_" + method + "_300_" + cohort + "_train_" + str(fold) + ".csv", index_col=0).columns.to_list()[:n]
                cross_fold.append(panel)
            
            # pass nested list to robust rank aggregation function
            cross_fold_consensus = robust_rank_agg(cross_fold)
            # output is the consensus panel for a given panel across all folds 
            # ex) combine 5 rankings (for each fold scheme) for svm_rfe_top_300 --> 1 svm_rfe_top_300 panel

            # name panel and add to dictionary
            name = method + "_" + str(n)
            # if you are cross-checking, add cohort name to the string 
            if cross_check: 
                name = name + "_" + cohort 

            # save top 300 panel if argument is true
            if save and n ==  300: 
                cross_fold_consensus.to_csv(".\\evaluation\\" + training + "_" + correction + "_" + cohort + "_" + name + ".csv")
            
            consensus_panels[name] = cross_fold_consensus["Name"].to_list()[:n]

    # get aggregate panel, aggregate all 20 rankings (5 fold schemes * 4 selection methods)
    aggy_panel = cross_fold_ranking(correction, training, cohort)
    
    # subset aggregate panel
    for top_n in [300, 200, 100, 50, 25, 10]: 

        # add name and gene panel to dictionary
        name = "aggregate_" + str(top_n)

        # if cross-checking, add cohort to name 
        if cross_check: 
            name = name + "_" + cohort 
        top_n_aggy = aggy_panel[:top_n]
        consensus_panels[name] = top_n_aggy
    
    # export dictionary
    return consensus_panels


def cross_method_venn(cohort, training, correction): 

    # get all 4 consensus top 300s for the given cohort 

    rf = pd.read_csv(".\\evaluation\\" + training + "_" + correction + "_" + cohort + "_" + "opt_rf_300" + ".csv", index_col=0)['Name'].to_list()[:300]

    svm_rfe = pd.read_csv(".\\evaluation\\" + training + "_" + correction + "_" + cohort + "_" + "svm_rfe_300" + ".csv", index_col=0)['Name'].to_list()[:300]

    adjp = pd.read_csv(".\\evaluation\\" + training + "_" + correction + "_" + cohort + "_" + "adjp_300" + ".csv", index_col=0)['Name'].to_list()[:300]

    logit = pd.read_csv(".\\evaluation\\" + training + "_" + correction + "_" + cohort + "_" + "logit_cv_300" + ".csv", index_col=0)['Name'].to_list()[:300]

    venn_dict = {"Optimized RF": set(rf), "SVM-RFE": set(svm_rfe), "DGEA adj.p": set(adjp), "Logit": set(logit)}

    venn(venn_dict)

    if cohort == "male": 
        title = "Male"
    elif cohort == "female": 
        title = "Female"
    elif cohort == "tumor": 
        title = "Tumor"

    plt.title(title)
    plt.savefig(".\\figures\\" + cohort + "_venn.png", dpi=300)

    return


def cross_fold_ranking(correction_method, training, cohort, save=False): 

    all_rankings = []


    for fold_num in [1, 2, 3, 4, 5]:

        # DGEA adj.p
        adjp_ranks = pd.read_csv(".\\post_selection_data\\" + training + "_" + correction_method + "_adjp_300_" + cohort + "_train_" + str(fold_num) + ".csv", index_col=0).columns.to_list()

        # optimized RF
        rf_ranks = pd.read_csv(".\\post_selection_data\\" + training + "_" + correction_method + "_opt_rf_300_" + cohort + "_train_" + str(fold_num) + ".csv", index_col=0).columns.to_list()

        # svm-rfe
        svm_ranks = pd.read_csv(".\\post_selection_data\\" + training + "_" + correction_method + "_svm_rfe_300_" + cohort + "_train_" + str(fold_num) + ".csv", index_col=0).columns.to_list()

        # logit 
        logit_ranks = pd.read_csv(".\\post_selection_data\\" + training + "_" + correction_method + "_logit_cv_300_" + cohort + "_train_" + str(fold_num) + ".csv", index_col=0).columns.to_list()

        # put in running nested list 
        all_rankings.extend([adjp_ranks, rf_ranks, svm_ranks, logit_ranks])
        
    # combine all 20 rankings for the cohort 
    aggregated = robust_rank_agg(all_rankings)
    if save: 
        aggregated.to_csv(".\\evaluation\\" + training + "_" + correction_method + "_" + cohort + "_aggregated_ranking.csv")
    aggregated_panel = aggregated["Name"].to_list()
    
    # graph RRA scores for given cohort 
    graph_rra(aggregated, "cross_fold_cross_method_" + cohort)
    
    # determine GO/KEGG pathway enrichments for given cohort 
    find_significant_enrichments(aggregated_panel, cohort, top_n=len(aggregated_panel))

    # export aggregated panel top 300
    return aggregated_panel[:300]


def inspect_enrichments(training, correction, cohort):
    
    aggy = pd.read_csv(".\\evaluation\\" + training + "_" + correction + "_" + cohort + "_aggregated_ranking.csv", index_col=0)["Name"].to_list()
    find_significant_enrichments(aggy, cohort, top_n=500)

    return 


def find_significant_enrichments(aggregated_panel, cohort, top_n=50):

    name = "cross_fold_cross_method_" + cohort

    aggregated_panel = aggregated_panel[:top_n]

    try:
        all_kegg, all_go = find_pathways(aggregated_panel, name)
        print(all_kegg.head(50))
        print(all_go.head(50))
    except KeyError:
        print("No significant enrichments")
    
    return 


def graph_rra(ranking, cohort): 

    # get top 20 genes in aggregate ranking
    top_agg = ranking.head(20).copy()
    # take -log10 of RRA score --> smaller, more significant values have larger bar in graph
    top_agg['neg_log10'] = -np.log10(top_agg['Score'])

    #bar graph of aggregated ranking
    plt.figure(figsize=(8,6))
    plt.barh(top_agg['Name'], top_agg['neg_log10'], color='teal')
    plt.gca().invert_yaxis()  # top gene on top

    # Labels and title
    plt.xlabel("-log10(RRA Score)")
    plt.ylabel("Gene")
    if cohort == "cross_fold_cross_method_male": 
        title = "Male Tumor vs. Non-Tumor"
    elif cohort == "cross_fold_cross_method_female": 
        title = "Female Tumor vs. Non-Tumor"
    else: 
        title = "Male vs. Female Tumor"
    plt.title(title)

    plt.tight_layout()
    plt.savefig(".\\figures\\" + cohort + "_rra.png", dpi=300)
    plt.close()

    return


def external_evaluation(cohort, training, correction, ex_num, dgea=False, cross_check=False): 

    # load full training dataset depending on cohort - no longer folded 
    X_train, y_train = get_full_training(cohort, training, correction, dgea=dgea)

    # for the given cohort, get the gene panels as a dictionary
    # key = name
    # value = list of genes 
    panel_dict = get_consensus_panels(cohort, training, correction)
    # cross-checking: test both gendered panels on the given cohort 
    if cross_check: 

        if cohort == "male": 
            other_cohort = "female"
        else: 
            other_cohort= "male"

        other_panel_dict = get_consensus_panels(other_cohort, training, correction, cross_check=cross_check)

        # ensure there are no overlapping keys 
        if set(panel_dict) & set(other_panel_dict):
            raise ValueError("Dictionaries have overlapping keys!")
    
        panel_dict = {**panel_dict, **other_panel_dict}

    # get external expressions and external target based on 
    external_expr, external_target = get_stratified_external(ex_num, correction, cohort)

    tasks = []

    for name, panel in panel_dict.items(): 

        # limit X_train to the panel 
        X_train_selected = X_train[panel]

        # only consider genes in the panel that also appear in the external test set
        X_train_selected, X_test_selected = compare_train_to_ex(X_train_selected, external_expr)

        # append parameters for a given panel to nested list 
        tasks.append((name, X_train_selected, X_test_selected, y_train, external_target))

    # Execute evaluation of all panels in parallel, extract results
    result = Parallel(n_jobs=-1, verbose=1)(delayed(run_eval)(*args) for args in tasks)
    all_results, all_curves = zip(*result)

    # create one table for performance metrics, take average across folds, sort by compsite metric, and save to CSV 
    mega_df = pd.concat(all_results, ignore_index=True)
    mega_df = mega_df.groupby('name').mean().sort_values(by='Composite Score', ascending=False)
    mega_df.to_csv(".\\evaluation\\external_testing_" + training + "_" + correction + "_" + cohort + "_" + str(ex_num) + ".csv")
    
    # flatten then sort resulting nested list of curves by auroc and get the top 5 
    top_curves = [c for group in all_curves for c in group]

    # get the best panel for each method 
    best_by_method = {}
    for curve in top_curves:
        name = curve[0]
        auroc = curve[1]
        
        # break up name so you can interpret method, top_n, and evaluator separately
        parts = name.split('_')
        # Find where the number is (top_n) and take everything before it
        for i, part in enumerate(parts):
            if part.isdigit():
                selection_method = '_'.join(parts[:i])
                break
        
        # Keep only if this is the best AUROC for this method
        if selection_method not in best_by_method or auroc > best_by_method[selection_method][1]:
            best_by_method[selection_method] = curve  # curve still has full name with top_n

    # Get the specific methods you want (just the 3 best)
    top_curves = [
        best_by_method.get('svm_rfe'),
        best_by_method.get('opt_rf'),
        best_by_method.get('aggregate')
    ]

    # pass to plotting function
    plot_best_rocs(top_curves, training, correction, cohort, ex_num)

    return


def plot_best_rocs(top_curves, training, correction, cohort, ex_num): 

    plt.figure(figsize=(6, 6))

    for curve in top_curves:

        name = get_name(curve[0])
        auroc = curve[1]
        fpr = curve[2][0]
        tpr = curve[2][1]
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auroc:.2f})')
    

    plt.plot([0, 1], [0, 1], 'k--', label='Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    if ex_num == 1: 
        title = 'GSE236932 - ' + cohort.capitalize()
    else: 
        title = 'GSE188715 - ' + cohort.capitalize()
    plt.title(title)
    plt.savefig(".\\figures\\" + training + "_" + correction + "_" + cohort + "_" + str(ex_num) + "_top_rocs.png")
    plt.close()

    return 


def get_name(raw_name): 

    parts = raw_name.split('_')
    selector = "_".join(parts[:2]) if parts[0] in ["opt", "svm"] else parts[0]
    top_n = next((p for p in parts if p.isdigit()), None)
    eval_method = parts[-1]

    selector_labels = {
        "opt_rf": "Opt RF",
        "svm_rfe": "SVM-RFE",
        "aggregate": "Aggregate"
    }
    eval_labels = {
        "rf": "RF eval",
        "svm": "SVM eval",
        "knn": "KNN eval",
        "xgb": "XGB eval"
    }

    s = selector_labels.get(selector, selector.upper())
    e = eval_labels.get(eval_method, eval_method.upper())
    return f"{s} (Top {top_n}, {e})" if top_n else f"{s} ({e})"


def get_stratified_external(ex_num, correction, cohort): 

    external = get_external_for_merged(ex_num, correction)

    # stratify external data based on cohort and separate target variable from expressions

    if cohort == "female": 

        external = external[external["Gender"] == 0]
        external_expr = external.drop("Gender", axis=1)
        external_target = external_expr.pop('Category')
    
    elif cohort == "male": 

        external = external[external["Gender"] == 1]
        external_expr = external.drop("Gender", axis=1)
        external_target = external_expr.pop('Category')
    
    else: 

        external = external[external["Category"] == 1]
        external_expr = external.drop("Category", axis=1)
        external_target = external_expr.pop('Gender')

    return external_expr, external_target


def compare_train_to_ex(X_train, ex_test): 

    common_columns = X_train.columns.intersection(ex_test.columns)
    test = ex_test[common_columns]
    train = X_train[common_columns]

    return train, test


def get_full_training(cohort, training, correction, dgea=False): 

    if cohort == "female" or cohort == "male": 
        mode = "gender_stratified"
    else: 
        mode = "disease_stratified"
    
    tpm_1, tpm_2, tpm_target_1, tpm_target_2 = get_stratified_tpm(mode, training, correction)
    counts_1, counts_2, target_1, target_2 = get_stratified_counts(correction, mode, training)
    
    if cohort == "male" or cohort == "tumor": 
        
        X_train = tpm_1
        y_train = tpm_target_1
        counts = counts_1
        labels = target_1
        cohort_int = 1
    
    else: 

        X_train = tpm_2
        y_train = tpm_target_2
        counts = counts_2
        labels = target_2
        cohort_int = 2
    
    if dgea: 
        
        result = run_dgea(counts, labels, mode, training, correction, volcano_int=1, cohort_int=cohort_int)
        result.to_csv(".\\dgea_results\\" + training + "_" + correction + "_" + cohort + "_dgea.csv")
    
    return X_train, y_train



r''' 

def merged_eval_dict_constructor(cohort, num, training, correction, consensus_only=False):
    # get test and train sets
    logit_train = pd.read_csv('./post_selection_data//' + training + "_" + correction + '_logit_cv_300_' + cohort + '_train.csv', index_col=0)
    print(len(logit_train.columns))
    logit_test = pd.read_csv('./post_selection_data//' + training + "_" + correction + '_logit_cv_300_' + cohort + '_test.csv', index_col=0)
    rf_train = pd.read_csv('./post_selection_data//' + training + "_" + correction + '_opt_rf_300_' + cohort + '_train.csv', index_col=0)
    print(len(rf_train.columns))
    rf_test = pd.read_csv('./post_selection_data//' + training + "_" + correction + '_opt_rf_300_' + cohort + '_test.csv', index_col=0)
    svm_train = pd.read_csv('./post_selection_data//' + training + "_" + correction + '_svm_rfe_300_' + cohort + '_train.csv', index_col=0)
    print(len(svm_train.columns))
    svm_test = pd.read_csv('./post_selection_data//' + training + "_" + correction + '_svm_rfe_300_' + cohort + '_test.csv', index_col=0)
    dge_train = pd.read_csv('./post_selection_data//' + training + "_" + correction + '_adjp_300_' + cohort + '_train.csv', index_col=0)
    print(len(dge_train.columns))
    dge_test = pd.read_csv('./post_selection_data//' + training + "_" + correction + '_adjp_300_' + cohort + '_test.csv', index_col=0)
    train_target = pd.read_csv('./expression_data//' + training + "_" + correction + '_' + cohort + '_train_target.csv', index_col=0)
    test_target = pd.read_csv('./expression_data//' + training + "_" + correction + '_' + cohort + '_test_target.csv', index_col=0)

    # get overlap between all gene panels and limit a dataframe to those genes 
    gene_overlap = list(set(logit_train.columns.to_list()) & set(rf_train.columns.to_list()) & set(svm_train.columns.to_list()) & set(dge_train.columns.to_list()))
    overlap_train = rf_train[gene_overlap]
    overlap_test = rf_test[gene_overlap]

    # hand pick genes based on knowledge of evaluation, much smaller panel 
    if consensus_only:
        if cohort == "female": 
            picked = ["FRMD3", "ZNF804A", "MAPK4", "MMP10", "LRR1", "MIDN", "CDK5"]
        elif cohort == "male": 
            picked = ["IL1RAPL1", "LMX1A", "PAPPA", "ANGPTL5", "PRAC1", "IL2", "EYA1", "PCDH11Y", "NOS1", "TOX2"]
        elif cohort == "tumor": 
            picked = ["SRY", "RPS4Y2", "NLGN4Y-AS1", "PROK2", "OR10AD1", "CCDC33", "HSPA6", "CLC"]
        else: 
            raise Exception("Input a valid cohort...")
        picked_train = rf_train[picked]
        picked_test = rf_test[picked]
        int_picked_dict = divvy_subsets([len(picked_train.columns)], picked_train, picked_test)

    # get dictionary containing dataframes of top n genes for each feature subset
    # top 100, 50, 25, and 10 for DGEA and best imputer
    # All overlapped, then top 25 and 10.
    int_logit_dict = divvy_subsets([300, 250, 200, 150, 100, 75, 50, 25, 10], logit_train, logit_test)
    int_rf_dict = divvy_subsets([300, 250, 200, 150, 100, 75, 50, 25, 10], rf_train, rf_test)
    int_svm_dict = divvy_subsets([300, 250, 200, 150, 100, 75, 50, 25, 10], svm_train, svm_test)
    int_dge_dict = divvy_subsets([300, 250, 200, 150, 100, 75, 50, 25, 10], dge_train, dge_test)
    int_overlap_dict = divvy_subsets([len(overlap_train.columns)], overlap_train, overlap_test)
    


   # get external validation and subset based on gender
    external_data = get_external_for_merged(num, correction)
    if cohort == 'male':
        external_expr = external_data[external_data['Gender'] ==  1]
        external_expr.drop('Gender', axis=1, inplace=True)
        external_target = external_expr.pop('Category')
    elif cohort == 'female':
        external_expr = external_data[external_data['Gender'] == 0]
        external_expr.drop('Gender', axis=1, inplace=True)
        external_target = external_expr.pop('Category')
    elif cohort == 'tumor':
        external_expr = external_data[external_data['Category'] == 1]
        external_expr.drop('Category', axis=1, inplace=True)
        external_target = external_expr.pop('Gender')
    elif cohort == 'healthy':
        external_expr = external_data[external_data['Category'] == 0]
        external_expr.drop('Category', axis=1, inplace=True)
        external_target = external_expr.pop('Gender')
    else:
        raise Exception("Valid cohort not inputted...")


    # limit gene panels based on genes present in the external validation set
    logit_train, logit_test = compare_to_ex(logit_train, logit_test, external_expr, external_target, num, cohort)

    if num == 3 and cohort == 'tumor':
        rf_train, rf_test, rf_train_target, rf_test_target = compare_to_ex(rf_train, rf_test, external_expr, external_target, num, cohort, get_targets=True)
    else: 
        rf_train, rf_test = compare_to_ex(rf_train, rf_test, external_expr, external_target, num, cohort)
    
    dge_train, dge_test = compare_to_ex(dge_train, dge_test, external_expr, external_target, num, cohort)
  
    svm_train, svm_test = compare_to_ex(svm_train, svm_test, external_expr, external_target, num, cohort)
   
    overlap_train, overlap_test = compare_to_ex(overlap_train, overlap_test, external_expr, external_target, num, cohort)

    if consensus_only:
        picked_train, picked_test = compare_to_ex(picked_train, picked_test, external_expr, external_target, num, cohort)
        ex_picked_dict = divvy_subsets([len(picked_train.columns)], picked_train, picked_test)

    # get dictionary containing dataframes of top n genes for each feature subset
    # top 100, 50, 25, and 10 for DGEA and best imputer
    # All overlapped, then top 25 and 10.
    ex_logit_dict = divvy_subsets([300, 250, 200, 150, 100, 75, 50, 25, 10], logit_train, logit_test)
    ex_rf_dict = divvy_subsets([300, 250, 200, 150, 100, 75, 50, 25, 10], rf_train, rf_test)
    ex_svm_dict = divvy_subsets([300, 250, 200, 150, 100, 75, 50, 25, 10], svm_train, svm_test)
    ex_dge_dict = divvy_subsets([300, 250, 200, 150, 100, 75, 50, 25, 10], dge_train, dge_test)
    ex_overlap_dict = divvy_subsets([len(overlap_train.columns)], overlap_train, overlap_test)

    if consensus_only: 

        if num == 3 and cohort == 'tumor':

            return int_overlap_dict, int_picked_dict, ex_overlap_dict, ex_picked_dict, train_target, test_target, pd.DataFrame(rf_train_target), pd.DataFrame(rf_test_target)

        else:

            return int_overlap_dict, int_picked_dict, ex_overlap_dict, ex_picked_dict, train_target, test_target, external_target

    else: 

        if num == 3 and cohort == 'tumor':

            return int_logit_dict, int_rf_dict, int_dge_dict, int_svm_dict, int_overlap_dict, ex_logit_dict, ex_rf_dict, ex_svm_dict, ex_dge_dict, ex_overlap_dict, train_target, test_target, pd.DataFrame(rf_train_target), pd.DataFrame(rf_test_target)

        else:

            return int_logit_dict, int_rf_dict, int_dge_dict, int_svm_dict, int_overlap_dict, ex_logit_dict, ex_rf_dict, ex_svm_dict, ex_dge_dict, ex_overlap_dict, train_target, test_target, external_target

def compare_to_ex(train, test, external, target, ex_num, cohort, get_targets=False):

    common_columns = train.columns.intersection(external.columns)
    test = external[common_columns]
    train = train[common_columns]
    if ex_num == 3 and cohort == "tumor":
        train, test, train_target, test_target = train_test_split(test, target,
                                                                                    test_size=0.3,
                                                                                    random_state=3,
                                                                                    stratify=target)
    assert list(test.columns) == list(train.columns)

    if get_targets: 
        return train, test, train_target, test_target
    else: 
        return train, test
    

def plot_roc_comparison(nested_rocs, model_type, cohort, ex_num, training, correction_method, concise=False):
   
    plt.figure(figsize=(8, 8))
    
    # Iterate over the ROC data for each feature subset
    for roc_data in nested_rocs:
        row_str, auroc, fpr, tpr = roc_data
        
        # Extract the label, which describes the feature selection method and gene count
        # e.g., 'int_opt_rf_300'
        label = row_str.replace('int_', 'Internal ').replace('ex_', 'External ').replace('_', ' ').strip()
        
        plt.plot(fpr, tpr, 
                 label=f'{label} (AUC = {auroc:.3f})', 
                 lw=2)

    # Plot the 'No Skill' line
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='No Skill (AUC = 0.5)') 

    # Customize the plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    
    # Put legend outside the plot
    plt.legend(loc='lower right', fontsize='small')
    plt.grid(True, linestyle='--')
    
    # Save the plot
    if concise:
        filename = f".//roc_auc//{training}_{correction_method}_{cohort}_{model_type.lower().replace(' ', '_')}_ROC_Top_{ex_num}_concise.png"
    else:
        filename = f".//roc_auc//{training}_{correction_method}_{cohort}_{model_type.lower().replace(' ', '_')}_ROC_Top_{ex_num}.png"
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved ROC plot for {model_type} to: {filename}")

    return




def merged_eval(cohort, ex_num, training, correction_method, consensus_only=False):

    # create list of dictionaries to store performance metrics for each feature subset
    val_table = [dict(), dict(), dict(), dict()]

    # get subsetted data for all combinations of selection method and test set
    if consensus_only: 

        if cohort == "tumor" and ex_num == 3:
            int_overlap_dict, int_picked_dict, ex_overlap_dict, ex_picked_dict, train_target, test_target, ex_train_target, ex_test_target = merged_eval_dict_constructor(cohort, ex_num, training, correction_method, consensus_only=True)

        else:
            int_overlap_dict, int_picked_dict, ex_overlap_dict, ex_picked_dict, train_target, test_target, external_target = merged_eval_dict_constructor(cohort, ex_num, training, correction_method, consensus_only=True)
        
        if ex_num == 0:

            
            row_label_overlap = "int_overlap_"
            row_label_picked = "int_picked_"

            val_table, overlap_rf_roc, overlap_svm_roc = merged_top_n_eval(int_overlap_dict, test_target, train_target, val_table, row_label_overlap, cohort, ex_num, training=training)
            val_table, picked_rf_roc, picked_svm_roc = merged_top_n_eval(int_picked_dict, test_target, train_target, val_table, row_label_picked, cohort, ex_num, training=training)

        
        row_label_overlap = "ex_overlap_"
        row_label_picked = "ex_picked_"

        if cohort == "tumor" and ex_num == 3:

            val_table, overlap_rf_roc, overlap_svm_roc = merged_top_n_eval(ex_overlap_dict, ex_test_target, ex_train_target, val_table,
                                                                row_label_overlap, cohort, ex_num, training=training)
            val_table, picked_rf_roc, picked_svm_roc = merged_top_n_eval(ex_picked_dict, ex_test_target, ex_train_target, val_table,
                                                                row_label_picked, cohort, ex_num, training=training)
            

        elif ex_num == 1 or ex_num == 2:

            val_table, overlap_rf_roc, overlap_svm_roc = merged_top_n_eval(ex_overlap_dict, external_target, train_target, val_table, row_label_overlap, cohort, ex_num, training=training)
            val_table, picked_rf_roc, picked_svm_roc = merged_top_n_eval(ex_picked_dict, external_target, train_target, val_table, row_label_picked, cohort, ex_num, training=training)

        # combine nested lists containing TPR and FPRs for SVM and RF evaluation models for all single selection method panels 
        nested_rocs_rf = overlap_rf_roc + picked_rf_roc
        nested_rocs_svm = overlap_svm_roc + picked_svm_roc

        # Random Forest: sort by AUROC
        final_rocs_rf = sorted(nested_rocs_rf, key=lambda x: float(x[1]), reverse=True)

        # SVM: Sort by AUROC
        final_rocs_svm = sorted(nested_rocs_svm, key=lambda x: float(x[1]), reverse=True)


    else:

        if cohort == "tumor" and ex_num == 3:
            int_logit_dict, int_rf_dict, int_dge_dict, int_svm_dict, int_overlap_dict, ex_logit_dict, ex_rf_dict, ex_svm_dict, ex_dge_dict, ex_overlap_dict, train_target, test_target, ex_train_target, ex_test_target = merged_eval_dict_constructor(cohort, ex_num, training, correction_method)
        else:
            int_logit_dict, int_rf_dict, int_dge_dict, int_svm_dict, int_overlap_dict, ex_logit_dict, ex_rf_dict, ex_svm_dict, ex_dge_dict, ex_overlap_dict, train_target, test_target, external_target = merged_eval_dict_constructor(cohort, ex_num, training, correction_method)

        if ex_num == 0:

            row_label_rf = "int_opt_rf_"
            row_label_dge = "int_dge_"
            row_label_svm = "int_svm_rfe_"
            row_label_logit = "int_logit_cv_"
            row_label_overlap = "int_overlap_"

            val_table, logit_rf_roc, logit_svm_roc = merged_top_n_eval(int_logit_dict, test_target, train_target, val_table, row_label_logit, cohort, ex_num, training=training)
            
            val_table, rf_rf_roc, rf_svm_roc = merged_top_n_eval(int_rf_dict, test_target, train_target, val_table, row_label_rf, cohort, ex_num, training=training)

            val_table, dge_rf_roc, dge_svm_roc = merged_top_n_eval(int_dge_dict, test_target, train_target, val_table, row_label_dge, cohort, ex_num, training=training)

            val_table, svm_rf_roc, svm_svm_roc = merged_top_n_eval(int_svm_dict, test_target, train_target, val_table, row_label_svm, cohort, ex_num, training=training)

            val_table, overlap_rf_roc, overlap_svm_roc = merged_top_n_eval(int_overlap_dict, test_target, train_target, val_table, row_label_overlap, cohort, ex_num, training=training)

        row_label_rf = "ex_opt_rf_"
        row_label_dge = "ex_dge_"
        row_label_svm = "ex_svm_rfe_"
        row_label_logit = "ex_logit_cv_"
        row_label_overlap = "ex_overlap_"

        if cohort == "tumor" and ex_num == 3:

            val_table, logit_rf_roc, logit_svm_roc = merged_top_n_eval(ex_logit_dict, ex_test_target, ex_train_target, val_table, row_label_logit, cohort, ex_num, training=training)
            
            val_table, rf_rf_roc, rf_svm_roc = merged_top_n_eval(ex_rf_dict, ex_test_target, ex_train_target, val_table,
                                                                row_label_rf, cohort, ex_num, training=training)

            val_table, dge_rf_roc, dge_svm_roc = merged_top_n_eval(ex_dge_dict, ex_test_target, ex_train_target, val_table,
                                                                row_label_dge, cohort, ex_num, training=training)

            val_table, svm_rf_roc, svm_svm_roc = merged_top_n_eval(ex_svm_dict, ex_test_target, ex_train_target, val_table,
                                                                row_label_svm, cohort, ex_num, training=training)
            
            val_table, overlap_rf_roc, overlap_svm_roc = merged_top_n_eval(ex_overlap_dict, ex_test_target, ex_train_target, val_table,
                                                                row_label_overlap, cohort, ex_num, training=training)
            

        elif ex_num == 1 or ex_num == 2:

            val_table, logit_rf_roc, logit_svm_roc = merged_top_n_eval(ex_logit_dict, external_target, train_target, val_table, row_label_logit, cohort, ex_num, training=training)

            val_table, rf_rf_roc, rf_svm_roc = merged_top_n_eval(ex_rf_dict, external_target, train_target, val_table, row_label_rf, cohort, ex_num, training=training)

            val_table, dge_rf_roc, dge_svm_roc = merged_top_n_eval(ex_dge_dict, external_target, train_target, val_table, row_label_dge, cohort, ex_num, training=training)

            val_table, svm_rf_roc, svm_svm_roc = merged_top_n_eval(ex_svm_dict, external_target, train_target, val_table, row_label_svm, cohort, ex_num, training=training)

            val_table, overlap_rf_roc, overlap_svm_roc = merged_top_n_eval(ex_overlap_dict, external_target, train_target, val_table, row_label_overlap, cohort, ex_num, training=training)

        # combine nested lists containing TPR and FPRs for SVM and RF evaluation models for all single selection method panels 
        nested_rocs_rf = rf_rf_roc + dge_rf_roc + svm_rf_roc + logit_rf_roc
        nested_rocs_svm = rf_svm_roc + dge_svm_roc + svm_svm_roc + logit_svm_roc

        # Random Forest: Select top 4 non-overlap
        sorted_rocs_rf = sorted(nested_rocs_rf, key=lambda x: float(x[1]), reverse=True)
        final_rocs_rf = sorted_rocs_rf[0:4]
        final_rocs_rf.append(overlap_rf_roc[0])

        # SVM: Select top 4 non-overlap
        sorted_rocs_svm = sorted(nested_rocs_svm, key=lambda x: float(x[1]), reverse=True)
        final_rocs_svm = sorted_rocs_svm[0:4]
        final_rocs_svm.append(overlap_svm_roc[0])

    # plot top 4 curves for each evaluation model + overlapped curve 
    
    plot_roc_comparison(
        final_rocs_rf, 
        'Random Forest', 
        cohort, ex_num, training, correction_method, consensus_only
    )

    plot_roc_comparison(
        final_rocs_svm, 
        'Support Vector Machine', 
        cohort, ex_num, training, correction_method, consensus_only
    )


    performance_dfs = []
    for table in val_table:
            performance_df = pd.DataFrame.from_dict(table, orient='index', columns=['Accuracy', 'F1 Score',
                                                                                            'AUROC', 'Composite']).sort_values(by='Composite', ascending=False)
            performance_dfs.append(performance_df)

    if consensus_only:
        filename = "evaluation//" + training + "_" + correction_method + "_" + cohort + "_validation_" + str(ex_num) + "_concise.csv"
    else: 
        filename = "evaluation//" + training + "_" + correction_method + "_" + cohort + "_validation_" + str(ex_num) + ".csv"

    with open(filename, "w", newline='') as f_stream:
        f_stream.write("Random Forest:\n\n")
        performance_dfs[0].to_csv(f_stream)

        f_stream.write("\n\nSupport Vector Machine:\n\n")
        performance_dfs[1].to_csv(f_stream)

        f_stream.write("\n\nK-Nearest Neighbors:\n\n")
        performance_dfs[2].to_csv(f_stream)

        f_stream.write("\n\nXGBoost:\n\n")
        performance_dfs[3].to_csv(f_stream)
        f_stream.close()

    return





def merged_top_n_eval(dfs, Y_test, Y_train, dicty, tech, cohort, ex_num, training=None):
   
    # extract integer keys of the dfs dictionary, which corresponding how many top features were selected
    keys = dfs.keys()

    Y_train = Y_train.iloc[:,0].to_list()
    try:
        Y_test = Y_test.iloc[:,0].to_list()
    except:
        Y_test = Y_test.to_list()


    rf_roc = []
    svm_roc = []
    
    for i in keys:
        # announce how many top genes we are working with
        print("Analyzing model performance with the top", i, "genes:")
        # create row label based on technique string and top n
        row_str = tech + str(i)

        # for each key, attain the associated training and testing set
        expr_dfs = dfs[i]
        X_train = expr_dfs[0]
        X_test = expr_dfs[1]


        print(ex_num)
        print(cohort)
        print(row_str)

        # Instantiate a random forest classifier
        rf = RandomForestClassifier(n_jobs=1, random_state=3)
        rf = optimize_hyperparameters(rf, X_train, Y_train)
        # fit to training data
        rf.fit(X_train, Y_train)
        # get feature importances
        all_importances = pd.Series(rf.feature_importances_, index=rf.feature_names_in_).sort_values(ascending=False)
        importances = all_importances[:30]
        gene_ranking_rf = all_importances.index.to_list()
        #bar_graph.write_image(bar_direc)
        # print(pd.Series(rf.feature_importances_, index=X_train.columns.to_list()).sort_values(ascending=False))
        # predict the class labels of the test set expression values, as well as the specific probabilities
        y_pred = rf.predict(X_test)
        y_pred_p = rf.predict_proba(X_test)[:, 1]
        # Determine the accuracy by comparing actual test labels to predictions
        acc = balanced_accuracy_score(Y_test, y_pred)
        # Do the same but for F1 score
        f1 = f1_score(Y_test, y_pred)
        # Use the predicted class probabilities to determine the AUROC curve score
        auroc = roc_auc_score(Y_test, y_pred_p)
        fpr, tpr, thresh = roc_curve(Y_test, y_pred_p)
        rf_roc.append([row_str, auroc, fpr, tpr])
        # get score for composite metric
        comp_score = 0.5*f1 + 0.3*auroc + 0.2*acc
        # output results of each metric to the first position of a list of dictionaries
        dicty[0][row_str] = [acc, f1, auroc, comp_score]


        # next, an SVM classifier. The linear kernel (k(x,z) = x^T*z) is chosen since the number
        # of features dwarfs the number of samples in the dataset, so increasing dimensions will not improve
        # seperability.

        # Instantiate the model
        svm = make_pipeline(StandardScaler(), SVC(probability=True, random_state=3, kernel='linear'))
        svm = optimize_hyperparameters(svm, X_train, Y_train)
        # Fit the model to the training data
        svm.fit(X_train, Y_train)
        # get feature importances
        coefs = np.abs(svm.named_steps['svc'].coef_).flatten()
        coef_series = pd.Series(coefs, index=X_train.columns.to_list()).sort_values(ascending=False)
        gene_ranking_svm = coef_series.index.to_list()
        # Use the fitted SVM to predict the label of the test training set as well as the class probability
        y_pred = svm.predict(X_test)
        y_pred_p = svm.predict_proba(X_test)[:, 1]
        # Compare actuality vs. predictions to get accuracy and F1
        acc = balanced_accuracy_score(Y_test, y_pred)
        f1 = f1_score(Y_test, y_pred)
        # Compare labels to probabilities to determine AUROC curve score
        auroc = roc_auc_score(Y_test, y_pred_p)
        fpr, tpr, thresh = roc_curve(Y_test, y_pred_p)
        svm_roc.append([row_str, auroc, fpr, tpr])
        # get composite metric
        comp_score = 0.5*f1 + 0.3*auroc + 0.2*acc
        # store SVM metrics in second position of list of dictionaries
        dicty[1][row_str] = [acc, f1, auroc, comp_score]


        # create the model
        knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_jobs=1))
        knn = optimize_hyperparameters(knn, X_train, Y_train)
        # fit to training data
        knn.fit(X_train, Y_train)
        # generate predictions and class probability
        y_pred = knn.predict(X_test)
        y_pred_p = knn.predict_proba(X_test)[:, 1]
        # attain performance metrics
        acc = balanced_accuracy_score(Y_test, y_pred)
        f1 = f1_score(Y_test, y_pred)
        auroc = roc_auc_score(Y_test, y_pred_p)
        # no AUROC plot for KNN, predicted probabilities are based on number of neighbors - too discrete for low
        # neighbor count
        # get composite metric
        comp_score = 0.5*f1 + 0.3*auroc + 0.2*acc
        # store the results for all metrics
        dicty[2][row_str] = [acc, f1, auroc, comp_score]

        # XGB

        xgb = XGBClassifier(random_state=3, objective="binary:logistic", eval_metric='logloss')
        xgb = optimize_hyperparameters(xgb, X_train, Y_train)
        weights = compute_sample_weight(class_weight='balanced', y=Y_train)
        xgb.fit(X_train, Y_train, sample_weight=weights)
        # get feature importances
        importances = xgb.get_booster().get_score(importance_type='gain')
        importances = pd.Series(importances)
        # BUT XGBoost doesn't use all features, so assign unused ones an importance of 0 
        all_importances = importances.reindex(X_train.columns, fill_value=0).sort_values(ascending=False)
        gene_ranking_xgb = all_importances.index.to_list()
        y_pred = xgb.predict(X_test)
        y_pred_p = xgb.predict_proba(X_test)[:, 1]
        # attain performance metrics
        acc = balanced_accuracy_score(Y_test, y_pred)
        f1 = f1_score(Y_test, y_pred)
        auroc = roc_auc_score(Y_test, y_pred_p)
        # get composite metric
        comp_score = 0.5*f1 + 0.3*auroc + 0.2*acc
        # store the results for all metrics
        dicty[3][row_str] = [acc, f1, auroc, comp_score]

        # if judging the top 100 genes, perform robust rank aggregation between the importances from
        # each evaluation model

        combined_ranking = pd.DataFrame()
        combined_ranking['rf'] = gene_ranking_rf 
        combined_ranking['svm'] = gene_ranking_svm
        combined_ranking['xgb'] = gene_ranking_xgb

        
        combined_ranking.to_csv(r'.\\evaluation_rankings\\' + training + '_' + row_str + 'aggregate_ranking_' + cohort + '_' + str(ex_num) + '.csv', index=False)


    return dicty, rf_roc, svm_roc 



def divvy_subsets(top_ns, train, test):
    # create a dictionary of top n genes
    top_dict = dict()
    # for each specified n, make it a key in the dictionary and limit the training and testing to those genes
    for n in top_ns:
        top_train = train.iloc[:, :n]
        top_dict[n] = [top_train, test[top_train.columns]]
    return top_dict

'''