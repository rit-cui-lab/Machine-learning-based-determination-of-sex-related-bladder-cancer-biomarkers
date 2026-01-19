from preprocessing import *
from gene_selection import *
from eval import *
import matplotlib
matplotlib.use("Agg")  # non-GUI backend
import matplotlib.pyplot as plt
from venn import venn 
from collections import Counter 
import os
import random
from multiprocessing import Pool
import time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



# stop annoying timezone warning
r('options(warn = -1)')

r('rm(list = ls())')

# establish source file as the script housing all the Bioconductor-based functions
r.source("./biomart.R")

# set random seeds in addition to random state throughout the code 
random.seed(3)
np.random.seed(3)

def convert_to_protein(gene_list):

    r_func = ro.globalenv['convert_symbol_2_protein']

    r_list = StrVector(gene_list)

    result_r = r_func(r_list)

    # Convert R dataframes to pandas using localconverter
    with localconverter(ro.default_converter + pandas2ri.converter):
        converted = ro.conversion.rpy2py(result_r)

    print(converted)
    
    return 


# simple function to unpack configuration tuple and input into folded_selection
def cohort_selection(config): 

    correction, training, mode = config

    folded_selection(correction, training, mode)

    return


# process all 5 fold schemes in parallel with joblib 
def folded_selection(correction_method, training, mode): 

    # get counts and TPMS, either stratified by gender or disease 
    counts_1, counts_2, target_1, target_2 = get_stratified_counts(correction_method, mode, training)
    tpm_1, tpm_2, tpm_target_1, tpm_target_2 = get_stratified_tpm(mode, training, correction_method)

    # need control male vs. tumor for disease stratified analysis, using GTEx because TCGA control samples are too few 
    # can execute DGEA outside of nested CV, it will not change based on fold and is not used for machine learning 
    if mode == "disease_stratified": 
        gtex_degs = gtex_dgea()
    else: 
        gtex_degs = None

    # set up 5-fold stratified CV for each cohort 
    folds_1 = list(get_folded_data(counts_1, target_1))
    folds_2 = list(get_folded_data(counts_2, target_2))

    # Process all fold schemes in parallel 
    Parallel(n_jobs=5, verbose=10, backend='loky')(
        delayed(process_fold)(
            fold_num,
            train_1, test_1, train_target_1, test_target_1,
            train_2, test_2, train_target_2, test_target_2,
            tpm_1, tpm_2, tpm_target_1, tpm_target_2,
            mode, training, correction_method,
            gtex_degs
        )
        for fold_num, ((train_1, test_1, train_target_1, test_target_1),
                       (train_2, test_2, train_target_2, test_target_2))
        in enumerate(zip(folds_1, folds_2), start=1)
    )

    return 


# function to perform DGEA, DEG processing, and feature selection for an individual fold scheme 
def process_fold(fold_num, train_1, test_1, train_target_1, test_target_1, train_2, test_2, train_target_2, test_target_2, tpm_1, tpm_2, tpm_target_1, tpm_target_2, mode, training, correction_method, gtex=None): 

    # perform DGEA on male healthy vs. tumor or tumor male vs. female 
    results_1 = run_dgea(train_1, train_target_1, mode, training, correction_method, cohort_int=1)

    # if doing gendered analysis, female info will be in train_2
    if mode == "gender_stratified":
        results_2 = run_dgea(train_2, train_target_2, mode, training, correction_method, cohort_int=2)
    else: 
        # otherwise for tumor use GTEx data as point of comparison, using DGEA results calculated outside CV scheme 
        results_2 = gtex

    # restrict to specific DEGs 
    specific_1, specific_2 = control_degs(mode, results_1, results_2)

    # restrict TPM data to DEGs 
    train_tpm_1 = tpm_1.loc[train_1.index, specific_1]
    test_tpm_1 = tpm_1.loc[test_1.index, specific_1]
    
    train_tpm_2 = tpm_2.loc[train_2.index, specific_2]
    test_tpm_2 = tpm_2.loc[test_2.index, specific_2]
    
    train_tpm_target_1 = tpm_target_1.loc[train_1.index]
    test_tpm_target_1 = tpm_target_1.loc[test_1.index]
    
    train_tpm_target_2 = tpm_target_2.loc[train_2.index]
    test_tpm_target_2 = tpm_target_2.loc[test_2.index]

    # make sure p-value order is maintained 
    assert list(train_tpm_1.columns) == specific_1[:len(train_tpm_1.columns)]

    # run feature selection for this fold scheme 
    # Feature selection and save results
    merged_dataset_gene_selection(
        train_tpm_1, train_tpm_2, test_tpm_1, test_tpm_2,
        train_tpm_target_1, train_tpm_target_2,
        test_tpm_target_1, test_tpm_target_2,
        mode, training, correction_method, fold_num
    )

    return


# function to run folded feature selection for all cohorts in parallel
# n_jobs=1 for all internal classifiers and grid searches though - 8 hour runtime 
def run_feature_selection():

    # define training set, correction method, and variable of stratification for each cohort 
    configs = [
        ('z-score', 'merged', 'gender_stratified'),
        ('', 'TCGA', 'disease_stratified')
    ]

    # process the two configurations in parallel with multiprocessing 
    with Pool(processes=2) as pool: 
        pool.map(cohort_selection, configs)

    return

# function to run cross-fold aggregation of all rankings (4 selection methods * 5 fold schemes)
# plots bar graph for top 20 in terms of RRA score 
# performs pathway enrichment for genes with RRA score < 0.05 
def inspect_cross_fold_ranking(): 

    configs = [("z-score", "merged", "male"), ("z-score", "merged", "female"), ("", "TCGA", "tumor")]

    for config in configs: 

        cross_fold_ranking(config[0], config[1], config[2], save=True)
        get_consensus_panels(config[2], config[1], config[0], save=True)
        cross_method_venn(config[2], config[1], config[0])
    
    return


def compare_aggy_rankings(): 

    male = pd.read_csv(".\\evaluation\\merged_z-score_male_aggregated_ranking.csv", index_col=0)['Name'].to_list()[:300]
    female = pd.read_csv(".\\evaluation\\merged_z-score_female_aggregated_ranking.csv", index_col=0)['Name'].to_list()[:300]
    tumor = pd.read_csv(".\\evaluation\\TCGA__tumor_aggregated_ranking.csv", index_col=0)['Name'].to_list()[:300]

    print(male)
    print(female)
    print(tumor)

    male_overlap_tumor = set(male) & set(tumor)
    print("Genes overlapping between male-specific development and sex-related progression:")
    print(male_overlap_tumor)

    for gene in male_overlap_tumor: 
        print("Ranks for", gene)
        print("In male:", male.index(gene) + 1)
        print("In tumor:", tumor.index(gene) + 1)

    female_overlap_tumor = set(female) & set(tumor)
    print("Genes overlapping between female-specific development and sex-related progression:")
    print(female_overlap_tumor)

    for gene in female_overlap_tumor: 
        print("Ranks for", gene)
        print("In female:", female.index(gene) + 1)
        print("In tumor:", tumor.index(gene) + 1)

    return


def run_internal_eval(): 

    internal_evaluation("male", "merged", "z-score")
    internal_evaluation("female", "merged", "z-score")
    internal_evaluation("tumor", "TCGA", "")

    return


def run_external_eval(cross=False): 

    for config in [("male", "merged", "z-score"), ("female", "merged", "z-score"), ("tumor", "TCGA", "")]:

            external_evaluation(config[0], config[1], config[2], 1, dgea=True, cross_check=cross)
            external_evaluation(config[0], config[1], config[2], 2, cross_check=cross)

    return




def run_complete_analysis(): 

    #run_feature_selection()
    #run_internal_eval()
    run_external_eval()

    return

def avg_ex(): 

    df1 = pd.read_csv(".\\evaluation\\external_testing_merged_z-score_female_1.csv", index_col=0)
    df2 = pd.read_csv(".\\evaluation\\external_testing_merged_z-score_female_2.csv", index_col=0)

    df_avg = pd.concat([df1, df2]).groupby(level=0).mean().sort_values(by='Composite Score', ascending=False)
    df_avg.to_csv(".\\evaluation\\external_testing_merged_z-score_female_avg.csv")

    df1 = pd.read_csv(".\\evaluation\\external_testing_merged_z-score_male_1.csv", index_col=0)
    df2 = pd.read_csv(".\\evaluation\\external_testing_merged_z-score_male_2.csv", index_col=0)

    df_avg = pd.concat([df1, df2]).groupby(level=0).mean().sort_values(by='Composite Score', ascending=False)
    df_avg.to_csv(".\\evaluation\\external_testing_merged_z-score_male_avg.csv")

    df1 = pd.read_csv(".\\evaluation\\external_testing_TCGA__tumor_1.csv", index_col=0)
    df2 = pd.read_csv(".\\evaluation\\external_testing_TCGA__tumor_2.csv", index_col=0)

    df_avg = pd.concat([df1, df2]).groupby(level=0).mean().sort_values(by='Composite Score', ascending=False)
    df_avg.to_csv(".\\evaluation\\external_testing_TCGA__tumor_avg.csv")

    return

def compare_to_wang(cohort): 

    wang_results = pd.read_excel(".\\wang_results.xlsx", sheet_name=cohort + "_validated_hub_genes")

    wang_genes = wang_results['Gene'].to_list()

    my_genes = pd.read_csv(".\\evaluation\\merged_z-score_" + cohort + "_aggregated_ranking.csv", index_col=0).index.to_list()[:300]

    overlap = set(wang_genes) & set(my_genes)
    print(overlap)
    aggy_rank = [i for i, val in enumerate(my_genes) if val in overlap]
    print(aggy_rank)  

    return

def dgea_stats(): 

    male = pd.read_csv(".\\dgea_results\\merged_z-score_male_dgea.csv", index_col=0)

    female = pd.read_csv(".\\dgea_results\\merged_z-score_female_dgea.csv", index_col=0)

    tumor = pd.read_csv(".\\dgea_results\\TCGA__tumor_dgea.csv", index_col=0)

    healthy = gtex_dgea()


    male_specific, female_specific = control_degs("gender_stratified", male, female)
    tumor_specific, healthy_specific = control_degs("disease_stratified", tumor, healthy)

    male_specific_results = male[male.index.isin(male_specific)]
    print("The number of male-specific upregulated is:", len(male_specific_results[male_specific_results['log2FoldChange'] > 0]))
    print("The number of male-specific downregulated is:", len(male_specific_results[male_specific_results['log2FoldChange'] < 0]))

    female_specific_results = female[female.index.isin(female_specific)]
    print("The number of female-specific upregulated is:", len(female_specific_results[female_specific_results['log2FoldChange'] > 0]))
    print("The number of female-specific downregulated is:", len(female_specific_results[female_specific_results['log2FoldChange'] < 0]))
    
    tumor_specific_results = tumor[tumor.index.isin(tumor_specific)]
    print("The number of tumor-specific upregulated is:", len(tumor_specific_results[tumor_specific_results['log2FoldChange'] > 0]))
    print("The number of tumor-specific downregulated is:", len(tumor_specific_results[tumor_specific_results['log2FoldChange'] < 0]))
    return

if __name__ == '__main__':

    # run the entire pipeline
    run_complete_analysis()





    









