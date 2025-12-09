import pandas as pd
import numpy as np
import os
import sys
import json
from pathlib import Path
import subprocess
os.environ["R_HOME"] = os.path.join(sys.prefix, "Lib", "R")
import rpy2.robjects as ro
from rpy2.robjects.vectors import ListVector, StrVector, IntVector, FloatVector
from rpy2.robjects import r
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
import csv
from collections import Counter
from sklearn.manifold import TSNE 
import matplotlib 
matplotlib.use("Agg")  # non-GUI backend
import matplotlib.pyplot as plt 
from sklearn.model_selection import StratifiedKFold
import random

# suppress warnings
r('options(warn=-1)')

# suppress package startup messages
r('suppressMessages(library(clusterProfiler))')
r('suppressMessages(library(dplyr))')

r('suppressPackageStartupMessages(library(org.Hs.eg.db))')

# clear environment
r('rm(list = ls())')

# establish source file as the script housing all the Bioconductor-based functions
r.source("./biomart.R")

plt.style.use('ggplot')

# set random seeds in addition to random state throughout the code 
random.seed(3)
np.random.seed(3)



def get_paths(dir):
    # identify all subdirectories
    dirs = sorted(os.listdir(dir))
    # add parent directory string and return list
    dirs = [dir + chr(92) + e for e in dirs]
    return dirs


def deseq2(expr_df, meta_df, str_label, volcano_int):

    # set random state in R 
    ro.r('set.seed(3)')

    r_func = ro.globalenv['run_deseq2']

    # Use a local conversion context
    with localconverter(ro.default_converter + pandas2ri.converter):
        expr_df_r = ro.conversion.py2rpy(expr_df.copy())
        meta_df_r = ro.conversion.py2rpy(meta_df.copy())
    
    # Call the R function
    result_r = r_func(expr_data=expr_df_r, meta_data=meta_df_r, str_label=str_label, volcano=volcano_int)
    
    # Convert the result back to pandas
    with localconverter(ro.default_converter + pandas2ri.converter):
        result_df = ro.conversion.rpy2py(result_r)

    return result_df


# custom transformer for true quantile normalization, for later use once computational resources are not limited
class QuantileNorm(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.rank_means = None

    def fit(self, X, y=None):

        # ensure data is in dataframe format
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        elif not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be numpy array or pandas dataframe...")

        # calculate rank means
        sort_in_sample = np.sort(X.values, axis=1)

        # average the expression of each rank
        self.rank_means = np.mean(sort_in_sample, axis=0)

        return self

    def transform(self, X):

        # ensure input is dataframe for easier manipulation
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        elif not isinstance(X, pd.DataFrame):
            raise ValueError("Input should be a ndarray or dataframe")

        # rank within sample
        ranked_x = X.rank(axis=1, method='min').astype(int) - 1

        # replace each value in ranked_x with the rank means in the np.array
        mapped_means = np.array(self.rank_means)[ranked_x.values]

        # convert back to df
        normed_x = pd.DataFrame(mapped_means, index=ranked_x.index, columns=ranked_x.columns)

        return normed_x


def robust_rank_agg(list_of_lists):
    r_func = ro.globalenv['robust_rank_agg']

    r_list_of_lists = ListVector({
        f"list{i+1}": StrVector(ranking)
        for i, ranking in enumerate(list_of_lists)
    })

    r_robust_rank = r_func(r_list_of_lists)

    with localconverter(ro.default_converter + pandas2ri.converter):
        result = ro.conversion.rpy2py(r_robust_rank)

    return result


def find_pathways(gene_list, cohort):

    r_func = ro.globalenv['pathway_analysis']

    r_list = StrVector(gene_list)

    r_hits = r_func(r_list, cohort)

    # Convert R dataframes to pandas using localconverter
    with localconverter(ro.default_converter + pandas2ri.converter):
        kegg_hits = ro.conversion.rpy2py(r_hits.rx2('kegg'))
        go_hits = ro.conversion.rpy2py(r_hits.rx2('go'))

    return kegg_hits[['Description', 'qvalue', 'geneID']], go_hits[['Description', 'qvalue', 'geneID']]


def run_gsea(rra_df, cohort): 

     # Create ranked vector: higher rank = smaller RRA score
    rra_series = pd.Series(-np.log10(rra_df['Score'].values), index=rra_df['Name'].values)
    rra_series = rra_series.replace([np.inf, -np.inf], np.nan).dropna()
    rra_series = rra_series[~rra_series.index.duplicated()]
    rra_series = rra_series.sort_values(ascending=False)
    print(rra_series)
    rra_series = rra_series[rra_series > 0]
    print(rra_series)

    # Convert to R named vector
    gene_names = StrVector(rra_series.index.tolist())
    gene_values = FloatVector(rra_series.values)
    r.assign("gene_list", r['setNames'](gene_values, gene_names))

    # Call your R GSEA function
    r_func = r['gsea_analysis']  # assumes this handles GSEA when given ranked vector
    r_hits = r_func(r["gene_list"], cohort)

    # Convert R dataframes to pandas
    with localconverter(pandas2ri.converter):
        kegg_hits = r.conversion.rpy2py(r_hits.rx2('kegg'))
        go_hits = r.conversion.rpy2py(r_hits.rx2('go'))

    # Select relevant columns
    return kegg_hits[['Description', 'qvalue', 'geneID']], go_hits[['Description', 'qvalue', 'geneID']]


def get_tcga(ids, format='counts', limit=False):

    # get all files within the directory for tumor/nontumor
    healthy_tumor_dir = get_paths(r"raw_data/TCGA Data")

    # create empty dataframes to hold gene counts and TPM-normalized counts
    all_counts = pd.DataFrame()
    all_tpms = pd.DataFrame()

    # empty list to store target variable binary
    encoded_classes = []
    encoded_disease = []
    encoded_gender = []

    # load annotation data
    clinical_table = pd.read_csv(r"raw_data/clinical.tsv", sep="\t")
    with open(r'raw_data/tcga_all_metadata.json', 'r') as metadata_file:
        metadata = json.load(metadata_file)

    # create lookup dictionary for JSON
    metadata_dict = {item["file_name"]: item for item in metadata}

    # list containing stage of each 
    stage_info = []
    invasive = []
    grades = []

    for condition in healthy_tumor_dir:
        tcga_files = get_paths(condition)
        for file in tcga_files:
            # load each patient RNA-seq TSV, which includes raw counts and TPM values
            patient = pd.read_csv(file, sep="\t", skiprows=[0, 2, 3, 4, 5])
            # extract raw counts and tpm
            gene_counts = patient[[ids, 'unstranded']].set_index(ids)
            tpms = patient[[ids, 'tpm_unstranded']].set_index(ids)
            # append to running dataframes
            all_counts = pd.concat([all_counts, gene_counts], axis=1)
            all_tpms = pd.concat([all_tpms, tpms], axis=1)
            # if doing male/female, the binary target must be determined based on the filename
            filename = str(Path(file).name)
            # use filename to get annotation data, then extract case ID
            caseid = metadata_dict.get(filename, {}).get("associated_entities")[0]['case_id']
            # use the case ID to query the clinical info TSV
            patient = clinical_table[clinical_table['cases.case_id'] == caseid]
            # get tumor stage and append to running list
            try:
                tumor_stage = patient[patient['diagnoses.diagnosis_is_primary_disease'] == 'true']['diagnoses.ajcc_pathologic_stage'].iloc[0]
                metastat =  patient[patient['diagnoses.diagnosis_is_primary_disease'] == 'true']['diagnoses.ajcc_pathologic_m'].iloc[0]
                grade = patient[patient['diagnoses.diagnosis_is_primary_disease'] == 'true']['diagnoses.tumor_grade'].iloc[0]
            except IndexError: 
                tumor_stage = patient['diagnoses.ajcc_pathologic_stage'].iloc[0]
                metastat = patient['diagnoses.ajcc_pathologic_m'].iloc[0]
                grade = patient['diagnoses.tumor_grade'].iloc[0]
            stage_info.append(tumor_stage)
            invasive.append(metastat)
            grades.append(grade)
            # get gender 
            gender = patient['demographic.gender'].iloc[0]
            # encode male/female as binary variable for DGEA purposes
            if gender == 'male':
                gender_binary = 1
            else:
                gender_binary = 0
            # add to running list 
            encoded_gender.append(gender_binary)

            # but also create string for one of four classes for machine learning
            if str(condition).find('Control') != -1:
                class_str = gender + "_healthy"
            else:
                class_str = gender + "_tumor"
            # add to running list of patient genders
            encoded_classes.append(class_str)

        # if doing tumor/non-tumor, binary target depends on the parent directory of the RNA-seq file
        if str(condition).find('Control') != -1:
            tumor_binary = 0
        else:
            tumor_binary = 1
        # create list of n_samples of the appropriate binary value
        encoded_disease += [tumor_binary] * len(tcga_files)

    # remove version number from ENSEMBL ID then sum counts of duplicate indices or average them for TPM
    if ids == "gene_id":
        if format == 'counts':
            all_counts.index = all_counts.index.str.replace(r'\..*', '', regex=True)
        else:
            all_tpms.index = all_tpms.index.str.replace(r'\..*', '', regex=True)

    if format == 'counts':
        all_counts = all_counts.groupby(all_counts.index).sum()
        assert not all_counts.index.duplicated().any()
    else: 
        all_tpms = all_tpms.groupby(all_tpms.index).mean()
        assert not all_tpms.index.duplicated().any()

   

    # encode classes - this is backup for pivot to multiclass 
    #print(Counter(encoded_classes))
    #labeler = LabelEncoder()
    #encoded_classes = labeler.fit_transform(encoded_classes)

    #all_counts.columns = encoded_classes

    if limit == True: 

        # if limiting, take out metastatic, low grade, and stage I tumors 
        if format == 'counts':
            transposed = all_counts.T.reset_index(drop=True)
        else:
            transposed = all_tpms.T.reset_index(drop=True)
        # add clinical info
        transposed['stage'] = stage_info 
        transposed['metastasis'] = invasive 
        transposed['grade'] = grades
        transposed['gender'] = encoded_gender
        transposed['tumor'] = encoded_disease
        # limit tumor samples to exclude Stage I, metastasized, and low grade tumors
        transposed_tumor = transposed[(transposed['tumor'] == 1) & (transposed['stage'].isin(["Stage II", "Stage III", "Stage IV"])) & (transposed['metastasis'].isin(["M0", "MX"])) & (transposed['grade'] == "High Grade")]
        # extract normal samples as well
        transposed_control = transposed[transposed['tumor'] == 0]
        # re-merge normal samples with limited tumor samples 
        transposed = pd.concat([transposed_tumor, transposed_control], axis=0)
        # extract target variable 
        encoded_gender = transposed['gender'].to_list()
        encoded_disease = transposed['tumor'].to_list()
        # drop clinical info and go back to gene=row format, make sure length of metadata matches number of patients 
        if format == 'counts': 
            all_counts = transposed.drop(['stage', 'metastasis', 'grade', 'gender', 'tumor'], axis=1)
            all_counts = all_counts.T
            assert len(encoded_gender) == len(encoded_disease) ==  len(all_counts.columns)
        else: 
            all_tpms = transposed.drop(['stage', 'metastasis', 'grade', 'gender', 'tumor'], axis=1)
            all_tpms = all_tpms.T
            assert len(encoded_gender) == len(encoded_disease) ==  len(all_tpms.columns)

    print("limit is set to:", str(limit))
    print("TCGA dist")
    demo = Counter(f"{g}_{d}" for g, d in zip(encoded_gender, encoded_disease))
    print(demo)

    if format == "counts":

        return all_counts, encoded_gender, encoded_disease

    elif format == "tpm":

        return all_tpms, encoded_gender, encoded_disease

    else:

        raise KeyError("Input valid format for expressions...")


def get_gtex():

    counts = pd.read_csv(r'raw_data/gene_reads_bladder.gct', sep="\t", header=2).set_index('Name')

    # remove version number from Ensembl ID for overlapping between datasets then sum duplicates
    counts.index = counts.index.str.replace(r'\..*', '', regex=True)
    counts = counts.groupby(counts.index).sum()

    assert not counts.index.duplicated().any()

    # store gene symbol then drop it
    id_to_symbol = counts['Description']
    counts.drop('Description', axis=1, inplace=True)

    # create list N of 0 (all normal samples)
    disease = [0] * len(counts.columns)

    # determine gender of each observation
    ids = [col[:9] for col in counts.columns]


    # load metadata
    metadata = pd.read_csv(r"raw_data/GTEx_Analysis_v10_Annotations_SubjectPhenotypesDS.txt", sep="\t")

    gender_df = pd.DataFrame({"SUBJID": ids}).merge(
        metadata[["SUBJID", "SEX"]],
        on="SUBJID",
        how="left"
    )

    # make encoding consistent, 1 for male 0 for female
    print('gtex gender dist:')
    print(gender_df['SEX'].value_counts())
    gender_df['SEX'] = gender_df['SEX'].replace(2, 0)

    print('gtex is all healthy samples')

    gender = gender_df['SEX'].to_list()

    return counts, gender, disease


def gtex_dgea(): 

    # get counts and gender info 
    counts, gender, disease = get_gtex()

    deseq_prepped = counts.astype(int)
    deseq_prepped.index.name = "GeneID"
    deseq_prepped.columns = [f"sample{i+1}" for i in range(deseq_prepped.shape[1])]
    deseq_prepped = deseq_prepped.reset_index()

    gender = pd.DataFrame(gender, columns=['Condition'])
    gender.index = [f"sample{i+1}" for i in range(gender.shape[0])]

    results = deseq2(deseq_prepped, gender, "gtex", 0)
    
    # convert ENSEMBL IDs to gene symbols with reference genome - only want DEGs appearing in GRch38.p13
    # It is what we will use to get TPMs later 
    grc = pd.read_csv(r"./Human.GRCh38.p13.annot.tsv", sep="\t")
    grc = grc[['EnsemblGeneID', 'Symbol']].dropna().reset_index(drop=True)
    convert_ensembl = dict(zip(grc['EnsemblGeneID'], grc['Symbol']))

    # match ENSEMBL ID to gene symbol and convert
    results = results[results.index.isin(convert_ensembl.keys())]
    results.rename(index=convert_ensembl, inplace=True)

    return results
    

def get_gse133624():

    counts = pd.read_csv(r'raw_data/GSE133624_reads-count-all-sample.txt', sep="\t").set_index('GeneID')

    # sum duplicate indices
    counts = counts.groupby(counts.index).sum()
    assert not counts.index.duplicated().any()

    # get disease status from sample IDs
    sample_ids = counts.columns.to_list()
    disease = [0 if col.find('N') != -1 else 1 for col in sample_ids]
    print("GSE133624 disease:")
    print(Counter(disease))

    # get gender from TXT files
    gender_key = pd.read_csv(r"raw_data/GSE133624_gender_key.txt", sep="\t")
    sample_key = pd.read_csv(r"raw_data/GSE133624_sample_key.txt", sep="\t")

    key = gender_key.merge(sample_key, on='Sample_ID')
    key['Tissue_ID'] = key['Tissue_ID'].str[:-2]
    gender_convert = dict(zip(key['Tissue_ID'], key['Gender']))
    counts.rename(columns=gender_convert, inplace=True)

    patient_gender = counts.columns.to_list()

    print("GSE133624 gender:")
    print(counts.columns.value_counts())
    gender = [1 if sex == 'Male' else 0 for sex in patient_gender]

    combined = [str(gender[i]) + "_" + str(disease[i]) for i in range(len(gender))]
    print(Counter(combined))

    return counts, gender, disease



def get_stratified_counts(correction_method, mode, training): 

    if training == "merged":

        merged_gender = []
        merged_disease = []

        tcga_counts, tcga_gender, tcga_disease = get_tcga(ids='gene_id', format='counts')
        merged_gender += tcga_gender
        merged_disease += tcga_disease

        gtex_counts, gtex_gender, gtex_disease = get_gtex()
        merged_gender += gtex_gender
        merged_disease += gtex_disease

        gse_counts, gse_gender, gse_disease = get_gse133624()
        merged_gender += gse_gender
        merged_disease += gse_disease

        merged_counts = pd.concat([tcga_counts, gtex_counts, gse_counts], axis=1).dropna()
        new_cols = [f"sample{i+1}" for i in range(merged_counts.shape[1])]
        merged_counts.columns = new_cols

        batch_list = (len(tcga_counts.columns) * ['TCGA']) + (len(gtex_counts.columns) * ['GTEx']) + (len(gse_counts.columns) * ['GSE133624'])

        if correction_method == 'ComBat-seq':
            corrected_counts = combat_seq(merged_counts, batch_list, merged_disease, merged_gender)
        elif correction_method == 'ComBat-ref':
            # export merged counts as TSV for ComBat-ref script
            merged_counts.to_csv('./Combat-ref-main/merged_counts.tsv', sep="\t", index=True, header=True, quoting=csv.QUOTE_NONE)
            # use batch list to create sample info CSV
            sample_info = pd.DataFrame()
            # get identifier for each sample
            sample_info.index = merged_counts.columns
            # per advice of ComBat-ref creator, include covariate in the treatment group
            # ex: male tumor, female healthy
            sample_info['group'] = [str(merged_disease[i]) + "_" + str(merged_gender[i]) for i in range(len(merged_disease))]
            sample_info['batch'] = batch_list
            sample_info.to_csv('./Combat-ref-main/sample_info.csv')
            # create command to run ComBat-ref
            script = r"/Combat-ref-main/batch_correct.R"
            counts_matrix = r"Combat-ref-main/merged_counts.tsv"
            sample_file = r"/Combat-ref-main/sample_info.csv"
            batch_command = ["Rscript", script, counts_matrix, sample_file]
            result = subprocess.run(batch_command, capture_output=True, text=True, check=True, cwd=r"/Combat-ref-main")
            corrected_counts = pd.read_csv(r"/Combat-ref-main/merged_counts_corrected.tsv", sep="\t", header=0)

        elif correction_method == "z-score":

            corrected_counts = merged_counts

        else:
            raise Exception("Input valid correction method!")




        # create column of merged class labels for stratifying train/test split
        corrected_counts.loc['Gender'] =  merged_gender
        corrected_counts.loc['Disease State'] = merged_disease
        combined_label = [f"{a}_{b}" for a, b in zip(merged_gender, merged_disease)]


        corrected_counts = corrected_counts.T
        corrected_counts['Batch'] =  batch_list


        # save counts as csv

        corrected_counts.to_csv('.\\expression_data\\merged_dataset_counts.csv')

        # keep batch in the dataframe for later if z-scoring

        if not correction_method == "z-score":

            corrected_counts.drop('Batch', axis=1, inplace=True)

    elif training == "TCGA":

        # when using TCGA data only for male tumor vs. female tumor, also use ENSEMBL IDs counts for consistency 
        # however, limit the tumor data - limiting to Stages II-IV, no metastasis, and high grade tumors 
        tcga_counts, tcga_gender, tcga_disease = get_tcga('gene_id', limit=True)
        # get ENSEMBL IDs as columns
        corrected_counts = tcga_counts.T.reset_index(drop=True)
        # add metadata
        corrected_counts['Gender'] = tcga_gender
        corrected_counts['Disease State'] = tcga_disease
        # save as CSV
        corrected_counts.to_csv('.\\expression_data\\tcga_counts.csv')

    else:

        raise KeyError("Input valid training cohort please...")

    # split into male and female
    if mode == "gender_stratified":

        counts_1 = corrected_counts[corrected_counts['Gender'] == 1]
        counts_2 = corrected_counts[corrected_counts['Gender'] == 0]
        counts_1.drop('Gender', axis=1, inplace=True)
        counts_2.drop('Gender', axis=1, inplace=True)

        # extract target variable and do train/test split
        target_1 = counts_1.pop('Disease State')
        target_2 = counts_2.pop('Disease State')

    # or split into tumor and non-tumor
    elif mode =="disease_stratified":

        counts_1 = corrected_counts[corrected_counts['Disease State'] == 1]
        counts_2 = corrected_counts[corrected_counts['Disease State'] == 0]
        counts_1.drop('Disease State', axis=1, inplace=True)
        counts_2.drop('Disease State', axis=1, inplace=True)

        # extract target variable and do train/test split
        target_1 = counts_1.pop('Gender')
        target_2 = counts_2.pop('Gender')

    else:

        raise Exception("Valid comparison not inputted...")
    
    return counts_1, counts_2, target_1, target_2

def get_folded_data(data, target): 

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)

    for fold, (train, test) in enumerate(cv.split(data, target), start=1):

        data_train, data_test = data.iloc[train], data.iloc[test]
        target_train, target_test = target.iloc[train], target.iloc[test]
        yield data_train, data_test, target_train, target_test


def run_dgea(train_counts, train_label, mode, training, correction, volcano_int=0, cohort_int=1): 

    if training == "merged": 
        batch = train_counts.pop('Batch')

    if cohort_int == 1 and mode == "gender_stratified": 
        cohort_str = "male"
    elif cohort_int == 2 and mode == "gender_stratified":
        cohort_str = "female"
    elif cohort_int == 1 and mode == "disease_stratified": 
        cohort_str = "tumor"
    elif cohort_int == 2 and mode == "disease_stratified": 
        cohort_str = "healthy"
    else: 
        raise Exception("Input valid cohort number and mode combination...")
        
    # prepare matrix - convert to int and rename/reset index
    deseq_prepped = train_counts.T.astype(int)
    deseq_prepped.index.name = "GeneID"
    deseq_prepped = deseq_prepped.reset_index()

    if mode == "gender_stratified": 

        # if not z-scoring, the only metadata you need is the condition of interest (tumor/healthy or male/female)
        if not correction == "z-score":

            results = deseq2(deseq_prepped, pd.DataFrame(train_label).rename(columns={'Disease State': 'Condition'}), cohort_str, volcano_int=volcano_int)

        # if z-scoring, also provide batch as a covariate for the DESeq2 model
        else:

            meta_df = pd.concat([train_label, batch], axis=1).rename(columns={'Disease State': 'Condition'})
            results = deseq2(deseq_prepped, meta_df, cohort_str, volcano_int=volcano_int)

    # same as above gender is just the condition so names must change 
    elif mode == "disease_stratified":

        if not correction == "z-score":

            results = deseq2(deseq_prepped, pd.DataFrame(train_label).rename(columns={'Gender': 'Condition'}), cohort_str, volcano_int=volcano_int)

        else:

            meta_df = pd.concat([train_label, batch], axis=1).rename(columns={'Gender': 'Condition'})
            results = deseq2(deseq_prepped, meta_df, cohort_str, volcano_int=volcano_int)
    else:

        raise Exception("Valid comparison not inputted...")

    # convert ENSEMBL IDs to gene symbols with reference genome - only want DEGs appearing in GRch38.p13
    # It is what we will use to get TPMs later 
    grc = pd.read_csv(r"./Human.GRCh38.p13.annot.tsv", sep="\t")
    grc = grc[['EnsemblGeneID', 'Symbol']].dropna().reset_index(drop=True)
    convert_ensembl = dict(zip(grc['EnsemblGeneID'], grc['Symbol']))

    # match ENSEMBL ID to gene symbol and convert
    results = results[results.index.isin(convert_ensembl.keys())]
    results.rename(index=convert_ensembl, inplace=True)

    # if you're getting the volcano plot, then this is the entire training set 
    # save results to CSV
    if volcano_int == 1:
        results.to_csv(".\\dgea_results\\" + training + "_" + correction + "_" + cohort_str + ".csv")
    
    return results 



def control_degs(mode, results_1, results_2): 

    if mode == "disease_stratified": 
        # if using another dataset's results to verify, you need to make sure you're only considering those appearing in both datasets
        genes_in_2 = set(results_2.index)
        results_1 = results_1[results_1.index.isin(genes_in_2)]
        
        genes_in_1 = set(results_1.index)
        results_2 = results_2[results_2.index.isin(genes_in_1)]
        # if results_1 and results_2 originate from the same dataset this is not necessary
    

    # results contain info for all gene s
    # filter to log2FC > |1.0| and p.adj < 0.05
    results_1, results_2 = results_1[(results_1['padj'] < 0.05) & (np.abs(results_1['log2FoldChange']) > 1.0)], results_2[(results_2['padj'] < 0.05) & (np.abs(results_2['log2FoldChange']) > 1.0)]

    degs_1 = set(results_1.index)
    degs_2 = set(results_2.index)
    

    # determine genes that are DEGs in cohort but not the other 
    specific_1 = degs_1 - degs_2
    specific_2 = degs_2 - degs_1

    # get genes appearing in both cohorts 
    overlap = degs_1 & degs_2
    # if they are oppositely regulated between cohorts extract them
    oppositely_regulated = [gene for gene in overlap
                            if np.sign(results_1.loc[gene, "log2FoldChange"]) != np.sign(results_2.loc[gene, "log2FoldChange"])]
    
    # combine oppositely regulated and cohort-specific together
    specific_1 = set(specific_1) | set(oppositely_regulated)
    specific_2 = set(specific_2) | set(oppositely_regulated)

    # need to retain order of results dataframes (adjusted p-value)
    specific_1 = [gene for gene in results_1.index if gene in specific_1]
    specific_2 = [gene for gene in results_2.index if gene in specific_2]
    
    return specific_1, specific_2


def get_stratified_tpm(mode, training, correction_method): 

    # load counts matrix

    # load reference genome
    grc = pd.read_csv(r"./Human.GRCh38.p13.annot.tsv", sep="\t")

    if training == "merged":

        if correction_method == "ComBat-ref":

            counts = pd.read_csv(r"/expression_data/merged_dataset_counts.csv", index_col=0)

            targets = counts[['Gender', 'Disease State']]
            batch = counts['Batch']
            counts.drop(['Gender', 'Disease State', 'Batch'], axis=1, inplace=True)

            tpm_normed = counts_2_tpm(counts, grc)

        elif correction_method == "z-score":

            # for z-score pipeline, process each dataset independently then z-score and concatenate
            # use the same reference genome for all of them (GRCh38.p13)

            merged_gender = []
            merged_disease = []

            tcga_counts, tcga_gender, tcga_disease = get_tcga(ids='gene_id', format='counts')
            tcga_tpm = counts_2_tpm(tcga_counts.T, grc)
            z_score = StandardScaler()
            tcga_tpm_z = pd.DataFrame(z_score.fit_transform(tcga_tpm), columns=tcga_tpm.columns, index=tcga_tpm.index)
            merged_gender += tcga_gender
            merged_disease += tcga_disease

            gtex_counts, gtex_gender, gtex_disease = get_gtex()
            gtex_tpm = counts_2_tpm(gtex_counts.T, grc)
            z_score = StandardScaler()
            gtex_tpm_z = pd.DataFrame(z_score.fit_transform(gtex_tpm), columns=gtex_tpm.columns, index=gtex_tpm.index)
            merged_gender += gtex_gender
            merged_disease += gtex_disease

            gse_counts, gse_gender, gse_disease = get_gse133624()
            gse_tpm = counts_2_tpm(gse_counts.T, grc)
            z_score = StandardScaler()
            gse_tpm_z = pd.DataFrame(z_score.fit_transform(gse_tpm), columns=gse_tpm.columns, index=gse_tpm.index)
            merged_gender += gse_gender
            merged_disease += gse_disease

            tpm_normed = pd.concat([tcga_tpm_z, gtex_tpm_z, gse_tpm_z], axis=0).dropna(axis=1)
            tpm = pd.concat([tcga_tpm, gtex_tpm, gse_tpm], axis=0).dropna(axis=1)
            new_index = [f"sample{i + 1}" for i in range(tpm_normed.shape[0])]
            tpm_normed.index = new_index
            tpm.index = new_index

            # make target dataframe
            gender_series = pd.Series(merged_gender, index=new_index)
            disease_series = pd.Series(merged_disease, index=new_index)
            targets = pd.concat([gender_series, disease_series], axis=1)
            targets.columns = ['Gender', 'Disease State']

            # make batch series
            batch_list = (len(tcga_counts.columns) * ['TCGA']) + (len(gtex_counts.columns) * ['GTEx']) + (
                        len(gse_counts.columns) * ['GSE133624'])
            batch = pd.Series(batch_list)


        # check for batch effects before and after z-scoring before merging 
        check_batch_effect(tpm, batch, training + "_" + correction_method + "_" + "prebatch_tpm_")
        check_batch_effect(tpm_normed, batch, training + "_" + correction_method + "_" + 'postbatch_tpm_')
        check_batch_effect(tpm, merged_disease, training + "_" + correction_method + "_" + "prebatch_tpm_tnt_")
        check_batch_effect(tpm_normed, merged_disease, training + "_" + correction_method + "_" + 'postbatch_tpm_tnt_')

    elif training == "TCGA":

        counts = pd.read_csv('.\\expression_data\\tcga_counts.csv')
        targets = counts[['Gender', 'Disease State']]

        counts = counts.drop(['Gender', 'Disease State'], axis=1)
        tpm_normed = counts_2_tpm(counts, grc)


    else:

        raise KeyError("Please input valid training set...")
    

    # add target var back to dataframe no matter what the training/correction are 
    tpm_norm_w_gender = pd.concat([tpm_normed, targets], axis=1)


    if mode == "gender_stratified":
        tpm_1 = tpm_norm_w_gender[tpm_norm_w_gender['Gender'] == 1]
        tpm_2 = tpm_norm_w_gender[tpm_norm_w_gender['Gender'] == 0]

    elif mode == "disease_stratified":
        tpm_1 = tpm_norm_w_gender[tpm_norm_w_gender['Disease State'] == 1]
        tpm_2 = tpm_norm_w_gender[tpm_norm_w_gender['Disease State'] == 0]
    else:

        raise Exception("Valid comparison not inputted...")


    # train/test split for each cohort
    if mode == "gender_stratified":

        target_1 = tpm_1.pop('Disease State')
        target_2 = tpm_2.pop('Disease State')

    elif mode == "disease_stratified":

        target_1 = tpm_1.pop('Gender')
        target_2 = tpm_2.pop('Gender')

    else:

        raise Exception("Valid comparison not inputted...")
    
    return tpm_1, tpm_2, target_1, target_2

def counts_2_tpm(counts, grch):

    # create key to convert ensembl ID to symbol
    ensembl_2_symbol = grch[['EnsemblGeneID', 'Symbol']].dropna().reset_index(drop=True)
    ensembl_2_symbol = dict(zip(ensembl_2_symbol['EnsemblGeneID'], ensembl_2_symbol['Symbol']))

    # include only genes that are in GRCh38.p13 and convert them to the appropriate symbol
    valid_genes = set(ensembl_2_symbol.keys())
    counts = counts[[col for col in counts.columns if col in valid_genes]]
    counts.rename(columns=ensembl_2_symbol, inplace=True)

    # get lengths for matched genes from GRCh38, convert to kB, and reorder
    matched_genes = counts.columns
    transcript_lengths = grch[grch['Symbol'].isin(matched_genes)][['Symbol', 'Length']].set_index('Symbol')
    transcript_lengths['Length'] = transcript_lengths['Length'] / 1000
    transcript_lengths = transcript_lengths.reindex(counts.columns)

    # convert counts to RPK
    rpk = counts.divide(transcript_lengths['Length'], axis=1)

    # sum RPK per sample and scale by 1e6
    rpk_sum_scaled = rpk.sum(axis=1) / 1000000

    # divide RPK by scaling factor to get TPM
    tpm = rpk.divide(rpk_sum_scaled, axis=0)

    # scale with log transformation
    tpm_normed = np.log2(tpm + 1)

    return tpm_normed

def combat_seq(merged_counts, batch_list, disease_list, gender_list):

    r_func = ro.globalenv['batch_correct']

    r_df = pandas2ri.py2rpy(merged_counts)

    r_list = StrVector(batch_list)

    disease_list = IntVector(disease_list)

    gender_list = IntVector(gender_list)

    corrected_r = r_func(r_df, r_list, disease_list, gender_list)

    corrected_df = pandas2ri.rpy2py(corrected_r)



    return corrected_df

def get_external_for_merged(which=1, correction="ComBat-ref"):

    if which == 1:
        # import TPM values from GSE236932 - they were generated with GRCh38.p13
        rna_ex = pd.read_csv(r"./external_datasets/GSE236932_norm_counts_TPM_GRCh38.p13_NCBI (1).tsv", sep="\t").set_index(
            'GeneID')

        # convert GeneID to symbol
        rna_ex_annot = pd.read_csv(r"Human.GRCh38.p13.annot.tsv", sep="\t").set_index('GeneID')
        probe2symbol = dict(zip(rna_ex_annot.index, rna_ex_annot['Symbol']))
        rna_ex = rna_ex.rename(index=probe2symbol)
        symbol_matches = set(probe2symbol.values())

        # average duplicate probes since we are working with RNA-seq data
        rna_ex = rna_ex.groupby(rna_ex.index).mean()
        rna_ex = rna_ex[rna_ex.index.isin(symbol_matches)].T

        # treat the same as the merged dataset - log2 with pseudocount of 1
        rna_ex = np.log2(rna_ex + 1)

        # sample key
        columns = ["Sample", "Age", "Gender", "Organ", "Category", "Histopathology", "Tumor stage", "N stage", "M stage", "GEO ID"]
        sample_key = pd.read_csv(r"./external_datasets/GSE236932_sample_key.txt", sep="\t", names=columns, header=0, engine="python", index_col=False)
        sample_key = sample_key[['GEO ID', 'Gender', 'Category']].set_index('GEO ID')
        ids = sample_key.index.intersection(rna_ex.index)
        rna_ex = rna_ex.loc[ids]
        sample_key = sample_key.loc[ids]
        rna_ex = pd.merge(rna_ex, sample_key, left_index=True, right_index=True)

        rna_ex['Gender'] = [1 if gender == 'Male' else 0 for gender in rna_ex['Gender']]
        rna_ex['Category'] = [1 if gender == 'Tumor' else 0 for gender in rna_ex['Category']]

        print("GSE236932 gender + disease dist")
        print("Male tumor:", len(rna_ex[(rna_ex['Category'] == 1) & (rna_ex['Gender'] == 1)]))
        print("Female tumor:", len(rna_ex[(rna_ex['Category'] == 1) & (rna_ex['Gender'] == 0)]))
        print("Male healthy:", len(rna_ex[(rna_ex['Category'] == 0) & (rna_ex['Gender'] == 1)]))
        print("Female healthy:", len(rna_ex[(rna_ex['Category'] == 0) & (rna_ex['Gender'] == 0)]))



    elif which == 2:

        # load TPM data (used GRCh38)
        rna_ex = pd.read_csv(r"./external_datasets/GSE188715_tpm.txt", sep="\t", header=0).set_index("ID_REF")
        # average duplicate probes to be safe
        rna_ex = rna_ex.groupby(rna_ex.index).mean().T

        # normalize by log2 transformation w/ pseudocount
        rna_ex = np.log2(rna_ex + 1)


        # determine gender and tumor status
        sample_key = pd.read_csv(r"./external_datasets/GSE188715_key.txt", sep="\t", header=0)[['Sample_ID', 'Category', 'Gender']].set_index("Sample_ID")
        ids = sample_key.index.intersection(rna_ex.index)
        rna_ex = rna_ex.loc[ids]
        sample_key = sample_key.loc[ids]
        rna_ex = pd.merge(rna_ex, sample_key, left_index=True, right_index=True)
        rna_ex['Gender'] = [1 if gender == 'Male' else 0 for gender in rna_ex['Gender']]
        rna_ex['Category'] = [1 if gender == 'Tumor' else 0 for gender in rna_ex['Category']]

        print("GSE188715 gender + disease dist")
        print("Male tumor:", len(rna_ex[(rna_ex['Category'] == 1) & (rna_ex['Gender'] == 1)]))
        print("Female tumor:", len(rna_ex[(rna_ex['Category'] == 1) & (rna_ex['Gender'] == 0)]))
        print("Male healthy:", len(rna_ex[(rna_ex['Category'] == 0) & (rna_ex['Gender'] == 1)]))
        print("Female healthy:", len(rna_ex[(rna_ex['Category'] == 0) & (rna_ex['Gender'] == 0)]))
        

    else:

        # read csv with probe IDs as index and convert to numeric
        project_bg_adj = pd.read_csv(r".\\external_datasets/GSE13507_RAW\\GSE13507_illumina_raw.txt", sep="\t").set_index('GEO ACCESSIONS').apply(pd.to_numeric, errors='raise')
        # drop samples that also appear in the external validation set (same author)
        # it is not explicitly stated whether the data is background corrected or not, but Illumina BeadStudio was used
        # and there are no small expressions, so it will be assumed for now
        # drop samples that also appear in the external validation set (same author)
        # project_bg_adj = limma_bg_ill(project)
        #project_bg_adj = project.drop(
            #["GSM340545", "GSM340538", "GSM340539", "GSM340645", "GSM340541", "GSM340651", "GSM340662", "GSM340711",
             #"GSM340737", "GSM340642", "GSM340717", "GSM340686", "GSM340740", "GSM340652", "GSM340544", "GSM340713",
             #"GSM340682", "GSM340698", "GSM340710", "GSM340542", "GSM340663", "GSM340666", "GSM340667", "GSM340749"],
            #axis=1)
        # gene symbols for probes are provided for GSE13507, read in csv to convert
        probe_info_0 = pd.read_csv(r".\\external_datasets\\GSE13507_RAW\\GPL6102_Illumina_HumanWG-6_V2_0_R1_11223189_A.bgx", sep="\t", on_bad_lines="skip", header=8, nrows=48702)
        # only include symbols that have a probe ID
        probe_info = probe_info_0[['Probe_Id', 'Symbol']].dropna()
        # convert to a dictionary and use to rename expression dataframe
        probe2symbol = dict(zip(probe_info['Probe_Id'], probe_info['Symbol']))
        converted_illum = set(probe2symbol.values())
        project = project_bg_adj.rename(index=probe2symbol)
        # average duplicate probes for the same gene
        project = project.groupby(project.index).mean()
        # make sure only successful conversions are retained
        project = project[project.index.isin(converted_illum)]
        # get NCBI GEO labels based on sample ID
        sample_key = pd.read_csv(r".\\external_datasets\\GSE13507_RAW\\GSE13507_key.txt", sep="\t", header=None)
        # convert to a dictionary and convert
        key_dict = sample_key.set_index(0)[1].to_dict()
        project = project.rename(columns=key_dict)

        # first, drop all columns that do not start with BT - want tumor only
        tumor_samples = project.columns[project.columns.str.contains('BT')]
        project = project[tumor_samples]
        # extract sample labels and integer encode to 0 or 1 based on gender
        # now rename columns to only include last 5 characters: e.g. BT001
        project.columns = [name[-5:] for name in project.columns]
        no_recurrent = project.columns[~project.columns.str.contains(r'\.')]
        project = project[no_recurrent].T
        # normalize before adding target
        qn = QuantileNorm()
        qn.fit(project)
        project = np.log2(qn.transform(project) + 1)
        # get gender and disease status
        gender_key = pd.read_csv(r".\\external_datasets\\GSE13507_RAW\\GSE13507_clinical_info.csv", sep=",")[['Sample name', 'SEX']]
        gender_key = dict(zip(gender_key['Sample name'].to_list(), gender_key['SEX'].to_list()))
        letter_2_num = {'M':1, 'F':0}
        project['Gender'] = project.index.map(gender_key)
        project['Gender'] = project['Gender'].map(letter_2_num)
        project['Category'] = [1]*len(project)

        rna_ex = project

    if correction == "z-score":

        targets = rna_ex[['Gender', 'Category']]
        rna_ex.drop(['Gender', 'Category'], axis=1, inplace=True)
        z_score = StandardScaler()
        rna_ex = pd.DataFrame(z_score.fit_transform(rna_ex), columns=rna_ex.columns, index=rna_ex.index)
        rna_ex = pd.concat([rna_ex, targets], axis=1)

    return rna_ex

def check_batch_effect(expr_df, batch_labels, label):
    import umap 
    # scale data, but genes must be converted back to columns beforehand
    scaled_df = StandardScaler().fit_transform(expr_df)

    # PCA 
    pca = PCA(n_components=2, random_state=3)
    pca_transformed = pca.fit_transform(scaled_df)
    plot_comps(pca_transformed, batch_labels, 'PC1', 'PC2', label + '_pca.png')
    
    # tSNE
    tsne = TSNE(n_components=2, random_state=3)
    tsne_transformed = tsne.fit_transform(scaled_df)
    plot_comps(tsne_transformed, batch_labels, 'tSNE', 'tSNE2', label + '_tsne.png')

    #UMAP
    umapper = umap.UMAP(n_components=2, random_state=3)
    umap_transformed = umapper.fit_transform(scaled_df)
    plot_comps(umap_transformed, batch_labels, 'UMAP1', 'UMAP2', label + '_umap.png')

    return


def plot_comps(embedding, labels, xlabel, ylabel, filename):
    # Convert labels to categorical codes for coloring
    labels_categorical = pd.Categorical(labels)
    label_codes = labels_categorical.codes
    
    plt.figure(figsize=(8,6))
    
    # create the scatter and capture the object
    scatter = plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=label_codes,            
        alpha=0.7,
        s=50
    )

    plt.xlabel(xlabel, fontsize=14, family='Arial')
    plt.ylabel(ylabel, fontsize=14, family='Arial')
    
    # Create legend with original label names
    handles, _ = scatter.legend_elements()
    legend_labels = labels_categorical.categories
    plt.legend(handles, legend_labels, title="Dataset", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(".\\figures\\" + filename, dpi=300)
    plt.close()

    return
