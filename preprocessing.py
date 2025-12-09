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




r'''


def write_expr_data_raw(source, mode):
    # set the path to the file location with all GEO txt matrices
    os.chdir(source)
    # extract all the filenames of the data matrices
    proj_dirs = get_paths(os.curdir)
    # read table for each project
    for d in proj_dirs:
        if str(d).find("TCGA") != -1:
            if mode == 'm-f':
                counts_rna, tpm_rna, gender_list, tumor_list = process_project_raw(d, mode)
                print("Raw expression dataframe for project ", d, ":")
                print(counts_rna)
                counts_rna = counts_rna.T.reset_index()
                counts_rna['gender'] = gender_list
                counts_rna['disease state'] = tumor_list
                tpm_rna = tpm_rna.T.reset_index()
                tpm_rna['gender'] = gender_list
                tpm_rna['disease state'] = tumor_list
            else:
                counts_rna, tpm_rna = process_project_raw(d, mode)
                print("Raw expression dataframe for project ", d, ":")
                print(counts_rna)
                counts_rna = counts_rna.T.reset_index()
                tpm_rna = tpm_rna.T.reset_index()
        elif str(d).find("GSE13507") != -1:
            # save the one microarray dataset
            # turn the index (binary cancer value) to a column by resetting the index
            microarray = process_project_raw(d, mode)
            microarray = microarray.T.reset_index()

        else:
            raise Exception("Invalid dataset processed, decide its platform")

    os.chdir(r'C:\Users\joepi\Code\Thesis\Python\BCa_ML_DL')

    # classify the genes in the TCGA data as protein coding, ncRNA, or pseudogene
    all_genes = counts_rna.columns.to_list()
    classification_table = find_protein_coding(all_genes)
    classification_table = classification_table.drop_duplicates(subset="hgnc_symbol", keep="first")
    classified_genes = classification_table['hgnc_symbol'].to_list()
    if mode == 'm-f':
        classified_genes = classified_genes + ['index', 'gender', 'disease state']
        protein_coding = classification_table[classification_table['gene_biotype'] == 'protein_coding'][
                             'hgnc_symbol'].to_list() + ['index', 'gender', 'disease state']
    else:
        classified_genes = classified_genes + ['index']
        protein_coding = classification_table[classification_table['gene_biotype'] == 'protein_coding']['hgnc_symbol'].to_list() + ['index']

    process_rna(counts_rna[protein_coding], tpm_rna[protein_coding], 'protein_coding', mode)
    process_rna(counts_rna[classified_genes], tpm_rna[classified_genes], 'all', mode)


    # skip microarray data for gendered analysis - GSE13507 only has clinical info for tumor samples

    if mode == "t-nt":
        process_microarray(microarray, mode)

    return

def process_project_raw(direc, mode):
    # get all file paths within project directory
    print("Now processing", direc)
    file_dirs = get_paths(direc)

    if str(direc).find('TCGA Data') != -1:
        # get all files within the directory for tumor/nontumor
        healthy_tumor_dir = get_paths(direc)

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

        for condition in healthy_tumor_dir:
            tcga_files = get_paths(condition)
            for file in tcga_files:
                # load each patient RNA-seq TSV, which includes raw counts and TPM values
                patient = pd.read_csv(file, sep="\t", skiprows=[0, 2, 3, 4, 5])
                # extract raw counts and tpm
                gene_counts = patient[['gene_name', 'unstranded']].set_index('gene_name')
                tpms = patient[['gene_name', 'tpm_unstranded']].set_index('gene_name')
                # append to running dataframes
                all_counts = pd.concat([all_counts, gene_counts], axis=1)
                all_tpms = pd.concat([all_tpms, tpms], axis=1)
                # if doing male/female, the binary target must be determined based on the filename
                filename = str(Path(file).name)
                # use filename to get annotation data, then extract case ID
                caseid = metadata_dict.get(filename, {}).get("associated_entities")[0]['case_id']
                # use the case ID to query the clinical info TSV
                gender = clinical_table[clinical_table['cases.case_id'] == caseid]['demographic.gender'].iloc[0]
                # encode male/female as binary variable for DGEA purposes
                if gender == 'male':
                    gender_binary = 1
                else:
                    gender_binary = 0
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
            encoded_disease += [tumor_binary]*len(tcga_files)

        # average duplicate gene names, but round for counts such that they remain integers
        all_counts = all_counts.groupby(all_counts.index).mean().round(0)
        all_tpms = all_tpms.groupby(all_tpms.index).mean()

        # encode classes
        labeler = LabelEncoder()
        encoded_classes = labeler.fit_transform(encoded_classes)

        # encoded binary target becomes the column identifier
        if mode == 't-nt':

            all_counts.columns = encoded_disease
            all_tpms.columns = encoded_disease

            return all_counts, all_tpms

        if mode == 'm-f':

            all_counts.columns = encoded_classes
            all_tpms.columns = encoded_classes

            return all_counts, all_tpms, encoded_gender, encoded_disease

    elif str(direc).find("GSE13507") != -1:
        # read csv with probe IDs as index and convert to numeric
        project = pd.read_csv(file_dirs[2], sep="\t").set_index('GEO ACCESSIONS').apply(pd.to_numeric, errors='raise')
        # drop samples that also appear in the external validation set (same author)
        # it is not explicitly stated whether the data is background corrected or not, but Illumina BeadStudio was used
        # and there are no small expressions, so it will be assumed for now
        # drop samples that also appear in the external validation set (same author)
        #project_bg_adj = limma_bg_ill(project)
        project_bg_adj = project.drop(["GSM340545","GSM340538","GSM340539","GSM340645","GSM340541","GSM340651","GSM340662","GSM340711","GSM340737","GSM340642","GSM340717","GSM340686","GSM340740","GSM340652","GSM340544","GSM340713","GSM340682","GSM340698","GSM340710","GSM340542","GSM340663","GSM340666","GSM340667","GSM340749"], axis=1)
        # gene symbols for probes are provided for GSE13507, read in csv to convert
        probe_info_0 = pd.read_csv(file_dirs[0], sep="\t", on_bad_lines="skip", header=8, nrows=48702)
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
        sample_key = pd.read_csv(file_dirs[3], sep="\t", header=None)
        # convert to a dictionary and convert
        key_dict = sample_key.set_index(0)[1].to_dict()
        project = project.rename(columns=key_dict)
        # extract sample labels and integer encode to 0 or 1 based on disease status or gender
        if mode == "t-nt":
            sample_names = project.columns
            # Control is healthy, BS... is surrounding healthy tissue, and BT is tumor
            encoded_names = [1 if name.find('BT') != -1 else 0 if name.startswith('Surrounding') != -1 or name.find(
            'Control') != -1 else name for name in sample_names]
        else:
            # first, drop all columns that do not start with BT - want tumor only
            tumor_samples = project.columns[project.columns.str.contains('BT')]
            project = project[tumor_samples]
            # now rename columns to only include last 5 characters: e.g. BT001
            project.columns = [name[-5:] for name in project.columns]
            no_recurrent = project.columns[~project.columns.str.contains(r'\.')]
            project = project[no_recurrent]
            gender_key = pd.read_csv(file_dirs[1], sep=",")[['Sample name', 'SEX']]
            gender_key = dict(zip(gender_key['Sample name'].to_list(), gender_key['SEX'].to_list()))
            project = project.rename(columns=gender_key)
            encoded_names = [1 if gender == 'M' else 0 for gender in project.columns]

        processed_proj = project
        processed_proj.columns = encoded_names

        return processed_proj

    else:
        # raise an error if the dataset is not recognized, each requires unique parsing
        raise Exception("Dataset unknown")

def find_protein_coding(all_names):
    r_func = ro.globalenv['filter_for_coding']
    # executes the same protocol as the biomaRt function
    r_vect = ro.StrVector(all_names)
    result = r_func(r_vect)
    classified_genes = pandas2ri.rpy2py(result)
    return classified_genes

def process_rna(tcga_counts, tcga_tpms, save_str, mode):

    # since no batch correction is needed for a single dataset, perform the same train/test split on both
    # the raw counts and TPMs

    tcga_targets = tcga_counts['index']
    tcga_counts.drop('index', axis=1, inplace=True)
    tcga_tpms.drop('index', axis=1, inplace=True)

    counts_train, counts_test, counts_train_target, counts_test_target = train_test_split(tcga_counts, tcga_targets, test_size=0.3,
                                                                      random_state=3, stratify=tcga_targets)

    tpm_train, tpm_test, tpm_train_target, tpm_test_target = train_test_split(tcga_tpms, tcga_targets,
                                                                                          test_size=0.3,
                                                                                          random_state=3,
                                                                                          stratify=tcga_targets)

    # make sure features are the same between counts and TPMs
    assert(tpm_train_target.equals(counts_train_target))

    # TEMPORARY FIX


    if mode == 't-nt':
        # construct metadata matrix for DGEA
        meta_df = counts_train_target.to_frame().rename(columns={'index': 'Condition'})
    else:
        #meta_df = counts_train[['gender', 'disease state']]
        #counts_train.drop('gender', axis=1, inplace=True)
        #counts_train.drop('disease state', axis=1, inplace=True)
        #meta_df.rename(columns={'gender': 'Gender', 'disease state': 'Condition'}, inplace=True)
        counts_train = counts_train[counts_train['gender'] == 1]
        print(counts_train['disease state'].value_counts())
        input('yo')
        counts_train.drop('gender', axis=1, inplace=True)
        meta_df = counts_train[['disease state']]
        counts_train.drop('disease state', axis=1, inplace=True)
        meta_df.rename(columns={'disease state': 'Condition'}, inplace=True)


    # convert to integer - averaging duplicate genes creates decimal points in counts
    counts_4_deseq = counts_train.T.astype(int)
    counts_4_deseq.index.name = "GeneID"
    counts_4_deseq = counts_4_deseq.reset_index()

    # perform DGEA with DESeq2 and extract results
    dgea_results = deseq2(counts_4_deseq, meta_df)
    dgea_results.to_csv("./dgea_results//" + mode + "_" + save_str + "_rna_dgea_female_results.csv")

    degs = dgea_results.index.to_list()

    if mode == "m-f":

        deg_dgea = find_pathways(degs)
        deg_dgea.to_csv('./Enrichment Analysis//' + mode + "_" + save_str + "_deg_pathway_analysis_female.csv")
        print(deg_dgea)
        input('yo')

    # Limit TPMs to differentially expressed genes and perform log2 transformation with a pseudocount of 0.1
    degs_tpm_train = np.log2(tpm_train[degs]+.1)
    degs_tpm_test = np.log2(tpm_test[degs]+.1)


    print("Processed RNA-seq training set for", mode, "analysis:")
    # save the training and testing subsets of the normalized, merged RNA-seq expressions
    degs_tpm_train.to_csv("Post-processing/" + mode + "_" + save_str + '_rna_exprs_train.csv', index=False)
    degs_tpm_test.to_csv("Post-processing/" + mode + "_" + save_str + '_rna_exprs_test.csv', index=False)

    # save target variables
    tpm_train_target.to_csv("Post-processing/" + mode + "_" + save_str + '_rna_targets_train.csv', index=False)
    tpm_test_target.to_csv("Post-processing/" + mode + "_" + save_str + '_rna_targets_test.csv', index=False)

    # get and save external validation sets
    ex_list = get_rna_ex(mode)

    # targets have already been saved
    for i in range(len(ex_list)):

        rna_ex = ex_list[i]

        rna_ex.to_csv('Post-processing/' + mode + '_rna_expr_external_' + str(i + 1) + '.csv',
                             index=False)

    return

def get_rna_ex(task):


    ex_exprs = []


    if task == 't-nt':

        # load external validation sets for the RNA analysis in TPM format

        # GSE188715 is already in TPM format

        # Log transform it to be consistent with training set normalization
        rna_ex1 = pd.read_csv(r"external_datasets/GSE188715_tpm.txt", sep='\t').set_index('ID_REF').T
        log_rna_ex1 = np.log2(rna_ex1 + .1).reset_index()

        # encode samples - sample IDs starting with X are control and those starting with tpm are cancer
        ex_target1 = pd.DataFrame(
            [1 if sample_id.startswith('tpm') else 0 if sample_id.startswith('X') else sample_id for sample_id in
             log_rna_ex1['index']], columns=['index'])

        ex_target1.to_csv('Post-processing/t-nt_rna_targets_external_1.csv', index=False)

        # drop target from df
        ex_exprs1 = log_rna_ex1.drop('index', axis=1)
        ex_exprs.append(ex_exprs1)

        # GSE236932 can also be obtained in TPM format from GEO
        rna_ex2 = pd.read_csv(r"BCa_Validation/GSE236932_norm_counts_TPM_GRCh38.p13_NCBI.tsv", sep="\t").set_index(
            'GeneID')
        rna_ex2 = np.log2(rna_ex2 + .1)
        # convert Gene IDs to HGNC symbol using the provided annotation file
        rna_ex2_annot = pd.read_csv(r"Human.GRCh38.p13.annot.tsv", sep="\t").set_index('GeneID')
        probe2symbol_2 = dict(zip(rna_ex2_annot.index, rna_ex2_annot['Symbol']))
        rna_ex2 = rna_ex2.rename(index=probe2symbol_2)
        hgnc_matches = set(probe2symbol_2.values())

        # take the average of probes corresponding to the same gene symbol
        rna_ex2 = rna_ex2.groupby(rna_ex2.index).mean()

        # ensure dataframe is just HGNC symbols
        rna_ex2 = rna_ex2[rna_ex2.index.isin(hgnc_matches)]

        # convert sample IDs to tissue type then transpose
        sample_key_2 = pd.read_csv(r"BCa_Validation/GSE236932_sample_key.txt", sep="\t")
        sample_key_2 = dict(zip(sample_key_2['Sample ID'], sample_key_2['Sample Type']))
        rna_ex2 = rna_ex2.rename(columns=sample_key_2).T.reset_index()

        # map tissue type to cancer binary
        ex_target2 = pd.DataFrame(
            [1 if name.find('bladder') != -1 else 0 if (name.find('normal') != -1 or name.find('para') != -1) else name
             for name in
             rna_ex2['index']], columns=['index'])

        ex_target2.to_csv('Post-processing/t-nt_rna_targets_external_2.csv', index=False)

        # drop target variable and add to list
        ex_exprs2 = rna_ex2.drop('index', axis=1)
        ex_exprs.append(ex_exprs2)

    else:

        rna_ex = pd.read_csv(r"BCa_Validation/GSE224248_norm_counts_TPM_GRCh38.p13_NCBI.tsv", sep="\t").set_index(
            'GeneID')
        rna_ex = np.log2(rna_ex + .1)
        rna_ex_annot = pd.read_csv(r"Human.GRCh38.p13.annot.tsv", sep="\t").set_index('GeneID')
        probe2symbol = dict(zip(rna_ex_annot.index, rna_ex_annot['Symbol']))
        rna_ex = rna_ex.rename(index=probe2symbol)
        symbol_matches = set(probe2symbol.values())
        rna_ex = rna_ex.groupby(rna_ex.index).mean()
        rna_ex = rna_ex[rna_ex.index.isin(symbol_matches)]
        # sample key
        sample_key = pd.read_csv(r"BCa_Validation/GSE224248_sample_key.txt", sep="\t", header=None, on_bad_lines='warn')
        sample_key.iloc[:, 1] = [0 if sample == 'female' else 1 for sample in sample_key.iloc[:, 1]]
        sample_convert = dict(zip(sample_key.iloc[:, 0], sample_key.iloc[:, 1]))

        rna_ex = rna_ex.rename(columns=sample_convert).T.reset_index()
        ex_target = rna_ex['index']

        ex_target.to_csv('Post-processing/m-f_rna_targets_external_1.csv', index=False)

        ex_exprs = [rna_ex.drop('index', axis=1)]

    return ex_exprs


def process_microarray(microarray_expr, mode):

    # if working with the male/female task, the external validation set must be imported beforehand
    # the feature counts are not the same between the training and validation sets,
    # so each must be limited to intersecting genes
    if mode == 'm-f':
        microarray_exes = get_mf_microarray_ex()
    else:
        microarray_exes = get_tnt_microarray_ex()




    # take in transposed microarray gene expressions, with columns as genes besides the 'index' column which has the
    # cancer binary

    # save target variable as a series
    microarray_target = microarray_expr['index']
    # drop the target from the expression dataframe
    microarray_expr.drop('index', axis=1, inplace=True)

    # first generate training and internal testing sets

    microarray_train, microarray_test, target_train, target_test = train_test_split(microarray_expr,
                                                                                    microarray_target,
                                                                                    test_size=0.3,
                                                                                    random_state=3,
                                                                                    stratify=microarray_target)

    assert len(microarray_expr) == (len(microarray_train) + len(microarray_test))

    # quantile normalization to make distribution of each probe the same
    qn = QuantileNorm()
    qn.fit(microarray_train)

    # log transformation to reduce the variance of expressions, with pseudocount to adjust for null expressions
    norm_microarray_train = np.log2(qn.transform(microarray_train) + 1)
    norm_microarray_test = np.log2(qn.transform(microarray_test) + 1)

    # z-score for consistency in feature selection but don't use for DGEA
    z_transform = StandardScaler()
    z_transform.fit(norm_microarray_train)
    microarray_train_final_all = pd.DataFrame(z_transform.transform(norm_microarray_train), index=norm_microarray_train.index, columns=norm_microarray_train.columns)
    microarray_test_final_all = pd.DataFrame(z_transform.transform(norm_microarray_test), index=norm_microarray_test.index, columns=norm_microarray_test.columns)

    # perform DGEA on the training
    dgea_results = limma_dge(norm_microarray_train, target_train)
    dgea_results.to_csv("./dgea_results//" + mode + "_microarray_dgea_results.csv", index=True)
    degs = dgea_results.index.to_list()

    # limit to DEGs, then perform z-score normalization again
    microarray_train_deg = norm_microarray_train[degs]
    microarray_test_deg = norm_microarray_test[degs]

    z_transform = StandardScaler()
    z_transform.fit(microarray_train_deg)
    microarray_train_final = pd.DataFrame(z_transform.transform(microarray_train_deg), index=microarray_train_deg.index, columns=microarray_train_deg.columns)
    microarray_test_final = pd.DataFrame(z_transform.transform(microarray_test_deg), index=microarray_test_deg.index, columns=microarray_test_deg.columns)

    if len(degs) <= 100:

        microarray_train_final_all.to_csv('Post-processing/' + mode + '_microarray_exprs_train_0.csv', index=False)
        microarray_test_final_all.to_csv('Post-processing/' + mode + '_microarray_exprs_test_0.csv', index=False)
        microarray_train_final.to_csv('Post-selection/' + mode + '_dge_microarray_exprs_train_0.csv', index=False)
        microarray_test_final.to_csv('Post-selection/' + mode + '_dge_microarray_exprs_test_0.csv', index=False)

    else:

        microarray_train_final.to_csv('Post-processing/' + mode + '_microarray_exprs_train_0.csv', index=False)
        microarray_test_final.to_csv('Post-processing/' + mode + '_microarray_exprs_test_0.csv', index=False)

    # fit quantile normalizer to the training data then use it transform both the training and internal test data

    for i in range(len(microarray_exes)):

        microarray_ex = microarray_exes[i]

        # limit training set and external validation set to the same genes, don't want to train on features not appearing
        # in the evaluated dataset
        overlap = norm_microarray_train.columns.intersection(microarray_ex.columns)
        microarray_train = microarray_train[overlap]
        microarray_test = microarray_test[overlap]
        microarray_ex = microarray_ex[overlap]

        # normalize training set limited to overlapping genes with external validation set
        qn = QuantileNorm()
        qn.fit(microarray_train)
        norm_microarray_train = np.log2(qn.transform(microarray_train) + 1)
        norm_microarray_test = np.log2(qn.transform(microarray_test) + 1)

        # z-score for consistency in feature selection but don't use for DGEA
        z_transform = StandardScaler()
        z_transform.fit(norm_microarray_train)
        microarray_train_final_all = pd.DataFrame(z_transform.transform(norm_microarray_train), index=norm_microarray_train.index, columns=norm_microarray_train.columns)
        microarray_test_final_all = pd.DataFrame(z_transform.transform(norm_microarray_test), index=norm_microarray_test.index, columns=norm_microarray_test.columns)

        # perform DGEA and limit to train/test to DEGs
        dgea_results = limma_dge(norm_microarray_train, target_train)
        degs = dgea_results.index.to_list()

        microarray_train_deg = norm_microarray_train[degs]
        microarray_test_deg = norm_microarray_test[degs]

        # z-score
        z_transform = StandardScaler()
        z_transform.fit(microarray_train_deg)
        microarray_train_final = pd.DataFrame(z_transform.transform(microarray_train_deg), index=microarray_train_deg.index, columns=microarray_train_deg.columns)
        microarray_test_final = pd.DataFrame(z_transform.transform(microarray_test_deg), index=microarray_test_deg.index, columns=microarray_test_deg.columns)


        # save as CSV

        if len(degs) <= 100:

            microarray_train_final_all.to_csv('Post-processing/' + mode + '_microarray_exprs_train_' + str(i+1) + '.csv', index=False)
            microarray_test_final_all.to_csv('Post-processing/' + mode + '_microarray_exprs_test_' + str(i+1) + '.csv', index=False)
            microarray_train_final.to_csv('Post-selection/' + mode + '_dge_microarray_exprs_train_' + str(i+1) + '.csv', index=False)
            microarray_test_final.to_csv('Post-selection/' + mode + '_dge_microarray_exprs_test_' + str(i+1) + '.csv', index=False)

        else:

            microarray_train_final.to_csv('Post-processing/' + mode + '_microarray_exprs_train_' + str(i+1) + '.csv', index=False)
            microarray_test_final.to_csv('Post-processing/' + mode + '_microarray_exprs_test_' + str(i+1) + '.csv', index=False)

        # make sure training and external validation features are the same
        assert list(microarray_ex.columns) == list(norm_microarray_train.columns)

        # transform the external test with quantile normalization and log transformation from the training data

        microarray_ex = np.log2(qn.transform(microarray_ex) + 1)

        # regardless of platform, apply z-score with means/SDs from the DEG-limited training set
        microarray_ex = pd.DataFrame(z_transform.transform(microarray_ex[degs]), index=microarray_ex[degs].index, columns=microarray_ex[degs].columns)

        microarray_ex.to_csv('Post-processing/' + mode + '_microarray_expr_external_' + str(i+1) + '.csv', index=False)

    target_train.to_csv('Post-processing/' + mode + '_microarray_targets_train.csv', index=False)
    target_test.to_csv('Post-processing/' + mode + '_microarray_targets_test.csv', index=False)

    return


def get_mf_microarray_ex():


    ex_list = []


    # GSE48276 - Illumina but not same as GSE13507


    # read in CSV of un-normalized microarray data and clean it up to only get MDACC cohort
    microarray_ex1 = pd.read_csv(r"BCa_Validation/GSE48276_non-normalized.txt", sep="\t", skiprows=2).set_index(
        'ID_REF')
    no_pvals1 = [col for col in microarray_ex1.columns if col.find('Pval') == -1]
    final_cols1 = [col for col in no_pvals1 if col.find('MDA') != -1]
    microarray_ex1 = microarray_ex1[final_cols1]
    # use conversion table from the platform NCBI GEO entry to create a conversion dictionary for probe names to gene symbol
    illum_key1 = pd.read_csv(r"BCa_Validation/GPL14951-11332.txt", sep="\t", skiprows=28)[['ID', 'Symbol']]
    illum_convert1 = dict(zip(illum_key1['ID'], illum_key1['Symbol']))
    converted1 = illum_convert1.values()
    microarray_ex1 = microarray_ex1.rename(index=illum_convert1)
    # average probes corresponding to the same gene
    microarray_ex1 = microarray_ex1.groupby(microarray_ex1.index).mean()
    microarray_ex1 = microarray_ex1[microarray_ex1.index.isin(converted1)]
    # gender binary conversion using manually created key which is then utilized as a conversion dictionary
    sample_key1 = pd.read_csv(r"BCa_Validation/GSE48276_sample_key.txt", sep="\t", header=None)
    sample_key1.iloc[:, 1] = [0 if sample == 'F' else 1 for sample in sample_key1.iloc[:, 1]]
    sample_convert1 = dict(zip(sample_key1.iloc[:, 0], sample_key1.iloc[:, 1]))
    microarray_ex1 = microarray_ex1[sample_convert1.keys()]
    microarray_ex1 = microarray_ex1.rename(columns=sample_convert1).T.reset_index()
    # extract target variable and save it, then drop from expression matrix
    target1 = microarray_ex1['index']
    target1.to_csv('Post-processing/m-f_microarray_targets_external_1.csv', index=False)
    microarray_ex1.drop('index', axis=1, inplace=True)

    ex_list.append(microarray_ex1)

    #GSE31684

    # external dataset  for m/f
    microarray_ex2 = get_affy(r"C:\Users\joepi\Code\Thesis\Python\BCa_ML_DL\BCa_Validation\GSE31684_RAW", "hgu133plus2")
    probe_key2 = pd.read_csv(r"C:\Users\joepi\Code\Thesis\Python\BCa_ML_DL\BCa_Validation\GPL570-55999.txt", sep="\t", on_bad_lines='skip',
                            header=16)[['ID', 'Gene Symbol']]
    # map probe IDs to gene symbols
    probe_key2 = dict(zip(probe_key2['ID'], probe_key2['Gene Symbol']))
    microarray_ex2.rename(index=probe_key2, inplace=True)
    microarray_ex2 = microarray_ex2[microarray_ex2.index.notnull()]
    converted2 = set(probe_key2.values())
    microarray_ex2 = microarray_ex2[microarray_ex2.index.isin(converted2)]
    # account for multiple symbols in one index value, only use the last one
    microarray_ex2.index = microarray_ex2.index.str.strip().str.split('///').str[-1].str.strip()
    microarray_ex2 = microarray_ex2.groupby(microarray_ex2.index).mean()
    # change sample names
    microarray_ex2.rename(columns=lambda x: str(x)[:-10], inplace=True)
    # map sample names to gender
    sample_key2 = pd.read_csv(r"C:\Users\joepi\Code\Thesis\Python\BCa_ML_DL\BCa_Validation\GSE31684_table_of_clinical_details.txt", sep="\t")
    sample_key2 = sample_key2[['GEO', 'Gender']]
    sample_key2 = dict(zip(sample_key2['GEO'], sample_key2['Gender']))
    microarray_ex2.rename(columns=sample_key2, inplace=True)
    microarray_ex2.columns = [1 if col == "male" else 0 for col in microarray_ex2.columns]
    microarray_ex2 = microarray_ex2.T.reset_index()
    target2 = microarray_ex2['index']
    target2.to_csv('Post-processing/m-f_microarray_targets_external_2.csv', index=False)
    microarray_ex2.drop('index', axis=1, inplace=True)

    ex_list.append(microarray_ex2)

    return ex_list

def get_tnt_microarray_ex():

    ex_list = []

    # GSE37815 - same platform as GSE13507

    # process the external test set here too, as the rank means from the internal training set will be needed
    # first get the unnormalized data itself
    microarray_ex1 = pd.read_csv(r"BCa_Validation\GSE37815_non-normalized.txt", sep="\t").set_index(
        'GEO ACCESSIONS').drop('PROBE_ID', axis=0).T.astype(float)
    microarray_ex_key1 = pd.read_csv(r"BCa_Validation/GPL6102_Illumina_HumanWG-6_V2_0_R1_11223189_A.bgx", sep="\t", header=8)[
        ['Probe_Id', 'Symbol']].dropna(axis=0)
    convert_key1 = dict(zip(microarray_ex_key1['Probe_Id'], microarray_ex_key1['Symbol']))
    microarray_ex1 = microarray_ex1.rename(columns=convert_key1)[list(convert_key1.values())]
    microarray_ex1 = microarray_ex1.groupby(level=0, axis=1).mean()
    # convert sample IDs to cancer binary
    sample_key1 = pd.read_csv("BCa_Validation/GSE37815_sample_key.txt", sep="\t", header=None)
    sample_key1.iloc[:, 1] = [0 if sample.find('Control') != -1 else 1 if sample.find('cancer') != -1 else sample for
                             sample in sample_key1.iloc[:, 1]]
    sample_key1 = dict(zip(sample_key1.iloc[:, 0], sample_key1.iloc[:, 1]))
    microarray_ex1 = microarray_ex1.rename(index=sample_key1).reset_index()
    microarray_ex1['index'].to_csv('Post-processing/t-nt_microarray_targets_external_1.csv', index=False)
    microarray_ex1.drop('index', axis=1, inplace=True)

    ex_list.append(microarray_ex1)

    # GSE31189 - Affymetrix

    microarray_ex2 = get_affy(r"C:\Users\joepi\Code\Thesis\Python\BCa_ML_DL\BCa_Validation\GSE31189_RAW", "hgu133plus2")
    probe_key2 = pd.read_csv(r"C:\Users\joepi\Code\Thesis\Python\BCa_ML_DL\BCa_Validation\GPL570-55999.txt", sep="\t", on_bad_lines='skip',
                            header=16)[['ID', 'Gene Symbol']]

    # map probe IDs to gene symbols, only retain successful conversions
    probe_key2 = dict(zip(probe_key2['ID'], probe_key2['Gene Symbol']))
    microarray_ex2.rename(index=probe_key2, inplace=True)
    microarray_ex2 = microarray_ex2[microarray_ex2.index.notnull()]
    converted2 = set(probe_key2.values())
    microarray_ex2 = microarray_ex2[microarray_ex2.index.isin(converted2)]

    # account for multiple symbols in one index value, only use the last one
    microarray_ex2.index = microarray_ex2.index.str.strip().str.split('///').str[-1].str.strip()
    microarray_ex2 = microarray_ex2.groupby(microarray_ex2.index).mean()

    # load sample key to convert sample ID to tissue type
    sample_key2 = pd.read_csv(r"C:\Users\joepi\Code\Thesis\Python\BCa_ML_DL\BCa_Validation\GSE31189_sample_key.txt", sep="\t")
    sample_key2 = dict(zip(sample_key2['Sample_ID'], sample_key2['Tissue_Type']))

    # change CEL filename to only sample ID
    microarray_ex2.columns = microarray_ex2.columns.str.split('.').str[0]
    microarray_ex2 = microarray_ex2.rename(columns=sample_key2)
    microarray_ex2.columns = [0 if str(sample).find('Non-Cancer') != -1 else 1 for sample in microarray_ex2.columns]
    microarray_ex2 = microarray_ex2.T.reset_index()

    microarray_ex2['index'].to_csv('Post-processing/t-nt_microarray_targets_external_2.csv', index=False)
    microarray_ex2.drop('index', axis=1, inplace=True)

    ex_list.append(microarray_ex2)

    #GSE3167

    path_to_cel = r"C:\Users\joepi\Code\Thesis\Python\BCa_ML_DL\BCa_Validation\GSE3167_RAW"

    microarray_ex3 = get_affy(path_to_cel, "hgu133a")

    probe_key3 = pd.read_csv(r"C:\Users\joepi\Code\Thesis\Python\BCa_ML_DL\BCa_Train_Affy_Keys\GPL96-57554.txt",
                            sep="\t", on_bad_lines='skip', header=16)

    probe_key3 = dict(zip(probe_key3['ID'], probe_key3['Gene Symbol']))
    microarray_ex3.rename(index=probe_key3, inplace=True)
    microarray_ex3 = microarray_ex3[microarray_ex3.index.notnull()]
    converted3 = set(probe_key3.values())
    microarray_ex3 = microarray_ex3[microarray_ex3.index.isin(converted3)]
    # account for multiple symbols in one index value, only use the last one
    microarray_ex3.index = microarray_ex3.index.str.strip().str.split('///').str[-1].str.strip()
    microarray_ex3 = microarray_ex3.groupby(microarray_ex3.index).mean()

    # convert sample names to tumor/non-tumor
    sample_key3 = pd.read_csv(r"C:\Users\joepi\Code\Thesis\Python\BCa_ML_DL\BCa_Train_Affy_Keys\GSE3167_sample_key.txt",
                             sep="\t")
    sample_key3 = dict(zip(sample_key3['Sample_ID'], sample_key3['Tissue_Type']))
    microarray_ex3.rename(columns=lambda x: str(x)[:-10], inplace=True)
    microarray_ex3.rename(columns=sample_key3, inplace=True)
    microarray_ex3.columns = [0 if col.find("Normal") != -1 else 1 if col.find("C") else col for col in
                                 microarray_ex3.columns]

    microarray_ex3 = microarray_ex3.drop(microarray_ex3.filter(regex='C').columns, axis=1).T.astype(
        float).reset_index()

    microarray_ex3['index'].to_csv('Post-processing/t-nt_microarray_targets_external_3.csv', index=False)

    microarray_ex3.drop('index', axis=1, inplace=True)

    ex_list.append(microarray_ex3)

    return ex_list

def limma_dge(expr_df, labels):
    # access deseq2 function
    r_func = ro.globalenv['dge']
    # convert python dataframe to R dataframe
    # make sure the labels input is a list
    labels_r = ro.IntVector(labels.to_list())
    expr_df_r = pandas2ri.py2rpy(expr_df.copy())
    dge_result = r_func(expr_df_r, labels_r)
    dge_df = pandas2ri.rpy2py(dge_result)
    # threshold is adjusted p value less than 0.05 and |logFC| > 1.0, sort by lowest adjusted p values first
    dge_table = dge_df[(dge_df['adj.P.Val'] < 0.05) & (dge_df['logFC'].abs() > 1.0)].sort_values(by='adj.P.Val', ascending=True)
    dge_genes = dge_table.index.to_list()
    print(dge_df[(dge_df['adj.P.Val'] < 0.05) & (dge_df['logFC'].abs() > 1.0)])
    return dge_table


    def id_2_hgnc_raw(df, d, lengths=True):
    # lengths binary variable determines if sequence lengths must be extracted for TPM calculation
    df_cols = df.index.unique()
    if lengths:
        sequence_lengths = pd.DataFrame()
    # extract the names that are UCSC coded and sort them
    uc_list = df_cols[df_cols.str.startswith('uc')].sort_values().to_list()
    if len(uc_list) > 0:
        # since the UCSC IDs in GSE55433 do not end with a number, we must test multiple possibilities
        # biomaRt will not find matches unless the UCSC ID ends with a number
        if any([x[-1].isdigit() for x in uc_list]):
            ucs = [[uc_list]]
        else:
            print("UCSC IDs have no version number, testing all possibilities...")
            ucs = [[x + '.1' for x in uc_list], [x + '.2' for x in uc_list], [x + '.3' for x in uc_list]]
        uc_converted = []
        for version in ucs:
            # use biomaRt helper function to create a map to replace UCSC IDs with HGNC symbols
            uc_replace, uc_lengths = ensembl_biomart_raw(version, "ucsc", lengths=lengths)
            # the function also only extracts HGNC symbols with an associated sequence length
            # add the info to the cumulative dataframe
            if lengths:
                sequence_lengths = pd.concat([sequence_lengths, uc_lengths], axis=0)
            # use the mapping dictionary to rename the columns
            df.rename(index=uc_replace, inplace=True)
            # track which column values in the updated dataframe have been converted
            uc_converted.append(list(uc_replace.values()))
        # merge the nested list of converted IDs
        uc_converted = list(itertools.chain(*uc_converted))
    else:
        print("There were no UCSC IDs in", d)
        uc_converted = []
        uc_replace = {}

    # use regex to identify the columns that are GenBank accession IDs
    gb_pattern = r"^[a-zA-Z]{2}\d{6}$"
    regx = re.compile(gb_pattern)
    # at this point, all UCSC IDs with a match were replaced with their HGNC symbol
    uc_replaced_cols = df.index.unique().to_list()
    gb_list = [x for x in uc_replaced_cols if regx.match(x)]
    # remove GB accessions from uc_converted column names list
    uc_converted = [x for x in uc_converted if x not in gb_list]

    if len(gb_list) > 0:
        # use another helper function to create a mapping dictionary for GenBank accessions to HGNC symbols
        gb_replace, gb_lengths = gb_org_Hs_eg_db_raw(gb_list)
        if lengths:
            sequence_lengths = pd.concat([sequence_lengths, gb_lengths], axis=0)
        # use the mapping dictionary to replace GB accessions with HGNC symbols
        df.rename(index=gb_replace, inplace=True)
        # track which columns in the updated dataframe have been converted
        gb_converted = list(gb_replace.values())

    else:
        print("There were no GenBank Accession IDs in", d)
        gb_converted = []
        gb_replace = {}

    # use a different regex pattern to identify ENSEMBL IDs
    ensembl_pattern = r"^[a-zA-Z]{4}\d{11}$"
    regx2 = re.compile(ensembl_pattern)
    # at this point, all possible UCSC IDs and GenBank accessions have been mapped
    uc_gb_replaced_cols = df.index.unique().to_list()
    ensembl_list = [x for x in uc_gb_replaced_cols if regx2.match(x)]
    # make sure there are no ENSEMBL IDs in the other conversion lists
    uc_converted = [x for x in uc_converted if x not in ensembl_list]
    gb_converted = [x for x in gb_converted if x not in ensembl_list]

    if len(ensembl_list) > 0:
        # use the biomaRt helper function to find all HGNC symbol matches for the ENSEMBL IDs
        ensembl_replace, ens_lengths = ensembl_biomart_raw(ensembl_list, "ensembl_gene_id", lengths=lengths)
        # add the gene types of each HGNC symbol to the cumulative dataframe
        if lengths:
            sequence_lengths = pd.concat([sequence_lengths, ens_lengths], axis=0)
        # use the mapping dictionary to replace ENSEMBL IDs with HGNC symbols
        df.rename(index=ensembl_replace, inplace=True)
        # keep track of what columns in the dataframe have been converted
        ensembl_converted = list(ensembl_replace.values())
    else:
        print("There were no ENSEMBL IDs in", d)
        ensembl_converted = []
        ensembl_replace = {}

    # combine the converted column names for each type of ID (UCSC, GenBank, ENSEMBL)
    # identify the column names that WERE NOT converted and drop them from the frame
    converted_cols = uc_converted + gb_converted + ensembl_converted
    if len(converted_cols) > 0:
        df = df.loc[converted_cols]
    else:
        print("No IDs were detected, assuming that HGNC gene symbols are already present...")

    # this new dataframe will still have repeated column names due to two factors:
    # 1. Duplicate probes in the RNAseq data
    # 2. Multiple IDs corresponding to the same gene name
    # the initial strategy is to take the mean of duplicate columns
    df = df.groupby(df.index).mean()

    if lengths:
        sequence_lengths = sequence_lengths.set_index('hgnc_symbol')
        sequence_lengths = sequence_lengths.groupby(sequence_lengths.index).mean()
        df = pd.concat([df, sequence_lengths], axis=1, ignore_index=False)

    return df


def get_tpm(df, length_series):
    # this function will get transcript per million values for raw counts
    # assumes that gene names are the index and column names are the sample names
    converted_w_length = pd.concat([df, length_series], axis=1)
    # first, convert sequence length from bases to kilobases
    converted_w_length['transcript_length'] = converted_w_length['transcript_length'] / 1000
    # convert raw counts to reads per kilobase by dividing by the transcript length
    converted_rpk = converted_w_length.div(converted_w_length['transcript_length'], axis=0)
    # remove sequence lengths from the dataframe
    converted_rpk.drop('transcript_length', axis=1, inplace=True)
    # sum rpk values for each sample - represents total gene expression
    rpk_sums_sample = converted_rpk.sum()
    # divide the rpk values in each sample by the sum and multiply by 1 million to get TPM
    converted_tpm = (converted_rpk / rpk_sums_sample) * 1000000

    return converted_tpm


def ensembl_biomart_raw(ens_list, scope, lengths=True):
    r_func = ro.globalenv['convert_ensembl']
    # convert the list of IDs to an R string vector
    r_vect = ro.StrVector(ens_list)
    # input the R string vector of IDs into the biomaRt function to generate an R dataframe of IDs with HGNC symbols
    # The function now includes a third column corresponding to the molecule type of the HGNC symbol (ncRNA, pseudogene)
    while True:
        try:
            r_convert_table = r_func(r_vect, scope, lengths)
        except RRuntimeError:
            print("ENSEMBL query unsuccessful, trying again!")
        else:
            break
    # convert the R dataframe to a pandas dataframe
    r_to_pd = pandas2ri.rpy2py(r_convert_table)
    # remove the last two characters from the ID if it is UCSC
    if scope == "ucsc":
        r_to_pd[scope] = r_to_pd[scope].str[:-2]
    # convert the IDs and HGNC symbols to a list, zip them together, then convert to a dictionary for mapping
    name_map_dict = dict(zip(r_to_pd[scope].to_list(), r_to_pd["hgnc_symbol"].to_list()))
    # store the symbol and molecule type
    length_mapper = r_to_pd[["hgnc_symbol", "transcript_length"]]
    return name_map_dict, length_mapper


def gb_org_Hs_eg_db_raw(gb_list):
    # GenBank accession numbers are not included in biomaRt, so we need a different function accessing Entrez gene IDs
    r_func = ro.globalenv['convert_gb']
    # executes the same protocol as the biomaRt function
    r_vect = ro.StrVector(gb_list)
    result = r_func(r_vect)
    r_convert_table = result.rx2('mapper')
    sequence_lengths = result.rx2('result')
    r_to_pd = pandas2ri.rpy2py(r_convert_table)
    seq_lengths = pandas2ri.rpy2py(sequence_lengths)
    name_mapper = dict(zip(r_to_pd["ACCNUM"].to_list(), r_to_pd["SYMBOL"].to_list()))
    return name_mapper, seq_lengths


def limma_bg_ill(raw_intensity):
    # use R function to execute single channel background correction on the illumina expression dataset
    r_func = ro.globalenv['illumina_background']
    # convert DataFrame to data.frame
    raw_intensity = pandas2ri.py2rpy(raw_intensity.copy())
    # input and convert background-corrected output back to DataFrame
    bg_adj_expr_r = r_func(raw_intensity)
    bg_adj_expr = pandas2ri.rpy2py(bg_adj_expr_r)
    return bg_adj_expr


def limma_bg_agi(sample_list, sample_labels):
    # use R function to execute single channel background correction on the illumina expression dataset
    r_func = ro.globalenv['agilent_background']
    # convert each DataFrame corresponding to each sample to a data.frame
    sample_fg_bg_list = [pandas2ri.py2rpy(sample) for sample in sample_list]
    # convert sample name list to an R string vector
    sample_labels = ro.StrVector(sample_labels)
    # input, then convert background-corrected output to DataFrame
    bg_adj_expr_r = r_func(sample_fg_bg_list, sample_labels)
    bg_adj_expr = pandas2ri.rpy2py(bg_adj_expr_r)
    return bg_adj_expr


def limma_batch(expr_df, meta_df):
    # access ComBat function
    r_func = ro.globalenv['limma_remove_batch']
    # convert python dataframe to R dataframe
    expr_df_r = pandas2ri.py2rpy(expr_df.copy())
    meta_df_r = pandas2ri.py2rpy(meta_df.copy())
    corrected_expr = r_func(expr_df_r, meta_df_r)
    corrected_df = pd.DataFrame(corrected_expr, index=expr_df.index, columns=expr_df.columns)
    return corrected_df


def get_affy(path_to_CELs, tech):
    r_func = ro.globalenv['process_affy']
    affy_r_df = r_func(path_to_CELs, tech)
    affy_df = pandas2ri.rpy2py(affy_r_df)
    return affy_df

    
def get_tcga_gtex_data():
    # load phenotype info
    metadata = pd.read_csv(r"C:\Users\joepi\Code\Thesis\Python\BCa_ML_DL\TcgaTargetGTEX_phenotype.txt", sep="\t").set_index('sample')
    metadata_bca = metadata[metadata['_primary_site'].str.contains('Bladder', case=False, na=False)]
    male = metadata_bca[metadata_bca['_gender'] == 'Male']
    female = metadata_bca[metadata_bca['_gender'] == 'Female']
    print(female[female['_sample_type'].str.contains('Normal', case=False, na=False)])
    print(female[~female['_sample_type'].str.contains('Normal', case=False, na=False)])
    input('yo')
    # load data - it is pan-cancer
    tcga_target_gtex = pd.read_csv(r"C:\Users\joepi\Code\Thesis\Python\BCa_ML_DL\TcgaTargetGtex_gene_expected_count", sep = "\t").set_index('sample')
    print(tcga_target_gtex)

    return

    def counts_to_tpm(mode, training, correction_method):

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
        results_1 = pd.read_csv(r".\dgea_results\\" + training + "_" + correction_method + "_dataset_dgea_results_male.csv", index_col=0)
        degs_1 = results_1.index
        results_2 = pd.read_csv(
            r".\dgea_results\\" + training + "_" + correction_method + "_dataset_dgea_results_female.csv",
            index_col=0)
        degs_2 = results_2.index
        specific_1 = list(degs_1[[gene not in degs_2 for gene in degs_1]])
        specific_2 = list(degs_2[[gene not in degs_1 for gene in degs_2]])
        # determine oppositely regulated
        overlap = set(degs_1).intersection(degs_2)
        oppositely_regulated = [gene for gene in overlap
                                if np.sign(results_1.loc[gene, "log2FoldChange"]) != np.sign(results_2.loc[gene, "log2FoldChange"])]
        # merge oppositely regulated and cohort-specific together
        specific_1 = list(set(specific_1) | {g for g in oppositely_regulated if g in degs_1})
        specific_2 = list(set(specific_2) | {g for g in oppositely_regulated if g in degs_2})

    elif mode == "disease_stratified":
        tpm_1 = tpm_norm_w_gender[tpm_norm_w_gender['Disease State'] == 1]
        tpm_2 = tpm_norm_w_gender[tpm_norm_w_gender['Disease State'] == 0]
        results_1 = pd.read_csv(
            r".\dgea_results\\" + training + "_" + correction_method + "_dataset_dgea_results_tumor.csv",
            index_col=0)
        degs_1 = results_1.index
        results_2 = pd.read_csv(
            r".\dgea_results\\" + training + "_" + correction_method + "_dataset_dgea_results_healthy.csv",
            index_col=0)
        degs_2 = results_2.index
        specific_1 = list(degs_1[[gene not in degs_2 for gene in degs_1]])
        specific_2 = list(degs_2[[gene not in degs_1 for gene in degs_2]])
        # determine oppositely regulated
        overlap = set(degs_1).intersection(degs_2)
        oppositely_regulated = [gene for gene in overlap
                                if np.sign(results_1.loc[gene, "log2FoldChange"]) != np.sign(results_2.loc[gene, "log2FoldChange"])]
        # merge oppositely regulated and cohort-specific together
        specific_1 = list(set(specific_1) | {g for g in oppositely_regulated if g in degs_1})
        specific_2 = list(set(specific_2) | {g for g in oppositely_regulated if g in degs_2})
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

    train_expr_1, test_expr_1, train_target_1, test_target_1 = train_test_split(tpm_1,
                                                                                                target_1,
                                                                                                test_size=0.3,
                                                                                                random_state=3,
                                                                                                stratify=target_1)

    train_expr_2, test_expr_2, train_target_2, test_target_2 = train_test_split(tpm_2,
                                                                                target_2,
                                                                                test_size=0.3,
                                                                                random_state=3,
                                                                                stratify=target_2)

    # limit training data to specific DEGs
    train_expr_1 = train_expr_1[specific_1]
    train_expr_2 = train_expr_2[specific_2]
    if mode == "gender_stratified":
        check_batch_effect(train_expr_1, train_target_1.to_list(), training + "_" + correction_method + "_" + 'postbatch_tpm_tnt_deg_male_')
    else: 
        check_batch_effect(train_expr_1, train_target_1.to_list(), training + "_" + correction_method + "_" + 'postbatch_tpm_mf_deg_tumor_')
    test_expr_1 = test_expr_1[specific_1]
    test_expr_2 = test_expr_2[specific_2]
    if mode == "gender_stratified":
        check_batch_effect(train_expr_2, train_target_2.to_list(), training + "_" + correction_method + "_" + 'postbatch_tpm_tnt_deg_female_')

    # if TCGA training, use SMOTE after feature selection but before normalization. Oversampling with too much
    # noise can be problematic
    #if training == "TCGA":
        #sm = SMOTEENN(random_state=3)
        #train_expr_1, train_target_1 = sm.fit_resample(train_expr_1, train_target_1)
        #train_expr_2, train_target_2 = sm.fit_resample(train_expr_2, train_target_2)
        #train_expr_1 = np.log2(train_expr_1 + 1)
        #train_expr_2 = np.log2(train_expr_2 + 1)
        #test_expr_1 = np.log2(test_expr_1 + 1)
        #test_expr_2 = np.log2(test_expr_2 + 1)


    # save to CSV for feature selection
    if mode == "gender_stratified":
        train_expr_1.to_csv("./expression_data//" + training + "_" + correction_method + "_male_train_expr.csv")
        train_expr_2.to_csv("./expression_data//" + training + "_" + correction_method + "_female_train_expr.csv")
        test_expr_1.to_csv("./expression_data//" + training + "_" + correction_method + "_male_test_expr.csv")
        test_expr_2.to_csv("./expression_data//" + training + "_" + correction_method + "_female_test_expr.csv")
        train_target_1.to_csv("./expression_data//" + training + "_" + correction_method + "_male_train_target.csv")
        train_target_2.to_csv("./expression_data//" + training + "_" + correction_method + "_female_train_target.csv")
        test_target_1.to_csv("./expression_data//" + training + "_" + correction_method + "_male_test_target.csv")
        test_target_2.to_csv("./expression_data//" + training + "_" + correction_method + "_female_test_target.csv")
    elif mode == "disease_stratified":
        train_expr_1.to_csv("./expression_data//" + training + "_" + correction_method + "_tumor_train_expr.csv")
        train_expr_2.to_csv("./expression_data//" + training + "_" + correction_method + "_healthy_train_expr.csv")
        test_expr_1.to_csv("./expression_data//" + training + "_" + correction_method + "_tumor_test_expr.csv")
        test_expr_2.to_csv("./expression_data//" + training + "_" + correction_method + "_healthy_test_expr.csv")
        train_target_1.to_csv("./expression_data//" + training + "_" + correction_method + "_tumor_train_target.csv")
        train_target_2.to_csv("./expression_data//" + training + "_" + correction_method + "_healthy_train_target.csv")
        test_target_1.to_csv("./expression_data//" + training + "_" + correction_method + "_tumor_test_target.csv")
        test_target_2.to_csv("./expression_data//" + training + "_" + correction_method + "_healthy_test_target.csv")
    else:

        raise Exception("Valid comparison not inputted...")

    return
    
'''