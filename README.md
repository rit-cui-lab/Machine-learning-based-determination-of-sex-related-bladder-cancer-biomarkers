# Bioinformatics-based Discovery of Sex-related Bladder Cancer Biomarkers 

## IMPORTANT NOTE: The TCGA data was too large to be uploaded to GitHub. In the "raw_data" directory, there needs to be a "TCGA" directory with "TCGA Control" and "TCGA Tumor" subdirectories. The control and tumor sample RNA-seq counts from TCGA-BLCA can be downloaded as TSVs from the genomic data commons and placed in the appropriate directories. The sample sheets are available in the main directory as "tumor_sample_sheet.tsv" and "control_sample_sheet.tsv."

### Execution 
1. Create the following directories: dgea_results, evaluation, expression_data, figures, and post_selection_data. Make the TCGA RNA-seq counts have been downloaded as described above and that you have downloaded the "external_datasets" folder.
2. Download the required packages using the requirements.txt file.
3. Download all scripts and the reference genome in the main directory.
4. Run execution.py to perform preprocessing, feature selection, and internal/external evaluation.
5. Observe the desired outputs. 
