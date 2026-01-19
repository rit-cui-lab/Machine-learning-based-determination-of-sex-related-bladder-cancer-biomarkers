 

# function for converting ENSEMBL IDs to their appropriate gene names

convert_ensembl <- function(feature_names, scope, lengths=TRUE){
  # import inside function call for rpy2 
  library(biomaRt) 
  
  # instantiate ENSEMBL mart for human genome
  ensembl <- useEnsembl(biomart = "genes", dataset = "hsapiens_gene_ensembl", mirror="useast")
  
  # filter results based on those with an ENSEMBL gene ID, looking only for ENSEMBL ID and HGNC symbol 
  # the list inputted in the function call is the query 
  result <- getBM(attributes = c(scope, "hgnc_symbol", "transcript_length", "gene_biotype"), values = feature_names, mart = ensembl, filter=scope)
  
  # we only want the results for IDs that have a symbol 
  if (lengths) {
  result <- result[result[,"hgnc_symbol"] != "" & result[,"transcript_length"] != "",]
  } 
  else { 
  result <- result[result[,"hgnc_symbol"] != "",]
    }
  
  return(result)
}



convert_symbol_2_protein <- function(feature_names){
  # import inside function call for rpy2 
  library(biomaRt) 
  
  # instantiate ENSEMBL mart for human genome
  ensembl <- useEnsembl(biomart = "genes", dataset = "hsapiens_gene_ensembl", mirror="useast")
  
  # filter results based on those with an ENSEMBL gene ID, looking only for ENSEMBL ID and HGNC symbol 
  # the list inputted in the function call is the query 
  result <- getBM(attributes = c('hgnc_symbol', 'uniprotswissprot'), values = feature_names, mart = ensembl, filter='hgnc_symbol')
  
  # we only want the results for IDs that have a symbol 
  
  
  return(result)

}





filter_for_coding <- function(feature_names){
  # import inside function call for rpy2 
  library(biomaRt) 
  
  # instantiate ENSEMBL mart for human genome
  ensembl <- useEnsembl(biomart = "genes", dataset = "hsapiens_gene_ensembl", mirror="useast")
  
  # filter results based on those with an ENSEMBL gene ID, looking only for ENSEMBL ID and HGNC symbol 
  # the list inputted in the function call is the query 
  result <- getBM(attributes = c("hgnc_symbol", "gene_biotype"), values = feature_names, mart = ensembl, filter="hgnc_symbol")
  
  # we only want the results for IDs that have a classification
  result <- result[result[,"hgnc_symbol"] != "" & result[,"gene_biotype"] != "",]
  
  return(result)
}


convert_gb <- function(names){ 
  # import inside function call
  library(org.Hs.eg.db)
  library(biomaRt)
  
  # select gene type and hgnc symbol based on GenBank accession #
  mapper <- select(org.Hs.eg.db, keys = names, keytype="ACCNUM", columns = "SYMBOL")
  
  # ignore all queries where there was not a symbol match
  mapper <- mapper[!is.na(mapper[,"SYMBOL"]),]
  
  # return the mapping object
  
  # now, use HGNC symbol names to get transcript lengths
  hgncs <- mapper[,"SYMBOL"]
  
  # instantiate ENSEMBL mart for human genome
  ensembl <- useEnsembl(biomart = "genes", dataset = "hsapiens_gene_ensembl", mirror="useast") 
  
  result <- getBM(attributes = c("hgnc_symbol", "transcript_length"), values = hgncs, mart = ensembl, filter="hgnc_symbol")
  
  both_results <- list(mapper = mapper, result = result)
  
  return(both_results) 
}

dge <- function(expr_data, target){ 
  
  library(limma)
  
  # first, convert the cancer/healthy integer label to a factor
  target <- factor(target, levels = c(0, 1), labels = c("healthy", "tumor"))
  
  # ensure that expression data is a matrix, transpose if needed 
  expr_data <- as.matrix(expr_data)
  if (ncol(expr_data) != length(target)) {
    expr_data <- t(expr_data)
  }
  
  # create metadata object to house gene expression and cancer status 
  metadata <- data.frame(condition = factor(target), row.names = colnames(expr_data)) 

  
  # create design matrix 
  d_matrix <- model.matrix(~ 0 + condition, data = metadata) 
  colnames(d_matrix) <- levels(metadata$condition)
  c_matrix <- makeContrasts(tumor - healthy, levels = d_matrix)
  
  
  # apply limma, looks at the mean-variance relationship to compute weights
  lin_fit <- lmFit(expr_data, d_matrix)
  lin_fit <- contrasts.fit(lin_fit, c_matrix)
  lin_fit <- eBayes(lin_fit)
  
  # get results 
  result <- topTable(lin_fit, number = Inf)
  
  return(result)
} 

batch_correct <- function(counts_data, batch_vector, group_vector, gender_vector){ 
  
  library(sva)
  library(dplyr)
  
  counts_matrix <- as.matrix(counts_data)
  
  batch_vector <- factor(batch_vector)
  
  group_vector <- factor(group_vector)
  
  gender_vector <- factor(gender_vector)
  
  covars <- data.frame(group = group_vector, gender = gender_vector)
  
  covar_matrix <- model.matrix(~ group + gender, data = covars)
  
  adjusted <- ComBat_seq(counts_matrix, batch = batch_vector, group = NULL, covar_mod = covar_matrix)
  
  adjusted <- as.data.frame(adjusted)
  
  colnames(adjusted) <- colnames(counts_data)
  
  rownames(adjusted) <- rownames(counts_data)
  
  return(adjusted)
  
  }

run_deseq2 <- function(expr_data, meta_data, str_label, volcano) {

  library("DESeq2")
  library(EnhancedVolcano)
  library(ggplot2)


  rownames(expr_data) <- expr_data$GeneID 
  expr_data$GeneID <- NULL

  # first convert condition and batch to factors in the meta dataframe 
  meta_data$Condition <- factor(meta_data$Condition, levels = c(0, 1), labels = c("healthy", "tumor")) 

  if (length(colnames(meta_data)) > 1) {

    meta_data$Batch <- as.factor(meta_data$Batch)
    dds <- DESeqDataSetFromMatrix(countData = expr_data, colData = meta_data, design = ~ Batch + Condition)

  } else { 
    
    dds <- DESeqDataSetFromMatrix(countData = expr_data, colData = meta_data, design = ~Condition)
    
    }
  
  
  
  # run DESeq2, get results, filter out genes with invalid p-val or log2FC
  dds <- DESeq(dds)
  results <- results(dds, contrast = c("Condition", "tumor", "healthy"))
  results <- results[!is.na(results$padj) & !is.na(results$log2FoldChange), ]

  # visualize results with volcano plot 
  if (volcano == 1) {

    results_df <- as.data.frame(results)
    volcano_plot(results_df, str_label)

  }
  
  # filter based on adjusted p value and log2FC
  #results <- subset(results, abs(log2FoldChange) > 1 & padj < 0.05)
  
  # return dataframe of DEGs from most to least significant 
  
  return (as.data.frame(results[order(results$padj, decreasing = FALSE), ]))
  
}

volcano_plot <- function(results_df, str_label){ 
  
  plot <- EnhancedVolcano(results_df, lab = NA, x = "log2FoldChange", y = "padj", pCutoff = 0.05, FCcutoff = 1.0, title = "", subtitle = "", legendLabels = c("NS", "log2FC", "padj", "Both"), caption = "") +
  theme(
    text = element_text(family = "Arial", size = 14),       # sets font for all text
    axis.title = element_text(family = "Arial", size = 14), # axis titles
    axis.text = element_text(family = "Arial", size = 12),  # axis labels
    legend.text = element_text(family = "Arial", size = 12),# legend
    legend.title = element_text(family = "Arial", size = 14)
  )

  panel_width_mm <- 28
  panel_width_in <- panel_width_mm / 25.4 
  ggsave(paste0("./figures/", str_label, "_volcano.tiff"), plot = plot, device = "tiff", width = panel_width_in, units = "in", dpi= 300, bg = "white")

  }

limma_remove_batch <- function(expr_df, meta_data){ 
  library("limma")
  
  # extract batch label from meta data
  batch_list <- meta_data$Batch
  
  # correct log-TPM expressions with removeBatchEffect, transpose 
  corrected <- removeBatchEffect(expr_df, batch = batch_list)
  
  return(corrected)
  
  
  }

illumina_background <- function(expr_data){
  library(limma)
  
  expr_matrix <- as.matrix(expr_data) 
  
  # Create an EListRaw object for single channel correction
  el <- new("EListRaw")
  el$E <- expr_matrix
  
  # Apply background correction
  expr_data_bg_adj <- backgroundCorrect(el, method = "normexp")
  
  # convert to data.frame
  bg_adj_df <- as.data.frame(expr_data_bg_adj$E)
  
  return(bg_adj_df)
  
}

agilent_background <- function(expr_df_list, sample_labels){
  library(limma)
  
  # lapply applies the correction function to each dataframe in the list, 
  # the output is a list of background-corrected expression values for each sample
  corrected_dfs <- lapply(expr_df_list, function(df) { 
    # identify foreground and background signals of array as well as names 
    fg <- as.matrix(df$gMedianSignal)
    bg <- as.matrix(df$gBGMedianSignal)
    probes <- df$ProbeName
    
    # create RGList object for each sample, specifying foreground (gMedian) and background (gBGMedian) signals 
    fg_bg <- list(R = fg, G = bg, genes = data.frame(ProbeName = probes)) 
    class(fg_bg) <- "RGList"
    # apply background correction 
    limma_correct <- suppressWarnings(backgroundCorrect(fg_bg, method = "normexp"))
    # extract corrected expression values into vector
    corrected_expr <- as.vector(limma_correct$R) 
    # assign vector label as probe names 
    names(corrected_expr) <- probes  
    return(corrected_expr)
    })
  
  # merge corrected values for each sample together
  corrected_df <- do.call(cbind, corrected_dfs)
  
  # assign sample labels to columns 
  colnames(corrected_df) <- sample_labels 
  
  # convert to dataframe, row names as probe names 
  corrected_df <- data.frame(corrected_df)
  
  return(corrected_df)
} 

process_affy <- function(path_to_CELs, plat){  
  
  if (plat == "HuGene-1_0-st") {
    
    library(oligo)
    library(AnnotationDbi)
    library(hugene10sttranscriptcluster.db)
    library(dplyr)
    
    # Read CEL files
    cel_files <- list.celfiles(path_to_CELs, full.names = TRUE)
    raw_affy <- read.celfiles(cel_files)
    
    # RMA normalization
    rma_normalized <- oligo::rma(raw_affy)
    normalized_exprs <- as.data.frame(exprs(rma_normalized))
    
    # Map probe IDs to gene symbols
    probes <- rownames(normalized_exprs)
    symbols <- mapIds(
      hugene10sttranscriptcluster.db,
      keys = probes,
      column = "SYMBOL",
      keytype = "PROBEID",
      multiVals = "first"
    )
    
    # Keep only probes with valid gene symbols
    valid_symbols <- !is.na(symbols)
    normalized_exprs <- normalized_exprs[valid_symbols, ]
    symbols <- symbols[valid_symbols]
    
    # Temporarily store symbols as a column (don't assign rownames yet)
    normalized_exprs$GeneSymbol <- symbols
    
    # Collapse expression values by gene symbol (mean of duplicates)
    avgd_exprs <- normalized_exprs %>%
      group_by(GeneSymbol) %>%
      summarise(across(where(is.numeric), mean), .groups = "drop") %>%
      as.data.frame()  # ensure it's a data.frame, not tibble
    
    # Move gene symbol to rownames and clean up
    rownames(avgd_exprs) <- as.character(avgd_exprs$GeneSymbol)
    avgd_exprs$GeneSymbol <- NULL
    
    # Transpose and convert to data.frame
    normalized_exprs <- as.data.frame(t(avgd_exprs))
    
    # Set column names to gene symbols (which are rownames of avgd_exprs)
    colnames(normalized_exprs) <- rownames(avgd_exprs)
    
    
  } else {
    
    library(affy)
    library(hgu133plus2cdf)
    
    affy_data <- ReadAffy(cdfname = plat, celfile.path = path_to_CELs)
    
    rma_normalized <- affy::rma(affy_data) 
    
    normalized_exprs <- as.data.frame(exprs(rma_normalized))
  
  }
  
  return(normalized_exprs)
  
}  


robust_rank_agg <- function(input){ 
  
  library(RobustRankAggreg) 
    
  all_genes <- unique(unlist(input, recursive = TRUE, use.names = FALSE))
  N <- length(all_genes)
  
  result <- aggregateRanks(input, method = "RRA", N = N)
  
  
  return(as.data.frame(result))
  
}


pathway_analysis <- function(genes_of_interest, cohort){ 
  
  library(clusterProfiler)
  library(org.Hs.eg.db)
  library(dplyr)
  library(stringr)
  library(ggplot2)
  
  # first convert gene symbols to entrez ID, compatible input with most enrichment functions 
  gene_table <- bitr(genes_of_interest, fromType = 'SYMBOL', toType = 'ENTREZID', OrgDb = org.Hs.eg.db)
  genes_of_interest <- gene_table$ENTREZID

  # Create capitalized cohort string for titles
  cohort_title <- paste0(toupper(substring(cohort, 1, 1)), substring(cohort, 2))
  
  # set dimensions for saving images 
  panel_width <- 28 / 25.4

  # kegg analysis
  
  kegg_results <- enrichKEGG(gene = genes_of_interest, organism = 'hsa', pvalueCutoff = 0.05, qvalueCutoff = 0.05)
  kegg_plot <- dotplot(kegg_results, showCategory = 10) + theme_bw(base_size = 14, base_family = "Arial") + ggtitle(paste(cohort_title, "KEGG"))
  ggsave(paste0(".\\figures\\", cohort, '_kegg_dotplot.tiff'), plot = kegg_plot, device = "tiff", width = panel_width, units = "in", dpi = 300, bg = "white")
  
  kegg_results <- as.data.frame(kegg_results)
  
  # GO biological process analysis 
  
  go_results <- enrichGO(gene = genes_of_interest, OrgDb = org.Hs.eg.db, ont="BP", pvalueCutoff = 0.05, qvalueCutoff = 0.05)
  go_plot <- dotplot(go_results, showCategory = 10) + theme_bw(base_size = 14, base_family = "Arial") + ggtitle(paste(cohort_title, "GO BP"))
  ggsave(paste0(".\\figures\\", cohort, '_go_dotplot.tiff'), plot = go_plot, device = "tiff", width = panel_width, units = "in", dpi = 300, bg = "white")
  
  go_results <- as.data.frame(go_results)
  
  kegg_results$geneID <- as.character(kegg_results$geneID)
  go_results$geneID <- as.character(go_results$geneID)
  kegg_entrez_ids <- unique(unlist(strsplit(kegg_results$geneID, "/")))
  go_entrez_ids <- unique(unlist(strsplit(go_results$geneID, "/")))
  
  kegg_symbols <- bitr(kegg_entrez_ids, fromType = 'ENTREZID', toType = 'SYMBOL', OrgDb = org.Hs.eg.db)
  go_symbols <- bitr(go_entrez_ids, fromType = 'ENTREZID', toType = 'SYMBOL', OrgDb = org.Hs.eg.db)
  kegg_id_to_symbol <- setNames(kegg_symbols$SYMBOL, kegg_symbols$ENTREZID)
  go_id_to_symbol <- setNames(go_symbols$SYMBOL, go_symbols$ENTREZID)
  
  kegg_results$geneID <- sapply(kegg_results$geneID, function(x) { 
    paste(kegg_id_to_symbol[strsplit(x, "/")[[1]]], collapse = "/")
  }) 
  
  go_results$geneID <- sapply(go_results$geneID, function(x) { 
    paste(go_id_to_symbol[strsplit(x, "/")[[1]]], collapse = "/")
  })
  
  return(list(kegg = kegg_results, go = go_results))
  
  } 


gsea_analysis <- function(gene_scores, cohort){ 
  suppressMessages({
    library(clusterProfiler)
    library(org.Hs.eg.db)
    library(dplyr)
    library(stringr)
    library(ggplot2)
  })
  
  # gene_scores: named numeric vector, names = SYMBOL, values = -log10(RRA Score)
  
  # convert SYMBOL -> ENTREZID
  gene_table <- bitr(names(gene_scores), fromType = 'SYMBOL', toType = 'ENTREZID', OrgDb = org.Hs.eg.db)
  
  # Keep only mapped genes
  gene_scores <- gene_scores[gene_table$SYMBOL]
  names(gene_scores) <- gene_table$ENTREZID
  
  # Remove duplicates and NAs
  gene_scores <- gene_scores[!duplicated(names(gene_scores))]
  gene_scores <- gene_scores[!is.na(names(gene_scores))]
  
  # Sort decreasing
  gene_scores <- sort(gene_scores, decreasing = TRUE)

  print(length(gene_scores))
  
  # Skip GSEA if too few genes
  if(length(gene_scores) < 10){
    warning("Not enough mapped genes for GSEA. Returning empty tables.")
    return(list(kegg = data.frame(), go = data.frame()))
  }

  # Create capitalized cohort string for titles
  cohort_title <- paste0(toupper(substring(cohort, 1, 1)), substring(cohort, 2))

  # KEGG GSEA
  kegg_results <- suppressMessages(
    gseKEGG(geneList = gene_scores, organism = 'hsa', pvalueCutoff = 0.2)
  )
  if(length(kegg_results) > 0){
    kegg_plot <- suppressMessages(dotplot(kegg_results, showCategory = 10) + theme_bw(base_size = 14) + ggtitle(paste(cohort_title, "KEGG"))) 
    ggsave(paste0(".\\figures\\", cohort, '_kegg_dotplot.png'), plot = kegg_plot, width = 8, height = 6, dpi = 300)
  }
  
  # GO GSEA
  go_results <- suppressMessages(
    gseGO(geneList = gene_scores, OrgDb = org.Hs.eg.db, ont="BP", pvalueCutoff = 0.2)
  )
  if(length(go_results) > 0){
    go_plot <- suppressMessages(dotplot(go_results, showCategory = 10) + theme_bw(base_size = 14) + + ggtitle(paste(cohort_title, "GO")))
    ggsave(paste0(".\\figures\\", cohort, '_go_dotplot.png'), plot = go_plot, width = 8, height = 6, dpi = 300)
  }
  
  # Convert results to data.frame
  kegg_results_df <- if(length(kegg_results) > 0) as.data.frame(kegg_results) else data.frame()
  go_results_df <- if(length(go_results) > 0) as.data.frame(go_results) else data.frame()
  
  # Map ENTREZ -> SYMBOL for readability
  if(nrow(kegg_results_df) > 0){
    kegg_entrez_ids <- unique(unlist(strsplit(kegg_results_df$core_enrichment, "/")))
    kegg_symbols <- bitr(kegg_entrez_ids, fromType='ENTREZID', toType='SYMBOL', OrgDb=org.Hs.eg.db)
    kegg_id_to_symbol <- setNames(kegg_symbols$SYMBOL, kegg_symbols$ENTREZID)
    kegg_results_df$geneID <- sapply(kegg_results_df$core_enrichment, function(x) {
      paste(kegg_id_to_symbol[strsplit(x, "/")[[1]]], collapse = "/")
    })
  }
  
  if(nrow(go_results_df) > 0){
    go_entrez_ids <- unique(unlist(strsplit(go_results_df$core_enrichment, "/")))
    go_symbols <- bitr(go_entrez_ids, fromType='ENTREZID', toType='SYMBOL', OrgDb=org.Hs.eg.db)
    go_id_to_symbol <- setNames(go_symbols$SYMBOL, go_symbols$ENTREZID)
    go_results_df$geneID <- sapply(go_results_df$core_enrichment, function(x) {
      paste(go_id_to_symbol[strsplit(x, "/")[[1]]], collapse = "/")
    })
  }
  
  return(list(kegg = kegg_results_df, go = go_results_df))
}
  
  



