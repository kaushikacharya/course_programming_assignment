library(nnet)
library(Rtsne)

# Parse the six feature set files and rewrite in csv format
# Also combine to create the feature data frame
# This is required as exisiting feature set files don't have uniform spacing between the columns
# which makes it difficult to use read.csv
create_data_set <- function(data_folder="mfeat/", data_csv_folder="mfeat_csv/"){
  con.fac <- file(file.path(data_folder,"mfeat-fac"), open = "r")
  con.fou <- file(file.path(data_folder,"mfeat-fou"), open = "r")
  con.kar <- file(file.path(data_folder,"mfeat-kar"), open = "r")
  con.mor <- file(file.path(data_folder,"mfeat-mor"), open = "r")
  con.pix <- file(file.path(data_folder,"mfeat-pix"), open = "r")
  con.zer <- file(file.path(data_folder,"mfeat-zer"), open = "r")
  
  # create output directory if not exists
  dir.create(path = file.path(data_csv_folder))
  
  con.csv.fac <- file(file.path(data_csv_folder,"mfeat-fac"), open = "w")
  con.csv.fou <- file(file.path(data_csv_folder,"mfeat-fou"), open = "w")
  con.csv.kar <- file(file.path(data_csv_folder,"mfeat-kar"), open = "w")
  con.csv.mor <- file(file.path(data_csv_folder,"mfeat-mor"), open = "w")
  con.csv.pix <- file(file.path(data_csv_folder,"mfeat-pix"), open = "w")
  con.csv.zer <- file(file.path(data_csv_folder,"mfeat-zer"), open = "w")
  con.csv.combined <- file(file.path(data_csv_folder,"mfeat-combined"), open = "w")
  
  line_i = 0
  
  while (TRUE){
    line.fac <- trimws(readLines(con = con.fac, n = 1))
    line.fou <- trimws(readLines(con = con.fou, n = 1))
    line.kar <- trimws(readLines(con = con.kar, n = 1))
    line.mor <- trimws(readLines(con = con.mor, n = 1))
    line.pix <- trimws(readLines(con = con.pix, n = 1))
    line.zer <- trimws(readLines(con = con.zer, n = 1))
    
    if ( (length(line.fac) == 0) | (length(line.fou) == 0) | (length(line.kar) == 0) | 
        (length(line.mor) == 0) | (length(line.pix) == 0) | (length(line.zer) == 0) ){
      break
    }
    
    line_i = line_i + 1
    
    tokens.fac <- strsplit(x = line.fac, split = "\\s+")
    tokens.fou <- strsplit(x = line.fou, split = "\\s+")
    tokens.kar <- strsplit(x = line.kar, split = "\\s+")
    tokens.mor <- strsplit(x = line.mor, split = "\\s+")
    tokens.pix <- strsplit(x = line.pix, split = "\\s+")
    tokens.zer <- strsplit(x = line.zer, split = "\\s+")
    
    if (line_i == 1){
      # write header
      header.fac <- paste(paste0("fac",seq(1:length(tokens.fac[[1]]))), collapse = "," )
      header.fou <- paste(paste0("fou",seq(1:length(tokens.fou[[1]]))), collapse = "," )
      header.kar <- paste(paste0("kar",seq(1:length(tokens.kar[[1]]))), collapse = "," )
      header.mor <- paste(paste0("mor",seq(1:length(tokens.mor[[1]]))), collapse = "," )
      header.pix <- paste(paste0("pix",seq(1:length(tokens.pix[[1]]))), collapse = "," )
      header.zer <- paste(paste0("zer",seq(1:length(tokens.zer[[1]]))), collapse = "," )
      
      header.combined = paste0( header.fac, ",",
                                header.fou, ",",
                                header.kar, ",",
                                header.mor, ",",
                                header.pix, ",",
                                header.zer )
      
      writeLines(text = header.fac, con = con.csv.fac)
      writeLines(text = header.fou, con = con.csv.fou)
      writeLines(text = header.kar, con = con.csv.kar)
      writeLines(text = header.mor, con = con.csv.mor)
      writeLines(text = header.pix, con = con.csv.pix)
      writeLines(text = header.zer, con = con.csv.zer)
      writeLines(text = header.combined, con = con.csv.combined)
    }
    
    csv.line.fac <- paste(tokens.fac[[1]], collapse = ",")
    csv.line.fou <- paste(tokens.fou[[1]], collapse = ",")
    csv.line.kar <- paste(tokens.kar[[1]], collapse = ",")
    csv.line.mor <- paste(tokens.mor[[1]], collapse = ",")
    csv.line.pix <- paste(tokens.pix[[1]], collapse = ",")
    csv.line.zer <- paste(tokens.zer[[1]], collapse = ",")
    
    csv.line.combined = paste0(csv.line.fac, ",", csv.line.fou, ",",
                               csv.line.kar, ",", csv.line.mor, ",",
                               csv.line.pix, ",", csv.line.zer)
    
    writeLines(text = csv.line.fac, con = con.csv.fac)
    writeLines(text = csv.line.fou, con = con.csv.fou)
    writeLines(text = csv.line.kar, con = con.csv.kar)
    writeLines(text = csv.line.mor, con = con.csv.mor)
    writeLines(text = csv.line.pix, con = con.csv.pix)
    writeLines(text = csv.line.zer, con = con.csv.zer)
    writeLines(text = csv.line.combined, con = con.csv.combined)
  }
  
  close(con.fac)
  close(con.fou)
  close(con.kar)
  close(con.mor)
  close(con.pix)
  close(con.zer)
  
  close(con.csv.fac)
  close(con.csv.fou)
  close(con.csv.kar)
  close(con.csv.mor)
  close(con.csv.pix)
  close(con.csv.zer)
  close(con.csv.combined)
}

#' split row indices into train and validation set
#' TBD: Also create option to select equal number of samples from each class
split_train_validation <- function(data_csv_folder="mfeat_csv/", train_frac=0.7, seed_val=100){
  df <- read.csv(file = file.path(data_csv_folder,"mfeat-combined"), header = TRUE, sep = ",")
  set.seed(seed_val)
  train = sort.int(sample(nrow(df), size = train_frac*nrow(df)))
  validation = setdiff(1:nrow(df),train)
  return(list(train,validation))
}

#' For each feature set we compute and select the top principal components
#'
#' @param train Row indices for creating train subset
compute_principal_components <- function(data_csv_folder="mfeat_csv/", feature_set_csv_file, train, pov_threshold = 0.85){
  df <- read.csv(file = file.path(data_csv_folder,feature_set_csv_file), header = TRUE, sep = ",")
  pca.train <- prcomp(x = df[train,], center = TRUE, scale. = TRUE)
  
  # select top principal components based on proportion of variance
  cum_sum_pov <- cumsum(pca.train$sdev^2)/sum(pca.train$sdev^2)
  n_top_prin_comp <- NULL
  for (i in 1:length(cum_sum_pov)){
    if (cum_sum_pov[i] > pov_threshold){
      n_top_prin_comp <- i
      break()
    }
  }
  
  df.validation <- predict(object = pca.train, newdata = df[-train,])
  
  df.pca <- rbind(pca.train$x, df.validation)
  # re-order the rows in its original position
  # https://stackoverflow.com/questions/20295787/how-can-i-use-the-row-names-attribute-to-order-the-rows-of-my-dataframe-in-r
  df.pca <- df.pca[order(as.numeric(row.names(df.pca))),]
  
  # return(list(df.train.top_principal_components, df.validation.top_principal_components))
  return(list(df.pca, n_top_prin_comp))
}

#' Extract top principal components of feature sets having lot of features
#' 
#' @param train Row indices for creating train subset
transform_feature_using_pca <- function(data_csv_folder="mfeat_csv/", train, n_validation){
  df.pca <- data.frame(matrix(,nrow = length(train) + n_validation,ncol = 0))
  # This vector of objects maps key: feature set name to value: n top principal components
  feature_set_to_ncomp_map <- list()
  
  # Ignoring "mfeat-mor" as there are only 6 features
  for (feature_set in c("mfeat-fac","mfeat-fou","mfeat-kar","mfeat-pix","mfeat-zer")){
    df.feature.pca.output <- compute_principal_components(data_csv_folder,feature_set,train)
    df.feature.pca <- df.feature.pca.output[[1]]
    feature.n_top_prin_comp <- df.feature.pca.output[[2]]
    df.feature.toppc <- df.feature.pca[,1:feature.n_top_prin_comp]
    
    print(paste0("Feature set: ", feature_set, " : top principal components(based on proportion of variance): ", feature.n_top_prin_comp))
    
    # append feature name in the principal components column names
    tokens_feature_name <- strsplit(x = feature_set, split = "-")
    colnames(df.feature.toppc) <- paste0(colnames(df.feature.toppc),tokens_feature_name[[1]][2])
    
    # Now append these columns to the final dataframe
    df.pca <- cbind(df.pca, df.feature.toppc)
    
    feature_set_to_ncomp_map[[feature_set]] <- feature.n_top_prin_comp
  }
  
  # Since there are only 6 morphological features, we are taking the entire set
  df.mor <- read.csv(file = file.path(data_csv_folder,"mfeat-mor"), header = TRUE, sep = ",")
  df.pca <- cbind(df.pca, df.mor)
  
  output <- list()
  output[["df.pca"]] <- df.pca
  output[["feature_set_to_ncomp_map"]] <- feature_set_to_ncomp_map
  
  return(output)
}

#' Create Class column
append_class_to_df <- function(df,index_vec){
  #  The first 200 patterns are of class `0', followed by sets of 200 patterns for each of the classes `1' - `9'.
  vec_class <- c()
  for (index in index_vec){
    vec_class <- c(vec_class,floor((index-1)/200))
  }
  df <- cbind(df, data.frame(Class=vec_class))
  df$Class <- as.factor(df$Class)
  
  return(df)
}

#' Create a list of column names based on 1st k principal components for the given feature set
#' These column names should be in sync with column names of df.pca of transform_feature_using_pca
#' 
#' @example feature_set "mfeat-fac"
create_topk_pca_feature_set <- function(feature_set,k){
  # append feature name in the principal components column names
  tokens_feature_name <- strsplit(x = feature_set, split = "-")
  topk_pca_features <- paste0("PC",1:k,tokens_feature_name[[1]][[2]])
  
  return(topk_pca_features)
}

create_tsne_plot <- function(data_csv_folder="mfeat_csv/",feature_set="mfeat-pix",train){
  df <- read.csv(file = file.path(data_csv_folder,feature_set), header = TRUE, sep = ",")
  # https://stats.stackexchange.com/questions/223602/why-does-the-implementation-of-t-sne-in-r-default-to-the-removal-of-duplicates
  # For large dataset, duplicates shouldn't be checked
  tsne_model <- Rtsne(as.matrix(df[train,]), check_duplicates = FALSE, pca = TRUE, perplexity = 30,
                      dims = 2, max_iter = 1000, verbose = TRUE)
  df <- append_class_to_df(df, 1:nrow(df))
  colors <- rainbow(n = length(unique(df[train,]$Class)))
  names(colors) <- unique(df[train,]$Class)
  plot(tsne_model$Y, t='n', main = "tsne")
  text(tsne_model$Y, labels = df[train,"Class"], col = colors[df[train,"Class"]])
}

# source: https://archive.ics.uci.edu/ml/datasets/Multiple+Features

# PCA on iris dataset example:
# https://stats.stackexchange.com/questions/72839/how-to-use-r-prcomp-results-for-prediction

# https://stats.stackexchange.com/questions/44060/choosing-number-of-principal-components-to-retain
# Refers
# a) Component retention in principal component analysis with application to cDNA microarray data (2007) by Cangelosi and Goriely
#    https://stats.stackexchange.com/questions/44060/choosing-number-of-principal-components-to-retain
# b) Automatic choice of dimensionality for PCA (uses Bayesian model selection) (matlab code also provided)
#    https://tminka.github.io/papers/pca/

# https://www.r-bloggers.com/how-to-multinomial-regression-models-in-r/
# https://stats.idre.ucla.edu/r/dae/multinomial-logistic-regression/

# https://stackoverflow.com/questions/31702132/r-create-empty-data-frame-with-200-rows-and-no-columns
# https://stackoverflow.com/questions/24741541/split-a-string-by-any-number-of-spaces
# https://stackoverflow.com/questions/12626637/reading-a-text-file-in-r-line-by-line
# https://stackoverflow.com/questions/7466023/how-to-give-color-to-each-class-in-scatter-plot-in-r

# TBD:
#   - https://www.kaggle.com/c/digit-recognizer/data
#     Do this in another project
#   - For each set of features plot scatter plot with top two principal components and save the plots.
#   - Plot proportion of variance explained wrt to principal components for each feature set
#   - For KNN classification function pass parameter which decides how many top principal components to pick.
#     Advancements:
#     - Function should take single feature set. This will in iteration take 1st PCA, 1st two PCA and further till accuracy doesn't improves significantly.
#     - This can be further improved to do using cross-validation.
#   - transform_feature_using_pca(): return entire pca
#   - plot mfeat-pix
