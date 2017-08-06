
#' Plot a sample
#' 
#' @param df train/test csv loaded dataframe
plot.sample <- function(sample_index, df){
  # convert the data frame row correspoding to the sample into matrix form
  sample.matrix <- matrix(as.numeric(unlist(df[sample_index,!(colnames(df) %in% c("label"))]/255)), nrow = 28, ncol = 28, byrow = TRUE)
  sample.label <- paste0("label: ", df[sample_index,"label"])
  image(t(sample.matrix[nrow(sample.matrix):1,]), col = grey(seq(0, 1, length = 256)), axes = TRUE, main = sample.label)
}


# Handling initial error using image
#   https://stackoverflow.com/questions/16518428/right-way-to-convert-data-frame-to-a-numeric-matrix-when-df-also-contains-strin
#   https://stackoverflow.com/questions/12384071/how-to-coerce-a-list-object-to-type-double
#   Was facing similar issue: https://bugs.r-project.org/bugzilla/show_bug.cgi?id=16217
#   https://stackoverflow.com/questions/5638462/r-image-of-a-pixel-matrix