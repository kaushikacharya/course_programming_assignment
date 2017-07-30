https://archive.ics.uci.edu/ml/datasets/Multiple+Features

number of features >> number of samples:
https://archive.ics.uci.edu/ml/datasets/Arcene

http://machinelearningmastery.com/feature-selection-with-the-caret-r-package/
http://r-statistics.co/Variable-Selection-and-Importance-With-R.html

Suggestion for using random projection:
    https://stackoverflow.com/questions/10485809/dimension-reduction-methods-for-images
    Points to 2001 paper with 995 citations according to Google:
    http://users.ics.aalto.fi/ella/publications/randproj_kdd.pdf
    "However, using random projections is computationally signicantly less expensive than using, e.g., principal component analysis."

Non-linear dimensionality reduction:
    https://www.quora.com/What-are-some-important-methods-of-dimension-reduction-used-in-image-processing
    
    https://en.wikipedia.org/wiki/Nonlinear_dimensionality_reduction
    "Although the idea of autoencoders is quite old, training of deep autoencoders has only recently become possible through
    the use of restricted Boltzmann machines and stacked denoising autoencoders."
    
Articles on t-sne:
    https://www.analyticsvidhya.com/blog/2017/01/t-sne-implementation-r-python/
    https://www.r-bloggers.com/playing-with-dimensions-from-clustering-pca-t-sne-to-carl-sagan/ (Links to other interesting articles)
    http://colah.github.io/posts/2015-01-Visualizing-Representations/
    Laurens van der Maaten(author)'s page:
        https://lvdmaaten.github.io/tsne/
    
rm(list = ls())

source("./handwritten_numerals.R")
split.data <- split_train_validation()
train <- split.data[[1]]
validation <- split.data[[2]]

# PCA
transform_feature.pca <- transform_feature_using_pca("mfeat_csv",train,length(validation))
df.pca <- transform_feature.pca[["df.pca"]]
df.pca <- append_class_to_df(df.pca, 1:nrow(df.pca))
feature_set_to_ncomp.map <- transform_feature.pca[["feature_set_to_ncomp_map"]]

# Applying multinomial logit
library(nnet)
mod <- multinom(formula = Class~PC1fac+PC2fac+PC3fac+PC4fac, data = df.toppc, subset = train)
# Note: using all the columns results in error: too many weights
pred.df <- predict(object = mod, newdata = df.toppc[-train,], type = "probs")
pred.df <- cbind(pred.df,df.toppc[-train,"Class"])

# plot pair of pca components
library(ggplot2)
qplot(PC1fou, PC2fou, colour=Class, data = df.toppc[train,])

# KNN
# feature_set = c("PC1fac","PC2fac","PC1fou","PC2fou","PC1kar","PC2kar","PC1pix","PC2pix","PC1zer","PC2zer")
feature_set <- create_topk_pca_feature_set("mfeat-fac",5)
knn.pred = knn(train = df.pca[train,feature_set], test = df.pca[validation,feature_set],cl = df.pca[train,"Class"], k = 5)
table(knn.pred, df.pca[validation,"Class"])
mean((knn.pred == df.pca[validation,"Class"]))

# t-sne
library(Rtsne)
df.pix <- read.csv(file = file.path("mfeat_csv/","mfeat-pix"), header = TRUE, sep = ",")
set.seed(100)
tsne_model <- Rtsne(as.matrix(df.pix), check_duplicates = FALSE, pca = TRUE, perplexity = 30, dims = 2, max_iter = 500, verbose = TRUE)
plot(tsne_model$Y, main = "tsne")
df.pix <- append_class_to_df(df.pix,1:2000)
> colors = rainbow(length(unique(df.pix$Class)))
> names(colors) <- unique(df.pix$Class)
> plot(tsne_model$Y, t='n', main = "tsne")
> text(tsne_model$Y, labels = df.pix$Class, col = colors[df.pix$Class])



df.pca.pair <- transform_feature_using_pca("mfeat_csv",train,length(validation))
df.pca.train <- df.pca.pair[[1]]
df.pca.validation <- df.pca.pair[[2]]
df.pca.train <- append_class_to_df(df.pca.train,train)
df.pca.validation <- append_class_to_df(df.pca.validation,validation)
# Applying multinomial logit
mod <- multinom(formula = Class~PC1fac+PC2fac+PC3fac+PC4fac, data = df.pca.train)
# Note: using all the columns results in error: too many weights
pred.df <- predict(object = mod, newdata = df.pca.train,type = "probs")
pred.df <- cbind(pred.df,df.pca.train$Class)
df.pca.train$Class <- as.factor(df.pca.train$Class)
control <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
model <- train(Class~., data=df.pca.train, method="lvq", preProcess="scale", trControl=control)
# estimate variable importance
importance <- varImp(model, scale=FALSE)
# summarize importance
print(importance)
# plot importance
plot(importance)



df.fac <- read.csv(file = file.path("mfeat_csv/","mfeat-fac"), header = TRUE)

pca.out.fac <- prcomp(x = df.fac[train,], scale. = TRUE)
cumsum(pca.out.fac$sdev^2)
plot(cumsum(pca.out.fac$sdev^2))
# This also provides cumulative proportion of variance
summary(pca.out.fac)

# choose first 13 principal components based on 85% of variation covered
pca.out.fac$rotation[,1:13]

pred.fac = predict(object = pca.out.fac, newdata = df.fac[-train,])

