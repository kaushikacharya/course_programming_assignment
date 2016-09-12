require(boot)
load("F:/ReadingMaterial//video_lectures/Statistical_Learning_Hastie_Tibshirani/Chapter5_Resampling_Methods/5.R.RData")

boot_standard.fn = function(data, index){
  return (coef(lm(y~X1+X2, data=data, subset=index)))
}

set.seed(1)
boot(Xy, boot_standard.fn, 1000)

# Standard Boosting
beta_1_vec = c()
for (i in 1:1000){
  index = sample(dim(Xy)[1], dim(Xy)[1], replace=T)
  res = coef(lm(y~X1+X2, data=Xy, subset=index))
  beta_1_vec = c(beta_1_vec, res[2])
}

# Block Boosting
beta_1_vec = c()
for (i in 1:1000){
  block_index = sample(10, 10, replace=T)
  index = c()
  for (block_i in block_index){
    index = c(index, (100*(block_i-1)+1):(100*block_i))
  }
  res = coef(lm(y~X1+X2, data=Xy, subset=index))
  beta_1_vec = c(beta_1_vec, res[2])
}

# standard error
sd(beta_1_vec)