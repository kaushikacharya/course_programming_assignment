source:
    https://www.kaggle.com/c/digit-recognizer/data
    
Commands:
    df.train <- read.csv(file = file.path("data/","train.csv"), header = TRUE, sep = ",")
    source("./digit_recognizer.R")
    # plot a sample
    plot.sample(600,df.train)