#---------------------------------------------------
# Template File for CF Prediction
#---------------------------------------------------

#---------------------------------------------------
# Read in command line arguments
#---------------------------------------------------
for (e in commandArgs()) {
  if(substring(e, 1, 7) == "inFile="){
    # Store input file name
    inFile <- substring(e, 8)
    print(inFile)
  }
  else if(substring(e, 1, 8) == "outFile="){
    # Store output file name
    outFile <- substring(e, 9)
    print(outFile)
  }
}

#---------------------------------------------------
# Load test data from input file
#---------------------------------------------------
data <- read.csv(file=inFile,head=FALSE,sep=" ")

#---------------------------------------------------
# Load any model parameters saved in learn.R from file
#---------------------------------------------------

# ADD CODE HERE

#---------------------------------------------------
# Form predictions for each data point
#---------------------------------------------------

# Find the number of datapoints
numExamples <- nrow(data)

# Create predictions array
preds <- array(0, c(1, numExamples)) 

# ADD CODE HERE


#---------------------------------------------------
# Write predictions to output file
#---------------------------------------------------
write(preds, file=outFile, ncolumns=1)