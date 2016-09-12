#---------------------------------------------------
# Template File for CF Learning
#---------------------------------------------------

#---------------------------------------------------
# Read in command line arguments
#---------------------------------------------------
for (e in commandArgs()) {
  if(substring(e, 1, 7) == "inFile="){
    # Store input file name
    inFile <- substring(e, 8)
  }
  else if(substring(e, 1, 8) == "outFile="){
    # Store output file name
    outFile <- substring(e, 89)
  }
}

#---------------------------------------------------
# Load training data from input file
#---------------------------------------------------
data <- read.csv(file=inFile,head=FALSE,sep=" ")


#---------------------------------------------------
# Learn model parameters using training data
#---------------------------------------------------

# ADD CODE HERE

#---------------------------------------------------
# Output model parameters to file
#---------------------------------------------------

# ADD CODE HERE