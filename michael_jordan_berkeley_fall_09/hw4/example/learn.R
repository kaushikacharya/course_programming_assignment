#---------------------------------------------------
# Shrunk Item Mean for CF (Learn)
# -For each item, computes and saves a mean rating 
#  shrunken toward the global mean
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

# Find the number of datapoints
numExamples <- nrow(data)

# Find the max item ID
maxItem <- max(data[,2])

# Global mean
globalMean <- 0

# Array to store mean item ratings 
itemMean <- array(0, c(1, maxItem))

# Array to store number of ratings per item
itemCounts <- array(0, c(1, maxItem))

# Compute global mean, item counts, and sum
# of ratings for each item
for (e in 1:numExamples) {
  itemID <- data[e,2]
  rating <- data[e,3]
  itemCounts[itemID] <- itemCounts[itemID] + 1
  itemMean[itemID] <- itemMean[itemID] + rating
  globalMean <- globalMean + rating
}
globalMean <- globalMean / numExamples

# Choose shrinkage parameter
alpha <- 20

# Compute shrunk item means by shrinking 
# toward global mean
itemMean <- (itemMean + globalMean*alpha) / (itemCounts + alpha);

#---------------------------------------------------
# Output model parameters to file
#---------------------------------------------------
save(itemMean, file = "itemMean.Rdata")