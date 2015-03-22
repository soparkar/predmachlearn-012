library(caret)
library(kernlab)
library(randomForest)
library(corrplot)
library(RCurl)
library(parallel); 
library(doParallel);

# read the data
trdata <- getURL("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", ssl.verifypeer=0L, followlocation=1L)
training_data <- read.csv(text=trdata, na.strings= c("NA",""," "))
tsdata <- getURL("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", ssl.verifypeer=0L, followlocation=1L)
testing_data <- read.csv(text=tsdata, na.strings= c("NA",""," "))

# remove columns with NAs
training_data_NAs <- apply(training_data, 2, function(x) {sum(is.na(x))})
training_data_clean <- training_data[,which(training_data_NAs == 0)]

# remove identifier columns such as names and timestamps
training_data_clean <- training_data_clean[8:length(training_data_clean)]

# split the cleaned testing data into training and cross validation
inTrain <- createDataPartition(y = training_data_clean$classe, p = 0.7, list = FALSE)
training <- training_data_clean[inTrain, ]
cv <- training_data_clean[-inTrain, ]

# plot the correlation matrix
correlMatrix <- cor(training[, -length(training)])
corrplot(correlMatrix, order = "FPC", method = "circle", type = "lower", tl.cex = 0.8,  tl.col = rgb(0, 1, 0))

# fit the model to predict the classe using everything else as a predictor
registerDoParallel(clust <- makeForkCluster(detectCores()));
modelFit <- randomForest(classe ~ ., data = training)
stopCluster(clust);

# cross-validate the model using the remaining 30% of trainig data
predictCrossVal <- predict(modelFit, cv)
confusionMatrix(cv$classe, predictCrossVal)

# clean the final testing data in the same way
testing_data_NAs <- apply(testing_data, 2, function(x) {sum(is.na(x))})
testing_data_clean <- testing_data[,which(testing_data_NAs == 0)]
testing_data_clean <- testing_data_clean[8:length(testing_data_clean)]

# predict the classes of the test set
predictTest <- predict(modelFit, testing_data_clean)
