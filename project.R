library(tidyverse)
library(caret)
library(cluster)
library(factoextra)
library(randomForest)
library(corrplot)
library(gridExtra)
library(nnet)
library(dplyr)
library(ggplot2)
library(rpart)
library(rpart.plot)
library(e1071)
library(pROC)

#Load Data
data <- read.csv("data/Amazon Customer Behavior Survey.csv")
View(data)

#View Structure
str(data)

#Data Cleaning
data_clean <- data %>% drop_na()
data_clean <- data_clean %>% mutate(across(where(is.character), as.factor))

#Exploratory Data Analysis (EDA)
summary(data_clean)
glimpse(data_clean)

#Correlation Heatmap for Numeric Variables
corrplot(cor(select_if(data_clean, is.numeric)), method = "square")

#Create heatmap
corrplot(
  cor(select_if(data_clean, is.numeric)), 
  method = "color",            
  col = colorRampPalette(c("#AB0926", "white", "#313696"))(10),  
  type = "full",               
  tl.col = "black",
  tl.cex = 0.7,                
  number.cex = 0.7,            
  addCoef.col = "black",       
  
)

#Convert satisfaction to binary: High (4–5) vs Low (1–3)
data_clean$Satisfaction_Level <- ifelse(data_clean$Shopping_Satisfaction > 3, "High", "Low")
data_clean$Satisfaction_Level <- as.factor(data_clean$Satisfaction_Level)

#Logistic regression 
log_model <- glm(Satisfaction_Level ~ Service_Appreciation + 
                   Recommendation_Helpfulness +
                   Rating_Accuracy +
                   Personalized_Recommendation_Frequency,
                 data = data_clean, family = binomial)

summary(log_model)


#K-means clustering
num_data <- scale(select_if(data_clean, is.numeric))  # or rename to num_data consistently

set.seed(123)
km <- kmeans(num_data, centers = 3, nstart = 25)

fviz_cluster(km, data = num_data) + ggtitle("Customer Segments via K-means Clustering")


#decision tree
tree_model <- rpart(
  Shopping_Satisfaction ~ Service_Appreciation + 
    Recommendation_Helpfulness +
    Rating_Accuracy +
    Personalized_Recommendation_Frequency,
  data = data_clean,
  method = "anova",  
  control = rpart.control(cp = 0.01, minsplit = 20, maxdepth = 10)  
)

rpart.plot(tree_model,
           type = 2,             
           extra = 101,          
           box.palette = "Red",
           fallen.leaves = TRUE,
           varlen = 0,           
           faclen = 0,           
         
           main = "Regression Tree: Predicting Shopping Satisfaction")


#SVM
#Train-test split (70-30)
set.seed(123)
train_index <- createDataPartition(data_clean$Satisfaction_Level, p = 0.7, list = FALSE)
train <- data_clean[train_index, ]
test <- data_clean[-train_index, ]

#Fit SVM model
svm_model <- svm(Satisfaction_Level ~ Service_Appreciation + 
                   Recommendation_Helpfulness +
                   Rating_Accuracy +
                   Personalized_Recommendation_Frequency,
                 data = train, kernel = "linear", probability = TRUE, 
                 class.weights = c("High" = 6, "Low" = 1))

svm_pred <- predict(svm_model, newdata = test)

confusionMatrix(svm_pred, test$Satisfaction_Level)

acc <- cm$overall["Accuracy"]
sens <- cm$byClass["Sensitivity"]
spec <- cm$byClass["Specificity"]

metrics <- data.frame(
  Metric = c("Accuracy", "Sensitivity", "Specificity"),
  Value = c(acc, sens, spec)
)

ggplot(metrics, aes(x = Metric, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", width = 0.6) +
  scale_fill_brewer(palette = "Set2") +
  ylim(0, 1) +
  labs(title = "SVM Performance Metrics", y = "Score") +
  theme_minimal()

#Generate confusion matrix
cm <- confusionMatrix(svm_pred, test$Satisfaction_Level, positive = "High")

cm_df <- as.data.frame(cm$table)

#heatmap
ggplot(cm_df, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 5, color = "black") +
  scale_fill_gradient(low = "lightblue", high = "steelblue") +
  labs(title = "SVM Confusion Matrix", x = "Actual", y = "Predicted") +
  theme_minimal()

svm_probs <- attr(predict(svm_model, newdata = test, probability = TRUE), "probabilities")[, "High"]




#---------------------------------Performance Summary------------------------------------------------


actual <- test$Satisfaction_Level

#Logistic Regression 
log_probs <- predict(log_model, newdata = test, type = "response")
log_pred_class <- ifelse(log_probs > 0.5, "High", "Low") %>% as.factor()
log_roc <- roc(actual, log_probs)

#Decision Tree 
tree_pred_vals <- predict(tree_model, newdata = test)
tree_pred_class <- ifelse(tree_pred_vals > 3, "High", "Low") %>% as.factor()
tree_roc <- roc(actual, tree_pred_vals)

#SVM 
svm_probs <- attr(predict(svm_model, newdata = test, probability = TRUE), "probabilities")[, "High"]
svm_pred_class <- predict(svm_model, newdata = test)
svm_roc <- roc(actual, svm_probs)


Accuracy <- function(pred, actual) mean(pred == actual)

#Summary Table
perf_summary <- data.frame(
  Model = c("Logistic", "Decision Tree", "SVM"),
  F1_Score = c(F1_Score(log_pred_class, actual, positive = "High"),
               F1_Score(tree_pred_class, actual, positive = "High"),
               F1_Score(svm_pred_class, actual, positive = "High")),
  ROC_AUC = c(pROC::auc(log_roc), 
              pROC::auc(tree_roc), 
              pROC::auc(svm_roc)),
  
  Accuracy = c(Accuracy(log_pred_class, actual),
               Accuracy(tree_pred_class, actual),
               Accuracy(svm_pred_class, actual))
  
)

print(perf_summary)


#Compute silhouette score
num_data <- scale(select_if(data_clean, is.numeric))

set.seed(123)
km <- kmeans(num_data, centers = 3, nstart = 25)

sil <- silhouette(km$cluster, dist(num_data))

avg_silhouette <- mean(sil[, 3])
cat("Average Silhouette Score for K-Means Clustering:", round(avg_silhouette, 3), "\n")



















































































