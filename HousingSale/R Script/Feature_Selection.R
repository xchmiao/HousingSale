library(moments)
library(readr)
library(dplyr)
library(mlr)
library(parallelMap)
library(doMC)
registerDoMC(cores = 8)
options(rf.cores = detectCores(), mc.cores = detectCores())

feature_importance_rf <- function(train, n = 10, verbose = TRUE){
  regr.task = makeRegrTask(id = 'rf_feat_imp', data = train, target = 'SalePrice')
  fv = generateFilterValuesData(regr.task)
  plotFilterValues(fv)
  feat_imp = fv$data
  feat_imp$imp = feat_imp$randomForestSRC.rfsrc/max(feat_imp$randomForestSRC.rfsrc)
  feat_imp = feat_imp[order(-feat_imp$imp), ]
  top_features = feat_imp$name[1:n]
  if (verbose){
    sprintf("Top %s features:\n", n)
    cat(top_features, sep = '\n')
  }
  feat_imp
}


feature_selection_rf <- function(train, threshold = 2e-3){
  feat_imp = feature_importance_rf(train)
  minor_features = subset(feat_imp, imp <= threshold)$name
  cat("Those features are minor features, will drop:\n")
  cat(minor_features, sep = '\n')
  minor_features
}

drop_features <- function(df, feature_names){
  df = df[, !colnames(df) %in% feature_names]
  df
}

transform_skewness <- function(df, df_test){
  df_num = df[sapply(df, is.numeric)]
  skew = sapply(df_num, skewness)
  over_skewed_features = names(skew[abs(skew) >= 9])
  #print(skew)
  cat("Drop those over skewed features:\n")
  cat(over_skewed_features, sep = "\n")
  skewed_features = names(skew[abs(skew) >= 0.75])
  cat("Transform those skewed features:\n")
  cat(skewed_features, sep = "\n")
  df[, skewed_features] = log(df[, skewed_features] + 1)
  df = drop_features(df, over_skewed_features)
  df_test[, skewed_features] = log(df_test[, skewed_features] + 1)
  df_test = drop_features(df_test, over_skewed_features)
  list("train" = df, "test" = df_test)
}

feature_engineering <- function(train, test){
  data <- subset(train, select = -c(SalePrice))
  data <- rbind(data, test)
  minor_features <- feature_selection_rf(train)
  data <- drop_featuers(data, minor_features)
  data$BsmtQC <- as.factor(data$BsmtQC)
  list("train" = train, "test" = test)
}

param_tuning_rf <- function(X_train){
  regr.task = makeRegrTask(data = X_train, target = 'SalePrice')
  rf_lrn <- makeLearner('regr.randomForestSRC', par.vals = list(ntree = 1000))
  rf_param <- makeParamSet(
    makeDiscreteParam("mtry", values = c(10, 15, 20)),
    makeDiscreteParam("nodesize", values = c(5)),
    makeDiscreteParam("bootstrap", values = c("by.root"))
  )
  rf_ctrl = makeTuneControlGrid()
  set.seed(10)
  kfoldCV = makeResampleDesc("CV", iters = 3)
  #parallelStartMulticore(8, logging = FALSE)
  gridsearch = tuneParams(rf_lrn, task = regr.task, resampling = kfoldCV, par.set = rf_param,
                          control = rf_ctrl, measures = rmse)
  best_rf_lrn = setHyperPars(rf_lrn, par.vals = gridsearch$x)
  best_rf_model = train(best_rf_lrn, regr.task)
  best_rf_model
  #parallelStop()
}


param_tuning_lasso <- function(X_train){
  regr.task = makeRegrTask(data = X_train, target = 'SalePrice')
  lasso_lrn <- makeLearner('regr.glmnet', par.vals = list(alpha =1)) # alpha = 1 fit a lasso model
  lasso_param <- makeParamSet(
    makeDiscreteParam("lambda", values = c(0.001, 0.003, 0.01, 0.03, 0.1)),
    makeDiscreteParam("alpha", values = c(0, 0.1, 0.5, 0.7,0.9, 1.0))
  )
  lasso_ctrl = makeTuneControlGrid()
  set.seed(18)
  kfoldCV = makeResampleDesc("CV", iters = 5)
  gridsearch = tuneParams(lasso_lrn, task = regr.task, resampling = kfoldCV, par.set = lasso_param,
                          control = lasso_ctrl, measures = rmse)
  #data = generateHyperParsEffectData(gridsearch)
  #plotHyperParsEffect(data, x = "lambda", y = "iteration", plot.type = "line")
  #print(data)
  best_lasso_lrn = setHyperPars(lasso_lrn, par.vals = gridsearch$x)
  best_lasso_model = train(best_lasso_lrn, regr.task)
  best_lasso_model
}

predict_model<-function(model, X_test, output_filename = 'RF2.csv'){
  submission <- read.csv('sample_submission.csv')
  y_test <- predict(model, newdata = X_test)$data$response
  submission$SalePrice <- exp(y_test)-1
  write_csv(submission, output_filename)
}


X_train <- read.csv('train_clean.csv')
X_test <- read.csv('test_clean.csv')

X_train$SalePrice <- log(X_train$SalePrice + 1)
X_train$BsmtQC <- as.factor(X_train$BsmtQC)
X_test$BsmtQC <- as.factor(X_test$BsmtQC)
minor_features <- feature_selection_rf(X_train)

#X_train <- drop_features(X_train, minor_features)
#X_test <- drop_features(X_test, minor_features)
data <- transform_skewness(X_train, X_test)
X_train <- data$train
X_test <- data$test
best_lasso_model <- param_tuning_lasso(X_train)
predict_model(best_lasso_model, X_test, output_filename = "lasso3.csv")

