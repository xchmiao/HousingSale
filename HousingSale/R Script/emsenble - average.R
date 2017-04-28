library(readr)
library(dplyr)

path = "/Users/xiaochangmiao/Documents/Course/Kaggle/HousingSale"
setwd(dir = path)

files = c("lasso_rs.csv", "lasso2.csv")
data = read.csv("lasso_3.csv")
data$v0 = log(data$SalePrice)
i = 1
for (fname in files){
  pred = read.csv(fname)
  data[paste("v", i, sep = "")] = log(pred$SalePrice)
  i = i+1
}

data$logmean <- rowMeans(subset(data, select = c(v0, v1, v2)))
data$SalePrice <- exp(data$logmean)
head(data)
write_csv(data[, 1:2], "avg_lasso.csv")
