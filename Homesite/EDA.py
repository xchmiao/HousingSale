import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from biokit.viz import corrplot
import seaborn as sns

seed = 260681
path = r""
print "Reading the data from file..."
train = pd.read_csv(path + "\\" + "train.csv")

train["Date"] = pd.to_datetime(train['Original_Quote_Date'])
train["Year"] = train.Date.map(lambda x: x.year)
train["Month"] = train.Date.map(lambda x: x.month)
train["WeekDay"] = train.Date.map(lambda x: x.dayofweek)
train.drop(["Date", "Original_Quote_Date"], axis = 1, inplace = True)

# sns.factorplot("QuoteConversion_Flag", col = "Year", data = train, kind = "count")
# sns.factorplot("QuoteConversion_Flag", col = "Month",
#                col_order=range(1, 13), col_wrap = 4, data = train, kind = "count")
#
# sns.factorplot("QuoteConversion_Flag", col = "WeekDay", col_order = range(0, 7),
#                col_wrap = 4, data = train, kind = "count")

train.drop(["GeographicField6A", "GeographicField10A"], axis =1, inplace = True)
num_train = train.select_dtypes(include = ["float64", "int64"])
cols = num_train.columns.values

feats = cols[1:-3]
# cr = num_train[feats[:70]].corr()
# c = corrplot.Corrplot(cr)
# c.plot(method = "color")

ls = cols[157:177] # GeographicField6B to Geographic16B
ls = np.delete(ls, [7, 14])
print ls
pca = PCA(n_components = len(ls))
x_new = pca.fit_transform(train[ls].values)
print pca.explained_variance_ratio_

for i in range(5):
    train["pca1_" + str(i)] = x_new[:, i]

# for col in ls:
#     plt.figure()
#     sns.boxplot(x = "QuoteConversion_Flag", y = col, data = train )
#     plt.savefig(col + ".jpg")

train.drop(ls, axis = 1, inplace = True)





# print "Filling Missing values..."
# df = df.fillna(-1)
#
# print "Converting data types..."
# for col in df.columns:
#     if df[col].dtype == 'object':
#         print col
#         lbl = preprocessing.LabelEncoder()
#         lbl.fit(list(df[col].values))
#         df[col] = lbl.transform(list(df[col].values))