# import matplotlib
# matplotlib.use('tkagg')
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
# import seaborn as sns
from time import time
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.grid_search import GridSearchCV

path = r"E:\Kaggle\Titanic\Data"
# datafile = r"/Users/Angela/Documents/Kaggle/Titanic/Data/train.csv"
# testfile = r"/Users/Angela/Documents/Kaggle/Titanic/Data/test.csv"
datafile = path + "\\" + "train.csv"
testfile = path + "\\" + "test.csv"
df = pd.read_csv(datafile)
df_test = pd.read_csv(testfile)

'''
g = sns.FacetGrid(df, col = "Pclass", row = "Sex", col_order = [1, 2, 3])
g = (g.map(plt.hist, "Survived", color = "c").set(xlim = (-0.5, 1.5), xticks = [0, 1]))
plt.show()
'''

#----------------------------- Replacing the missing Embarked Values ------------------------------------
'''
g = sns.FacetGrid(df, col = "Pclass", col_order = [1, 2, 3])
g = (g.map(sns.boxplot, "Embarked", "Fare")).set(ylim = (0, 300))
plt.show()

bins = np.arange(10, 300, 20)
g = sns.FacetGrid(df[df.Pclass == 1], col = "Embarked")
g = (g.map(plt.hist, "Fare", bins = bins)).set(xlim = (10, 300))
plt.show()
'''

#------------------------ Embark -------------------------------------
df.loc[df.Embarked.isnull(), "Embarked"] = "S" # Stone is a English last name and Fare

median_fare = df[["Fare", "Pclass", "Embarked"]].groupby(["Pclass", "Embarked"]).median()

def datacleaning(df, median_fare):
    #------------------------ Embark to dummy variables ------------------
    #df["Em"] = df[df["Embarked"].notnull()]["Embarked"].map({"S":0, "C": 1, "Q": 2}).astype(int)
    embark = pd.get_dummies(df.Embarked)
    df = pd.concat([df, embark], axis = 1)
    #---------------------- Gender ---------------------------------------
    df["Gender"] = df["Sex"].map({"female":0, "male": 1}).astype(int)
    #----------------------- Add Title ------------------------------------
    df["Title"] = df.Name.map(lambda x: re.search('\w+\.', x).group()[:-1])

    # ------------ Replacing the Cabin missing values -------------------
    df.Cabin = df.Cabin.map(lambda x: "Unknown" if pd.isnull(x) else x[0].upper())

    # ------------ Fare value cleaning ----------------------------------
    df.Fare = df.Fare.map(lambda x: np.nan if x < 5 else x)

    f = lambda x: median_fare.loc[x["Pclass"], x[ "Embarked"]].values[0] if pd.isnull(x["Fare"]) else x["Fare"]
    df["fare"] = df[["Fare", "Pclass", "Embarked"]].apply(f, axis = 1)
    df = df.drop(["Embarked", "Fare"], axis=1)
    
    return df

df = datacleaning(df, median_fare)
df_test = datacleaning(df_test, median_fare)

# ------------ Missing Age values ---------------------------------------

# g = sns.FacetGrid(df, row = "Sex", col = "Pclass", col_order = [1, 2, 3])
# g = (g.map(sns.boxplot, "Embarked", "Age")).set(ylim = (0, 80))
# plt.show()
#
# g = sns.FacetGrid(df, row = "Sex", col = "Pclass",hue = "Embarked", col_order = [1, 2, 3])
# g = (g.map(sns.distplot,"Age", hist = False).add_legend()).set(xlim = (0, 80))
# plt.show()

'''
#------------- Plot Age vs Title ------------------------------------------
f, axes = plt.subplots(2, 1, sharey = True)
data = df[df.Age.notnull() & (df.Sex == "female")]
sns.boxplot(x = "Title", y = "Age", data = data, color = "m", ax = axes[0])

data = df[df.Age.notnull() & (df.Sex == "male")]
sns.boxplot(x = "Title", y = "Age", data = data, color = "b", ax = axes[1])
plt.tight_layout()
plt.show()
'''

#------------- Replace the Missing value by age -------------------------------

def AgeFitData(df):
    age_df = df[["Age", "Pclass", "S", "Q", "C", "fare", "Sex", "Gender", "Parch", "SibSp",
                 "Title"]]

    #----------Group Titles by Age and Gender
    age_df.loc[(age_df.Title == "Dr") & (age_df.Sex == "female"), "Title"] = "Mrs"
    age_df.loc[age_df.Title.isin(["Lady", "Countess", "Mrs", "Dona"]), "Title"] = "Mrs"
    age_df.loc[age_df.Title.isin(["Miss", "Mme", "Ms", "Mlle"]), "Title"] = "Miss"

    age_df.loc[age_df.Title.isin(["Mr", "Jonkheer", "Don", "Rev", "Major",
    "Sir", "Col", "Capt", "Dr"]),
           "Title"] = "Mr"
    age_df.loc[age_df.Title.isin(["Master"]), "Title"] = "Master"

    title = pd.get_dummies(age_df.Title).astype("int")
    age_df = pd.concat([age_df, title], axis = 1).\
        drop(["Title", "Sex"], axis =1)

    return age_df

def setMissingAges(df, df_test):

    age_df = AgeFitData(df)
    age_df_test = AgeFitData(df_test)

    knownAge = age_df.loc[age_df.Age.notnull()]
    unknownAge = age_df.loc[age_df.Age.isnull()]
    unknownAge_test = age_df_test[age_df_test.Age.isnull()]

    y_train = knownAge.values[:, 0]
    X_train = knownAge.values[:, 1::]

    X_predict = unknownAge.values[:, 1::]
    X_predict_test = unknownAge_test.values[:, 1::]

    # Create and fit a model
    rft = RandomForestRegressor(n_estimators=2000, n_jobs = -1)
    rft.fit(X_train, y_train) # ------------ Later on, need to use this fit to predict the missing value in test set also

    # Use the fitted model to predict the missing values
    y_predict = rft.predict(X_predict)
    y_predict_test = rft.predict(X_predict_test)

    # Assign those predictions to the full data set
    df.loc[df.Age.isnull(), "Age"] = y_predict
    df_test.loc[df_test.Age.isnull(), "Age"] = y_predict_test

    return df, df_test

df, df_test = setMissingAges(df, df_test)

def Del_col(df):

    df["FamSize"] = df["SibSp"] + df["Parch"] + 1

    df.loc[df.Title.isin(["Don", "Dona", "Sir", "Major", "Col", "Capt", "Jonkheer", "Countess", "Lady"])
    , "Title"]="Sir"
    df.loc[df.Title.isin(["Ms"]), "Title"] = "Mrs"
    df.loc[df.Title.isin(["Mme", "Mlle"]), "Title"] = "Miss"
    title = pd.get_dummies(df.Title).astype("int")
    cabin = pd.get_dummies(df.Cabin).astype("int")
    df = pd.concat([df, title, cabin], axis = 1)
    df = df.drop(["Sex", "PassengerId", "Cabin", "Name", "SibSp", "Parch", "Title", "Ticket"], axis = 1)
    return df

df = Del_col(df)
df_test = Del_col(df_test)
df_test.insert(23, "T", 0)

X_train = df.values[:, 1:]
y_train = df.values[:, 0]

X_predict = df_test.values


# clf = RandomForestClassifier(n_jobs=-1)
#
# param_grid = {"n_estimators": [100, 300, 1000],
#               "max_depth": [1, 5, 10],
#               "max_features": [1, "sqrt", None],
#               "min_samples_split": [1, 3, 10],
#               "min_samples_leaf": [5, 10],
#               "bootstrap": [True],
#               "oob_score": [True, False],
#               "criterion": ["gini"]}

# start = time()
# grid_search = GridSearchCV(clf, param_grid = param_grid, cv = 5)
# grid_search.fit(X_train, y_train)
#
# print "GridSearchCV took %.2f seconds for %d candidate parameter settings."\
#     %(time()-start, len(grid_search.grid_scores_))
#
# # best_est = grid_search.best_estimator_
# #
# savefile = r"/Users/Angela/Documents/Kaggle/Titanic/Data/result_2.csv"
# result = pd.read_csv(testfile)
# y_predict = grid_search.predict(X_predict).astype("int")
# result.insert(2, "Survived", y_predict)
# result[["PassengerId","Survived"]].to_csv(savefile, index = False)
