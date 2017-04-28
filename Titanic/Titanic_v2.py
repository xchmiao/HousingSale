# import matplotlib
# matplotlib.use('tkagg')
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.grid_search import GridSearchCV

# path = r"E:\Titanic\Data"
path = r"E:\Kaggle\Titanic\Data"
# datafile = r"/Users/Angela/Documents/Kaggle/Titanic/Data/train.csv"
# testfile = r"/Users/Angela/Documents/Kaggle/Titanic/Data/test.csv"
datafile = path + "\\" + "train.csv"
testfile = path + "\\" + "test.csv"
df_train = pd.read_csv(datafile)
df_test = pd.read_csv(testfile)
survived = df_train.Survived
df_train = df_train.drop(["Survived"], axis = 1)
df = pd.concat([df_train, df_test], ignore_index = True)

#----------------------------------- Clean the outliners in Fare --------------------------


# ------------ Tinny Fare number cleaning ----------------------------------
median_fare = df[["Fare", "Pclass", "Embarked"]].groupby(["Pclass", "Embarked"]).median()
df.Fare = df.Fare.map(lambda x: np.nan if x < 5 else x)
f = lambda x: median_fare.loc[x["Pclass"], x[ "Embarked"]].values[0] if pd.isnull(x["Fare"]) else x["Fare"]
df["Fare"] = df[["Fare", "Pclass", "Embarked"]].apply(f, axis = 1)

df["fare"] = df["Fare"]
# ------------ Hunge Fare value cleaning ------------------------------------
def f1(x):
    count = group.count()["PassengerId"]
    dict = count.to_dict()
    if x in dict.keys():
        return (x/dict[x])
    else:
        return x
table = df.pivot_table(values = "Fare", index = ["Embarked"],
                       columns = ["Pclass"], aggfunc = np.median)
df.loc[df.fare > 500, "fare"] = table.loc["C", 1]
group = df[(df.fare > 160) & (df.Pclass == 1)].groupby("fare")
df.loc[df.fare > 160, "fare"] = df.loc[df.fare > 160, "fare"].apply(f1)
group = df[(df.fare > 50) & (df.Pclass == 2)].groupby("fare")
df.loc[(df.fare > 50) & (df.Pclass == 2), "fare"] = df.loc[(df.fare > 50) & (df.Pclass == 2), "fare"].apply(f1)
group = df[(df.fare > 25) & (df.Pclass == 3)].groupby("fare")
df.loc[(df.fare>25) & (df.Pclass == 3), "fare"] = df.loc[(df.fare>25) & (df.Pclass == 3), "fare"].apply(f1)

df.loc[df.Name.str.contains("Backstrom"), "fare"] = 15.85/2

g = sns.FacetGrid(df, col = "Embarked", col_order = ["C", "S", "Q"])
g = g.map(sns.boxplot, "Pclass", "fare", order = [1, 2, 3])
plt.show()


#------------------------ Embark -------------------------------------
df.loc[df.Embarked.isnull(), "Embarked"] = "S" # Stone is a English last name and Fare
"""
#------------------------ Embark to dummy variables ------------------
#df["Em"] = df[df["Embarked"].notnull()]["Embarked"].map({"S":0, "C": 1, "Q": 2}).astype(int)
embark = pd.get_dummies(df.Embarked)
df = pd.concat([df, embark], axis = 1)
"""
#---------------------- Gender ---------------------------------------
df["Gender"] = df["Sex"].map({"female":0, "male": 1}).astype(int)
#----------------------- Add Title ------------------------------------
df["Title"] = df.Name.map(lambda x: re.search('\w+\.', x).group()[:-1])

# ------------ Replacing the Cabin missing values -------------------
df.loc[df.Cabin == "T", "Cabin"] = "C"
df["CabinLetter"] = df.Cabin.map(lambda x: "U" if pd.isnull(x) else x[0].upper())
df["CabinLetter"] = pd.factorize(df["CabinLetter"], sort = True)[0] + 1
# df = df.drop(["Embarked", "Fare"], axis=1)


#------------------------- Last Name ----------------------------------
df["LastName"] = df["Name"].map(lambda x: x.split(",")[0])
df.loc[df.LastName == "Frolicher-Stehli", "LastName"] = "Frolicher"
df.loc[df.Name.str.contains("Lamson"), "LastName"] = "Lamson"
df.loc[df.Name.str.contains("Gustafsson")& (df.SibSp == 2), "LastName"] = "Backstrom"


#--------------------- Familiy Size ------------------------------------
group = df.groupby(["Ticket"])
dict = group.count()["PassengerId"].to_dict()
df["TicketGroup"] = df["Ticket"].map(dict)
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df.loc[df.LastName == "Backstrom", "GroupSize"] = 4
df.loc[df.LastName == "Frauenthal", "GroupSize"] = 3
df.loc[df.LastName == "Davidson", "GroupSize"] = 4
df.loc[(df.LastName == "Hansen") & (df.FamillySize > 1), "GroupSize"] = 3
df.loc[(df.LastName == "Hirvonen"), "GroupSize"] = 3
df.loc[(df.LastName == "Jacobsohn"), "GroupSize"] = 4
df["GroupSize"] = df[["FamilySize", "TicketGroup"]].\
    apply(lambda x: max(x["FamilySize"], x["TicketGroup"]), axis = 1)


group = df[df.FamilySize >=3].groupby("LastName").count()["PassengerId"]
"""
#---------------------- Ticket ------------------------------------------
def GetTicketPrefix(ticket):
    match = re.compile("([a-zA-Z\.\/]+)").search(ticket)
    if match:
        return match.group()
    else:
        return 'U'

def GetTicketNumber(ticket):
    match = re.compile("([\d]+$)").search(ticket)
    if match:
        return match.group()
    else:
        return '0'

df["TicketPrefix"] = df["Ticket"].map(lambda x: GetTicketPrefix(x.upper()))
df['TicketPrefix'] = df['TicketPrefix'].map(lambda x: re.sub('[\.?\/?]', '', x))
df['TicketPrefix'] = df['TicketPrefix'].map(lambda x: re.sub('STON', 'SOTON', x))
df['TicketPrefix'] = df['TicketPrefix'].map(lambda x: re.sub('FCC', 'FC', x))
# prefixes = pd.get_dummies(df['TicketPrefix']).rename(columns=lambda x: 'TicketPrefix_' + str(x))
# df = pd.concat([df, prefixes], axis=1)
df['TicketPrefixId'] = pd.factorize(df['TicketPrefix'])[0]
# extract Ticket number
df['TicketNumber'] = df['Ticket'].map(lambda x: GetTicketNumber(x))
df['TicketNumberDigits'] = df['TicketNumber'].map( lambda x: len(x) ).astype(np.int)
df['TicketNumberStart'] = df['TicketNumber'].map( lambda x: x[0:1] ).astype(np.int)
"""
'''
#------------- Plot Age vs Title ------------------------------------------
f, axes = plt.subplots(2, 1, sharey = True)
data = df[df.Age.notnull() & (df.Sex == "female")]
sns.boxplot(x = "Title", y = "Age", data = data, color = "m", ax = axes[0])

data = df[df.Age.notnull() & (df.Sex == "male")]
sns.boxplot(x = "Title", y = "Age", data = data, color = "b", ax = axes[1])
plt.show()
'''

#------------- Replace the Missing value by age -------------------------------



#----------- Look for couples --------

lastnames = df[df.Age.isnull() & (df.SibSp == 1) & (df.Title == "Mrs")]["LastName"].values
tickets = df[df.Age.isnull() & (df.SibSp == 1) & (df.Title == "Mrs")]["Ticket"].values

age_dict = df[(df.Title == "Mr") & (df.SibSp == 1) & df.LastName.isin(lastnames) &
              df.Ticket.isin(tickets)].set_index("LastName")["Age"].to_dict()

for key in age_dict.keys():
    age_dict[key] = age_dict[key] - 3

df.loc[df.Age.isnull() & (df.SibSp == 1) & (df.Title == "Mrs"), "Age"] = df.loc[df.Age.isnull() & (df.SibSp == 1)
                                                        & (df.Title == "Mrs")]["LastName"].map(age_dict)

df.loc[df.Name.str.contains("Frauenthal, Mrs. Henry William"), "Age"] = 47
df.loc[df.Name == "van Billiard, Master. James William", "Age"] = 11.5


def AgeFitData(df):
    age_df = df[["Age", "Pclass", "S", "Q", "C", "fare", "Sex", "Gender", "Parch", "SibSp",
                 "Title"]]

    #----------Group Titles by Age and Gender
    age_df.loc[(age_df.Title == "Dr") & (age_df.Sex == "female"), "Title"] = "Mrs"
    age_df.loc[age_df.Title.isin(["Lady", "Countess", "Mrs", "Dona"]), "Title"] = "Mrs"
    age_df.loc[age_df.Title.isin(["Miss", "Mme", "Ms", "Mlle"]), "Title"] = "Miss"

    age_df.loc[age_df.Title.isin(["Jonkheer", "Don", "Major",
    "Sir", "Col", "Capt"]),
           "Title"] = "Sir"
    age_df.loc[age_df.Title.isin(["Master"]), "Title"] = "Master"

    title = pd.get_dummies(age_df.Title).astype("int")
    age_df = pd.concat([age_df, title], axis = 1).\
        drop(["Title", "Sex"], axis =1)

    return age_df

def setMissingAges(df):

    age_df = AgeFitData(df)

    knownAge = age_df.loc[age_df.Age.notnull()]
    unknownAge = age_df.loc[age_df.Age.isnull()]

    y_train = knownAge.values[:, 0]
    X_train = knownAge.values[:, 1::]

    X_predict = unknownAge.values[:, 1::]

    # Create and fit a model
    rft = RandomForestRegressor(n_estimators=2000, n_jobs = -1)
    rft.fit(X_train, y_train) # ------------ Later on, need to use this fit to predict the missing value in test set also

    # Use the fitted model to predict the missing values
    y_predict = rft.predict(X_predict)

    # Assign those predictions to the full data set
    df.loc[df.Age.isnull(), "Age"] = y_predict

    return df

df = setMissingAges(df, df_test)

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
