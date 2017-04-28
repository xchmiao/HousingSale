import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from biokit.viz import corrplot

seed = 260681
path = ""
print "Reading the data from file..."
train = pd.read_csv(path + "//" + "train.csv")
test = pd.read_csv(path + "//" + "test.csv")
submission = pd.read_csv(path + "//" + "sample_submission.csv")
y_train = train.QuoteConversion_Flag.values
train = train.drop("QuoteConversion_Flag", axis = 1)
print "Combining the all the data into a big dataframe..."
df = pd.concat([train, test], ignore_index = True)

print "The size of the data is:"
print df.shape

df["Date"] = pd.to_datetime(df['Original_Quote_Date'])
df["Year"] = df.Date.map(lambda x: x.year)
df["Month"] = df.Date.map(lambda x: x.month)
df["WeekDay"] = df.Date.map(lambda x: x.dayofweek)
df.drop("Date", axis = 1, inplace = True)

print "Filling Missing values..."
df = df.fillna(-1)

print "Converting data types..."
for col in df.columns:
    if df[col].dtype == 'object':
        print col
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(df[col].values))
        df[col] = lbl.transform(list(df[col].values))

x_train = df.values[:len(y_train), 1:]
x_test = df.values[len(y_train):, 1:]

# x_train = x_train[:500, ]
# y_train = y_train[:500]
# x_test = x_test[:100, ]
# submission = submission.iloc[:100, ]

xgb_model = xgb.XGBClassifier(silent = True, seed = seed, nthread = 4)

parameters = {'objective': ['binary:logistic'],
              'n_estimators': [100, 500],
              'max_depth':[1, 3, 10],
              'min_child_weight':[1, 3, 10],
              'subsample':[0.3, 0.8],
              'colsample_bytree': [0.3, 0.8]}
			  
print "Start paramter grid search..."
start = time()
clf = GridSearchCV(xgb_model, parameters, n_jobs = -1, scoring = 'roc_auc',
                   cv = StratifiedKFold(y_train, n_folds = 5, shuffle = True, \
                                        random_state = 128))

clf.fit(x_train, y_train)
end = time()
print "It takes %d seconds to finish the training." %(end - start)

best_parameters, score, _ = max(clf.grid_scores_, key=lambda x: x[1])

print('Raw AUC score:', score)
for param_name in sorted(best_parameters.keys()):
    print("%s: %r" % (param_name, best_parameters[param_name]))

ypreds = clf.predict_proba(x_test)[:, 1]
submission.QuoteConversion_Flag = ypreds
submission.to_csv("submission.csv", index = False)
print "The predicted result is saved."

scores = clf.grid_scores_
scores = sorted(scores, key = lambda x: x[1], reverse = True)

f = open("GridSearch.txt", "wb")
for line in scores:
    f.write(str(line) + "\n")
f.close()
print "Grid Search result has been saved in the txt file."

