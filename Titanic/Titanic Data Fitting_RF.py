# import matplotlib
# matplotlib.use('tkagg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
from time import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from operator import itemgetter
from sklearn.cross_validation import *

def report(grid_scores, n_top=5):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")


def Del_col(df):

    df["FamSize"] = df["SibSp"] + df["Parch"] + 1
    df["Mother"] = 0
    df.loc[(df.Title!= "Miss") & (df.Sex =="female") & (df.Parch > 0) & (df.Age > 20), "Mother"] = 1
    df["Child"] = 0
    df["BigGroup"] = 0
    df.loc[df.GroupSize >= 5, "BigGroup"] = 1
    df.loc[(df.Age < 18) & (df.Parch > 0), "Child"] = 1
    df.loc[df.Title.isin(["Sir", "Major", "Col", "Capt", "Jonkheer"]), "Title"] = "Sir"
    df.loc[df.Title.isin(["Dona", "Countess", "Lady"]), "Title"] = "Lady"
    df.loc[df.Title.isin(["Ms"]), "Title"] = "Mrs"
    df.loc[df.Title.isin(["Mme", "Mlle"]), "Title"] = "Miss"
    #df["Group"] = df[["GroupSize", "GroupId"]].apply(lambda x: str(x["GroupSize"]) + x["GroupId"], axis = 1)
    #df.loc[df.GroupSize < 5, "Group"] = "Small"

    title = pd.get_dummies(df.Title).astype("int")
    #cabin = pd.get_dummies(df.Cabin).astype("int")
    #group = pd.get_dummies(df.Group).astype("int")
    #df = pd.concat([df, title, group], axis = 1)
    df = pd.concat([df, title], axis = 1)

    df = df.drop(["PassengerId", "Name", "Sex", "Ticket", "fare",
                  "Cabin", "Embarked", "TicketGroup", "Title", "LastName",
                  "GroupId","TicketNumber", "FamSize"], axis = 1)
    return df


if __name__ == '__main__':
#-------------------------------------- Read the clean data -----------------------------------------
    path = r"E:\Kaggle\Titanic\Data"
    # path = r"E:\Kaggle\Titanic\Data"
    # datafile = r"/Users/Angela/Documents/Kaggle/Titanic/Data/train.csv"
    # testfile = r"/Users/Angela/Documents/Kaggle/Titanic/Data/test.csv"
    print "Loading the data..."
    testfile = path + "\\" + "test.csv"
    clean_train = path + "\\" + "train_clean.csv"
    clean_test = path + "\\" + "test_clean.csv"
    df_train = pd.read_csv(clean_train)
    df_test = pd.read_csv(clean_test)
    survived = df_train.Survived
    df_train = df_train.drop(["Survived"], axis = 1)
    df = pd.concat([df_train, df_test], ignore_index = True)

    df = Del_col(df)

    X_train = df.values[:len(survived), :]
    y_train = survived.values

    X_predict = df.values[len(survived):, :]


    clf = RandomForestClassifier(random_state = 100)

    param_grid = {"n_estimators": [100, 300, 1000],
                  "max_depth": [1, 3, 10],
                  "max_features": ["auto", None],
                  "min_samples_split": [1, 3, 10],
                  "min_samples_leaf": [1, 3, 10],
                  "oob_score": [True],
                  "criterion": ["entropy"]}

    print "start model training..."
    start = time()
    grid_search = GridSearchCV(clf, param_grid = param_grid,
                               cv = StratifiedKFold(y_train, n_folds = 5, shuffle=True,
                                                    random_state= 100),
                               n_jobs = -1)
    grid_search.fit(X_train, y_train)

    print "GridSearchCV took %.1f mins for %d candidate parameter settings."\
        %((time()-start)/60, len(grid_search.grid_scores_))

    report(grid_search.grid_scores_, n_top = 3)

    best_estimator = grid_search.best_estimator_
    feature_importance = best_estimator.feature_importances_
    col = df.columns.values[np.argsort(feature_importance)[::-1]][:20]
    feature_importance = np.sort(feature_importance)[::-1][:20]
    pos = np.arange(len(col))
    plt.barh(pos, feature_importance[::-1], color = "blue", align = "center")
    plt.yticks(pos, col[::-1])
    plt.ylim([-1, 20])
    plt.title("Feature Importance Analysis")
    plt.savefig(path + "\\" + "Feat_Imp.jpg")

    # savefile = r"/Users/Angela/Documents/Kaggle/Titanic/Data/result_2.csv"
    savefile = path + "\\" + "result.csv"
    result = pd.read_csv(testfile)
    y_predict = grid_search.predict(X_predict).astype("int")
    result.insert(2, "Survived", y_predict)
    result[["PassengerId","Survived"]].to_csv(savefile, index = False)

    print "saving the grid search result..."
    scores = grid_search.grid_scores_
    scores = sorted(scores, key = lambda x: x[1], reverse = True)

    f = open(path + "\\" + "GridSearch.txt", "wb")
    for line in scores:
        f.write(str(line) + "\n")
    f.close()
    print "Grid Search result has been saved in the txt file."
    
