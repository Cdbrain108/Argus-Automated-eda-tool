from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from eli5.sklearn import PermutationImportance
import eli5
import numpy as np
import pandas as pd
import text_generation
import data_cleaning
import input_file
class FeatureImportance:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def sklearn_importance(self):
        model = RandomForestClassifier()
        model.fit(self.X, self.y)
        importance = model.feature_importances_
        return self.X.columns[np.argsort(importance)[::-1]]

    def xgboost_importance(self):
        model = XGBClassifier()
        model.fit(self.X, self.y)
        importance = model.feature_importances_
        return self.X.columns[np.argsort(importance)[::-1]]

    def lightgbm_importance(self):
        model = LGBMClassifier()
        model.fit(self.X, self.y)
        importance = model.feature_importances_
        return self.X.columns[np.argsort(importance)[::-1]]

    def eli5_importance(self):
        model = RandomForestClassifier()
        model.fit(self.X, self.y)
        perm = PermutationImportance(model, random_state=1).fit(self.X, self.y)
        importance = perm.feature_importances_
        return self.X.columns[np.argsort(importance)[::-1]]

    def most_similar_columns(self):
        sklearn_cols = self.sklearn_importance()
        xgboost_cols = self.xgboost_importance()
        lightgbm_cols = self.lightgbm_importance()
        eli5_cols = self.eli5_importance()

        # Find the most similar columns
        similar_cols = set(sklearn_cols).intersection(xgboost_cols, lightgbm_cols, eli5_cols)
        return similar_cols
dataset_name, target_variable, data = input_file.load_data()             
cleaner = data_cleaning.DatasetCleaning()
