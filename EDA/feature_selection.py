#Feature Selection
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

class FeatueSelection:
    def __init__(self, df, method):
        self.df = df
        if method == "corr":
            self.corr()
        elif method == "vt":
            self.VarianceThreshold()
        elif method == "uni":
            self.UnivariateSelection()
        elif method == "step":
            self.StepWise()
        else:
            self.rfe()

    def corr(self):
        corr_df = self.df.corr()
        print(corr_df.iloc[:,13])

    def VarianceThreshold(self):
        vt = VarianceThreshold()
        transformed_data = vt.fit_transform(self.df)
        print("Number of variables before selection is {}".format(len(self.df.columns)))
        print("Number of variables after selection is {}".format(transformed_data.shape[1]))

    def UnivariateSelection(self):
        X= self.df.iloc[:,0:13]
        y= self.df[['Exited']]
        available_tests= {"f_classif": f_classif,
                          "chi2": chi2,
                          "mutual_info_classif": mutual_info_classif}
        for uni, method in available_tests.items():
            selector = SelectKBest(method, k=5)
            selector.fit(X,y)
            features = selector.get_support(indices=True)
            print("Top 5 features selected from method {}:{}".format(uni,list(self.df.iloc[:,features].columns)))

    def StepWise(self):
        X= self.df.iloc[:,0:13]
        y= self.df[['Exited']]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
        features = list(X_train.columns)
        score_list = []
        score = 0.0
        best_features = []
        selected_features = []
        for feature in features:
            outputs = self.pipeline(X_train[[feature]], y_train, X_test[[feature]], y_test)
            if outputs[2] > score:
                first_feature = feature
                score = outputs[2]
        best_features.append(first_feature)
        features.remove(first_feature)
        score_list = []
        score_list.append(score)
        while len(features) > 0:
            score = 0.0
            for feature in features:
                selected_features = []
                selected_features = best_features + [feature]
                outputs = self.pipeline(X_train[selected_features], y_train, X_test[selected_features], y_test)
                if outputs[2] > score:
                    best_feature = feature
                    score = outputs[2]
            best_features.append(best_feature)
            score_list.append(score)
            features.remove(best_feature)

        self._plot(best_features, score_list)
        print(best_features)
        print(score_list)

    def rfe(self):
        X= self.df.iloc[:,0:13]
        y= self.df[['Exited']]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
        model = LogisticRegression()
        rfe = RFE(estimator = model, n_features_to_select = 5)
        selector = rfe.fit(X_train, y_train)
        features = selector.get_support()
        print(self.df.iloc[:,features].columns)
    
    def pipeline(self, X_train, y_train, X_test, y_test):
        model = self.fit(X_train, y_train)
        predictions = self.predict(model, X_test)
        score_value = self.score(y_test, predictions)
        return (model, predictions, score_value)

    def fit(self, X, y):
        model = LogisticRegression().fit(X,y)
        return model

    def predict(self, model, test):
        predictions = model.predict(test)
        return predictions

    def score(self,actual, predictions):
        score = f1_score(actual, predictions)
        return score

    def _plot(self, list1, list2):
        plt.plot(list1, list2)


if __name__ == "__main__":
    data = pd.read_csv("../Data/engineered_data.csv")
    print(data.head(1))
    ob_corr = FeatueSelection(data, "corr")
    ob_uni = FeatueSelection(data, "uni")
    ob_vt = FeatueSelection(data, "vt")
    ob_step = FeatueSelection(data, "step")
    ob_rfe = FeatueSelection(data, "rfe")

    #based on the analysis above done.
    #features I'm selecting are
    #1. Age
    #2. Geography_Germany
    #3. Is active member
    #4. Geography male
    #5. Balance
    final_data = data[['Age', 'Geography_Germany', 'IsActiveMember', 'Balance', 'Gender_Male', 'Exited']]
    final_data.to_csv('../Data/final_data.csv', index=False)