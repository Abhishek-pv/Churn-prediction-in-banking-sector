#Outlier detection using z-score and IQR.
import pandas as pd
from scipy.stats import zscore


class OutlierDetection:
    def __init__(self, method, df):
        self.method = method
        self.df = df
        self.cols = list(self.df.columns)
        self.clean_df = self.drop_columns()
        if self.method == "z-score":
            self.zScore()
        else:
            self.IQR()
    
    def drop_columns(self):
        clean_df = self.df.drop(['Gender', 'Geography'], axis=1)
        return clean_df

    def zScore(self):
        df_zscore = self.clean_df.apply(zscore)
        print(df_zscore.head(5))
        for j in df_zscore.iteritems(): 
            print(j)
            # if j > 3 or j < -3:
            #     print("outliers present and value is {}".format(j))
            # else:
            #     print("hurray")
        
    def IQR(self):
        print("iqr")

if __name__ == "__main__":
    data = pd.read_csv("../Data/cleaned_data.csv")
    ob = OutlierDetection("z-score", data)

