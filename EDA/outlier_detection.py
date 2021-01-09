#Outlier detection using z-score and IQR.
import pandas as pd
import numpy as np
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
        for col in df_zscore.columns:
            for i in range(0, len(df_zscore[col])):
                val = df_zscore[col].iloc[i]
                if val > 3 or val < -3:
                    print('Outliers present in the column {} and at the idex {}'.format(col,i))
            break
                    
    def IQR(self):
        for col in self.clean_df.columns:
            q1, q3, iqr = 0, 0, 0
            col_list = self.clean_df[col].to_list()
            q1 = np.quantile(col_list, .25)
            q3 = np.quantile(col_list, .75)
            iqr = q3 - q1
            for j in range(0, len(col_list)):
                if col_list[j] > (q3 + (1.5 * iqr)) or col_list[j] < (q1 - (1.5 * iqr)):
                    print('Outliers present in the column {} and at the index {}'.format(col,j))
            print("********")
            

if __name__ == "__main__":
    data = pd.read_csv("../Data/cleaned_data.csv")
    ob = OutlierDetection("IQR", data)
    #ob1 = OutlierDetection("z-score", data)