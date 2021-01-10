#Outlier detection using z-score and IQR.
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import zscore


class OutlierDetection:
    def __init__(self, method, df):
        self.method = method
        self.df = df
        self.cols = list(self.df.columns)
        self.clean_df = self.drop_columns()
        if self.method == "z-score":
            self.zScore()
        elif self.method == "iqr":
            self.IQR()
        elif self.method == "plots":
            self.box_plot()
    
    def drop_columns(self):
        clean_df = self.df.drop(['Gender', 'Geography'], axis=1)
        return clean_df

    def zScore(self):
        count_zscore = 0
        df_zscore = self.clean_df.apply(zscore)
        for col in df_zscore.columns:
            for i in range(0, len(df_zscore[col])):
                val = df_zscore[col].iloc[i]
                if val > 3 or val < -3:
                    count_zscore = count_zscore + 1
                    #print('Outliers present in the column {} and at the idex {}'.format(col,i))
        print("Outlier detected from z-score test is {}".format(count_zscore))
                    
    def IQR(self):
        count_iqr = 0
        for col in self.clean_df.columns:
            col_list = self.clean_df[col].to_list()
            col_list = np.sort(col_list)
            q1 = np.quantile(col_list, .25)
            q3 = np.quantile(col_list, .75)
            iqr = q3 - q1
            upper_limit = q3 + (1.5 *iqr)
            lower_limit = q1 - (1.5 * iqr)
            for j in range(0, len(col_list)):
                if (col_list[j] > upper_limit) or (col_list[j] < lower_limit):
                    count_iqr = count_iqr + 1
                    print('Outliers present in the column {} and at the index {}'.format(col,j))
        print("Outlier detected from IQR test is {}".format(count_iqr))


    def box_plot(self):
        self.clean_df.boxplot(column=list(self.clean_df.columns))

            

if __name__ == "__main__":
    data = pd.read_csv("../Data/cleaned_data.csv")
    #ob_iqr = OutlierDetection("IQR", data)
    #ob_zscore = OutlierDetection("z-score", data)
    ob_plot = OutlierDetection("plots", data)