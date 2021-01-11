#converting categorical variable into a numerical using on-hot encoding.
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

class OneHotEncoding:

    def __init__(self, df, columns):
        self.df = df[columns]
        self.columns = columns
        self.df1 = self.OneHotEncode()

    def OneHotEncode(self):
        enc = OneHotEncoder(dtype = int, sparse = False)
        encoded_arr = enc.fit_transform(self.df)
        new_col_names = enc.get_feature_names(self.columns)
        new_col_names = list(new_col_names)
        new_col_names = [str(col) for col in new_col_names]
        df = pd.DataFrame(encoded_arr, columns = new_col_names)
        return df

if __name__ == "__main__":
    data = pd.read_csv("../Data/cleaned_data.csv")
    ob = OneHotEncoding(data, ['Geography', 'Gender'])
    data = data.drop(['Geography', 'Gender'], axis = 1)
    final_data = pd.concat([ob.df1, data], axis = 1)
    final_data.to_csv('../Data/cleaned_data1.csv', index=False)