import pandas as pd

class Processor(object):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def fill_null(self, col, flag):
        if flag == "pad":
            return self.data[col].fillna(method='pad', inplace=True)
        elif flag == "bfill":
            return self.data[col].fillna(method='bfill', inplace=True)
        elif flag == "mean":
            return self.data[col].fillna(self.data[col].mean(), inplace=True)
        elif flag == "interpolate":
            return self.data[col].interpolate()

    def date_transfer(self, col, flag):
        self.data[col] = pd.to_datetime(self.data[col])
        if flag == "week":
            self.data.loc[:, 'day_of_week'] = self.data[col].dt.dayofweek
        elif flag == "year":
            self.data.loc[:, 'day_of_year'] = self.data[col].dt.dayofyear
        self.data.drop(columns=col, inplace=True)
        return self.data

    def drop_uni(self):
        for col in self.data.columns:
            if len(self.data[col].value_counts()) == 1:
                print(">> Drop uni col：%s" % col)
                return col
