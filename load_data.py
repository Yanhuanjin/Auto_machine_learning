import pandas as pd
"""
import class
Usage: 
myloader = LoadData()
data = myloader.load_data("target.csv")
return: pandas data_frame
"""
class LoadData(object):
    def __init__(self):
        self.data = None
    def load_data(self, data_name):
        """
        :param data_name: input the filename, define the file type
        :return: the content of the data
        """
        if data_name.endswith(".csv"):
            print("The input file is .csv file.")
            self.data = pd.read_csv(data_name)
        elif data_name.endswith(".xlsx"):
            print("The input file is excel file.")
            self.data = pd.read_excel(data_name)
        else:
            raise Exception("File Not Support")
        return self.data