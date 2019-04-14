#!/usr/bin/env python3
import os
from settings import Crawl_dict, FD_path
import logging
from datetime import datetime

# ------------------------ #
#  record.txt Maintenance  #
# ------------------------ #
class Record:

    def __init__(self, file_path):
        # file path
        self.file_path = file_path
        # record crawler function name, start date, end date.
        self.record_data = {}
        # record other information
        self.information = {} # 1. modified date
        # Init update_date
        self.update_date = None # datetime type

        # read file
        self.readfile()

    def readfile(self):
        assert os.path.isfile(self.file_path), self.file_path + " doesn't exist."

        rfile = open(self.file_path, 'r')
        for line in rfile:
            line = [x.strip() for x in line.split(',')]
            if len(line) == 0 or line[0][0] == '#': continue

            if len(line) == 3:
                self.record_data[line[0]] = (line[1], line[2])
            elif len(line) == 2:
                self.information[line[0]] = line[1]
            else:
                raise AssertionError("Error: can't read the {0}".format(self.path) )

        # check crawl_dict new keys
        for func_name, value in Crawl_dict.items():
            if self.record_data.get(func_name) is None:
                self.record_data[func_name] = (value['start'], value['start'])
                # build directory
                Write_directory = FD_path + "/" + func_name
                Check_directory_files(FD_path, Write_directory)

        # convert date
        if 'Update date' in self.information:
            self.update_date = datetime.strptime(self.information['Update date'], '%Y%m%d')

    def init_record_file(self):

        self.record_data = {}
        for func_name, value in Crawl_dict.items():
            self.record_data[func_name] = (value['start'], value['start'])

        self.information['Update date'] = '19000101'
        self.updatefile()

    def updatefile(self):
        print ('\n# Update ' + self.file_path)
        wfile = open(self.file_path, 'w')
        wfile.write('# Function name'.rjust(35) + ", " + '# Start'.rjust(10) + ", " + '# End'.rjust(10) + "\n")
        for key, value in self.record_data.items():
            wfile.write(key.rjust(35) + ", " + value[0].rjust(10) + ", " + value[1].rjust(10) + "\n")

        wfile.write('# Others'.rjust(35) + ", " + '# INFO'.rjust(22) + "\n")
        for key, value in self.information.items():
            wfile.write(key.rjust(35) + ", " + value.rjust(22) + "\n")

        wfile.close()

    def update_latest_date(self, merge_CSV = False, update_file = True):
        from Packing import Lastest_date
        for func_name, value in Crawl_dict.items():
            path = FD_path + "/" + func_name +"/"

            # only need to find latest_date (faster) #
            if not merge_CSV:
                latest_date = Lastest_date(path, value['date_format'], value['date_name'], self.update_date)
                latest_date = str(latest_date.date()).replace("-", "")

            # merge command: find latest_date first (slower) #
            elif merge_CSV:
                from Packing import Merge_csv_by_year, Concate_df
                df = Concate_df(path, value['date_format'], value['date_name'])
                if df is None or len(df.index) == 0: continue
                latest_date = str(df.index[-1].date()).replace("-", "")

            # update record.txt #
            if int(latest_date) > int(self.record_data[func_name][1]):

                print ("\t" + func_name + ": End date = " + latest_date)
                self.record_data[func_name] = (self.record_data[func_name][0], latest_date)

            # merge command: merge csv after updating record.txt #
            if merge_CSV:
                Merge_csv_by_year(df, path)

        # merge command: also need to update pkl files (S0119) #
        if merge_CSV:
            from Packing import Data_preprocessing
            Data_preprocessing(rebuild = True)

        # update file #
        if update_file: self.updatefile()

    def update_date_func(self, update_date:str,  update_file = True):
        self.information['Update date'] = update_date

        if update_file: self.updatefile()

    def get_each_data(self):
        for key, value in self.record_data.items():
            yield {'Func_name':key, 'Start': value[0], 'End': value[1]}


# ------------------- #
#   Check Functions   #
# ------------------- #
def Check_columns(col_array, filter_list, df = None):

    # Check columns
    #assert False == (False in [x in col_array for x in filter_list]), "There are some columns can't find."
    check_col = True
    for x in filter_list:
        assert x in col_array, "There are some columns can't find, example: " + x

    # Filter
    if type(df) != type(None):
         return df.filter(filter_list)

def Check_directory_files(FD_path, Write_directory, check_file = None):
    assert os.path.exists(FD_path) == True, "The directory: " + FD_path +" is not exist."

    if not os.path.exists(Write_directory):
        os.makedirs(Write_directory)
    if check_file is not None and os.path.isfile(check_file) == True:
        print ('the file: ' + check_file[len(Write_directory)+1:] + ' exists!')
        return 0

def Check_duplicated_date(df, name):
    if len(df.index.get_duplicates()) > 0:
        # find duplicated date
        logging.warning('{0} has duplicated date.\nDate : {1}'.format(name, str(df.index.get_duplicates().tolist())) )




# test
if __name__ == '__main__':
    # update test
    record = Record(FD_path + '/record.txt')
    record.update_latest_date()
