#!/usr/bin/env python3
import os
from settings import Crawl_dict, FD_path

##### Maintain record.txt
class Record:

    def __init__(self, file_path):
        self.file_path = file_path
        self.record_data = {}

    def readfile(self):
        assert os.path.isfile(self.file_path), self.file_path + " doesn't exist."

        rfile = open(self.file_path, 'r')
        for line in rfile:
            
            line = [x.strip() for x in line.split(',')]
            if line[0][0] == '#': continue

            assert len(line) == 3, len(line) 
            self.record_data[line[0]] = (line[1], line[2])

    def init_record_file(self):
        # assert os.path.isfile(self.file_path) != True, self.file_path + " already exists."

        self.record_data = {}
        for func_name, value in Crawl_dict.items():
            self.record_data[func_name] = (value['start'], value['start'])
        # for index, (value1, value2) in enumerate(zip(Func_l, start_l)):
        #     self.record_data[value1] = (value2, value2)
        self.updatefile()

    def updatefile(self):
        print ('update ' + self.file_path)
        wfile = open(self.file_path, 'w')
        wfile.write('# Function name'.rjust(35) + ", " + '# Start'.rjust(10) + ", " + '# End'.rjust(10) + "\n")
        for key, value in self.record_data.items():
            wfile.write(key.rjust(35) + ", " + value[0].rjust(10) + ", " + value[1].rjust(10) + "\n")
        wfile.close()

    def update_latest_date(self, merge_CSV = False):
        from Packing import Lastest_date
        for func_name, value in Crawl_dict.items():
            path = FD_path + "/" + func_name +"/"

            # only need to find latest_date (faster) #
            if not merge_CSV:
                latest_date = Lastest_date(path, value['date_format'], value['date_name'])
                latest_date = str(latest_date.date()).replace("-", "")
            
            # merge command: find latest_date first (slower) #
            elif merge_CSV:
                from Packing import Merge_csv_by_year, Concate_df
                df = Concate_df(path, value['date_format'], value['date_name'])
                if df is None or len(df.index) == 0: continue
                latest_date = str(df.index[-1].date()).replace("-", "")

            # update record.txt #
            if int(latest_date) > int(self.record_data[func_name][1]):

                print (func_name + ": End date = " + latest_date)
                self.record_data[func_name] = (self.record_data[func_name][0], latest_date)

            # merge command: merge csv after updating record.txt #
            if merge_CSV:
                Merge_csv_by_year(df, path)

        # merge command: also need to update pkl files (S0119)
        if merge_CSV:
            from Packing import Data_preprocessing
            Data_preprocessing(rebuild = True)

        self.updatefile()

    def get_each_data(self):
        for key, value in self.record_data.items():
            yield {'Func_name':key, 'Start': value[0], 'End': value[1]}


##### End



def Check_columns(col_array, filter_list, df = None):
    
    # Check columns
    #assert False == (False in [x in col_array for x in filter_list]), "There are some columns can't find."
    check_col = True
    for x in filter_list:
        assert  x in col_array, "There are some columns can't find, example: " + x

    # Filter
    if type(df) != type(None):
         return df.filter(filter_list)

def Check_directory_files(FD_path, Write_directory, check_file = "None"):
    assert os.path.exists(FD_path) == True, "The directory: " + FD_path +" is not exist."
    
    if not os.path.exists(Write_directory):
        os.makedirs(Write_directory)
    if check_file != "None" and os.path.isfile(check_file) == True:
        print ('the file: ' + check_file[len(Write_directory)+1:] + ' exists!')
        return 0

# test
if __name__ == '__main__':
    # pass
    record = Record('record.txt')
    record.readfile()
    record.update_latest_date()




