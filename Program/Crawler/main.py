#!/usr/bin/env python3

import os
import sys
import datetime
import argparse

from settings import *


def Doc():
    '''
    [OBJECTIVE]
        A crawler can crawl Taiwan Capitalization Weighted Stock Index and Taiwan Index futures.
    [USAGE]
        ./main.py [-h]
        ./main.py [-update] 
        ./main.py [-rebuild] [-r] 
        ./main.py [-check_m]
        ./main.py [-merge_csv]
    [EXAMPLE]
        Show more help message: ./main.py -h 

    '''
def str_to_date(string, formate = '%Y%m%d'):
    return datetime.datetime.strptime(str(string), formate)

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(Doc.__doc__)
        sys.exit()

    parser = argparse.ArgumentParser(usage=Doc.__doc__ )
    #Crawler
    parser.add_argument("-update",  action = 'store_true', dest = 'update',  help = "Update the data.")
    parser.add_argument("-rebuild", action = 'store_true', dest = 'rebuild', help = "Remove and rebuild all the data.")
    parser.add_argument("-checkm", action = 'store_true',  dest = 'checkm',  help = "Check all the missing data and execute \"merge_csv\".")    
    #parser.add_argument("-period",  nargs='+',           dest='period',  help= "Crawl data during a period.")
    #parser.add_argument("-verbose",  action='store_true', dest='verbose',  help="More detailed info.")

    #Packing
    parser.add_argument("-merge_csv", action = 'store_true',dest = 'merge',  help = "Merge CSV files (Generate *_year.csv)" +
                                                                                    " and remove old files. This command will also" + 
                                                                                    " update record.txt")

    args = parser.parse_args()

    # today
    today = datetime.datetime.now()
    if   today.weekday() == 6: # Sunday
        today = today - datetime.timedelta(days=2)
    elif today.weekday() == 5: # Saturday
        today = today - datetime.timedelta(days=1)
    elif today.hour < 15: # Some data maybe cannot update 
        today = today - datetime.timedelta(days=1)
    today = today.strftime("%Y%m%d")

    from Check import Record
    from Crawler import  Wrapper
    if args.update:

        #1.  update record first
        print ("# 1. Update End date in record.txt")
        record = Record(FD_path + '/record.txt')
        record.readfile()
        record.update_latest_date()

        #2.  Update data
        print ("# 2. Update data.")
        for func_data in record.get_each_data():
            if str_to_date(func_data['End']) >=  str_to_date(today): continue

            start_date = str_to_date(func_data['End']) + datetime.timedelta(days=1)
            start_date = str(start_date.date()).replace("-", "")
            Wrapper(start_date, today, [ func_data['Func_name'] ] ) 

        print ("\nDone.")
    elif args.merge:
        record = Record(FD_path + '/record.txt')
        record.readfile()
        record.update_latest_date(True)
    elif args.checkm:
        # 1. Check Missing Data
        print ("# 1. Check missing data.")
        from Packing import Check_miss_data
        Check_miss_data()
        # 2. Merge 
        print ("# 2. Merge data.")
        record = Record(FD_path + '/record.txt')
        record.readfile()
        record.update_latest_date(True)

    elif args.rebuild:
        reply = str(input('This command will remove all the data from the directory: Finance_data/Parse ' +
                          'and rebuild the data.\nAre you sure you want to do this? (y/n): ')).lower().strip()
        if reply == 'y':
            # 1. Delete 
            for (dirpath, dirnames, filenames) in os.walk(FD_path):
                print ("rm files in " + dirpath)
                for file in filenames:
                    if file.endswith(".csv"):
                        os.remove(os.path.join(dirpath, file))
            
            # 2. Init record.txt
            record = Record(FD_path + '/record.txt')
            record.init_record_file()

            # 3. Rebuild
            for func_data in record.get_each_data():
                start_date = datetime.datetime.strptime(str(func_data['End']), '%Y%m%d') + datetime.timedelta(days=1)
                start_date = str(next_date.date()).replace("-", "")
                Wrapper(start_date, today, [ func_data['Func_name'] ] ) 
    else:
        pass

'''

'''