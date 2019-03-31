#!/usr/bin/env python3

import os
import sys
import datetime
import argparse

from settings import *


def Doc():
    '''
    [OBJECTIVE]
        A crawler can crawl Taiwan Capitalization Weighted Stock Index / Taiwan Index futures / Stocks.
    [USAGE]
        ./main.py [-h]
        ./main.py [-update]
        ./main.py [-rebuild] [-r]
        ./main.py [-check_m]
        ./main.py [-merge_csv]
    [EXAMPLE]
        Show more help message: ./main.py -h

    '''
#def str_to_date(string, formate = '%Y%m%d'):
#    return datetime.datetime.strptime(str(string), formate)

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
    parser.add_argument("-verbose",  action='store_true', dest='verbose',  help = "More detailed info.")

    #Packing
    parser.add_argument("-merge_csv", action = 'store_true',dest = 'merge',  help = "Merge CSV files (Generate *_year.csv)" +
                                                                                    " and remove old files. This command will also" +
                                                                                    " update record.txt")

    args = parser.parse_args()


    from Crawler import Crawler
    Crawl = Crawler(verbose = args.verbose)

    if args.update:
        Crawl.UPDATE()
    elif args.merge:
        Crawl.MERGE()
    elif args.checkm:
        Crawl.CHECK_MISSING_DATA()
    elif args.rebuild:
        Crawl.REBUILD()
    else:
        pass

'''
#0313
Stock_MonthlyRevenue 沒寫check_miss_data
'''
