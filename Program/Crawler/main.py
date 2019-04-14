#!/usr/bin/env python3

#import os
import sys
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

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(Doc.__doc__)
        sys.exit()

    parser = argparse.ArgumentParser(usage=Doc.__doc__ )
    #Crawler
    parser.add_argument("-update",  action = 'store_true', dest = 'update',  help = "Update the data.")
    parser.add_argument("-force_update",  action = 'store_true', dest = 'update_f',  help = "Force to update the data.")
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
    elif args.update_f:
        Crawl.UPDATE(must_update = True)
    elif args.merge:
        Crawl.MERGE()
    elif args.checkm:
        Crawl.CHECK_MISSING_DATA()
    elif args.rebuild:
        Crawl.REBUILD()
    else:
        pass

'''
# TODO LIST
1. Iplotly + MA
評估交易費用
有300多股票的OHLCV 根本沒data

#0406
1. Stock_MonthlyRevenue/Financial_Income/Financial_Balance_Sheet 沒寫check_miss_data
2. 現金流量表 http://mops.twse.com.tw/mops/web/t164sb05

# 0407
1. DP: datapreprocessing 不需要每次都跑
2. DP update pkl 應該要保證pkl把xxx_year.csv都讀過了 (或是改成檢查modified date就好)

# Done
1. bug fixes
2. Plot_3_Investors 配futures: bug fix
add 財報損益表&負債表
'''
