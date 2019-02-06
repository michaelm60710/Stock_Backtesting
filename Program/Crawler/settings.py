#!/usr/bin/env python3
import os

PKL_path = os.path.abspath(""+"../../Finance_data/Pickle")
FD_path = os.path.abspath(""+"../../Finance_data/Parse")

Func_l = ['Institutional_investors', 'TaiExchange', 'TaiExchange_OHLC', \
          'Taifutures', 'Taifutures_Investors', 'Taifutures_LargeTrade', 'MTX', \
          'MTX_Investors', 'TX_Investors', \
          'Stock_Price', 'Stock_Investors']

Crawl_dict = {'Institutional_investors': {'start':'20040407','date_format':'%Y%m%d',   'date_name': '日期'    },
              'TaiExchange':             {'start':'20000101','date_format':'%Y-%m-%d', 'date_name': '日期'    },
              'TaiExchange_OHLC':        {'start':'20000101','date_format':'%Y-%m-%d', 'date_name': '日期'    },
              'Taifutures':              {'start':'20110901','date_format':'%Y/%m/%d', 'date_name': '交易日期' },
              'MTX':                     {'start':'20110901','date_format':'%Y/%m/%d', 'date_name': '交易日期' },
              'Taifutures_Investors':    {'start':'20070702','date_format':'%Y/%m/%d', 'date_name': '日期'     },
              'MTX_Investors':           {'start':'20151231','date_format':'%Y/%m/%d', 'date_name': '日期'     },
              'TX_Investors':            {'start':'20151231','date_format':'%Y/%m/%d', 'date_name': '日期'     },
              'Taifutures_LargeTrade':   {'start':'20040701','date_format':'%Y/%m/%d', 'date_name': '日期'     },
              # Stocks
              'Stock_Price':             {'start':'20050114','date_format':'%Y%m%d',   'date_name': '日期'     },
              'Stock_Investors':         {'start':'20120502','date_format':'%Y%m%d',   'date_name': '日期'     },
              }
'''
# Question: How to add new crawler function?
    1. Update Func_l, Crawl_dict
    2. Update Wrapper
    3. Update record.txt
    4. Make new dirctory
    5. Update Check_miss_data 
    6. Update datapreprocessing
'''