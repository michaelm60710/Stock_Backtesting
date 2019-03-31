#!/usr/bin/env python3
import pandas as pd
import numpy as np
import requests
from io import StringIO
import time
import os
import datetime
from dateutil.relativedelta import relativedelta

from settings import Crawl_dict, FD_path
from Check import *

#Global variables
global ua  # 偽瀏覽器
ua = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.186 Safari/537.36',
    'Connection':'Keep-Alive',
    'Accept-Language':'zh-CN,zh;q=0.8',
    'Accept-Encoding':'gzip,deflate,sdch',
    'Accept':'*/*',
    'Accept-Charset':'GBK,utf-8;q=0.7,*;q=0.3',
    'Cache-Control':'max-age=0'
    }

class Crawler:
    '''
        # 4 commands:
        1. UPDATE
        2. MERGE
        3. CHECK_MISSING_DATA
        4. REBUILD
    '''
    verbose = False
    record = None

    def __init__(self, verbose = False):
        self.verbose = verbose



    def UPDATE(self):
        today = self.Get_today()

        #1.  update record first
        print ("# 1. Update End date in record.txt")
        self.record = Record(FD_path + '/record.txt')
        self.record.readfile()
        self.record.update_latest_date()

        #2.  Update data
        print ("# 2. Update data.")
        for func_data in self.record.get_each_data():
            if self.str_to_date(func_data['End']) >=  self.str_to_date(today): continue

            start_date = self.str_to_date(func_data['End']) + datetime.timedelta(days=1)
            start_date = str(start_date.date()).replace("-", "")
            self.Wrapper(start_date, today, [ func_data['Func_name'] ] )

        print ("\n# 3. Done.")

    def MERGE(self):
        # Merge CSV files (Generate *_year.csv) and remove old files.
        # It will also update record.txt
        self.record = Record(FD_path + '/record.txt')
        self.record.readfile()
        self.record.update_latest_date(True)

    def CHECK_MISSING_DATA(self):
        # 1. Check Missing Data
        print ("# 1. Check missing data.")

        # 1.2 Get Missing data date
        from Packing import Check_miss_data
        Miss_data_date_dict = Check_miss_data()

        for func in Miss_data_date_dict.keys():
            # Month
            if func == 'Taifutures_LargeTrade' or func == 'Taifutures' \
            or func == 'TaiExchange_OHLC' or func == 'TaiExchange' or func == 'MTX':
                month_id = Miss_data_date_dict[func].copy()
                month_id = np.unique( month_id.map(lambda x: x.replace(day = 1)) )
                for date in month_id:
                    start = str(date.date()).replace("-", "")
                    end = str(date.date() + relativedelta(months= 1)).replace("-", "")

                    self.Wrapper(start, end, [ func ] )
            # Date
            elif func == 'Institutional_investors' or func == 'Taifutures_Investors' \
            or func == 'MTX_Investors' or func == 'TX_Investors' \
            or func == 'Stock_Price' or func == 'Stock_Investors':
                for date in Miss_data_date_dict[func]:
                    start = str(date.date()).replace("-", "")
                    end = start

                    self.Wrapper(start, end, [ func ] )
            else:
                print ("ERROR: didn't define " + func + " in def Check_miss_data")

        # 2. Merge
        print ("# 2. Merge data.")
        self.MERGE()

    def REBUILD(self):
        reply = str(input('This command will remove all the data from the directory: Finance_data/Parse ' +
                  'and rebuild the data.\nAre you sure you want to do this? (y/n): ')).lower().strip()

        if reply == 'y':
            # 1. Delete
            print ("# 1. Remove files.")
            for (dirpath, dirnames, filenames) in os.walk(FD_path):
                print ("\trm files in " + dirpath)
                for file in filenames:
                    if file.endswith(".csv"):
                        os.remove(os.path.join(dirpath, file))

            # 2. Init record.txt
            self.record = Record(FD_path + '/record.txt')
            self.record.init_record_file()

            # 3. build directories
            print ("# 2. Build/Check directories.")
            if not os.path.exists(PKL_path): os.makedirs(PKL_path)
            if not os.path.exists(FD_path): os.makedirs(FD_path)
            for func_data in self.record.get_each_data():
                Write_directory = FD_path + "/" + func_data['Func_name']
                if not os.path.exists(Write_directory):
                    os.makedirs(Write_directory)

            # 4. Use ./main -u
            print ("# 3. Rebuild the directories, please use command: \"./main.py -u\" to update DATA. ")

    def Get_today(self):
        # get today
        today = datetime.datetime.now()
        if   today.weekday() == 6: # Sunday
            today = today - datetime.timedelta(days=2)
        elif today.weekday() == 5: # Saturday
            today = today - datetime.timedelta(days=1)
        elif today.hour < 15:      # Some data maybe cannot update
            today = today - datetime.timedelta(days=1)
        today = today.strftime("%Y%m%d")
        return today

    def str_to_date(self, string, formate = '%Y%m%d'):
        return datetime.datetime.strptime(str(string), formate)

    def Gen_xMonth(self, start_date, end_date, month, Func, Func_type = ''):
        '''
            example: Gen_xMonth("20160101", "20180201", 3, Taifutures_LargeTrade)
        '''

        print("\n# {0}: {1} to {2}".format(Func.__name__, start_date, end_date) )

        start = datetime.datetime.strptime(str(start_date), '%Y%m%d')
        end = datetime.datetime.strptime(str(end_date), '%Y%m%d')
        next_month = relativedelta(months= month)
        Do_it_again = False
        if start == end: Do_it_again = True
        while start < end:
            start_next = start + next_month
            if(start_next > end ):
                start_next = end
                if Func_type == 'YM' and start.strftime('%m') != end.strftime('%m'): Do_it_again = True
            if Func_type == 'YM':
                if self.verbose: print (start.strftime('%Y%m%d')[:6])
                Func(start.strftime('%Y%m%d')[:4], start.strftime('%Y%m%d')[4:6], 1)
            else:
                if self.verbose: print (start.strftime('%Y%m%d')+ " to " + start_next.strftime('%Y%m%d'))
                Func(start.strftime('%Y%m%d'), start_next.strftime('%Y%m%d'), 1)
            start = start_next
        #Do one more time
        if Do_it_again:
            if Func_type == 'YM':
                if self.verbose: print ("last one: " + start.strftime('%Y%m%d')[:6])
                Func(start.strftime('%Y%m%d')[:4], start.strftime('%Y%m%d')[4:6], 1)
            else:
                if self.verbose: print ("last one: " + start.strftime('%Y%m%d')+ " to " + end.strftime('%Y%m%d'))
                Func(start.strftime('%Y%m%d'), end.strftime('%Y%m%d'), 1)

    def Gen_xDay(self, start_date, end_date, Func, Func_type = ''):
        '''
            example: Gen_xDay("20160101", "20180201", Institutional_investors)
        '''

        print("\n# {0}: {1} to {2}".format(Func.__name__, start_date, end_date) )

        start = datetime.datetime.strptime(str(start_date), '%Y%m%d')
        end = datetime.datetime.strptime(str(end_date), '%Y%m%d')
        step = datetime.timedelta(days=1)
        while start <= end:
            YMD = str(start.date()).replace("-", "")
            if self.verbose: print (YMD)
            Func(YMD, 1)
            start += step

    def Wrapper(self, start: str, end: str, Func_list):
        '''
        [OBJECTIVE]
            A wrapper run other functions in Crawler.py
        [USAGE]
            start & end can be int or string type
        Example:
            Wrapper('20181101', '20181109', [Institutional_investors, TaiExchange])

        [RETURN]

        '''
        for func_name in Func_list:
            if func_name == 'Taifutures_LargeTrade':
                self.Gen_xMonth(start, end, 2, Taifutures_LargeTrade)
            elif func_name == 'Taifutures':
                self.Gen_xMonth(start, end, 1, Taifutures)
            elif func_name == 'MTX':
                self.Gen_xMonth(start, end, 1, MTX)
            elif func_name == 'Taifutures_Investors':
                Taifutures_Investors(start, end, 1)
            elif func_name == 'MTX_Investors':
                MTX_Investors(start, end, 1)
            elif func_name == 'TX_Investors':
                TX_Investors(start, end, 1)
            elif func_name == 'TaiExchange_OHLC':
                self.Gen_xMonth(start, end, 1, TaiExchange_OHLC, "YM")
            elif func_name == 'TaiExchange':
                self.Gen_xMonth(start, end, 1, TaiExchange, "YM")
            elif func_name == 'Stock_MonthlyRevenue':
                end   = self.str_to_date(end)
                start = self.str_to_date(start)
                if end.day > 10: end -= relativedelta(months= 1)
                else           : end -= relativedelta(months= 2)
                if start.month != end.month and start < end:
                    self.Gen_xMonth(start, end, 1, Stock_MonthlyRevenue, "YM")
            elif func_name == 'Financial_C_Income':
                # quarterly
                pass
            elif func_name == 'Institutional_investors':
                self.Gen_xDay(start, end, Institutional_investors)
            elif func_name == 'Stock_Price':
                self.Gen_xDay(start, end, Stock_Price)
            elif func_name == 'Stock_Investors':
                self.Gen_xDay(start, end, Stock_Investors)
            else:
                raise AssertionError("Error: didn't define " + func_name )

####------------------------------------------------------------------------------------------------------------------------------
####------------------------------------------------------------------------------------------------------------------------------
####------------------------------------------------------------------------------------------------------------------------------
#當日三大法人 大盤買賣超額
def Institutional_investors(YMD, write=0):
    '''
    [OBJECTIVE]
    Parsinging Institutional investors:
        1. Foreign Investors
        2. Investment Trust
        3. Dealer
    [USAGE]
    Institutional_investors(YMD, write)
    YMD is string type, write is boolean(1 or 0)
    Example:
        Institutional_investors('20180430', 0)

    [RETURN]
    A dataframe with 3 columns
    '''
    YMD = str(YMD)
    if(len(YMD)!=8):
        print("Wrong INPUT format:",YMD )
        print("example: YMD = 20180430")
        return 0

    Foreign_Investors_path = FD_path + "/Institutional_investors"
    write_path = Foreign_Investors_path + "/" + YMD + ".csv"

    if(write == 1): #check directory & file
        if not os.path.exists(FD_path):
            print("The directory: " + FD_path +" is not exist.")
            return 0
        if not os.path.exists(Foreign_Investors_path):
            os.makedirs(Foreign_Investors_path)
        if os.path.isfile(write_path):
            print('the file: ' + YMD + ' exists!')
            return 0

    url = "http://www.twse.com.tw/fund/BFI82U?response=csv&dayDate="+YMD+"&type=day&_=1525186484440"

    #Check
    try:
        time.sleep(3)
        r = requests.post(url, headers = ua)
        # 如果響應狀態碼不是 200，就主動拋出異常
        r.raise_for_status()
        if len(r.text) < 5:
            raise Exception('the file '+ YMD + ' is empty!')
            return 0

        df = pd.read_csv(StringIO(r.text), header=1, thousands=',')
        #remove unnecessary columns
        df = df.filter(['單位名稱', '買賣差額'], axis = 1)

        #remove NaN rows
        df = df[~df['買賣差額'].isnull()]
        #to numeric
        df['買賣差額'] = pd.to_numeric(df['買賣差額'], errors='coerce')
        外資_volume = [df[df['單位名稱'].str.contains("外資")]['買賣差額'].sum()]
        自營商_volume = [df[df['單位名稱'].str.match("自營商")]['買賣差額'].sum()]
        投信_volume = [df[df['單位名稱'].str.contains("投信")]['買賣差額'].sum()]
        df = pd.DataFrame(data = {'外資買賣差額': 外資_volume, '自營商買賣差額': 自營商_volume,
                                  '投信買賣差額': 投信_volume})
        df['日期'] = [YMD]

        if(write == 1):#write the file
            df.to_csv(write_path, encoding = "big5", index = False)
            print('write the file: ' + YMD)


        return df

    except requests.RequestException as e:
        print(e)
        return 0
    except Exception as e:
        print(e)
        return 0

####------------------------------------------------------------------------------------------------------------------------------
#成交股數,成交金額,成交筆數,發行量加權股價指數,漲跌點數
def TaiExchange(year, month, write = 0):
    '''
    [OBJECTIVE]
    Parsinging TaiExchange:
        成交股數,成交金額,成交筆數,發行量加權股價指數,漲跌點數
    [USAGE]
    TaiExchange(year, month, write)
    year & month are string type, write is boolean(1 or 0)
    Example:
        TaiExchange('2017','01','1')

    [RETURN]
    A dataframe
    '''
    month = str(month)
    year = str(year)
    if(len(month)==1):
        month = "0" + month

    TaiEx2_path = FD_path + "/TaiExchange"
    write_path = TaiEx2_path + "/" + year + "_" + month + ".csv"

    if(write == 1): #check directory & file
        if not os.path.exists(FD_path):
            print("The directory: " + FD_path +" is not exist.")
            return 0
        if not os.path.exists(TaiEx2_path):
            os.makedirs(TaiEx2_path)

    url = "http://www.twse.com.tw/exchangeReport/FMTQIK?response=csv&date=" + year + month + "01&_=1525251000849"

    #Check
    try:
        # 偽停頓
        time.sleep(5)
        # 下載該年月的網站，並用pandas轉換成 dataframe
        r = requests.get(url, headers = ua)
        r.raise_for_status()
        if len(r.text) < 5:
            raise Exception('the file '+ year + "_" + month + ' is empty!')
            return 0

        df = pd.read_csv(StringIO(r.text), header=1)
        df = df.drop(df.columns[df.columns.str.contains('^Unnamed')], axis = 1)
        df = df[~df['成交金額'].isnull()]

        #convert "107/04/04" to "2018-04-04"
        df['日期'] = ['-'.join(tup[1] if(tup[0]!=0) else str(int(tup[1])+1911)
                        for tup in enumerate(date1.split('/'))) for date1 in df['日期'] ]

        if(write == 1): #write the file
            if os.path.isfile(write_path):
                read_df = pd.read_csv(write_path, encoding='big5')
                if(len(read_df) < len(df)):
                    df.to_csv(write_path, encoding='big5', index = False)
                    print('update the file: ' + year + "_" + month)
                else:
                    print('the file: ' + year + "_" + month + ' exists and don\'t need to update!')
            else:
                df.to_csv(write_path, encoding='big5', index = False)
                print('write the file: ' + year + "_" + month)

        return df

    except requests.RequestException as e:
        print(e)
        return 0
    except Exception as e:
        print(e)
        return 0

####------------------------------------------------------------------------------------------------------------------------------
#台灣加權指數 Open, High, Low, Close
def TaiExchange_OHLC(year, month, write = 0):
    '''
    [OBJECTIVE]
    Parsinging TaiExchange_OHLC:
        Open, High, Low, Close
    [USAGE]
    TaiExchange_OHLC(year, month, write)
    year & month are string type, write is boolean(1 or 0)
    Example:
        TaiExchange_OHLC('2017','01', 0)

    [RETURN]
    A dataframe
    '''

    month = str(month)
    year = str(year)
    if(len(month)==1):
        month = "0" + month

    TaiEx_path = FD_path + "/TaiExchange_OHLC"
    write_path = TaiEx_path + "/" + year + "_" + month + ".csv"

    if(write == 1): #check directory & file
        if not os.path.exists(FD_path):
            print("The directory: " + FD_path +" is not exist.")
            return 0
        if not os.path.exists(TaiEx_path):
            os.makedirs(TaiEx_path)
        #if os.path.isfile(write_path):
        #    print('the file: ' + year + "_" + month + ' exists!')
        #    return 0

    url = 'http://www.twse.com.tw/indicesReport/MI_5MINS_HIST?response=csv&date=' + year + month + '01&_=1522825377229';

    #Check
    try:
        # 偽停頓
        time.sleep(5)
        # 下載該年月的網站，並用pandas轉換成 dataframe
        r = requests.get(url, headers = ua)
        r.raise_for_status()
        if len(r.text) < 5:
            raise Exception('the file '+ year + "_" + month + ' is empty!')
            return 0

        df = pd.read_csv(StringIO(r.text), header=1)
        df = df.drop(df.columns[df.columns.str.contains('^Unnamed')], axis = 1)
        #convert "107/04/04" to "2018-04-04"
        df['日期']  = ['-'.join(tup[1] if(tup[0]!=0) else str(int(tup[1])+1911)
                           for tup in enumerate(date1.split('/'))) for date1 in df['日期'] ]

        if(write == 1): #write the file
            if os.path.isfile(write_path):
                read_df = pd.read_csv(write_path, encoding='big5')
                if(len(read_df) < len(df)):
                    df.to_csv(write_path, encoding='big5', index = False)
                    print('update the file: ' + year + "_" + month)
                else:
                    print('the file: ' + year + "_" + month + ' exists and don\'t need to update!')
            else:
                df.to_csv(write_path, encoding='big5', index = False)
                print('write the file: ' + year + "_" + month)

        return df

    except requests.RequestException as e:
        print(e)
        return 0
    except Exception as e:
        print(e)
        return 0

####------------------------------------------------------------------------------------------------------------------------------
#台指期資訊: '交易日期', '契約', '到期月份(週別)', '開盤價',  '最高價', '最低價', '收盤價',
#           '漲跌價', '漲跌%','成交量', '結算價', '未沖銷契約數', '交易時段'
def Taifutures(start, end, write = 0 , id_select = 'TX'):
    '''
    [OBJECTIVE]
    Parsinging Taiwan Index Futures
    [USAGE]
    taifex(start, end , write)
    start & end can be int or string type, write is boolean(1 or 0)
    Example:
        taifutures('20181101', '20181109', 0)

    [RETURN]
    A dataframe
    '''

    start = str(start)
    end   = str(end)

    if id_select == 'TX':
        taif_path = FD_path + "/Taifutures"
    elif id_select == 'MTX':
        taif_path = FD_path + "/MTX"
    else:
        raise AssertionError("Error: didn't define " + id_select )

    write_path = taif_path + "/" + start + "_" + end + ".csv"

    assert len(start)==8 and len(end)==8, "Wrong INPUT format: " + start + ' '+ end + \
                                          "\nexample: start (end) = 20180430"
    start = start[:4] + '/' + start[4:6] + '/' + start[6:]
    end   = end[:4] + '/' + end[4:6] + '/' + end[6:]

    if write == 1 and Check_directory_files(FD_path, taif_path, write_path) == 0:
        return 0



    url = 'http://www.taifex.com.tw/cht/3/dlFutDataDown'
    form_data = {'down_type': '1',
                 'commodity_id': id_select,
                 'queryStartDate': start,
                 'queryEndDate': end}
    #Check
    try:
        # 偽停頓
        time.sleep(5)

        r = requests.post(url, headers = ua, data = form_data)
        r.raise_for_status()

        df = pd.read_csv(StringIO(r.text), index_col = False)

        #Check columns & Filter
        filter_list = ['交易日期', '契約', '到期月份(週別)', '開盤價',  '最高價', '最低價', '收盤價',
                       '漲跌價', '漲跌%','成交量', '結算價', '未沖銷契約數', '交易時段']
        df = Check_columns(df.columns.values, filter_list, df)

        if(write == 1):#write the file
            df.to_csv(write_path, encoding = "big5", index = False)
            print('write the file: ' + start + " to " + end)

        return df

    except requests.RequestException as e:
        print(e)
        return 0
    except Exception as e:
        print(e)
        return 0

####------------------------------------------------------------------------------------------------------------------------------
#期貨總資訊: 三大法人 多空 交易口數, 金額, 未平倉口數
def Taifutures_Investors(start, end, write = 0 ):
    '''
    [OBJECTIVE]
    Parsinging Taiwan Index Futures Institutional Investors
    [USAGE]
    taifex(start, end , write)
    start & end can be int or string type, write is boolean(1 or 0)
    start date must be later than 2007/07/02
    Example:
        Taifutures_Investors('20181101', '20181109', 0)

    [RETURN]
    A dataframe
    '''
    start = str(start)
    end   = str(end)
    print("\n# Taifutures_Investors: " + start + " to " + end )

    taif_path = FD_path + "/Taifutures_Investors"
    write_path = taif_path + "/" + start + "_" + end + ".csv"

    assert len(start)==8 and len(end)==8, "Wrong INPUT format: " + start + ' '+ end + \
                                          "\nexample: start (end) = 20180430"
    start = start[:4] + '/' + start[4:6] + '/' + start[6:]
    end   = end[:4] + '/' + end[4:6] + '/' + end[6:]

    if write == 1 and Check_directory_files(FD_path, taif_path, write_path) == 0:
        return 0



    url = 'http://www.taifex.com.tw/cht/3/dlTotalTableDateDown'
    form_data = {'firstDate': '2000/11/05 00:00',
                 'queryStartDate': start,
                 'queryEndDate': end}
    #Check
    try:
        # 偽停頓
        time.sleep(2)

        r = requests.post(url, headers = ua, data = form_data)
        r.raise_for_status()

        df = pd.read_csv(StringIO(r.text), index_col = False)

        #Check columns & Filter
        filter_list = ['日期', '身份別', '多方交易口數', '多方交易契約金額(百萬元)', '空方交易口數', '空方交易契約金額(百萬元)',
                       '多空交易口數淨額', '多空交易契約金額淨額(百萬元)', '多方未平倉口數', '多方未平倉契約金額(百萬元)',
                       '空方未平倉口數','空方未平倉契約金額(百萬元)', '多空未平倉口數淨額', '多空未平倉契約金額淨額(百萬元)']
        df = Check_columns(df.columns.values, filter_list, df)

        if(write == 1):#write the file
            df.to_csv(write_path, encoding = "big5", index = False)
            print('write the file: ' + start + " to " + end)

        return df

    except requests.RequestException as e:
        print(e)
        return 0
    except Exception as e:
        print(e)
        return 0

####------------------------------------------------------------------------------------------------------------------------------
#台指期資訊: '日期', '商品(契約)', '商品名稱(契約名稱)', '到期月份(週別)', '交易人類別', '前五大交易人買方',
#           '前五大交易人賣方', '前十大交易人買方', '前十大交易人賣方', '全市場未沖銷部位數'
def Taifutures_LargeTrade(start, end, write = 0 ):
    '''
    [OBJECTIVE]
    Parsinging Taiwan Index Futures LargeTrade
    [USAGE]
    taifex(start, end , write)
    start & end can be int or string type, write is boolean(1 or 0)
    start date must be later than 2007/07/02
    Example:
        Taifutures_LargeTrade('20181101', '20181109', 0)

    [RETURN]
    A dataframe
    '''

    start = str(start)
    end   = str(end)

    taif_path = FD_path + "/Taifutures_LargeTrade"
    write_path = taif_path + "/" + start + "_" + end + ".csv"

    assert len(start)==8 and len(end)==8, "Wrong INPUT format: " + start + ' '+ end + \
                                          "\nexample: start (end) = 20180430"
    start = start[:4] + '/' + start[4:6] + '/' + start[6:]
    end   = end[:4] + '/' + end[4:6] + '/' + end[6:]

    if write == 1 and Check_directory_files(FD_path, taif_path, write_path) == 0:
        return 0



    url = 'http://www.taifex.com.tw/cht/3/dlLargeTraderFutDown'
    form_data = {'queryStartDate': start,
                 'queryEndDate': end}
    #Check
    try:
        # 偽停頓
        time.sleep(5)

        r = requests.post(url, headers = ua, data = form_data)
        r.raise_for_status()

        df = pd.read_csv(StringIO(r.text), index_col = False)

        #Check columns & Filter
        filter_list = ['日期', '商品(契約)', '商品名稱(契約名稱)', '到期月份(週別)', '交易人類別', '前五大交易人買方',
                       '前五大交易人賣方', '前十大交易人買方', '前十大交易人賣方', '全市場未沖銷部位數']
        df = Check_columns(df.columns.values, filter_list, df)
        df['商品(契約)'] = df['商品(契約)'].str.strip()
        df = df[df['商品(契約)'] == 'TX']

        if(write == 1):#write the file
            df.to_csv(write_path, encoding = "big5", index = False)
            print('write the file: ' + start + " to " + end)

        return df

    except requests.RequestException as e:
        print(e)
        return 0
    except Exception as e:
        print(e)
        return 0

####------------------------------------------------------------------------------------------------------------------------------
#小台指資訊: 三大法人 多空 交易口數, 金額, 未平倉口數
def MTX_Investors(start, end, write = 0, id_select = 'MXF'):
    '''
    [OBJECTIVE]
    Parsinging Institutional investors:
        1. Foreign Investors
        2. Investment Trust
        3. Dealer
    [USAGE]
    MTX_Investors(start, end , write)
    YMD is string type, write is boolean(1 or 0)
    Example:
        MTX_Investors('20181101', '20181109', 0)

    [RETURN]
    A dataframe
    '''
    start = str(start)
    end   = str(end)


    if id_select == 'MXF':
        taif_path = FD_path + "/MTX_Investors"
        print("\n# MTX_Investors: " + start + " to " + end )
    elif id_select == 'TXF':
        taif_path = FD_path + "/TX_Investors"
        print("\n# TX_Investors: " + start + " to " + end )
    else:
        raise AssertionError("Error: didn't define " + id_select )

    #taif_path = FD_path + "/MTX_Investors"
    write_path = taif_path + "/" + start + "_" + end + ".csv"

    assert len(start)==8 and len(end)==8, "Wrong INPUT format: " + start + ' '+ end + \
                                          "\nexample: start (end) = 20180430"
    start = start[:4] + '/' + start[4:6] + '/' + start[6:]
    end   = end[:4] + '/' + end[4:6] + '/' + end[6:]

    if write == 1 and Check_directory_files(FD_path, taif_path, write_path) == 0:
        return 0



    url = 'https://www.taifex.com.tw/cht/3/futContractsDateDown'
    # firstDate no use
    form_data = {'firstDate': '2000/11/05 00:00',
                 'queryStartDate': start,
                 'queryEndDate': end,
                 'commodityId': id_select}
    #Check
    try:
        # 偽停頓
        time.sleep(2)

        r = requests.post(url, headers = ua, data = form_data)
        r.raise_for_status()

        df = pd.read_csv(StringIO(r.text), index_col = False)
        df.columns = df.columns.str.strip()
        #Check columns & Filter
        filter_list = ['日期', '身份別', '多方交易口數', '多方交易契約金額(千元)', '空方交易口數', '空方交易契約金額(千元)',
                       '多空交易口數淨額', '多空交易契約金額淨額(千元)', '多方未平倉口數', '多方未平倉契約金額(千元)',
                       '空方未平倉口數','空方未平倉契約金額(千元)', '多空未平倉口數淨額', '多空未平倉契約金額淨額(千元)',
                       '商品名稱']
        df = Check_columns(df.columns.values, filter_list, df)

        if(write == 1):#write the file
            df.to_csv(write_path, encoding = "big5", index = False)
            print('write the file: ' + start + " to " + end)

        return df

    except requests.RequestException as e:
        print(e)
        return 0
    except Exception as e:
        print(e)
        return 0

####------------------------------------------------------------------------------------------------------------------------------
#小台指期資訊: '交易日期', '契約', '到期月份(週別)', '開盤價',  '最高價', '最低價', '收盤價',
#           '漲跌價', '漲跌%','成交量', '結算價', '未沖銷契約數', '交易時段'
def MTX(start, end, write = 0):
    Taifutures(start, end, write, id_select = 'MTX')

####------------------------------------------------------------------------------------------------------------------------------
#大台資訊: 三大法人 多空 交易口數, 金額, 未平倉口數
def TX_Investors(start, end, write = 0):
    MTX_Investors(start, end, write, id_select = 'TXF')

####------------------------------------------------------------------------------------------------------------------------------
#個股價格: 成交股數, 成交筆數, 成交金額, 開盤價, 最高價, 最低價, 收盤價, 本益比
def Stock_Price(YMD, write = 0):
    '''
    [OBJECTIVE]
    Parsinging Stocks:
        1. OHLC
        2. Volume

    [USAGE]
    YMD is string type, write is boolean(1 or 0)
    Example:
        example: Price('20180309', 1)

    [RETURN]
    A dataframe
    '''

    if(type(YMD)!=str or len(YMD)!=8):
        print("Wrong INPUT format:",YMD )
        print("example: YMD = 20180302")
        return 0

    Price_path = FD_path + "/Stock_Price"
    write_path = Price_path + "/" + YMD + ".csv"

    if write == 1 and Check_directory_files(FD_path, Price_path, write_path) == 0:
        return 0

    #Check
    try:
        # 偽停頓
        time.sleep(3)
        #Request
        r = requests.post('http://www.twse.com.tw/exchangeReport/MI_INDEX?response=csv&date=' + YMD + '&type=ALL')
        # 如果響應狀態碼不是 200，就主動拋出異常
        r.raise_for_status()
        if len(r.text) < 5:
            raise Exception('the file: ' + YMD + ' is empty!')

        df = pd.read_csv(StringIO("\n".join([i.translate({ord(c): None for c in ' '})
                                         for i in r.text.split('\n')
                                         if len(i.split('",')) == 17 and i[0] != '='])), header=0)

        # date
        #df['日期'] = YMD

        #remove unnecessary columns
        df = df.drop(['最後揭示買價', '最後揭示賣價', '最後揭示賣量', '最後揭示買量', '漲跌價差', '漲跌(+/-)', '證券名稱'], axis = 1)
        df = df.drop(df.columns[df.columns.str.contains('^Unnamed')], axis = 1)

        # remove comma
        df = df.astype(str)
        df = df.apply(lambda x: x.str.replace(',',''))

        # remove unnecessary rows
        df = df[df['證券代號'].str.len() == 4]

        # Rename
        df['Volume'] = (df['成交股數'].astype(int)/1000).astype(int)
        df = df.rename(columns={"開盤價": "Open", "最高價": "High", \
                                "最低價": "Low", "收盤價": "Close"})

        # transpose
        df.index = df['證券代號']
        df = df.drop(['證券代號'], axis = 1)
        df = df.transpose()

        df.insert(0, 'Data_Name', df.index)
        df.insert(0, '日期', YMD)

        if(write == 1):#write the file
            df.to_csv(write_path, encoding = "big5", index = False)
            print('write the file: ' + YMD)

        return df

    except requests.RequestException as e:
        print(e)
        return 0
    except Exception as e:
        print(e)
        return 0

####------------------------------------------------------------------------------------------------------------------------------
#個股三大法人: '外資', '投信', '自營商(僅避險)', '自營商', '總計'
def Stock_Investors(YMD, write = 0):
    '''
    [OBJECTIVE]
    Parsinging Stocks:
        1. Institutional investors

    [USAGE]
    YMD is string type, write is boolean(1 or 0)
    Example:
        example: Price('20180309', 1)

    [RETURN]
    A dataframe
    '''

    if(type(YMD)!=str or len(YMD)!=8):
        print("Wrong INPUT format:",YMD )
        print("example: YMD = 20180302")
        return 0

    Investors_path = FD_path + "/Stock_Investors"
    write_path = Investors_path + "/" + YMD + ".csv"


    if write == 1 and Check_directory_files(FD_path, Investors_path, write_path) == 0:
        return 0

    url = 'http://www.twse.com.tw/fund/T86?response=csv&date='+YMD+'&selectType=ALLBUT0999&_=1520679211958';


    #Check
    try:
        # 偽停頓
        time.sleep(3)
        r = requests.get(url, headers = ua)
        # 如果響應狀態碼不是 200，就主動拋出異常
        r.raise_for_status()
        if len(r.text) < 5:
            raise Exception('the file '+ YMD + ' is empty!')
            return 0

        # 下載該日三大法人的csv，並用pandas轉換成 dataframe
        df = pd.read_csv(StringIO(r.text), header=1)

        # remove unnecessary rows
        df = df[~df['證券名稱'].isnull()]
        df['證券名稱'] = df['證券名稱'].str.strip()
        df = df[df['證券代號'].astype(str).str[0] != '=']
        df = df[df['證券代號'].str.len() == 4]

        # sort
        df = df.sort_values(['證券代號'])

        # filter
        df = df.filter(['證券代號', '外陸資買賣超股數(不含外資自營商)', '外資買賣超股數', '投信買賣超股數' \
                                , '自營商買賣超股數(避險)', '自營商買賣超股數', '三大法人買賣超股數'])
        df = df.rename(columns={"外陸資買賣超股數(不含外資自營商)": "外資", "外資買賣超股數": "外資", \
                                "投信買賣超股數": "投信", "自營商買賣超股數(避險)": "自營商(僅避險)", \
                                "自營商買賣超股數": "自營商", "三大法人買賣超股數": "總計"})

        df = df.apply(lambda x: x.str.replace(',',''))
        df["外資"] = (df['外資'].astype(int)/1000).astype(int)
        df["投信"] = (df['投信'].astype(int)/1000).astype(int)
        df["自營商"] = (df['自營商'].astype(int)/1000).astype(int)
        df["總計"] = (df['總計'].astype(int)/1000).astype(int)
        if "自營商(僅避險)" in df:
            df["自營商(僅避險)"] = (df['自營商(僅避險)'].astype(int)/1000).astype(int)

        # check columns
        if df.shape[1] != 5 and df.shape[1] != 6:
            print(YMD + 'data is not correct. INFO: columns:'+ str(df.columns.values) )
            return 0


        # transpose
        df.index = df['證券代號']
        df = df.drop(['證券代號'], axis = 1)
        df = df.transpose()

        df.insert(0, 'Data_Name', df.index)
        df.insert(0, '日期', YMD)

        if write == 1: #write the file
            df.to_csv(write_path, encoding = "big5", index = False)
            print('write the file: ' + YMD)

        return df


    except requests.RequestException as e:
        print(e)
        return 0
    except Exception as e:
        print(e)
        return 0

####------------------------------------------------------------------------------------------------------------------------------
#個股月營收
def Stock_MonthlyRevenue(year, month, write = 0):
    '''
    [OBJECTIVE]
    Parsinging Stocks:
        1. Monthly revenue

    [USAGE]
    year/month is string type, write is boolean(1 or 0)
    Example:
        example: Price('2018', '03', 1)

    [RETURN]
        A dataframe
    '''

    # 假如是西元，轉成民國
    date = str(year)+str(month)+'11'
    year = int(year)
    if year > 1990:
        year -= 1911

    month = str(month)
    if(len(month) > 1 and int(month) < 10):
        month = month[-1]
    elif int(month) > 12:
        print("Wrong INPUT format:",year, month )
        return 0

    Monthly_revenue_path = FD_path + "/Stock_MonthlyRevenue"
    write_path = "{0}/{1}.csv".format(Monthly_revenue_path, date[:6])

    if write == 1 and Check_directory_files(FD_path, Monthly_revenue_path, write_path) == 0:
        return 0

    # http://mops.twse.com.tw/mops/web/t21sc04_ifrs
    url = 'http://mops.twse.com.tw/nas/t21/sii/t21sc03_'+str(year)+'_'+str(month)+'_0.html'
    if year <= 98:
        url = 'http://mops.twse.com.tw/nas/t21/sii/t21sc03_'+str(year)+'_'+str(month)+'.html'

        #Check
    try:
        # 偽停頓
        time.sleep(2)

        r = requests.get(url, headers = ua)
        r.raise_for_status()    # 如果響應狀態碼不是 200，就主動拋出異常
        if len(r.text) < 5:
            raise Exception('the file: ' + str(year) + "_" + str(month) + ' is empty!')

        r.encoding = 'big5'
        html_df = pd.read_html(StringIO(r.text))

        # 處理一下資料
        if html_df[0].shape[0] > 500:
            df = html_df[0].copy()
        else:
            df = pd.concat([df for df in html_df if df.shape[1] <= 11])
        df = df[list(range(0,10))]
        column_index = df.index[(df[0] == '公司代號')][0]
        df.columns = df.iloc[column_index]
        df['當月營收'] = pd.to_numeric(df['當月營收'], 'coerce') #'coerce' : invalid parsing will be set as NaN
        df = df[~df['當月營收'].isnull()] #remove NaN rows
        df = df[df['公司代號'] != '合計'] #remove ‘合計‘


        #Check columns & Filter
        filter_list = ['公司代號', '當月營收', '去年同月增減(%)']
        df = Check_columns(df.columns.values, filter_list, df)
        df = df.rename(columns={'公司代號':'證券代號', '當月營收':'當月千元營收', '去年同月增減(%)':'去年同月增減'})

        # transpose
        df.columns.name = ""
        df.index = df['證券代號']
        df = df.drop(['證券代號'], axis = 1)
        df = df.transpose()

        df.insert(0, 'Data_Name', df.index)
        df.insert(0, '日期', date)

        if(write == 1):#write the file
            df.to_csv(write_path, encoding='big5', index = False)
            print('write the file: ' + date[:6])

        return df

    except requests.RequestException as e:
        print(e)
        return 0
    except Exception as e:
        print(e)
        return 0

####------------------------------------------------------------------------------------------------------------------------------
#個股財報:綜合損益 # Not ready (S0331)
def Financial_C_Income(year, season, write = 0):
    '''
    [OBJECTIVE]
    Parsinging Financial_statements:
        1. Statement of Comprehensive Income

    [USAGE]
    year/season is string type, write is boolean(1 or 0)
    Example:
        example: Financial_statements_C_Income('2018', '3', 1)

    [RETURN]
        A dataframe
    [NOTE]
        營業費用 = 生產製造產品時所產生的費用 (ex: 購買原物料、包裝、運送)
        營業成本 = 出售商品或提供勞務所負擔之成本 (ex: 員工薪資、租金)
        營業外收支 = 公司的營業項目以外的活動 (ex: 利息收支、投資收益或損失)
        毛利 = 營收 - 營業成本
        營業利益 = 毛利 - 營業費用
        稅前淨利 ＝ 營業利益 ＋ 營業外收支
    '''
    year = int(year)
    origin_year = year
    season = str(season)

    if year > 1990:     year -= 1911
    if len(season) < 2: season = '0' + season

    if int(season) > 4 :
        print("Wrong INPUT format: season should be 1 to 4.")
        return 0

    if   int(season) == 1: YMD = str(origin_year) + '0516'
    elif int(season) == 2: YMD = str(origin_year) + '0901'
    elif int(season) == 3: YMD = str(origin_year) + '1130'
    elif int(season) == 4: YMD = str(origin_year) + '0401'

    Finance_Comprehensive_Income_path = FD_path + "/Finance_ComprehensiveIncome"
    write_path = "{0}/{1}_{2}.csv".format(Finance_Comprehensive_Income_path, origin_year, season)

    if write == 1 and Check_directory_files(FD_path, Finance_Comprehensive_Income_path, write_path) == 0:
        return 0

    # http://mops.twse.com.tw/mops/web/t163sb04
    url = 'http://mops.twse.com.tw/mops/web/ajax_t163sb04'
    form_data = {'encodeURIComponent': '1',
                 'step': '1',
                 'firstin': '1',
                 'off': '1',
                 'isQuery': 'Y',
                 'TYPEK': 'sii',
                 'year': year,
                 'season': season}
      #Check
    try:
        # 偽停頓
        time.sleep(2)
        r = requests.post(url, headers = ua, data = form_data)
        r.raise_for_status()

        df_list = pd.read_html(StringIO(r.text))

        df_list = [df for df in df_list if df.shape[1] > 5]
        for i, df in enumerate(df_list):
            column_index = df.index[(df[0] == '公司代號')][0]
            df.columns = df.iloc[column_index]
            df = df.rename(columns={'營業收入':'營收', '收入':'營收', '營業毛利（毛損）淨額':'毛利', \
                                    '營業利益（損失）':'營業利益', '稅前淨利（淨損）':'稅前淨利', \
                                    '繼續營業單位稅前淨利（淨損）': '稅前淨利', '繼續營業單位本期淨利（淨損）':'稅後淨利', \
                                    '基本每股盈餘（元）': '每股盈餘', '公司代號':'證券代號'} )
            df = df[df['公司名稱'] != '公司名稱']

            df_list[i] = df

        df_all = pd.concat([df for df in df_list ])
        df_all = df_all.sort_values(['證券代號'])
        df_all = df_all.replace('--', 0)
        df_all['季'] = '{0}_{1}'.format(origin_year, season)

        # check columns
        check_list = ['營收', '毛利', '營業利益', '稅前淨利', '稅前淨利', '稅後淨利', '每股盈餘', '證券代號']
        Check_columns(df_all.columns.values, check_list)

        # transpose
        df_all.columns.name = ""
        df_all.index = df_all['證券代號']
        df_all = df_all.drop(['證券代號'], axis = 1)
        df_all = df_all.transpose()

        df_all.insert(0, 'Data_Name', df_all.index)
        df_all.insert(0, '日期', YMD)

        if(write == 1):#write the file
            df_all.to_csv(write_path, index = False)
            print('write the file: {0}_{1}'.format(origin_year, season) )

        return df_all

    except requests.RequestException as e:
        print(e)
        return 0
    except Exception as e:
        print(e)
        return 0
