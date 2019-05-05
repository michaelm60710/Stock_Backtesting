#!/usr/bin/env python3
import sys
import pandas as pd
import numpy as np
import os
import datetime

from settings import Crawl_dict
from settings import FD_path, PKL_path


def trimAllColumns(df):
    """
    Trim whitespace from ends of each value across all series in dataframe
    """
    trimStrings = lambda x: x.strip() if type(x) is str else x
    return df.applymap(trimStrings)

def series_to_int_str(x):
    '''
    Convert series to str type without decimal (floating)
    '''
    if type(x) != str:
        return str(int(x))
    else:
        # Note: str(int(float('1.0'))) = '1'
        try:
            num = float(x)
        except ValueError:
            return x
        return str(int(num))

# Find lastest date
# modified_date: only need to read csv which is modified after modified_date
def Lastest_date(path, dateparse_str = '%Y/%m/%d', date_name = '日期', modified_date = None):
    '''
    find lastest date.
    return type: datetime.
    '''
    dateparse = lambda x: pd.datetime.strptime(x, dateparse_str)
    last_day = datetime.datetime(1000, 1, 1, 0, 0)
    for file in  os.listdir(path):
        if not file.endswith(".csv"): continue
        if file.endswith("year.csv"): continue
        if modified_date is not None:
            file_modified_date = datetime.datetime.fromtimestamp(os.path.getmtime(path + file))
            if file_modified_date < modified_date: continue

        df = pd.read_csv(path + file, encoding="big5", parse_dates=[date_name], date_parser=dateparse, thousands=',')

        if len(df.index) == 0: continue
        date = df[date_name].iloc[-1].to_pydatetime()

        if date > last_day: last_day = date

    return last_day

def Concate_all_df(path, dateparse_str = '%Y/%m/%d', date_name = '日期', Only_return_time = False):
    '''
    Concate_all_df
    If set Only_return_time = True, only return date columns. (More efficient)
    '''
    dateparse  = lambda x: pd.datetime.strptime(x, dateparse_str)
    dateparse1 = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
    dfs = []
    for file in  os.listdir(path):
        if file.endswith("year.csv"):
            df = pd.read_csv(path + file, encoding="big5", parse_dates=['time'],   date_parser=dateparse1, thousands=',', low_memory=False)
        elif file.endswith(".csv"):
            df = pd.read_csv(path + file, encoding="big5", parse_dates=[date_name], date_parser=dateparse, thousands=',')
            df = df.rename(index=str, columns={date_name: "time"})
        else:
            continue

        if Only_return_time:
            df = df[['time']].drop_duplicates()
        dfs.append(df)

    if len(dfs) < 1: return None
    frame = (pd.concat(dfs, ignore_index=True)).sort_values(by='time').drop_duplicates().set_index('time')

    if not Only_return_time:
        frame = frame.apply(pd.to_numeric, errors='ignore')  # Note: some variable type is float, if it convert to str directly, it will have decimal point.
        frame = trimAllColumns(frame)

    return frame

# Read origional csv files and Concate dataframe to new dataframe
# modified_date: only need to read csv which is modified after modified_date
def Concate_df(path, dateparse_str = '%Y/%m/%d', date_name = '日期', modified_date = None):
    dateparse = lambda x: pd.datetime.strptime(x, dateparse_str)
    dfs = []
    for file in  os.listdir(path):
        if not file.endswith(".csv"): continue
        if file.endswith("year.csv"): continue
        if modified_date is not None:
            file_modified_date = datetime.datetime.fromtimestamp(os.path.getmtime(path + file))
            if file_modified_date < modified_date: continue

        dfs.append(pd.read_csv(path + file, encoding="big5", parse_dates=[date_name], date_parser=dateparse, thousands=',') )

    if len(dfs) < 1: return None
    frame = (pd.concat(dfs, ignore_index=True)).sort_values(by=date_name).set_index(date_name)
    frame.index.names = ['time']
    return frame

# Merge CSV files. Generate new files (*_year.csv) and remove old files
def Merge_csv_by_year(df_all, path):

    dateparse_year = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
    first_year = int(df_all.index[0].strftime('%Y'))
    last_year = int(df_all.index[-1].strftime('%Y'))
    print(path)

    # merge data by year
    for i in range(first_year, last_year+1):
        start = datetime.datetime.strptime(str(i)+'0101', '%Y%m%d')
        end   = datetime.datetime.strptime(str(i)+'1231', '%Y%m%d')
        df = df_all.loc[start:end]
        if  len(df) == 0: continue
        file_name = str(i)+ "_year.csv"
        write_path = path + file_name
        if os.path.isfile(write_path):
            print ('merge to ' + file_name)
            df_year = pd.read_csv(write_path, encoding="big5", parse_dates=['time'],  date_parser=dateparse_year, \
                                              thousands=',', low_memory=False).set_index('time')
            df = pd.concat([df, df_year])#.drop_duplicates().sort_index()
            # remove duplicates row (need to consider date)
            df['temp'] = df.index
            df = df.drop_duplicates().sort_values(by='temp').drop('temp', axis = 1)

        df.to_csv(write_path, encoding = "big5", index = True)
        print('write the file: ' + file_name)

    # remove original files
    for file in  os.listdir(path):
        if not file.endswith(".csv"): continue
        if file.endswith("year.csv"): continue
        os.remove(path+ file)
        print ("rm " + file)

def Check_miss_data():
    from dateutil.relativedelta import relativedelta

    # TODO: How to check monthly/quarterly data
    dont_check_list = ['Stock_MonthlyRevenue', 'Financial_Income', 'Financial_Balance_Sheet'] # monthly/quarterly data

    # Merge all date_index
    dif_id = {}
    origin_dict = {}
    all_index = {}
    d_index = pd.DatetimeIndex([]).rename('time')
    for func_name, value in Crawl_dict.items():

        if func_name in dont_check_list: continue # don't need to consider
        path = FD_path + "/" + func_name +"/"
        origin_dict[func_name] = Concate_all_df(path, value['date_format'], value['date_name'], Only_return_time = True)
        all_index[func_name]   = origin_dict[func_name].index.drop_duplicates() # func_index
        d_index = d_index.union(all_index[func_name])

        #print(func_name)
        #print("    # index length: " + str(len(all_index[func_name] ) ) )

    d_index = d_index.drop_duplicates()
    print("# Total date_index length: " + str(len(d_index)) + "\n")

    # find the different date
    for func_name, nouse in Crawl_dict.items():

        if func_name in dont_check_list: continue # don't need to consider
        dif = d_index.difference(all_index[func_name])
        first_date = all_index[func_name][0]
        end_date = all_index[func_name][-1]

        # trim date
        dif_id[func_name] = dif[(dif > first_date) & (dif < end_date)]

        print ("# FUNCTION NAME: " + func_name)
        print ("\t# Length of missing data: " + str(len(dif_id[func_name])))
        if len(dif_id[func_name]) > 0:
            print ("\t# Missing date: " + str(dif_id[func_name]))

    return dif_id


####------------------------------------------------------------------------------------------------------------------------------
####------------------------------------------------------------------------------------------------------------------------------
####------------------------------------------------------------------------------------------------------------------------------
#### Data_preprocessing and  pickle maintenance
import pickle, re
class Data_preprocessing:
    modified_date = None

    def __init__(self, update = True, rebuild = False, verbose = True):
        '''
        rebuild: rebuild pkl
        update: only update partial data (Except the file name with xxx_year.csv). more efficient.
        '''
        self.origin_dict = {}
        self.data_dict = {}
        self.verbose = verbose

        if rebuild:
            if self.verbose: print ("# Rebuild data")
            self.update_pkl()
            self.data_preprocessing()
        elif update:
            if self.verbose: print ("# Update data")
            self.load_pkl()
            self.update_pkl(Only_unprocess_data = True)
            self.data_preprocessing()
        else:
            if self.verbose: print ("# Read pickle file")
            self.load_pkl()

        # Debug & Check
        self.Check()

    def read_csv_files(self, Only_unprocess_data = False):
        if 'Update date' in self.origin_dict: modified_date = datetime.datetime.strptime(self.origin_dict['Update date'], '%Y%m%d')
        else                                : modified_date = None
        #self.tmp_dict = {}

        for func_name, value in Crawl_dict.items():

            path = FD_path + "/" + func_name +"/"

            if not Only_unprocess_data:
                # Concate all data
                self.origin_dict[func_name] = Concate_all_df(path, value['date_format'], value['date_name'])
            else:
                # Only concate new data after merging csv
                new_df = Concate_df(path, value['date_format'], value['date_name'], modified_date)
                if new_df is None: continue

                #if func_name not in self.origin_dict:
                    #print('\tWarning: Missing {0}. Try to rebuild pkl.'.format(func_name)); continue
                #    self.origin_dict[func_name] = new_df

                if func_name in self.origin_dict:
                    last_origin_date = self.origin_dict[func_name].index[-1].to_pydatetime() + datetime.timedelta(days=1)
                    new_df = new_df[last_origin_date:]
                    self.origin_dict[func_name] = pd.concat([self.origin_dict[func_name], new_df])
                else:
                    self.origin_dict[func_name] = new_df

                if len(new_df) == 0: continue


            if self.verbose: print("\t{0}".format(func_name))

    def update_pkl(self, Only_unprocess_data = False):
        if self.verbose: print("Update original data pkl.")

        self.read_csv_files(Only_unprocess_data)
        self.origin_dict['Update date'] = datetime.datetime.now().strftime("%Y%m%d")

        file = open(PKL_path+'/origin_finance_data.pkl', 'wb')
        pickle.dump(self.origin_dict, file)

    def load_pkl(self):
        assert os.path.isfile(PKL_path+'/origin_finance_data.pkl'), "{0}/origin_finance_data.pkl doesn't exist. ".format(PKL_path) \
                                      +"Please try to use command: Data_preprocessing(rebuild = True)"
        if self.verbose: print ("loading original pkl file.")
        with open(PKL_path+'/origin_finance_data.pkl', 'rb') as file:
            self.origin_dict = pickle.load(file)

        if not os.path.isfile(PKL_path+'/finance_data.pkl'):
            print ("WARNING: {0}/finance_data.pkl doesn't exist. ".format(PKL_path) \
                  +"Try to use Data_preprocessing.data_preprocessing() to generate the file first.")
            return

        if self.verbose: print ("loading financial data pkl file.")
        with open(PKL_path+'/finance_data.pkl', 'rb') as file:
            self.data_dict = pickle.load(file)

    def data_preprocessing(self):
        from dateutil.relativedelta import relativedelta
        if self.verbose: print("# Data_preprocessing")
        #######################################
        # 1. Preprocess Taifutures_LargeTrade #
        #######################################
        temp = self.origin_dict['Taifutures_LargeTrade'].copy()
        temp['交易人類別'] = temp['交易人類別'].apply(series_to_int_str)
        temp['到期月份(週別)'] = temp['到期月份(週別)'].apply(series_to_int_str)

        # 一般
        temp_0 = temp[temp['交易人類別'] == '0']
        temp_0 = temp_0[((temp_0['到期月份(週別)'].str.slice(4,5) == '0') | (temp_0['到期月份(週別)'].str.slice(4,5) == '1')) \
                       & (temp_0['到期月份(週別)'].str.len() == 6 ) ]
        temp_0['date'] = temp_0.index
        temp_0 = temp_0.drop_duplicates(subset = 'date')
        temp_0 = temp_0.filter (regex='大交易人', axis=1)

        # 特法
        temp_1 = temp[temp['交易人類別'] == '1'].drop_duplicates()
        temp_1 = temp_1[((temp_1['到期月份(週別)'].str.slice(4,5) == '0') | (temp_1['到期月份(週別)'].str.slice(4,5) == '1')) \
                       & (temp_1['到期月份(週別)'].str.len() == 6 ) ]
        temp_1['date'] = temp_1.index
        temp_1 = temp_1.drop_duplicates(subset = 'date')
        temp_1.columns = temp_1.columns.str.replace('大交易人','大特定法人交易人')
        temp_1 = temp_1.filter (regex='法人交易人', axis=1)


        self.data_dict['台指期貨_大額交易人'] = pd.concat([temp_0, temp_1], axis=1).astype(float)

        #######################################
        # 2. Preprocess TaiExchange           #
        #######################################
        temp = pd.concat([self.origin_dict['TaiExchange'], self.origin_dict['TaiExchange_OHLC'], \
                          self.origin_dict['Institutional_investors']], axis = 1)
        temp = temp.rename(columns={"成交金額": "Volume", "開盤指數": "Open", "最高指數": "High", \
                            "最低指數": "Low", "收盤指數": "Close"})
        temp = temp.drop(['發行量加權股價指數'], axis = 1)
        self.data_dict['加權指數'] = temp

        #######################################
        # 3. Preprocess TaiFutures            #
        #######################################
        temp_copy = self.origin_dict['Taifutures'].copy()
        for i in range (0,2):
            if i == 0: temp_ord = temp_copy[temp_copy['交易時段'] == '一般']
            else:      temp_ord = temp_copy[temp_copy['交易時段'] == '盤後']

            temp_ord = temp_ord[(temp_ord['到期月份(週別)'].str.len() == 6 ) ]
            temp_ord['colFromIndex'] = temp_ord.index
            temp = temp_ord.sort_values(by=['colFromIndex', '成交量'])
            temp = temp.drop_duplicates(subset='colFromIndex', keep='last')
            temp = temp.drop(['契約', 'colFromIndex'], axis= 1)
            temp = temp.rename(columns={"成交量": "Volume", "開盤價": "Open", "最高價": "High", \
                                        "最低價": "Low", "收盤價": "Close"})
            temp['漲跌%'] = temp['漲跌%'].str.replace('%','')

            if i == 0: self.data_dict['台指期貨_一般'] = temp
            else:      self.data_dict['台指期貨_盤後'] = temp

        self.data_dict['台指期貨_盤後'] = self.data_dict['台指期貨_盤後'].drop(['結算價', '未沖銷契約數'], axis = 1)

        #######################################
        # 4. Preprocess TaiFutures (Merge)    #
        #######################################
        origin = self.data_dict['台指期貨_一般'].copy()
        night  = self.data_dict['台指期貨_盤後'].copy()
        origin = origin.drop(['漲跌價', '漲跌%', '結算價', '交易時段'], axis = 1) # 1569 rows
        night = night.drop(['漲跌價', '漲跌%', '交易時段', '到期月份(週別)'], axis = 1) \
                     .rename(columns=lambda x: re.sub('$','_夜盤',x) )

        origin = pd.concat([origin,night], axis= 1)

        # Volume
        def f_v(x):
            if pd.isnull(x['Volume_夜盤']):
                return x['Volume']
            return x['Volume'] + x['Volume_夜盤']
        origin['Volume'] = origin[['Volume','Volume_夜盤']].apply(f_v,axis=1)

        # Open
        def f_o(x):
            if pd.isnull(x['Open_夜盤']):
                return x['Open']
            return x['Open_夜盤']
        origin['Open'] = origin[['Open','Open_夜盤']].apply(f_o,axis=1)

        # Low
        def f_l(x):
            if pd.isnull(x['Low_夜盤']) or x['Low'] < x['Low_夜盤']:
                return x['Low']
            return x['Low_夜盤']
        origin['Low'] = origin[['Low','Low_夜盤']].apply(f_l,axis=1)

        # High
        def f_h(x):
            if pd.isnull(x['High_夜盤']) or x['High'] > x['High_夜盤']:
                return x['High']
            return x['High_夜盤']
        origin['High'] = origin[['High','High_夜盤']].apply(f_h,axis=1)

        # Close (same as Origin)

        # Remove 夜盤 columns
        origin = origin.drop(origin.columns[origin.columns.str.contains('_夜盤')], axis = 1)
        self.data_dict['台指期貨_合併'] = origin.astype(int)

        #######################################
        # 5. Preprocess TaiFutures Investors  #
        #######################################
        temp = self.origin_dict['Taifutures_Investors'].copy()
        MTX_temp = self.origin_dict['MTX_Investors'].copy()
        TX_temp = self.origin_dict['TX_Investors'].copy()

        self.data_dict['期貨法人'] = dict()
        self.data_dict['期貨法人']['期貨_投信'] = temp[temp['身份別'] == '投信']
        self.data_dict['期貨法人']['期貨_自營商'] = temp[temp['身份別'] == '自營商']
        self.data_dict['期貨法人']['期貨_外資'] = temp[temp['身份別'] == '外資及陸資']

        self.data_dict['期貨法人']['小台指_投信'] = MTX_temp[MTX_temp['身份別'] == '投信']
        self.data_dict['期貨法人']['小台指_自營商'] = MTX_temp[MTX_temp['身份別'] == '自營商']
        self.data_dict['期貨法人']['小台指_外資'] = MTX_temp[MTX_temp['身份別'] == '外資及陸資']

        self.data_dict['期貨法人']['大台_投信'] = TX_temp[TX_temp['身份別'] == '投信']
        self.data_dict['期貨法人']['大台_自營商'] = TX_temp[TX_temp['身份別'] == '自營商']
        self.data_dict['期貨法人']['大台_外資'] = TX_temp[TX_temp['身份別'] == '外資及陸資']

        #######################################
        # 6. Preprocess MTX Position          #
        #######################################
        Futures_Pos = self.origin_dict['MTX'][['未沖銷契約數']].copy()
        Futures_Pos = Futures_Pos[Futures_Pos['未沖銷契約數'] != '-']
        Futures_Pos = Futures_Pos.astype(int)
        # Sum the position in same date
        self.data_dict['小台指_總留倉數'] = pd.pivot_table(Futures_Pos, values = '未沖銷契約數', \
                                          index = Futures_Pos.index, aggfunc=np.sum).to_frame()
        #######################################
        # 7.1 Preprocess Stocks               #
        #######################################
        self.data_dict['台股個股'] = dict()
        stock_df_list = ['Stock_Price', 'Stock_Investors', 'Stock_MonthlyRevenue', \
                         'Financial_Balance_Sheet', 'Financial_Income']
        for stock_df in stock_df_list:
            stock_df = self.origin_dict[stock_df]
            Data_Name_list =  stock_df['Data_Name'].drop_duplicates().tolist()
            for data_name in Data_Name_list:
                frame = stock_df[stock_df['Data_Name'] == data_name]
                frame = frame.apply(pd.to_numeric, errors='coerce')
                self.data_dict['台股個股'][data_name] = frame.drop(['Data_Name'], axis = 1)

        # 月營收相關data, 日期需往後一個月
        for m_name in ['去年同月增減', '當月千元營收']:
            date_list = self.data_dict['台股個股'][m_name].index.copy()
            self.data_dict['台股個股'][m_name].index = [date + relativedelta(months= 1) for date in date_list]

        # Financial_Income: origin data每季會疊加
        Data_Name_list =  self.origin_dict['Financial_Income']['Data_Name'].drop_duplicates().tolist()
        for data_name in Data_Name_list:
            if data_name == '季': continue
            # 因為疊加, 所以2, 3, 4季data需減去上比data
            tmp = self.data_dict['台股個股'][data_name].shift(1)
            tmp[tmp.index.month == 5] = 0
            self.data_dict['台股個股'][data_name] = self.data_dict['台股個股'][data_name] - tmp

        # rename & delete
        self.data_dict['台股個股']["法人總計"] = self.data_dict['台股個股']["總計"].copy()
        del self.data_dict['台股個股']["總計"]
        del self.data_dict['台股個股']["季"]


        #######################################
        # 8. Fill NaN                         #
        #######################################
        self.FillNan()

        #######################################
        # 9. Rewrite financial data pkl         #
        #######################################
        if self.verbose: print ("update financial data pkl.")
        file = open(PKL_path+'/finance_data.pkl', 'wb')
        pickle.dump(self.data_dict, file)

    # not ready
    def data_preprocessing_new(self):
        from dateutil.relativedelta import relativedelta
        if self.verbose: print("# Data_preprocessing_new S0407")

        # use tmp_dict
        tmp_dict = self.origin_dict

        #######################################
        # 1. Preprocess Taifutures_LargeTrade #
        #######################################
        if 'Taifutures_LargeTrade' in tmp_dict:
            temp = self.tmp_dict['Taifutures_LargeTrade'].copy()
            temp['交易人類別'] = temp['交易人類別'].apply(series_to_int_str)
            temp['到期月份(週別)'] = temp['到期月份(週別)'].apply(series_to_int_str)

            # 一般
            temp_0 = temp[temp['交易人類別'] == '0']
            temp_0 = temp_0[((temp_0['到期月份(週別)'].str.slice(4,5) == '0') | (temp_0['到期月份(週別)'].str.slice(4,5) == '1')) \
                           & (temp_0['到期月份(週別)'].str.len() == 6 ) ]
            temp_0['date'] = temp_0.index
            temp_0 = temp_0.drop_duplicates(subset = 'date')
            temp_0 = temp_0.filter (regex='大交易人', axis=1)

            # 特法
            temp_1 = temp[temp['交易人類別'] == '1'].drop_duplicates()
            temp_1 = temp_1[((temp_1['到期月份(週別)'].str.slice(4,5) == '0') | (temp_1['到期月份(週別)'].str.slice(4,5) == '1')) \
                           & (temp_1['到期月份(週別)'].str.len() == 6 ) ]
            temp_1['date'] = temp_1.index
            temp_1 = temp_1.drop_duplicates(subset = 'date')
            temp_1.columns = temp_1.columns.str.replace('大交易人','大特定法人交易人')
            temp_1 = temp_1.filter (regex='法人交易人', axis=1)

            self.data_dict['台指期貨_大額交易人'] = pd.concat([temp_0, temp_1], axis=1).astype(float)

        #######################################
        # 2. Preprocess TaiExchange           #
        #######################################
        if 'TaiExchange' in tmp_dict or 'TaiExchange_OHLC' in tmp_dict or 'Institutional_investors' in tmp_dict:
            temp = pd.concat([self.tmp_dict['TaiExchange'], self.tmp_dict['TaiExchange_OHLC'], \
                              self.tmp_dict['Institutional_investors']], axis = 1)
            temp = temp.rename(columns={"成交金額": "Volume", "開盤指數": "Open", "最高指數": "High", \
                                "最低指數": "Low", "收盤指數": "Close"})
            temp = temp.drop(['發行量加權股價指數'], axis = 1)

            self.data_dict['加權指數'] = temp

        #######################################
        # 3. Preprocess TaiFutures            #
        #######################################
        if 'Taifutures' in tmp_dict:
            temp_copy = self.tmp_dict['Taifutures'].copy()
            for i in range (0,2):
                if i == 0: temp_ord = temp_copy[temp_copy['交易時段'] == '一般']
                else:      temp_ord = temp_copy[temp_copy['交易時段'] == '盤後']

                temp_ord = temp_ord[(temp_ord['到期月份(週別)'].str.len() == 6 ) ]
                temp_ord['colFromIndex'] = temp_ord.index
                temp = temp_ord.sort_values(by=['colFromIndex', '成交量'])
                temp = temp.drop_duplicates(subset='colFromIndex', keep='last')
                temp = temp.drop(['契約', 'colFromIndex'], axis= 1)
                temp = temp.rename(columns={"成交量": "Volume", "開盤價": "Open", "最高價": "High", \
                                            "最低價": "Low", "收盤價": "Close"})
                temp['漲跌%'] = temp['漲跌%'].str.replace('%','')

                if i == 0: self.data_dict['台指期貨_一般'] = temp
                else:      self.data_dict['台指期貨_盤後'] = temp

            self.data_dict['台指期貨_盤後'] = self.data_dict['台指期貨_盤後'].drop(['結算價', '未沖銷契約數'], axis = 1)

        #######################################
        # 4. Preprocess TaiFutures (Merge)    #
        #######################################
        origin = self.data_dict['台指期貨_一般'].copy()
        night  = self.data_dict['台指期貨_盤後'].copy()
        origin = origin.drop(['漲跌價', '漲跌%', '結算價', '交易時段'], axis = 1) # 1569 rows
        night = night.drop(['漲跌價', '漲跌%', '交易時段', '到期月份(週別)'], axis = 1) \
                     .rename(columns=lambda x: re.sub('$','_夜盤',x) )

        origin = pd.concat([origin,night], axis= 1)

        # Volume
        def f_v(x):
            if pd.isnull(x['Volume_夜盤']):
                return x['Volume']
            return x['Volume'] + x['Volume_夜盤']
        origin['Volume'] = origin[['Volume','Volume_夜盤']].apply(f_v,axis=1)

        # Open
        def f_o(x):
            if pd.isnull(x['Open_夜盤']):
                return x['Open']
            return x['Open_夜盤']
        origin['Open'] = origin[['Open','Open_夜盤']].apply(f_o,axis=1)

        # Low
        def f_l(x):
            if pd.isnull(x['Low_夜盤']) or x['Low'] < x['Low_夜盤']:
                return x['Low']
            return x['Low_夜盤']
        origin['Low'] = origin[['Low','Low_夜盤']].apply(f_l,axis=1)

        # High
        def f_h(x):
            if pd.isnull(x['High_夜盤']) or x['High'] > x['High_夜盤']:
                return x['High']
            return x['High_夜盤']
        origin['High'] = origin[['High','High_夜盤']].apply(f_h,axis=1)

        # Close (same as Origin)

        # Remove 夜盤 columns
        origin = origin.drop(origin.columns[origin.columns.str.contains('_夜盤')], axis = 1)
        self.data_dict['台指期貨_合併'] = origin.astype(int)

        #######################################
        # 5. Preprocess TaiFutures Investors  #
        #######################################
        temp = self.origin_dict['Taifutures_Investors'].copy()
        MTX_temp = self.origin_dict['MTX_Investors'].copy()
        TX_temp = self.origin_dict['TX_Investors'].copy()

        self.data_dict['期貨法人'] = dict()
        self.data_dict['期貨法人']['期貨_投信'] = temp[temp['身份別'] == '投信']
        self.data_dict['期貨法人']['期貨_自營商'] = temp[temp['身份別'] == '自營商']
        self.data_dict['期貨法人']['期貨_外資'] = temp[temp['身份別'] == '外資及陸資']

        self.data_dict['期貨法人']['小台指_投信'] = MTX_temp[MTX_temp['身份別'] == '投信']
        self.data_dict['期貨法人']['小台指_自營商'] = MTX_temp[MTX_temp['身份別'] == '自營商']
        self.data_dict['期貨法人']['小台指_外資'] = MTX_temp[MTX_temp['身份別'] == '外資及陸資']

        self.data_dict['期貨法人']['大台_投信'] = TX_temp[TX_temp['身份別'] == '投信']
        self.data_dict['期貨法人']['大台_自營商'] = TX_temp[TX_temp['身份別'] == '自營商']
        self.data_dict['期貨法人']['大台_外資'] = TX_temp[TX_temp['身份別'] == '外資及陸資']

        #######################################
        # 6. Preprocess MTX Position          #
        #######################################
        Futures_Pos = self.origin_dict['MTX'][['未沖銷契約數']].copy()
        Futures_Pos = Futures_Pos[Futures_Pos['未沖銷契約數'] != '-']
        Futures_Pos = Futures_Pos.astype(int)
        # Sum the position in same date
        self.data_dict['小台指_總留倉數'] = pd.pivot_table(Futures_Pos, values = '未沖銷契約數', \
                                          index = Futures_Pos.index, aggfunc=np.sum).to_frame()
        #######################################
        # 7. Preprocess Stocks                #
        #######################################
        self.data_dict['台股個股'] = dict()
        stock_df_list = ['Stock_Price', 'Stock_Investors', 'Stock_MonthlyRevenue']
        for stock_df in stock_df_list:
            stock_df = self.origin_dict[stock_df]
            Data_Name_list =  stock_df['Data_Name'].drop_duplicates().tolist()
            for data_name in Data_Name_list:
                frame = stock_df[stock_df['Data_Name'] == data_name]
                frame = frame.apply(pd.to_numeric, errors='coerce')
                self.data_dict['台股個股'][data_name] = frame.drop(['Data_Name'], axis = 1)
        # 月營收相關data, 日期需往後一個月
        for m_name in ['去年同月增減', '當月千元營收']:
            date_list = self.data_dict['台股個股'][m_name].index.copy()
            self.data_dict['台股個股'][m_name].index = [date + relativedelta(months= 1) for date in date_list]

        # rename
        self.data_dict['台股個股']["法人總計"] = self.data_dict['台股個股']["總計"].copy()
        del self.data_dict['台股個股']["總計"]

        #######################################
        # 8. Fill NaN                         #
        #######################################
        self.FillNan()

        #######################################
        # 9. Rewrite finance data pkl         #
        #######################################
        if self.verbose: print ("update finance data pkl.")
        file = open(PKL_path+'/finance_data.pkl', 'wb')
        pickle.dump(self.data_dict, file)

    def Check(self):
        # check duplicated
        from Check import Check_duplicated_date
        for x in self.data_dict:
            if type(self.data_dict[x]) is dict:
                for xx in self.data_dict[x]:
                    Check_duplicated_date(self.data_dict[x][xx],x+" -> "+xx)
            else:
                Check_duplicated_date(self.data_dict[x],x)

    def FillNan(self):
        # use fillna(method='ffill', limit = 5)
        limit = 5
        #print('# Fill Nan')
        for x in self.data_dict:
            if type(self.data_dict[x]) is dict:
                for xx in self.data_dict[x]:
                    self.data_dict[x][xx] = self.data_dict[x][xx].fillna(method='ffill', limit = limit)
            else:
                self.data_dict[x] = self.data_dict[x].fillna(method='ffill', limit = limit)

    def Print_data_columns(self):
        for x in self.data_dict:
            print ("\n# {0}, type = {1}".format(x, type(self.data_dict[x])))
            if type(self.data_dict[x]) is dict:
                print ("\tKeys: " + str(list(self.data_dict[x].keys())) + "\n")
            else:
                print ("\tColumns: " + str(self.data_dict[x].columns.values) + "\n")

####------------------------------------------------------------------------------------------------------------------------------
####------------------------------------------------------------------------------------------------------------------------------
####------------------------------------------------------------------------------------------------------------------------------
