#!/usr/bin/env python3
import pandas as pd
import numpy as np
import logging
import talib


#from Simulator import Component

'''
Note: You can use help(func) to get info

'''
class Component:
    def __init__(self, data):
        if type(data) != pd.core.series.Series and type(data) != pd.core.frame.DataFrame:
            logging.warning("Component: Wrong type, " + str(type(data)))
            #return
        # Init
        self.data = data

    def ConvertData(self, a, b):
        '''
        兩種情況: 一種為Dataframe, 為個股相關
                 一種為Series, 為大盤相關
        '''
        if isinstance(a, Component): a = a.data
        if isinstance(b, Component): b = b.data

        if not isinstance(a, (pd.core.frame.Series, pd.core.frame.DataFrame)) or \
           not isinstance(b, (pd.core.frame.Series, pd.core.frame.DataFrame)):
           return a, b

        if   (type(a) == pd.core.frame.Series and type(b) == pd.core.frame.DataFrame):
            a = pd.DataFrame({c:a for c in b.columns})
        elif (type(b) == pd.core.frame.Series and type(a) == pd.core.frame.DataFrame):
            b = pd.DataFrame({c:b for c in a.columns})

        # truncate
        if a.index[0] != b.index[0]:
            if a.index[0] > b.index[0]: b = b.loc[a.index[0]:]
            else:                       a = a.loc[b.index[0]:]
        if a.index[-1] < b.index[-1]:   b = b.loc[:a.index[-1]]
        else:                           a = a.loc[:b.index[-1]]

        # Monthly/Quarterly data convert
        convert_a = False
        convert_b = False
        if (a.index[-1] - a.index[0]).days/len(a) > 25: convert_a = True
        if (b.index[-1] - b.index[0]).days/len(b) > 25: convert_b = True
        if   convert_a and convert_b is False:
            a = a.reindex(b.index).fillna(method = 'ffill')
        elif convert_b and convert_a is False:
            b = b.reindex(a.index).fillna(method = 'ffill')

        return a, b

    @classmethod
    def getData(self, v):
        if type(v) == type(self):
            return v.data
        elif type(v) == pd.core.series.Series or type(v) == pd.core.frame.DataFrame:
            return v
        else:
            logging.warning("Wrong type, " + str(type(v)))
            return v

    def __and__(self, v):
        a, b = self.ConvertData(self.data, v)
        return Component(a & b)
    def __or__(self, v):
        a, b = self.ConvertData(self.data, v)
        return Component(a | b)
    def __invert__(self):
        return Component(~self.data)

    # Compare: >, <, >=, <=
    def __lt__(self, v): #less than <
        a, b = self.ConvertData(self.data, v)
        return Component(a < b)
    def __gt__(self, v): #greater than >
        a, b = self.ConvertData(self.data, v)
        return Component(a > b)
    def __ge__(self, v):
        a, b = self.ConvertData(self.data, v)
        return Component(a >= b)
    def __le__(self, v):
        a, b = self.ConvertData(self.data, v)
        return Component(a <= b)
    def __eq__(self, v): # ==
        a, b = self.ConvertData(self.data, v)
        return Component(a == b)
    def __ne__(self, v): # !=
        a, b = self.ConvertData(self.data, v)
        return Component(a != b)
    # + - * /
    def __add__(self, v):
        a, b = self.ConvertData(self.data, v)
        return Component(a + b)
    def __sub__(self, v):
        a, b = self.ConvertData(self.data, v)
        return Component(a - b)
    def __mul__(self, v):
        a, b = self.ConvertData(self.data, v)
        return Component(a * b)
    def __div__(self, v):
        a, b = self.ConvertData(self.data, v)
        return Component(a / b)

    def __str__(self):
        return str(self.data)

    def shift(self, N):
        return Component(self.data.shift(N))


# --------------------- #
#        技術指標        #
# --------------------- #

def N日均線(data, N = 3):
    '''
    INFO:
        data should be series type.
        N means the num of day.
    EXAMPLE:
        N日均線(sim.DATA['加權指數']['Close'], 5)
    '''
    data = Component.getData(data)
    return Component(data.rolling(N).mean())

def delta(data, shift_N = 1):
    '''
    data - data.shift(shift_N)
    '''
    data = data - data.shift(shift_N)
    if type(data) != Component:
        return Component(data)
    return data

# --------------------- #
#     talib related     #
# --------------------- #
'''
# list of functions
print talib.get_functions()

# dict of functions by group
print talib.get_function_groups()

# info
talib.abstract.STOCH.info

'''

def OHLCV_to_dict(OHLCV_data):
    # type 1: many Stocks/DATA
    if type(OHLCV_data) == dict:
        cols = OHLCV_data['High'].columns
        dict_dict_OHLCV = dict()
        for col in cols:
            dict_dict_OHLCV[col] = {
                'open': OHLCV_data['Open'][col].astype(float),
                'high': OHLCV_data['High'][col].astype(float),
                'low': OHLCV_data['Low'][col].astype(float),
                'close': OHLCV_data['Close'][col].astype(float),
                'volume': OHLCV_data['Volume'][col].astype(float)
            }
        return dict_dict_OHLCV

    # type 2: TWSI/single StockID (single OHLCV_data)
    elif type(OHLCV_data) == pd.core.frame.DataFrame:
        dict_data = {
            'open': OHLCV_data['Open'].astype(float),
            'high': OHLCV_data['High'].astype(float),
            'low': OHLCV_data['Low'].astype(float),
            'close': OHLCV_data['Close'].astype(float),
            'volume': OHLCV_data['Volume'].astype(float)
        }
        return {"TWSI":dict_data}

def talib2df(talib_output, index, split = False, column_name_list = None):
    '''
    split = False, Concate all arr to a Dataframe
    if there is only one array, convert to Series
    '''
    if type(talib_output) == list:
        if split or len(talib_output) == 1:
            ret = []
            for tmp_ret in talib_output:
                tmp_ret =  pd.Series(tmp_ret)
                tmp_ret.index = index
                ret.append(tmp_ret)
        else:
            ret = pd.DataFrame(talib_output).transpose()
            ret.index = index
    else:
        ret = pd.Series(talib_output)
        ret.index = index

    if type(ret) == pd.core.frame.DataFrame and column_name_list is not None:
        ret.columns = column_name_list

    return ret

def talib2component(talib_output, index, split = False, column_name_list = None):
    # convert to df/Series
    talib_output = talib2df(talib_output, index, split, column_name_list)

    if type(talib_output) == list:
        ret = []
        for df in talib_output: ret.append( Component(df) )
        if len(ret) == 1: ret = ret[0]
    else:
        ret = Component(talib_output)

    return ret

def talib_KD(OHLCV_data):
    '''
    INFO:
        Make sure OHLCV_data contains Open, High, Low, Close, Volume data
    EXAMPLE:
        Comp_slowk, Comp_slowd = talib_KD(sim.DATA['加權指數'])
        Comp_slowk, Comp_slowd = talib_KD(sim.DATA['台股個股'])
    '''
    logging.warning("Try to use function: talib_Output. You can use help(talib_Output) to get more details.")
    return talib_Output(OHLCV_data, talib.abstract.STOCH)

def talib_BBANDS(OHLCV_data):
    '''
    INFO:
        Make sure OHLCV_data contains Open, High, Low, Close, Volume data
    EXAMPLE:
        Upper, Middle, Lower = talib_BBANDS(sim.DATA['加權指數'])
        Upper, Middle, Lower = talib_BBANDS(sim.DATA['台股個股'])
    '''
    logging.warning("Try to use function: talib_Output. You can use help(talib_Output) to get more details.")
    talib_func_parameters = {'timeperiod':20, 'nbdevup':2, 'nbdevdn':2, 'matype':talib.MA_Type.T3 }
    return talib_Output(OHLCV_data, talib.abstract.BBANDS, talib_func_parameters = talib_func_parameters)

def talib_Output(OHLCV_data, talib_func, talib_func_parameters = None):
    '''
    INFO:
        Run talib function and return Component.
        Note that 'talib_func_parameters' is a dict type.
        You can use the command: talib.abstract.XXX.parameters to get parameters.

    EXAMPLE:
        1. KD :
            slowk, slowd = talib_KD(sim.DATA['加權指數'])
            slowk, slowd = talib_Output(sim.DATA['台股個股'], talib.abstract.STOCH)
        2. BBAND :
            talib_func_parameters = {'timeperiod':20, 'nbdevup':2, 'nbdevdn':2, 'matype':talib.MA_Type.T3 }
            upper, middle, lower = talib_Output(sim.DATA['加權指數'], talib.abstract.BBANDS, talib_func_parameters = talib_func_parameters)
    '''
    # Init variables
    talib_out_list = [ [] for x in range(len(talib_func.info['output_names']))]
    name_list = []
    d_index = []
    Com = []

    # Init talib_func paramters, NOTE: Talib set_parameters 有記憶性
    if talib_func_parameters is None:
        print('# Talib Function: {0}. Use default parameters. '.format(talib_func.info['display_name']) )
    else:
        talib_func.set_parameters(talib_func_parameters)

    # Get OHLCV_data
    OHLCV_data_dict = OHLCV_to_dict(OHLCV_data)

    # Get talib outputs
    for Key, Value in OHLCV_data_dict.items():
        name_list.append(Key)
        talib_output = talib_func(Value)
        for i, arr in enumerate(talib_output):
            talib_out_list[i].append(arr)

    # Get index
    for Key, Value in OHLCV_data_dict.items():
        d_index = Value['close'].index
        break

    # convert to Component type
    for arr in talib_out_list:
        Com.append(talib2component(arr, d_index, column_name_list = name_list) )

    return Com

# ---------------------- #
#   Components example   #
# ---------------------- #
'''
Note: DATA = sim.DATA
'''
class Components_lib:
    '''
    Components example:
        sim = Simulator()
        components = Components_lib(sim.DATA)
    '''
    def __init__(self, DATA):
        self._DATA = DATA
        #print("Construct Components_lib.")

    def 小外資_多方口數(self, DATA = None):
        if DATA is None: DATA = self._DATA
        return Component( DATA['期貨法人']['大台_外資']['多方未平倉口數'] - DATA['台指期貨_大額交易人']['前五大特定法人交易人買方'])

    def 小外資_空方口數(self, DATA = None):
        if DATA is None: DATA = self._DATA
        return Component( DATA['期貨法人']['大台_外資']['空方未平倉口數'] - DATA['台指期貨_大額交易人']['前五大特定法人交易人賣方'])

    def 小台_法人多方口數(self, DATA = None):
        if DATA is None: DATA = self._DATA
        期貨法人 = DATA['期貨法人']
        return Component(期貨法人['小台指_外資']['多方未平倉口數'] + 期貨法人['小台指_投信']['多方未平倉口數'] \
                       + 期貨法人['小台指_自營商']['多方未平倉口數'] )

    def 小台_法人空方口數(self, DATA = None):
        if DATA is None: DATA = self._DATA
        期貨法人 = DATA['期貨法人']
        return Component(期貨法人['小台指_外資']['空方未平倉口數'] + 期貨法人['小台指_投信']['空方未平倉口數'] \
                       + 期貨法人['小台指_自營商']['空方未平倉口數'] )

    def 散戶多空(self, DATA = None): # 散戶指標
        if DATA is None: DATA = self._DATA
        return self.小台_法人空方口數(DATA) - self.小台_法人多方口數(DATA)

    def 散戶留倉數(self, DATA = None):
        if DATA is None: DATA = self._DATA
        return Component(DATA['小台指_總留倉數']['未沖銷契約數']*2) - self.小台_法人多方口數(DATA) - self.小台_法人空方口數(DATA)

    def 價差(self, DATA = None): # 台指 - 加權指數
        if DATA is None: DATA = self._DATA
        return Component(DATA['台指期貨_合併']['Close'] - DATA['加權指數']['Close'])
