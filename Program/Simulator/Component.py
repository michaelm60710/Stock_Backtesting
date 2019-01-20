import pandas as pd
import numpy as np
import logging
import talib


#from Simulator import Component

'''
Note: You can use help(func) to get info

'''
class Component:
    def __init__(self, series_data):
        if type(series_data) != pd.core.series.Series:
            logging.warning("Wrong type")
            #return

        # Init
        self.data = series_data

    def getData(self, v):
        if type(v) == type(self):
            return v.data
        elif type(v) == pd.core.series.Series:
            return v
        return v
        #elif type(v) == list or :
        #    logging.warning("Wrong data type? Type: "+str(type(v)) )
        #    return v
    def __and__(self, v):
        other = self.getData(v)
        return Component(self.data & other)
    def __or__(self, v):
        other = self.getData(v)
        return Component(self.data | other)
    def __invert__(self):
        return Component(~self.data)

    # Compare: >, <, >=, <=
    def __lt__(self, v): #less than < 
        other = self.getData(v)
        return Component(self.data < other)
    def __gt__(self, v): #greater than >
        other = self.getData(v)
        return Component(self.data > other)
    def __ge__(self, v):
        other = self.getData(v)
        return Component(self.data >= other)
    def __le__(self, v):
        other = self.getData(v)
        return Component(self.data <= other)
    def __eq__(self, v): # ==
        other = self.getData(v)
        return Component(self.data == other)
    def __ne__(self, v): # !=
        other = self.getData(v)
        return Component(self.data != other)
    # + - * /
    def __add__(self, v):
        other = self.getData(v)
        return Component(self.data + other)
    def __sub__(self, v):
        other = self.getData(v)
        return Component(self.data - other)
    def __mul__(self, v):
        other = self.getData(v)
        return Component(self.data * other)
    def __div__(self, v):
        other = self.getData(v)
        return Component(self.data / other)

    def __str__(self):
        return str(self.data)

    def shift(self, N):
        return Component(self.data.shift(N))





def GetSeriesData(data, Name):
    '''
    If data is Dataframe, only return column = Name
    '''
    if type(data) == Component:
        data = data.data
    if type(data) == pd.core.series.Series:
        pass
    elif type(data) == pd.core.frame.DataFrame:
        if Name in data:
            data = data[Name]
        else:
            logging.warning("Can't find column name: " + Name)
    return data



# --------------------- #
#        技術指標        #
# --------------------- #

def N日均線(data, N = 3, Name = 'Close'):
    '''
    INFO:
        data should be series type.
        N means the num of day
        Default name is 'Close'
    EXAMPLE:
        N日均線(sim.DATA['加權指數']['Close'], 5)
    '''
    data = GetSeriesData(data, Name)
    return Component(data.rolling(N).mean())

def delta(data, shift_N = 1):
    '''
    data - data.shift(shift_N)
    '''
    data = data - data.shift(shift_N)
    if type(data) != Component: 
        data = Component(data)
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
    if type(OHLCV_data) == dict:
        return OHLCV_data
    elif type(OHLCV_data) == pd.core.frame.DataFrame:
        dict_data = {
            'open': OHLCV_data['Open'].astype(float),
            'high': OHLCV_data['High'].astype(float),
            'low': OHLCV_data['Low'].astype(float),
            'close': OHLCV_data['Close'].astype(float),
            'volume': OHLCV_data['Volume'].astype(float)
        }

    return dict_data

def talib2df(talib_output, index):
    if type(talib_output) == list:
        ret = pd.DataFrame(talib_output).transpose()
    else:
        ret = pd.Series(talib_output)
    ret.index = index

    return ret

def talib2component(talib_output, index):
    if type(talib_output) == list:
        ret = []
        for arr in talib_output:
            tmp = pd.Series(arr)
            tmp.index = index 
            ret.append( Component(tmp) )

    else:
        ret = pd.Series(talib_output)
        ret.index = ret
        ret = Component(ret)

    return ret

def talib_KD(OHLCV_data):
    '''
    INFO:
        Make sure OHLCV_data contains Open, High, Low, Close, Volume data
    EXAMPLE:
        Comp_slowk, Comp_slowd = talib_KD(sim.DATA['加權指數'])
c    '''
    OHLCV_data = OHLCV_to_dict(OHLCV_data)
    talib_output = talib.abstract.STOCH(OHLCV_data)
    return talib2component(talib_output, OHLCV_data['close'].index)

def talib_BBANDS(OHLCV_data):
    '''
    INFO:
        Make sure OHLCV_data contains Open, High, Low, Close, Volume data
    EXAMPLE:
        Comp_slowk, Comp_slowd = talib_KD(sim.DATA['加權指數'])
    '''
    OHLCV_data = OHLCV_to_dict(OHLCV_data)
    talib_output = talib.abstract.BBANDS(OHLCV_data, timeperiod=20, nbdevup=2, nbdevdn=2, matype=talib.MA_Type.T3)
    return talib2component(talib_output, OHLCV_data['close'].index)


# --------------------- #
#   Component example   #
# --------------------- #
'''
Note: DATA = sim.DATA
'''
def 小外資_多方口數(DATA):
    return Component( DATA['期貨法人']['大台_外資']['多方未平倉口數'] - DATA['台指期貨_大額交易人']['前五大特定法人交易人買方'])

def 小外資_空方口數(DATA):
    return Component( DATA['期貨法人']['大台_外資']['空方未平倉口數'] - DATA['台指期貨_大額交易人']['前五大特定法人交易人賣方'])

def 小台_法人多方口數(DATA):
    期貨法人 = DATA['期貨法人']
    return Component(期貨法人['小台指_外資']['多方未平倉口數'] + 期貨法人['小台指_投信']['多方未平倉口數'] \
                   + 期貨法人['小台指_自營商']['多方未平倉口數'] )

def 小台_法人空方口數(DATA):
    期貨法人 = DATA['期貨法人']
    return Component(期貨法人['小台指_外資']['空方未平倉口數'] + 期貨法人['小台指_投信']['空方未平倉口數'] \
                   + 期貨法人['小台指_自營商']['空方未平倉口數'] )

def 散戶多空(DATA): # 散戶指標
    return 小台_法人空方口數(DATA) - 小台_法人多方口數(DATA)

def 散戶留倉數(DATA):
    return Component(DATA['小台指_總留倉數']['未沖銷契約數']*2) - 小台_法人多方口數(DATA) - 小台_法人空方口數(DATA)

def 價差(DATA): # 台指 - 加權指數 
    return Component(DATA['台指期貨_合併']['Close'] - sim.DATA['加權指數']['Close'])

