#!/usr/bin/env python3
import pandas as pd
import numpy as np
import logging
import sys, os
import talib
from functools import wraps

# import plotly
import plotly

# Component
from Component import talib_Output
from Structure import date_convert

class Sim_func():

    # Information
    _buy  = None
    _sell = None
    _hold = None
    _hold_AllDateIndex = None
    _tax = None
    _long_short = 'long'
    Report_item = dict()
    _each_stocks_trade_count = None
    _Sell_info = None
    _Profit_rate_list = None
    _Report_str = ""

    def __init__(self,DATA):
        self.DATA = DATA
        self.logger = logging.getLogger(__name__)

    def _GetStock_OHLCV_data(self, StockID):
        name_list = ['Open', 'High', 'Low', 'Close', 'Volume']
        data_list = dict()
        for name in name_list:
            data_list[name] = self.DATA['台股個股'][name][StockID]
        OHLCV = pd.DataFrame(data = data_list)

        return OHLCV

    # Decorator
    @staticmethod
    def _input_wrapper(decorated):
        '''
            check input varaiables
        '''
        @wraps(decorated)
        def wrapper(*args, **kwargs):
            #print (decorated.__name__ + " was called" )

            # check date type
            if 'start_date' in kwargs and kwargs['start_date'] is not None:
                kwargs['start_date'] = date_convert(kwargs['start_date'])
            if 'end_date' in kwargs and kwargs['end_date'] is not None:
                kwargs['end_date'] = date_convert(kwargs['end_date'])
            # check Mode
            if 'Mode' in kwargs:
                Mode = kwargs['Mode'].lower()
                if   Mode[0] == 's':
                    Mode = 'stocks'
                elif Mode[0] == 'f':
                    Mode = 'futures'
                kwargs['Mode'] = Mode
            # check long_short
            if 'long_short' in kwargs:
                long_short = str(kwargs['long_short'])[0].lower()
                if long_short == 'l':
                    long_short = 'long'
                elif long_short == 's':
                    long_short = 'short'
                else:
                    self.logger.error("Wrong long_short input. Try to use: \'long\' or \'short\'.")
                kwargs['long_short'] = long_short

            return decorated(*args, **kwargs)
        return wrapper

    # Usage function
    def Print_data_columns(self):
        for x in self.DATA:
            print ("\n# {0}, type = {1}".format(x, type(self.DATA[x])))
            if type(self.DATA[x]) is dict:
                print ("\tKeys: " + str(list(self.DATA[x].keys())) + "\n")
            else:
                print ("\tColumns: " + str(self.DATA[x].columns.values) + "\n")


class Iplot(Sim_func):
    def __init__(self, DATA):
        self.DATA = DATA
        self.logger = logging.getLogger(__name__)
        Sim_func.__init__(self, DATA = self.DATA)

    def P_ADD_decorator(decorated):

        def wrapper(*args, **kwargs):
            print (decorated.__name__ + " was called by P_ADD_decorator" )


            return decorated(*args, **kwargs)
        return wrapper

    @Sim_func._input_wrapper
    def P_Add_BBAND(self, fig = None, StockID = None, Mode = 'futures', start_date = None, end_date = None):
        # settings
        if Mode == 'futures':
            OHLC_data = self.DATA['加權指數']
        else:
            OHLC_data = self._GetStock_OHLCV_data(StockID)

        if fig is None:
            fig = dict( data= [dict()], layout=dict() )

        if start_date is None: start_date = OHLC_data.index[0]
        if end_date   is None: end_date   = OHLC_data.index[-1]

        talib_func_parameters = {'timeperiod':20, 'nbdevup':2, 'nbdevdn':2, 'matype':talib.MA_Type.T3 }
        upperband, middleband, lowerband = talib_Output(OHLC_data, talib.abstract.BBANDS, talib_func_parameters = talib_func_parameters)
        upperband  = upperband.data.truncate(start_date, end_date, axis = 0)
        middleband = middleband.data.truncate(start_date, end_date, axis = 0)
        lowerband  = lowerband.data.truncate(start_date, end_date, axis = 0)

        # add upperBBAND & lowerBBAND
        fig = self.P_Add_y1(fig, upperband,  'UpperBAND')
        fig = self.P_Add_y1(fig, middleband, 'MiddleBAND')
        fig = self.P_Add_y1(fig, lowerband,  'LowerBAND')

        return fig

    @Sim_func._input_wrapper
    def P_Add_MV(self, mv_days = 5, fig = None, StockID = None, Mode = 'futures', start_date = None, end_date = None):
        # settings
        if Mode == 'futures':
            close_data = self.DATA['加權指數']['Close']
        else:
            close_data = self.DATA['台股個股']['Close'][StockID]

        if fig is None:
            fig = dict( data= [dict()], layout=dict() )

        if start_date is None: start_date = close_data.index[0]
        if end_date   is None: end_date   = close_data.index[-1]

        mv_close_data = close_data.rolling(mv_days).mean().truncate(start_date, end_date, axis = 0)

        # add close_data
        fig = self.P_Add_y1(fig, mv_close_data, "MV_{0}".format(mv_days))

        return fig


    @Sim_func._input_wrapper
    def P_Add_KD(self, fig = None, StockID = None, Mode = 'futures', start_date = None, end_date = None):

        # settings
        if Mode == 'futures':
            OHLC_data = self.DATA['加權指數']
        else:
            OHLC_data = self._GetStock_OHLCV_data(StockID)

        if fig is None:
            fig = dict( data= [dict()], layout=dict() )

        if start_date is None: start_date = OHLC_data.index[0]
        if end_date   is None: end_date   = OHLC_data.index[-1]

        slowk, slowd = talib_Output(OHLC_data, talib.abstract.STOCH, {'fastk_period':9})
        slowk = slowk.data.truncate(start_date, end_date, axis = 0)
        slowd = slowd.data.truncate(start_date, end_date, axis = 0)


        # add slowk & slowd
        fig['data'].append( dict(
                type = 'scatter',
                y = slowk,
                x = slowk.index,
                yaxis='y4',
                opacity = 0.5,
                name='slowk'
            ) )
        fig['data'].append( dict(
                type = 'scatter',
                y = slowd,
                x = slowd.index,
                yaxis='y4',
                opacity = 0.5,
                name='slowd'
            ) )

        fig['layout']['yaxis4'] = dict(
            range= [0, 500],
            overlaying= 'y',
            anchor= 'x',
            side= 'right',
        )

        return fig

    @Sim_func._input_wrapper
    def P_Add_Volume(self, fig = None, StockID = None, Mode = 'futures', start_date = None, end_date = None):
        # settings
        if Mode == 'futures':
            OHLC_data = self.DATA['加權指數']
        else:
            OHLC_data = self._GetStock_OHLCV_data(StockID)

        if fig is None:
            fig = dict( data= [dict()], layout=dict() )

        if start_date is None: start_date = OHLC_data.index[0]
        if end_date   is None: end_date   = OHLC_data.index[-1]

        Volume = OHLC_data['Volume'].truncate(start_date, end_date, axis = 0)
        V_max = Volume.max()


        # add volume
        fig['data'].append( dict(
                type = 'bar',
                y = Volume,
                x = Volume.index,
                yaxis='y2',
                opacity = 0.5,
                name='volume'
            ) )

        fig['layout']['yaxis2'] = dict(
            range= [0, V_max*5],
            overlaying= 'y',
            anchor= 'x',
            side= 'right',
        )

        return fig

    @Sim_func._input_wrapper
    def P_Add_3_Investors(self, fig = None, StockID = None, Mode = 'futures', start_date = None, end_date = None):

        # settings
        # self.logger.error(" No support future mode. func(P_Add_3_Investors)")
        if Mode == 'futures':
            外資 = self.DATA['加權指數']['外資買賣差額']
            投信 = self.DATA['加權指數']['投信買賣差額']
            自營商 = self.DATA['加權指數']['自營商買賣差額']
        else:
            外資 = self.DATA['台股個股']['外資'][StockID]
            投信 = self.DATA['台股個股']['投信'][StockID]
            自營商 = self.DATA['台股個股']['自營商'][StockID]

        if start_date is None: start_date = 外資.index[0]
        if end_date   is None: end_date   = 外資.index[-1]

        外資 = 外資.truncate(start_date, end_date, axis = 0)
        投信 = 投信.truncate(start_date, end_date, axis = 0)
        自營商 = 自營商.truncate(start_date, end_date, axis = 0)



        V_max = (外資+投信+自營商).max()
        V_min = (外資+投信+自營商).min()


        # add 三法人
        fig['data'].append( dict(
                type = 'bar',
                y = 外資,
                x = 外資.index,
                yaxis='y3',
                #opacity = 0.5,
                name='外資買賣超',
                marker=dict( color = 'rgba(114, 195, 192, 0.9)'),
            ) )
        fig['data'].append( dict(
                type = 'bar',
                y = 投信,
                x = 投信.index,
                yaxis='y3',
                name='投信買賣超',
                marker=dict( color = 'rgba(243, 180, 69, 0.9)'),
            ) )
        fig['data'].append( dict(
                type = 'bar',
                y = 自營商,
                x = 自營商.index,
                yaxis='y3',
                opacity = 0.5,
                name='自營商買賣超',
                marker=dict( color = 'rgba(237, 75, 68, 0.9)'),
            ) )

        fig['layout']['yaxis3'] = dict(
            range= [V_min, V_max*5],
            overlaying= 'y',
            anchor= 'x',
            side= 'left',
        )
        fig['layout']['barmode'] = 'relative'

        return fig

    def P_Add_y1(self, fig, series, naming):
        fig['data'].append( dict(
                type = 'scatter',
                y = series,
                x = series.index,
                yaxis = 'y1',
                name = naming
            ) )
        return fig

    def Plot_Correlation(self, df, ignore_outliers = True, std_n = 3):
        '''
            Scatter Plot: profit ratio & given df
            Observe the correlation between two array after the Simulation.
            Example:
                sim.Correlation_plot(df = sim.DATA.台股個股['稅後淨利'], ignore_outliers = True, std_n = 3)
        '''
        # Check _Profit_rate_list
        if self._Profit_rate_list is None:
            self.logger.error("Profit_rate_list is Empty.")
            return

        # Convert df data
        from Component import Component
        if isinstance(df, Component): df = df.data
        first_idx = df.index[0]
        df = df.reindex(self._hold.index).fillna(method = 'ffill')
        df = df.loc[first_idx:]

        # Get profit_list & data_list
        trade_info = self._Profit_rate_list[0]
        profit_list = [round(x[0].tolist(), 4) for x in self._Profit_rate_list]
        text_info = [ "Stock ID = {0}, Buy_date = {1}, idx = {2}" \
                    .format(x[1]['stockID'], x[1]['Buy_date'].strftime('%Y-%m-%d'), idx) \
                    for idx, x in enumerate(self._Profit_rate_list)]
        data_list   = list()
        for trade_info in self._Profit_rate_list:
            stock_ID = trade_info[1]['stockID']
            Buy_date = trade_info[1]['Buy_date']
            profit_rate = trade_info[0]
            if Buy_date < first_idx or stock_ID not in df.columns:
                data_list.append(np.nan)
            else:
                d = df[stock_ID][Buy_date]
                if type(d) is bool:
                    d = int(d)
                else: # Converting numpy dtypes to native python types
                    d = d.tolist()
                data_list.append(d)

        # ignore outliers
        if ignore_outliers:
            mean = np.nanmean(data_list)
            sd = np.nanstd(data_list)
            print('Remove outliers:\nmean = {0:.2f}, upper bound = {1:.2f}, lower bound = {2:.2f}'.format(mean, mean+std_n*sd, mean-std_n*sd))
            data_list = [ np.nan if x > mean+std_n*sd or x < mean-std_n*sd else x for x in data_list]
        else:
            print('Plot all data')

        # plot info
        fig = dict( data= [dict()], layout=dict() )
        fig['data'].append(
            dict(
                type = 'scatter',
                y = data_list,
                x = profit_list,
                mode = 'markers',
                marker = dict(
                    size = 3,
                    color = 'rgba(152, 0, 0, .8)',
                    line = dict(
                        width = 1,
                        color = 'rgb(0, 0, 0)'
                    )
                ),
                text=text_info,
            )
        )
        fig['layout']['xaxis'] = dict(
            title = 'profit rate',
        )
        return plotly.offline.iplot(fig, filename='Correlation')
