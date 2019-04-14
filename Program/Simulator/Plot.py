#!/usr/bin/env python3
import pandas as pd
import numpy as np
import logging
import sys, os
import talib
from functools import wraps

# import plotly

# Component
from Component import talib_Output
from Structure import date_convert

class Sim_func():
    def __init__(self,DATA):
        self.DATA = DATA

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
                elif Mode[0] == 'f': Mode = 'futures'
                kwargs['Mode'] = Mode

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
        upperband = upperband.data.truncate(start_date, end_date, axis = 0)
        lowerband = lowerband.data.truncate(start_date, end_date, axis = 0)

        # add upperBBAND & lowerBBAND
        fig['data'].append( dict(
                type = 'scatter',
                y = upperband,
                x = upperband.index,
                yaxis='y1',
                name='UpperBAND'
            ) )
        fig['data'].append( dict(
                type = 'scatter',
                y = lowerband,
                x = lowerband.index,
                yaxis='y1',
                name='LowerBAND'
            ) )

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
