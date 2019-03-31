#!/usr/bin/env python3
import pandas as pd
import numpy as np
import logging
import sys, os
import datetime
sys.path.insert(0, os.path.abspath(""+"../Crawler"))

from Packing import Data_preprocessing
from Component import Component, Components_lib
from Structure import  dotdict
from Plot import Iplot, Sim_func

# import plotly
import plotly
from plotly import tools

# Setting loggin
#logging.basicConfig(level=logging.info)

class Simulator(Iplot):
    _buy  = None
    _sell = None
    _long_short = 'long'
    _hold = None
    _hold_AllDateIndex = None

    _tax = None

    # Information
    Report_item = dict()
    _each_stocks_trade_count = None
    _Sell_info = None
    _Profit_rate_list = None
    _Report_str = ""

    # Class
    _DP = None        # Class Data_preprocessing
    Components = None # Class Components_lib

    def __init__(self, updatePKL = True, rebuildPKL = False, verbose = True):

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO) # set level: ALL < DEBUG < INFO < WARN < ERROR < FATAL < OFF

        self.verbose = verbose
        self._DP = Data_preprocessing(update = updatePKL, rebuild = rebuildPKL, verbose = verbose)
        self.DATA = dotdict(self._DP.data_dict)
        self.Components = Components_lib(self.DATA)
        Iplot.__init__(self, DATA = self.DATA)

    # Decorator
    #def _input_wrapper(decorated):
    #    pass

    def Setup(self, buy, sell, tax = 80, long_short = 'long'):
        '''
        set up : buy, sell, tax, hold, hold_AllDateIndex
        '''

        # long_short
        if type(long_short) is str and long_short.lower() == 'long':
            self._long_short = 'long'
        elif type(long_short) is str and long_short.lower() == 'short':
            self._long_short = 'short'
        else:
            self.logger.error("Wrong long_short input. Try to use: \'long\' or \'short\'.")

        if len(sell.data) != len(buy.data): self.logger.warning(" Length of sell and buy is different. ")

        # tax
        self._tax = tax

        # buy sell 選擇隔天交易
        old_index = buy.data.index
        new_index = old_index[1:].append(pd.DatetimeIndex([old_index[-1] + datetime.timedelta(days=1)])) # shift index
        self._buy  = Component(buy.data.copy() )
        self._sell = Component(sell.data.copy() )
        self._buy.data.index = new_index
        self._sell.data.index = new_index

        # old
        #self._buy = buy.shift(1)
        #self._sell = sell.shift(1)

        # buy sell 整理 （假設交易為'買多')
        # sell, buy不能同時為True. If sell is True, buy must be False
        self._buy = self._buy & (self._buy != self._sell)

        # set buy signal = 2, set sell signal = -1. # Buy_Sell = buy + sell
        # if Buy_Sell > 0: hold = True, elif Buy_Sell < 0: hold = False, elif Buy_Sell == 0: hold = previous_value
        Buy_Sell = (self._buy.data * 2).fillna(0) + (self._sell.data * -1).fillna(0)

        # hold
        Buy_Sell = AddFirstRow(Buy_Sell)
        self._hold = Buy_Sell.applymap(Hold_convert)

        # hold with all date
        Buy_Sell = Fillup_missing_date(Buy_Sell)
        self._hold_AllDateIndex = Buy_Sell.applymap(Hold_convert)

        return self

    @Sim_func._input_wrapper
    def Run(self, buy, sell, tax = 80, sell_price = None, buy_price = None, \
        start_date = None, end_date = None, long_short = 'long', Ratio = 0.5, \
        Mode = 'futures', OHLC_data = None, Futures_Unit = None, Money = None, Plot = True):
        '''
        Money: 本金
        buy_price: open | close
        sell_price: open | close
        Mode: Stocks | Futures
        Futures_Unit: 小台一點50 (Futures Mode)
        Ratio: The percentage of capital to be put into a single trade (0 < Ratio < 1). The default is 0.5.
        # Note: long_short, Money no use
        '''

        # Setup buy, sell, hold data, tax
        self.Setup(buy = buy, sell = sell, tax = tax, long_short = long_short)

        # ------------------- #
        #     Mode select     #
        # ------------------- #
        # default settings
        if Mode == 'futures':
            if Futures_Unit is None: Futures_Unit = 50
            if OHLC_data    is None: OHLC_data = self.DATA['台指期貨_一般']
            if Money        is None: Money = 30000
        elif Mode == 'stocks':
            Futures_Unit = 1
            if OHLC_data    is None: OHLC_data = self.DATA['台股個股']
            if Money        is None: Money = 1000000
        else:
            self.logger.error("Wrong Mode select. Try to use: \'futures\' or \'stocks\'.")
        self.logger.info("Mode : " + Mode)

        # ------------------- #
        #  Setting arguments  #
        # ------------------- #
        def price(use_price, default_use):
            '''
                for sell_price & buy_price
            '''
            if use_price is None:
                use_price = OHLC_data[default_use].astype(float)
            elif use_price == 'Open' or use_price == 'Close':
                use_price = OHLC_data[use_price].astype(float)
            elif type(use_price) != pd.core.series.Series:
                self.logger.error("Wrong {0} format. Try to use: \'Open\', \'Close\' or Series type.".format(use_price))
            if type(use_price) == pd.core.series.Series:
                use_price = use_price.to_frame()
            return use_price

        buy_price  = price(use_price = buy_price,  default_use = 'Open')
        sell_price = price(use_price = sell_price, default_use = 'Close')

        if start_date is None: start_date = self._hold.index[0]
        start_date = max(start_date, self._hold.index[0], buy_price.index[0])

        if end_date is None: end_date = self._hold.index[-1]
        end_date = min(end_date, self._hold.index[-1], buy_price.index[-1])

        Close_price = OHLC_data['Close'].astype(float)
        tax = self._tax

        # Index 整合
        buy_price = buy_price.truncate(start_date, end_date, axis = 0)
        sell_price = sell_price.truncate(start_date, end_date, axis = 0)
        Close_price = Close_price.truncate(start_date, end_date, axis = 0)
        hold_all = self._hold.truncate(start_date, end_date, axis = 0)

        # ---------------------------- #
        #  Calculate Profit/Drawdown   #
        # ---------------------------- #
        # Use hold_all, buy_price, sell_price
        # BuySell: buy = 1, sell = -1, others = 0
        BuySell = hold_all.apply(EachCol, axis = 0)

        # Init variable
        #trade_times = 0
        #exception_trades = 0
        #Ratio = 0.5
        buy_ratio = 0
        sell_hold_stocks = 0 # 算當日把持有全賣掉所獲的unit
        I_Unit = 1           # Ideal
        Unit_tmp = I_Unit    # 類似現有手頭資金
        Unit_Cost = []       # 類似換算總資金, 把所有持股也賣掉來換算
        Unit_Hold = []       # 類似持股
        Unit      = []       # list of Unit_tmp
        self._Sell_info = []  # sell stocks, unit, profit_rate
        self._Profit_rate_list = [] # all profit rate

        hold_info_list = [ 0 for i in BuySell.columns] # daily hold stocks info
        buyprice_info  = [ 0 for i in BuySell.columns] # hold stock's buy price

        # Sort dataframeby columns
        BuySell = BuySell.sort_index(axis = 1)
        sell_price = sell_price.sort_index(axis = 1)
        buy_price = buy_price.sort_index(axis = 1)

        for (idx1, row), (idx2, sell_row), (idx3, buy_row) in zip(BuySell.iterrows(), sell_price.iterrows(), buy_price.iterrows()):

            # Step 1. Calculate how many stocks have buy signals
            pos_count = 0
            for v in row:
                if v > 0: pos_count += 1
            # prevent division by zero
            if pos_count == 0: pos_count = 1

            # Step 2. Calculate Ideal buy ratio
            buy_ratio = Unit_tmp*Ratio/pos_count
            assert buy_ratio > 0, "buy_ratio = {0} {1} {2}".format(buy_ratio,Unit_tmp,Ratio)

            # Step 3. Run each StockID: to calculate daily Unit, Unit_Hold, Unit_Cost

            # setting before run pre_DF
            new_buyprice_info  = []
            new_hold_info_list = []
            daily_sell_info = dict()
            sell_hold_stocks = 0

            # Run each stocksID
            for stock_idx, Cur_BuySellSignal, Cur_sellprice, Cur_buyprice, pre_buyprice_info, hold_info \
                in zip(row.index, row.values, sell_row.values, buy_row.values, buyprice_info, hold_info_list):

                # refresh buyprice & hold info
                if Cur_BuySellSignal > 0: # New buy signal
                    if not np.isnan(Cur_buyprice): # if buyprice is Nan, dont trade
                        pre_buyprice_info = Cur_buyprice
                        hold_info = buy_ratio
                        Unit_tmp -= buy_ratio
                elif Cur_BuySellSignal < 0 and hold_info > 0: # New sell signal
                    if np.isnan(Cur_sellprice):
                        Unit_add = hold_info
                        #exception_trades += 1
                    else:
                        Unit_add = hold_info*Cur_sellprice/pre_buyprice_info
                        profit_rate = (Cur_sellprice-pre_buyprice_info)/pre_buyprice_info
                        daily_sell_info[stock_idx] = {'hold_info':hold_info, \
                                                      'profit_rate':profit_rate}
                        self._Profit_rate_list.append((profit_rate, dict(stockID = stock_idx, Sell_date = idx1)))
                        #trade_times += 1

                    Unit_tmp += Unit_add
                    pre_buyprice_info = 0
                    hold_info = 0

                # calculate sell_hold_stocks
                if hold_info > 0:
                    sell_hold_stocks += hold_info*Cur_buyprice/Cur_sellprice

                new_buyprice_info.append(pre_buyprice_info)
                new_hold_info_list.append(hold_info)

            # update buyprice & hold info
            buyprice_info  = new_buyprice_info
            hold_info_list = new_hold_info_list

            # Record unit info
            Unit.append(Unit_tmp)
            Unit_Cost.append(Unit_tmp + sell_hold_stocks) # cost = Unit_tmp + sell_hold_stocks
            Unit_Hold.append(hold_info_list)

            self._Sell_info.append(daily_sell_info)

        # Ideal info
        Unit_Hold = pd.DataFrame(Unit_Hold, columns=BuySell.columns, index = BuySell.index)
        Unit      = pd.DataFrame(Unit, index=BuySell.index)
        Unit_Cost = pd.DataFrame(Unit_Cost, index=BuySell.index).rename(columns={0:'Unit_Cost'}).round(3)

        # TODO (actual info) Hold, RemainMoney, Cost

        # Max_profit
        Max_profit = Unit_Cost.max()

        # drawdown, 虧損評估, Current/Max_current_profit
        Drawdown = Unit_Cost/Unit_Cost.cummax()
        Drawdown = Drawdown.rename(columns={'Unit_Cost':'Drawdown'})
        Drawdown[Drawdown < 0] = 0
        Max_Drawdown = Drawdown['Drawdown'].min() - 1

        # WeightIndex_profit
        WeightIndex_profit = self.DATA['加權指數']['Close'].copy()
        WeightIndex_profit = WeightIndex_profit.truncate(start_date, end_date, axis = 0)
        WeightIndex_profit = WeightIndex_profit/WeightIndex_profit[0]

        # round
        Drawdown = Drawdown.round(2)

        # Long or shot
        if self._long_short == 'short':
            pass # TODO

        # Report
        self._each_stocks_trade_count = BuySell.apply(pd.value_counts).loc[1]
        self._each_stocks_trade_count = self._each_stocks_trade_count[self._each_stocks_trade_count > 0]
        trade_times = int(self._each_stocks_trade_count.sum())
        ratio_list = [x[0] for x in self._Profit_rate_list]
        win_ratio = len(list(filter(lambda x: (x > 0), ratio_list)))/ len(ratio_list)
        self.Report(Unit_Cost['Unit_Cost'], Money = I_Unit, hold_all = hold_all, \
                    trade_times = trade_times, Max_Drawdown = Max_Drawdown, Win_ratio = win_ratio)

        # Plot
        #return WeightIndex_profit, Unit_Cost.squeeze(), Drawdown.squeeze()
        if Plot:
            return self.SubPlotly(WeightIndex_profit, Unit_Cost.squeeze(), Drawdown.squeeze(), Mode = Mode)

    def Report(self, Cost, Money, hold_all, **kwargs):
        '''
        交易開始日期: start_date, 交易結束日期: end_date
        獲利: Profit, 最大獲利: Max_Profit, 最小獲利(或虧損): Min_Profit
        報酬率: ROI, 年報酬率: Annualized_ROI
        交易次數: trade_times
        平均持倉時間: Ave_hold_days, 最長持倉時間: Max_hold_days (只算交易日)
        最短持倉時間: Min_hold_days (只算交易日)
        '''

        # Setttings report
        self.Report_item['start_date'] = Cost.index[0]
        self.Report_item['end_date'] = Cost.index[-1]
        self.Report_item['Money']  = Money
        self.Report_item['Profit'] = round(Cost[-1] - Money, 3)
        self.Report_item['Max_Profit'] = round(Cost.nlargest(1)[0] - Money, 3)
        self.Report_item['Min_Profit'] = round(Cost.nsmallest(1)[0] - Money, 3)
        self.Report_item['ROI'] = float(self.Report_item['Profit']/Money)
        self.Report_item['Annualized_ROI'] = 0
        self.Report_item['trade_times'] = 0
        self.Report_item['Ave_hold_days'] = 0
        self.Report_item['Max_hold_days'] = 0
        self.Report_item['Min_hold_days'] = 10000

        # TODO
        '''
        Max Drawdown: max(1 - 策略当日价值 / 当日之前虚拟账户最高价值)
        Annualized Returns:  (策略最终价值 / 策略初始价值 - 1) / 回测交易日数量 × 250
        Benchmark Returns:
        Alpha:
        Beta:
        Volatility:
        Information Ratio:
        Sharpe Ratio: (平均年化報酬率-無風險利率)/年化標準差 # 大小本身沒意義, 比較才有意義
        '''

        # add Report
        for key in kwargs:
            self.logger.info("Add new report info: {0}, {1}".format(key,kwargs[key]))
            self.Report_item[key] = kwargs[key]

        # Calculate Ave/Max/Min_hold_days, trade_times
        for col, Stock_series in hold_all.iteritems():
            trade_start = None
            Trading_day_idx = 0
            for i, v in Stock_series.iteritems():
                Trading_day_idx += 1
                if v:
                    if trade_start is None :
                        trade_start = Trading_day_idx

                elif trade_start is not None:
                    period = Trading_day_idx - trade_start
                    if period > self.Report_item['Max_hold_days']:
                        self.Report_item['Max_hold_days'] = period
                    if period < self.Report_item['Min_hold_days']:
                        self.Report_item['Min_hold_days'] = period
                    self.Report_item['Ave_hold_days'] += period
                    trade_start = None

        self.Report_item['Ave_hold_days'] = round(self.Report_item['Ave_hold_days']/self.Report_item['trade_times'], 2)

        # Calculate Annualized ROI: (（1＋ROI）^（1／year））－1
        years = round((self.Report_item['end_date'] - self.Report_item['start_date']).days/365, 2)
        if self.Report_item['ROI'] > 0:
            self.Report_item['Annualized_ROI'] = pow(1+self.Report_item['ROI'], 1/years) - 1
        else:
            self.Report_item['Annualized_ROI'] = -(pow(1-self.Report_item['ROI'], 1/years) - 1)

        # ------------------- #
        #    Report String    #
        # ------------------- #
        report_str = "# " + "RESULTS".center(54, "-") + " #\n"
        report_str += "#" + " "*56 + "#\n"

        start = self.Report_item['start_date'].strftime("%Y-%m-%d")
        end = self.Report_item['end_date'].strftime("%Y-%m-%d")
        if self._long_short == 'short':
            report_str += "# {:>20} = {:>15}{:>18}\n".format('Long or Short', self._long_short, "#")
        report_str += "# {:>20} = {:>15} to {:}{:>4}\n".format('Period', start, end, "#")
        report_str += "# {:>20} = {:>15}{:>18}\n".format('Num of Trade', self.Report_item['trade_times'],"#")

        report_str += "# {:>20} = {:>15}{:>18}\n".format('Money', self.Report_item['Money'], "#")
        report_str += "# {:>20} = {:>15}{:>18}\n".format('Profit', self.Report_item['Profit'], "#")
        report_str += "# {:>20} = {:>15}{:>18}\n".format('Max Profit', self.Report_item['Max_Profit'], "#")
        #report_str += "# {:>20} = {:>15}{:>18}\n".format('Min Profit', self.Report_item['Min_Profit'], "#")
        report_str += "# {:>20} = {:>14}%{:>18}\n".format('Win Ratio', round(self.Report_item['Win_ratio']*100,2), "#")
        report_str += "# {:>20} = {:>14}%{:>18}\n".format('Max Drawdown', round(self.Report_item['Max_Drawdown']*100,2), "#")

        report_str += "# {:>20} = {:>14}%{:>18}\n".format('Return On Investment', round(self.Report_item['ROI']*100,2), "#")
        report_str += "# {:>20} = {:>14}%{:>18}\n".format('Annualized ROI', round(self.Report_item['Annualized_ROI']*100,2), "#")

        report_str += "# {:>20} = {:>15}{:>18}\n".format('Ave_hold_days', self.Report_item['Ave_hold_days'], "#")
        report_str += "# {:>20} = {:>15}{:>18}\n".format('Max_hold_days', self.Report_item['Max_hold_days'], "#")
        report_str += "# {:>20} = {:>15}{:>18}\n".format('Min_hold_days', self.Report_item['Min_hold_days'], "#")

        report_str += "#" + " "*56 + "#\n"
        report_str += "# " + "".center(54, "-") + " #\n"

        self._Report_str = report_str
        print(report_str)

    def SubPlotly(self, WeightIndex_profit, Cost_risk, Drawdown, Mode):
        # ------------------------------ #
        #    First Fig: Compare Profit   #
        # ------------------------------ #
        # fig1: 1. 大盤獲利
        fig1 = []
        fig1.append( dict(
            type = 'scatter',
            y = WeightIndex_profit,
            x = WeightIndex_profit.index,
            mode = 'lines',
            marker = dict(size = 10,
                color = '#2F3C8E',
                line = dict(
                    width = 2,
                )),
            yaxis='y1',
            name='台指獲利'
        ) )
        # fig1: 2. 績效獲利
        fig1.append( dict(
            type = 'scatter',
            y = Cost_risk,
            x = Cost_risk.index,
            mode = 'lines',
            marker = dict(size = 10,
                color = '#8DF53D',
                line = dict(
                    width = 2,
                )),
            yaxis='y1',
            name='績效獲利'
        ) )
        if Mode == 'stocks':
            trade_info = pd.Series(data = self._Sell_info, index = Cost_risk.index)
            trade_info = trade_info[trade_info != {}]
            new_trade_info = self._Convert_tradeInfo_to_str(trade_info)
            fig1.append( dict(
                type = 'scatter',
                y = pd.Series(data = 1, index = trade_info.index),
                x = trade_info.index,
                text = new_trade_info,
                mode = 'lines+markers',
                marker = dict(size = 3,
                    color = 'black',
                    ),
                line = dict(width = 1,),
                yaxis='y1',
                name = 'Trade info',
            ) )

        # ---------------------------------- #
        #   Second Fig: Drawdown percentage  #
        # ---------------------------------- #
        # fig2: Drawdown
        fig2 = []
        fig2.append( dict(
            type = 'scatter',
            y = Drawdown,
            x = Drawdown.index,
            mode = 'lines',
            marker = dict(size = 10,
                color = '#FF0000',
                line = dict(
                    width = 2,
                )),
            yaxis='y1',
            name='虧損評估'
        ) )

        # ----------- #
        #   Sub Plot  #
        # ----------- #
        # add traces
        fig = tools.make_subplots(rows=2, cols=1,  subplot_titles=('績效','Drawdown') )
        for f in fig1:
            fig.add_trace(f, 1, 1)
        for f in fig2:
            fig.add_trace(f, 2, 1)

        # layout
        fig['layout']['yaxis1'].update(domain=[0.5, 1])
        fig['layout']['yaxis2'].update(domain=[0, 0.35])
        fig['layout']['xaxis2'].update(anchor='y2')
        fig['layout'].update(height=700, title='Results')

        # Plot
        return plotly.offline.iplot(fig, filename='Results')

    @Sim_func._input_wrapper
    def Iplotly(self, start_date = None, end_date = None, hold = None, return_iplot = True, StockID = None, Mode = 'futures', \
        ignore_non_trading_date = False, \
        Plot_KD = False, Plot_BBAND = False, Plot_Volume = False, Plot_3_Investors = False, \
        **kwargs):
        '''
            return_iplot: default is True
            ignore_non_trading_date: default is False
            Plot_KD
            Plot_BBAND
            Plot_Volume
            Plot_3_Investors 
        '''

        Future_mode = 'futures'
        Stock_mode  = 'stocks'

        # Init Mode & variable
        if Mode == Future_mode:
            OHLC_data = self.DATA['加權指數']
        elif Mode == Stock_mode:
            if StockID is None: self.logger.error("Please select a stockID.")
            OHLC_data = self._GetStock_OHLCV_data(StockID)
        else:
            self.logger.error("Wrong Mode select. Try to use: Mode = \'futures\' or \'stocks\'.")
        self.logger.info("Mode : " + Mode)

        if start_date is None: start_date = OHLC_data.index[0]
        if end_date   is None: end_date = OHLC_data.index[-1]


        # Index 整合
        OHLC_data = OHLC_data.loc[start_date:end_date]

        # hold_AllDateIndex
        hold_AllDateIndex = None
        if hold is None:
            if self._hold_AllDateIndex is not None:
                hold_AllDateIndex = self._hold_AllDateIndex.truncate(OHLC_data.index[0], OHLC_data.index[-1], axis = 0)
        else:
            hold_AllDateIndex = hold.truncate(OHLC_data.index[0], OHLC_data.index[-1], axis = 0)

        if hold_AllDateIndex is not None:
            if Mode == Stock_mode:
                hold_AllDateIndex = hold_AllDateIndex[StockID]
            else:
                hold_AllDateIndex = hold_AllDateIndex.squeeze() # convert DF to series

        #----------------------------#
        #        plotly setup        #
        #----------------------------#

        # 1. Settings
        fig = dict( data= [dict()], layout=dict() )
        INCREASING_COLOR = '#AA0000' # Red
        DECREASING_COLOR = '#227700' # Green

        # 2. 大盤
        fig['data'].append( dict(
            type = 'candlestick',
            open = OHLC_data.Open,
            high = OHLC_data.High,
            low = OHLC_data.Low,
            close = OHLC_data.Close,
            x = OHLC_data.index,
            yaxis='y1',
            name='OHLC',
            increasing = dict( line = dict( color = INCREASING_COLOR ) ),
            decreasing = dict( line = dict( color = DECREASING_COLOR ) ),
        ) )
        # 3. Hold siganls shape
        trade_times = 0
        if hold_AllDateIndex is not None:
            fig['layout']['shapes'] = self._Hold_shape_insert([], hold_AllDateIndex)
            trade_times = len(fig['layout']['shapes'])


        print("# trade times: {0}".format(trade_times) )

        # 4. ignore empty dates
        if ignore_non_trading_date:
            fig['layout']['xaxis'] = {'type':'category'}

        # 5. add plot
        if Plot_KD:
            fig = self.P_Add_KD(fig = fig, StockID = StockID, Mode = Mode, start_date = start_date, end_date = end_date)
        if Plot_Volume:
            fig = self.P_Add_Volume(fig = fig, StockID = StockID, Mode = Mode, start_date = start_date, end_date = end_date)
        if Plot_3_Investors:
            fig = self.P_Add_3_Investors(fig = fig, StockID = StockID, start_date = start_date, end_date = end_date)

        if return_iplot:
            return plotly.offline.iplot(fig, filename='TW_Futures')
        else:
            return fig

    def _Hold_shape_insert(self, shape, hold_AllDateIndex):

        start_date = False
        for i, v in hold_AllDateIndex.iteritems():
            if hold_AllDateIndex[i] and not start_date:
                start_date = i
            elif not hold_AllDateIndex[i] and start_date:
                end_date = i
                # TODO: insert shape
                shape.append({
                    'type': 'rect',
                    'xref': 'x',
                    'yref': 'paper',
                    'x0': start_date,
                    'y0': 0,
                    'x1': end_date,
                    'y1': 1,
                    'fillcolor': '#0080FF',
                    'opacity': 0.3,
                    'line': {
                        'width': 0,
                    }
                })
                start_date = False

        return shape

    def _Convert_tradeInfo_to_str(self, trade_info): #trade_info = list
        new_trade_info = []
        for idx in trade_info:
            string1 = ""
            for stockID, value in idx.items():
                string1 += stockID + ": "
                for v, value in value.items():
                    string1 += "{0}={1}, ".format(v, round(value,2))
                string1 += '<br>'
            new_trade_info.append(string1)
        return new_trade_info

    def Optimize_ratio(self, nun_of_trades:int, total_profit_ratio, profit_rate_list):
        '''
            Original Kelly format:
                W*R/(1+Rf) - (1-W)*1/(1-f) = 0
                -> Kelly % = W – [(1 – W)/R]
                NOTE: W is win ratio, R is average profit ratio, f is Kelly &
            New: W*RR/(1+RRf) - (1-W)*L/(1-Lf) = 0
                -> New % = W(RR+L)/L*RR - 1/RR
                NOTE: RR is average win profit ratio, L is average loss profit ratio
        '''
        pass


def date_convert(date):
    '''
    str/int/datetype to date type
    '''
    import datetime
    if type(date) == int: date = str(date)
    if type(date) == str:
        if   len(date) == 10: return datetime.datetime.strptime(date, '%Y-%m-%d')
        elif len(date) == 8:  return datetime.datetime.strptime(date, '%Y%m%d')
        logging.error("Can't convert {0} to date type. example: date = 20180101".format(date))

    else:
        return date


#-----------------------------#
#        For Sim setup        #
#-----------------------------#

def Fillup_missing_date(df, fill_value = 0):
    '''
    Fill missing date
    '''
    idx = pd.date_range(start = df.index[0],end = df.index[-1])
    return df.reindex(idx, fill_value = fill_value)

def static_var(**kwargs):
    '''
    Generate/Init Static variable
    '''
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

@static_var(hold=False)
def Hold_convert(x):
    '''
    Generate Static variable: 'hold' (Used to keep previous value)
    # NOTE: Need to init static value before using it
    '''
    if x > 0:
        Hold_convert.hold = True
    elif x < 0:
        Hold_convert.hold = False

    return Hold_convert.hold

def AddFirstRow(df, first_row_value = -1):
    '''
    To avoid wrong conditions during calculating 'hold' data, add a new row at the first row.
    '''
    import datetime
    if type(df) is pd.core.series.Series:
        df = df.to_frame()

    FirstDate = df.index[0].to_pydatetime() - datetime.timedelta(days=1)
    tmp_df = pd.DataFrame(data = first_row_value, columns = df.columns, index = [FirstDate])
    return  pd.concat([tmp_df, df])


#-----------------------------#
#     For Sim Run: Buy        #
#-----------------------------#

@static_var(prev=False)
def Update_hold(x):
    '''
    buy = 1, sell = -1, others = 0
    '''
    tmp = 0
    if Update_hold.prev != x:
        if x: tmp = 1
        else: tmp = -1
    Update_hold.prev = x

    return tmp

def EachCol(x):
    # Init Update_hold
    Update_hold.prev = x[0]

    return x.apply(Update_hold)



if __name__ == '__main__':
    # debug and test use
    Simulator()
