import pandas as pd
import logging
import sys, os
sys.path.insert(0, os.path.abspath(""+"../Crawler"))

from Packing import Data_preprocessing
from Component import Component

# import plotly
import plotly
from plotly import tools
import plotly.plotly as py
import plotly.graph_objs as go

# Setting loggin
#logging.basicConfig(level=logging.info,
#                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
#                    datefmt='%m-%d %H:%M',
#                    handlers = [logging.FileHandler('my.log', 'w', 'utf-8'),])

class Simulator():
    _buy  = None
    _sell = None
    _long_short = 'long'
    _hold = None
    _hold_AllDateIndex = None

    #Price = None
    #Start_date = None
    #End_date = None
    _tax = None

    Report_item = dict()
    _Report_str = ""

    _DP = None # Class Data_preprocessing

    def __init__(self, updatePKL = True, rebuildPKL = False):
        #logging.info("Init Simulator")
        self._DP = Data_preprocessing(update = updatePKL, rebuild = rebuildPKL)
        self.DATA = self._DP.data_dict
        #super().__init__(update = updatePKL, rebuild = rebuildPKL)
        #self.DATA = self.data_dict

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
            logging.error("Wrong long_short input. Try to use: \'long\' or \'short\'.")
        # tax 
        self._tax = tax 

        # buy sell 選擇隔天交易
        self._buy = buy.shift(1)
        self._sell = sell.shift(1)

        # buy sell 整理 （假設交易為'買多')
        # sell, buy不能同時為True. If sell == True: buy = False
        self._buy = self._buy & (self._buy != self._sell)

        # hold
        self._hold = self._buy.data.copy()
        sell_series = self._sell.data
        assert len(sell_series) == len(self._hold), "ERROR: length of sell and hold is different."
        prev = self._hold[0] 
        for i, v in self._hold.iteritems():
            if sell_series[i]: 
                self._hold[i] = False
            elif not self._hold[i]:
                self._hold[i] = prev
            prev = self._hold[i]

        # hold with all date
        idx = pd.date_range(start = self._buy.data.index[0],end = self._buy.data.index[-1])
        buy_AllDateIndex = self._buy.data.reindex(idx,fill_value=False)
        sell_AllDateIndex = self._sell.data.reindex(idx,fill_value=False)
        self._hold_AllDateIndex = pd.Series(data = 0, index=idx)
        prev = 0
        for i, v in self._hold_AllDateIndex.iteritems():
            if sell_AllDateIndex[i]: 
                self._hold_AllDateIndex[i] = 0
            elif buy_AllDateIndex[i]:
                self._hold_AllDateIndex[i] = 1
            else:
                self._hold_AllDateIndex[i] = prev
            prev = self._hold_AllDateIndex[i]

        return self

    def Run(self, buy, sell, tax = 80, long_short = 'long', OHLC_data = None, buy_price = None, sell_price = None, \
        Futures_Unit = 50, Principal = 30000, start_date = None, end_date = None):
        '''
        Futures_Unit: 小台一點50
        Principal: 本金
        buy_price: open | close
        sell_price: open | close
        '''

        # Setup buy, sell, hold data, tax
        self.Setup(buy = buy, sell = sell, tax = tax, long_short = long_short)

        # ------------------- #
        #  Setting arguments  #
        # ------------------- #
        if OHLC_data is None:
            OHLC_data = self.DATA['台指期貨_一般']
        
        if buy_price is None:
            buy_price = OHLC_data['Open'].astype(float)
        elif buy_price == 'Open' or buy_price == 'Close':
            buy_price = OHLC_data[buy_price].astype(float)
        elif type(buy_price) != pd.core.series.Series:
            logging.error("Wrong buy_price format. Try to use: \'Open\', \'Close\' or Series type.")
        
        if sell_price is None:
            sell_price = OHLC_data['Close'].astype(float)
        elif sell_price == 'Open' or sell_price == 'Close':
            sell_price = OHLC_data[sell_price].astype(float)
        elif type(sell_price) != pd.core.series.Series:
            logging.error("Wrong sell_price format. Try to use: \'Open\', \'Close\' or Series type.")

        if start_date is None:
            start_date = buy_price.index[0]

        if end_date is None:
            end_date = buy_price.index[-1]

        Close_price = OHLC_data['Close'].astype(float)
        tax = self._tax

        # Index 整合
        OHLC_data = Select_time(OHLC_data, start_date, end_date)
        buy_price = buy_price.truncate(start_date, end_date, axis = 0)
        sell_price = sell_price.truncate(start_date, end_date, axis = 0)
        Close_price = Close_price.truncate(start_date, end_date, axis = 0)
        hold_all = self._hold.truncate(start_date, end_date, axis = 0)

        # ------------------------ #
        #  Calculate Profit/Loss   #
        # ------------------------ #

        # Cost, 只算交易完的價值
        Cost = pd.Series(data = 0, index = hold_all.index)
        # Cost_risk, 算手上持有股每天收盤時的價值
        Cost_risk = pd.Series(data = 0, index = hold_all.index)
        # loss, 虧損評估, Current/Max_profit
        loss = pd.Series(data = 0.0, index = Cost_risk.index)

        price = None
        pre_Cost = 0
        for i, v in hold_all.iteritems():
            if v:
                if price is None:
                    price = buy_price[i]
                Cost_risk[i] = Close_price[i] - price
            elif price is not None:
                pre_Cost = pre_Cost + (sell_price[i] - price)*Futures_Unit - tax
                price = None
            
            Cost_risk[i] = Cost_risk[i] + pre_Cost 
            Cost[i] = pre_Cost
            pre_Cost = Cost[i]


        Max_profit = 0
        for i, v in Cost_risk.iteritems():
            if v > Max_profit:
                Max_profit = v
            loss[i] = (Principal + v)/(Principal + Max_profit)


        # WeightIndex_profit
        WeightIndex_profit = Close_price.copy()
        WeightIndex_profit = WeightIndex_profit.truncate(start_date, WeightIndex_profit.index[-1], axis = 0)
        WeightIndex_profit = (WeightIndex_profit - WeightIndex_profit[0])*Futures_Unit

        # round
        loss = loss.round(2)
        Cost_risk = Cost_risk.astype(int)
        Cost = Cost.astype(int)
        WeightIndex_profit = WeightIndex_profit.astype(int)

        # Long or shot
        if self._long_short == 'short':
            loss = -loss
            Cost_risk = -Cost_risk
            Cost = -Cost
            WeightIndex_profit = -WeightIndex_profit

        # Report
        self.Report(Cost, Principal, hold_all)

        # Plot
        return self.SubPlotly(WeightIndex_profit, Cost_risk, loss, OHLC_data)

    def Report(self, Cost, Principal, hold_all):
        '''
        交易開始日期: start_date, 交易結束日期: end_date
        獲利: Profit, 最大獲利: Max_Profit, 最小獲利(或虧損): Min_Profit
        報酬率: ROI, 年報酬率: Annualized_ROI
        交易次數: trade_times
        平均持倉時間: Ave_hold_days, 最長持倉時間: Max_hold_days
        最短持倉時間: Min_hold_days
        '''

        # Setttings report
        self.Report_item['start_date'] = Cost.index[0]
        self.Report_item['end_date'] = Cost.index[-1]
        self.Report_item['Profit'] = Cost[-1]
        self.Report_item['Max_Profit'] = Cost.nlargest(1)[0]
        self.Report_item['Min_Profit'] = Cost.nsmallest(1)[0]
        self.Report_item['ROI'] = float(self.Report_item['Profit']/Principal)
        self.Report_item['Annualized_ROI'] = 0
        self.Report_item['trade_times'] = 0
        self.Report_item['Ave_hold_days'] = 0
        self.Report_item['Max_hold_days'] = 0
        self.Report_item['Min_hold_days'] = 10000

        # Calculate Ave/Max/Min_hold_days, trade_times
        trade_start = None
        for i, v in hold_all.iteritems():
            if v:
                if trade_start is None :
                    self.Report_item['trade_times'] += 1
                    trade_start = i
                    
            elif trade_start is not None:
                period = (i - trade_start).days
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

        report_str += "# {:>20} = {:>15}{:>18}\n".format('Profit', self.Report_item['Profit'], "#")
        report_str += "# {:>20} = {:>15}{:>18}\n".format('Max Profit', self.Report_item['Max_Profit'], "#")
        report_str += "# {:>20} = {:>15}{:>18}\n".format('Min Profit', self.Report_item['Min_Profit'], "#")

        report_str += "# {:>20} = {:>14}%{:>18}\n".format('Return On Investment', round(self.Report_item['ROI']*100,2), "#")
        report_str += "# {:>20} = {:>14}%{:>18}\n".format('Annualized ROI', round(self.Report_item['Annualized_ROI']*100,2), "#")

        report_str += "# {:>20} = {:>15}{:>18}\n".format('Ave_hold_days', self.Report_item['Ave_hold_days'], "#")
        report_str += "# {:>20} = {:>15}{:>18}\n".format('Max_hold_days', self.Report_item['Max_hold_days'], "#")
        report_str += "# {:>20} = {:>15}{:>18}\n".format('Min_hold_days', self.Report_item['Min_hold_days'], "#")

        report_str += "#" + " "*56 + "#\n"
        report_str += "# " + "".center(54, "-") + " #\n"

        self._Report_str = report_str
        print(report_str)

    def SubPlotly(self, WeightIndex_profit, Cost_risk, loss, OHLC_data):
    
        # ------------------------------ #
        #    First Fig: Compaere Profi   #
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

        # ------------------------------ #
        #   Second Fig: loss percentage  #
        # ------------------------------ #
        # fig2: loss
        fig2 = []
        fig2.append( dict(
            type = 'scatter',
            y = loss,
            x = loss.index,
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
        fig = tools.make_subplots(rows=2, cols=1,  subplot_titles=('First Subplot','Second Subplot') )
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
        return py.iplot(fig, filename='Results')

    def Iplotly(self, OHLC_data = None, start_date = None, end_date = None, hold = None, return_iplot = True):
        
        # Init variable
        if OHLC_data  is None: OHLC_data = self.DATA['加權指數']
        if start_date is None: start_date = OHLC_data.index[0]
        if end_date   is None: end_date = OHLC_data.index[-1]


        # Index 整合
        OHLC_data = Select_time(OHLC_data, start_date, end_date)
        if hold is None:
            hold_AllDateIndex = self._hold_AllDateIndex.truncate(OHLC_data.index[0], OHLC_data.index[-1], axis = 0)
        elif type(hold) == int:
            hold_AllDateIndex = pd.Series(data = 0, index=OHLC_data.index)
        else:
            hold_AllDateIndex = hold.truncate(OHLC_data.index[0], OHLC_data.index[-1], axis = 0)

        #----------------------------#
        #        plotly setup        #
        #----------------------------#

        # 1. Settings
        fig = dict( data= [dict()] )
        INCREASING_COLOR = '#AA0000' #Red
        DECREASING_COLOR = '#227700' #Green

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
        fig['layout'] = dict(
        shapes = self._Hold_shape_insert([], hold_AllDateIndex)
        )
        if return_iplot:
            return py.iplot(fig, filename='TW_Futures')
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

    # Usage function
    def Print_data_columns(self):
        for x in self.DATA:
            print ("\n# {0}, type = {1}".format(x, type(self.DATA[x])))
            if type(self.DATA[x]) is dict:
                print ("\tKeys: " + str(list(self.DATA[x].keys())) + "\n")
            else:
                print ("\tColumns: " + str(self.DATA[x].columns.values) + "\n")


def Select_time(df, first_day, last_day):
    '''
    Summary line:
        A function used to select a period for dataframe
    Parameters:
        1. df, type: dataframe
        2. first_day, type: string or datetime.datetime
        3. last_day, type: string or datetime.datetime
    Example:
       TWSI = Select_time(TWSI, '2018-01-05', '2018-06-05')
    '''
    first_day = date_convert(first_day)
    last_day  = date_convert(last_day)

    return df.loc[first_day:last_day]

def date_convert(date):
    import datetime
    if type(date) == str:  
        return datetime.datetime.strptime(date, '%Y-%m-%d')
    else:
        return date


if __name__ == '__main__':
    Simulator()
        