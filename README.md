# Stock_Backtesting
股市/期貨 爬蟲和回測系統 
Stocks/Futures crawler & vectorized backtest


# usage
> Command: 初始環境

\> virtualenv futures_venv 

\> . futures_venv/bin/activate

\> pip3 install -r requirements.txt


> 爬蟲main程式: Program/Crawler/main.py

\> cd Program/Crawler
 

> 第一次須先執行: rebuild 來重建data資料夾

\> ./main.py -rebuild
> 之後: update DATA

\> ./main.py -update

> 檢查遺漏DATA和整理DATA: checkm

\> ./main.py -checkm

> Help INFO

\> ./main.py -h 

# Example
> 回測使用範例: Program/Jupyter/



# Update 1.1V_S0206

1. Simulator

> 新增個股回測
> 
> Simulator 優化
> 
> Plotly 新增 Trade_info (Only for stocks Mode)

2.  Component

> 優化 Operator
> 
> 買賣日期自動update (Operation, Run function)
> 
> 新增 talib_Output
> 
> 新增 components_lib
> 

3. Example

>
> 新增 backtest\_Stocks_example1
> 
> # Update 1.2V_S0331

1. Crawler

>  Add Monthly_revenue data
> 
>  Re-write Crawler data structure

2. Simulator

>  Update sim.Iplotly
> 
>  Update structure
>

> # Update 1.3V_S0414

1. Crawler

> Add Financial Statements data: Income & Balance Sheet
