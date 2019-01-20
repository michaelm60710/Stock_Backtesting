# Stock_Backtesting
股市/期貨 回測系統 (未完整)


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
> path: Program/Jupyter/
