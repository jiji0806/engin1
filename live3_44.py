# coding=utf-8
########################################################################################################
# stg6(4) -수정중-------------------6-v2.ccccc 테스트
# 실시간 peak 사용 -최신peak, 역지만 걸고 익절은 없음  1 
#
# find_peak() 사용.
# stg3 사용
#
# 진입
# stg3, stg9 절대 픽 진입.
#
# 물타기 x
#
# 불타기
# stg3, stg9
#
# 시장가 종료, 손절, 반대 진입(x)
#
# 역지 활성
# scale n 이 마지막일 경우 역지 생성
# 역지가(2:1 수익 손실 비율): # max(Stop Lose Limit Percent Range = interval volatility * 70% /2, 0.1)
#
# 장점
# 
# 단점
# 
########################################################################################################
import os.path
import ccxt
import traceback
import numpy as np
from dateutil import parser
from scipy.signal import find_peaks
import json
import time, sys
import math
import datetime as dt
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
import pytz
#from cs50 import SQL
import sqlite3
import boto3
from botocore.config import Config
import datetime
from botocore.exceptions import ClientError
from boto3.dynamodb.conditions import Key
import operator
import subprocess
import traceback
from finta import TA
import itertools
from findpeaks import findpeaks
from datetime import timedelta
import requests, base64

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model  # 수정된 부분: 모델 불러오기
from adtk.detector import MinClusterDetector
from sklearn.cluster import KMeans
from adtk.data import validate_series
from adtk.detector import OutlierDetector
from sklearn.neighbors import LocalOutlierFactor
from adtk.detector import RegressionAD
from sklearn.linear_model import LinearRegression
from adtk.detector import PcaAD
from adtk.detector import QuantileAD

# AnomalyDetection
from scipy import sparse, stats
from sklearn.metrics import f1_score

# 정규화
from sklearn.preprocessing import MinMaxScaler

# 스므딩
from scipy.signal import savgol_filter
import statsmodels.api as sm
#####################################################################################################################################
#from binance.client import Client
api_key = 'r4M6aBy24Ae5oGld7q'
api_secret = 'oYMoflkS6Pb4nvrAwfrm8hPFWLwRWvoLbtWo'
#market_id = 'YFIUSDT'
#client = Client(api_key, api_secret)
#
#plot_timeline_begain = dt.datetime.now().replace(microsecond=0) - dt.timedelta(minutes = 150)
#plot_timeline_end = dt.datetime.now().replace(microsecond=0) + dt.timedelta(minutes = 150)
#plot_price_range_low = float(client.futures_symbol_ticker(market_id = 'YFIUSDT')['price']) - 250
#plot_price_range_high = float(client.futures_symbol_ticker(market_id = 'YFIUSDT')['price']) + 250
#####################################################################################################################################

#boto3 ddb
my_config = Config(
    region_name = 'ap-northeast-1',
    signature_version = 'v4',
    retries = {
        'max_attempts': 10,
        'mode': 'standard'
    }
)

dynamodb_table_name = 'worldengin_configuration'
####################################################################################################################################
####################################################################################################################################
trader_name = 'live3_44'
####################################################################################################################################
####################################################################################################################################
# trader_name = 'binanceusdm_perpetual_1' # 3. delivatives_futures_usdt_perpetual
#trader_name = 'binanceusdm_perpetual_yfiusdt_stg3_macd_single_interval_r0' # 4. delivatives_futures_usdt_perpetual
#trader_name = 'bybit_perpetual_btcusdt_stg3_macd_single_interval_r0' # 5. delivatives_futures_usdt_perpetual
#trader_name = 'bitmex_inverse_perpetual_btcusd_stg3_macd_single_interval_r0' # 6. delivatives_futures_inverse_perpetual_btcusd
#trader_name = 'bitmex_perpetual_btcusd_stg3_macd_single_interval_r0' # 6. delivatives_futures_usdt_perpetual
#trader_name = 'huobi_perpetual_btcusdt_stg3_macd_single_interval_r0' # 8. delivatives_futures_usdt_perpetual
#trader_name = 'okx_perpetual_btcusdt_stg3_macd_single_interval_r0' # 9. delivatives_futures_usdt_perpetual
#trader_name = 'ftx_perpetual_btcusdt_stg3_r0' # 10. delivatives_futures_usdt_perpetual
#trader_name = 'bitfinex_perpetual_btcusdt_stg3_macd_single_interval_r0' # 11. delivatives_futures_usdt_perpetual
#trader_name = 'phemex_perpetual_btcusdt_stg3_macd_single_interval_r0' # 12. delivatives_futures_usdt_perpetual
#trader_name = 'poloniex_perpetual_btcusdt_stg3_macd_single_interval_r0' # 13. delivatives_futures_usdt_perpetual
#trader_name = 'Kraken_perpetual_btcusdt_stg3_macd_single_interval_r0' # 14. delivatives_futures_usdt_perpetual
####################################################################################################################################
def get_configuration(trader_name, dynamodb=None):
    if not dynamodb:
        dynamodb = boto3.resource('dynamodb', config=my_config)
    table = dynamodb.Table(dynamodb_table_name)
    try:
        response = table.get_item(Key={'trader_name': trader_name})
    except ClientError as e:
        print(e.response['Error']['Message'])
    else:
        return response['Item']

pd.set_option('display.max_columns', None)
default_time = pd.Timestamp(2019, 12, 22, 13, 30, 59)
# cpu = get_configuration(trader_name, )
loop = message = max_waiting_in_second = exit_status = ''
loop_counter = u = v = 0 # loop counter
interval = '1m'
# intervals_temp = cpu['info']['itvs']
# intervals = [] # cumulate_lv_calc 에 사용
# intervals = cpu['info']['itvs']
# intervals = ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h']
intervals = ['1m', '5m', '15m', '1h']
# intervals = ['1m', '15m']
valid_intervals = []
# intervals = ['1m', '5m', '15m', '1h']
# intervals = ['1m', '5m', '15m']
limit = 1000
#limit_leverage = int(cpu['info']['limit_leverage']) # 최대 담을수 있는 leverage 제한
#initial_order_amount = float(cpu['info']['initial_order_amount'])
stop_lose_limit_percent_range = 90
# api_key = cpu['info']['api_key']
# api_secret = cpu['info']['api_secret']
#scale_order_max_limit = int(cpu['info']['scale_order_max_limit']) # 최대 3번까지 시도
scale_order_waiting_seconds = 900 # 물타기 주기(약 16.6분)
open_position_stop_market_waiting_seconds = 60000 # 임계치 오버, 마지막 손절 대기시간
entry_order_waiting_seconds = 120 # A-2. entry_order 최장 x second(s) 대기
exit_order_waiting_seconds = 120
#exit_order_waiting_seconds = int(cpu['info']['exit_order_waiting_seconds']) # ※ B-1. exit_order 최장 x second(s) 대기

entry_order_timestamp = 0
exit_order_timestamp = 0
position_entry_time = 0

scale_order_max_limit = 3.9
# min_order_amount = float(cpu['info']['min_order_amount']) # volatility == False

# max_leverage = 15
# lev_limit = 15
#lev_limit = round(max_leverage/5) # 최대 leverage 125배일경우, 25배이상 수량못담게 제한

#exchange_id_list_df_1h = ['bitmex', 'huobi', 'kraken', 'ftx']
#exchange_id_list_df_2h = ['bybit', 'okx', 'binanceusdm', 'bitfinex2', 'phemex', 'poloniex']

exchange_id_list_df_1h = ['bitmex']
exchange_id_list_df_15m = ['bybit', 'huobi', 'kraken', 'okx', 'ftx', 'binanceusdm', 'bitfinex2', 'phemex', 'poloniex']


close=[] #close_price
volume=[] #total quantity
tb_base_av=[] #taker's quantity
quote_av=[] #total quantity in USDT
tb_quote_av=[] #taker's quantity in USDT
sma_7=[]
sma_30=[]
sma_50=[]
sma_100=[]
sma_200=[]
sma_999=[]
ema_7=[]
ema_30=[]
ema_50=[]
ema_100=[]
ema_200=[]
ema_999=[]
bbu=[]
bbm=[]
bbl=[]
bbb=[] #bandwidth
bbp=[] #percentage
rsi_14=[]
macd=[]
macds=[] #signal
macdh=[]
K_9_3=[]
d_9_3=[]
j_9_3=[] #difference between k and d
cumulate_current_time = []
cumulate_current_price =[]
cumulate_lv_pick_max = []
cumulate_lv_pick_min = []
elapsed_times = [0]
inverse_exchanges = []
globals()['volatility_macro_state'] = 0 # volatility confirm
globals()['volatility_micro_interval_pick'] = ''
globals()['volatility_atr_given'] = 0
globals()['trend_macro_state'] = '' # trend confirm
globals()['long_trend_micro_interval_pick'] = ''
globals()['short_trend_micro_interval_pick'] = ''
globals()['long_trend_atr_given'] = 0
globals()['short_trend_atr_given'] = 0
globals()['atr_pick'] = ''
globals()['atr_given'] = 0
scale_order_timestamp = dt.datetime.now().replace(microsecond=0) # 수동으로 포지잡고 봇돌릴시, 물타기 횟수는 1번이상일수 있으나, 시간변수가 존재하지 않을경우도 있으므로 선언해줌
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
db_path = r'/aws/database/t1_transaction.db'
#db_path = r't1_transaction.db'
####################################################################################################################################
#db = SQL('sqlite:///' + db_path)
# conn = sqlite3.connect(db_path)
# db = conn.cursor()

exchange_id = 'bybit'
exchange_class = getattr(ccxt, exchange_id)

if exchange_id in inverse_exchanges:
    inverse_exchange = True
else:
    inverse_exchange = False

if exchange_id == 'okx':
    my_password = 'Mega080^'
    exchange = exchange_class({
        'apiKey': api_key,
        'secret': api_secret,
        'password' : my_password,
        'enableRateLimit': True,
        'options': {
            'adjustForTimeDifference': True,
        },
    })    
elif exchange_id == 'huobi':
    exchange = exchange_class({
        'apiKey': api_key,
        'secret': api_secret,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'swap',
            'adjustForTimeDifference': True,
        },
    })
else:
    exchange = exchange_class({
        'apiKey': api_key,
        'secret': api_secret,
        'enableRateLimit': True,
        'options': {
            #'defaultType': 'future',
            'adjustForTimeDifference': True,
        },
    })

if len(sys.argv) > 1:
    symbol = sys.argv[1]

else:
    symbol = 'BTC/USDT'

balance_currency = 'USDT'
exchange.load_time_difference()
markets = exchange.load_markets()
market_id = exchange.market_id(symbol) # market['id']
market = exchange.market(market_id)
l_s = market['symbol']

if exchange_id == 'binanceusdm':
    min_cost_in_usdt = market['limits']['cost']['min']
    min_order_qty = market['limits']['amount']['min']
    market_amount_precision = float(market['precision']['amount'])
elif exchange_id == 'bybit':
    min_cost_in_usdt = market['info']['lotSizeFilter']['minNotionalValue']
    min_order_qty = market['info']['lotSizeFilter']['minOrderQty']
    market_amount_precision = float(market['precision']['amount'])
    if market_amount_precision < 1:
        decimal_part = str(market_amount_precision).split('.')[1]
        market_amount_precision = len(decimal_part)
    else:
        decimal_part = str(market_amount_precision).split('.')[0]
        market_amount_precision = (len(decimal_part)-1)*-1

if exchange_id == 'huobi':
    if len(exchange.fetch_positions([market_id])) > 0:
        current_leverage = int(exchange.fetch_positions([market_id])[0]['leverage'])
    else:
        current_leverage = 200
    params = {
        'offset': 'both', 'lever_rate': current_leverage
    }
elif exchange_id == 'bybit':
    params = {
        'position_idx': 0,
    }
else:
    params = ''

#print(exchange.symbols)
#if market['linear']:
#    print(market['limits']['amount']['min']) # 최소 주문수량 0.001
#else:
#    print(market['limits']['price']['min']) # 최소 주문수량 0.001

#if (hasattr(exchange, 'timeframes')):
# if exchange.has['fetchOHLCV']:
#     for key in exchange.timeframes.keys():
#         if key in intervals_temp:
#             intervals.append(key)
# else:
#     intervals = intervals_temp

side = {'long': 'buy', 'short': 'sell'}
# strategy = ['stg10']
# strategy = ['stg3', 'stg10', 'stg110']
# strategy = ['stg2']
# strategy = ['stg1', 'stg3', 'stg10', 'stg110']
# strategy = ['stg1', 'stg2', 'stg3', 'stg110']
# strategy = ['stg1', 'stg3', 'stg10']
strategy = ['stg1', 'stg2', 'stg3', 'stg10', 'stg110']
strategy_scalping = ['stg_scalping_0']
directions = ['forward', 'reverse']
current_last = {'current': 'current', 'last': 'last'}
point_frame = {'m': 1, 'h': 60, 'd': 1440, 'w': 10080, 'M': 43800, 'y': 525600}
x_to_seconds_frame = {'m':60, 'h':3600, 'd':86400, 'w':604800}
point_sum = 0
globals()['peaker_frame'] = pd.DataFrame(columns=['peaker_pk','peaked_time','open_time'])
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################

# counter_light_weight_long = 0
# counter_heavy_weight_long = 1
# counter_light_weight_short = 0
# counter_heavy_weight_short = -1



counter_light_weight_long = 1
counter_heavy_weight_long = 2
counter_light_weight_short = -1
counter_heavy_weight_short = -2

#counter_light_weight_long = 0
#counter_heavy_weight_long = 2
#counter_light_weight_short = 0
#counter_heavy_weight_short = -2
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
def base64_decrypt(string):
    return base64.b64decode(string).decode('utf-8').replace('\n','')

def send_to_telegram(*messages):
    message = "\n".join(messages)
    telegram_bot_token = '6521282593:AAEJWTaV6qavLac7Xu9_-iG_neHxm53F8KM'
    telegram_chat_id = '1051301724'
    url = f"https://api.telegram.org/bot{telegram_bot_token}/sendMessage?chat_id={telegram_chat_id}&text={message}"
    response = requests.get(url)

def round_up(n, decimals=0):
    multiplier = 10**decimals
    return math.ceil(n * multiplier) / multiplier

def set_max_leverage(current_leverage, max_leverage):
    if int(current_leverage) != int(max_leverage):
        exchange.set_leverage (leverage = int(max_leverage), symbol = market_id, params = {})
    return

# def klines(market_id, interval, limit):
#     klines = exchange.fetch_ohlcv(symbol=market_id, timeframe=interval, limit=limit)
#     df =pd.DataFrame(klines, columns = ['open_time', 'open', 'high', 'low', 'close', 'volume'])
#     df['close_time'] = df['open_time'].shift(-1)
#     df['open_time'] = pd.to_datetime(df['open_time'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('Asia/Seoul').dt.tz_localize(None)
#     df=df.set_index('open_time')
#     df = df.astype(float)
#     df.at[df.index[-1], 'close_time'] = datetime.datetime.now()  # Set the close_time of the last row to current time
#     df['close_time'] = pd.to_datetime(df['close_time'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('Asia/Seoul').dt.tz_localize(None)
#     print(df)
#     return df

def klines(market_id, interval, limit):
    klines = exchange.fetch_ohlcv(symbol=market_id, timeframe=interval, limit=limit)
    close_time_of_the_last_row = datetime.datetime.now()
    df = pd.DataFrame(klines, columns=['open_time', 'open', 'high', 'low', 'close', 'volume'])
    df['close_time'] = df['open_time'].shift(-1)
    df.at[df.index[-1], 'close_time'] = close_time_of_the_last_row  # Set the close_time of the last row to current time
    df['open_time2'] = df['open_time']
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', errors='coerce')
    df['open_time2'] = pd.to_datetime(df['open_time2'], unit='ms', errors='coerce')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms', errors='coerce')

    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', errors='coerce').astype('datetime64[ms]')
    df['open_time2'] = pd.to_datetime(df['open_time2'], unit='ms', errors='coerce').astype('datetime64[ms]')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms', errors='coerce').astype('datetime64[ms]')

    df['open_time'] = df['open_time'].dt.tz_localize('UTC').dt.tz_convert('Asia/Seoul').dt.tz_localize(None)
    df['open_time2'] = df['open_time2'].dt.tz_localize('UTC').dt.tz_convert('Asia/Seoul').dt.tz_localize(None)
    df['close_time'] = df['close_time'].dt.tz_localize('UTC').dt.tz_convert('Asia/Seoul').dt.tz_localize(None)
    
    df['peak_time'] = df['open_time']

    df['close_high'] = df['close']
    df.at[df.index[-2], 'close_high'] = df['high'].iloc[-1]
    df['close_low'] = df['close']
    df.at[df.index[-2], 'close_low'] = df['low'].iloc[-1]

    df = df.set_index('open_time')
    return df




def interval_cumulate_lv(df, cumulate_lv):
    if 'SMA_7' in df.columns:
        cumulate_lv.append(df.SMA_7.iloc[-1])
    if 'SMA_30' in df.columns:
        cumulate_lv.append(df.SMA_30.iloc[-1])
    if 'SMA_50' in df.columns:
        cumulate_lv.append(df.SMA_50.iloc[-1])
    # if 'SMA_100' in df.columns:
    #     cumulate_lv.append(df.SMA_100.iloc[-1])
    if 'SMA_200' in df.columns:
        cumulate_lv.append(df.SMA_200.iloc[-1])
    if 'EMA_7' in df.columns:
        cumulate_lv.append(df.EMA_7.iloc[-1])
    if 'EMA_30' in df.columns:        
        cumulate_lv.append(df.EMA_30.iloc[-1])
    if 'EMA_50' in df.columns:
        cumulate_lv.append(df.EMA_50.iloc[-1])
    # if 'EMA_100' in df.columns:
    #     cumulate_lv.append(df.EMA_100.iloc[-1])
    if 'EMA_200' in df.columns:
        cumulate_lv.append(df.EMA_200.iloc[-1])
    if 'BBU_21_2.0' in df.columns:
        cumulate_lv.append(df['BBU_21_2.0'].iloc[-1])
    if 'BBL_21_2.0' in df.columns:
        cumulate_lv.append(df['BBL_21_2.0'].iloc[-1])
    if 'PSARl_0.02_0.2' in df.columns:
        cumulate_lv.append(df['PSARl_0.02_0.2'].iloc[-1])
    if 'PSARs_0.02_0.2' in df.columns:
        cumulate_lv.append(df['PSARs_0.02_0.2'].iloc[-1])
        
    cumulate_lv.append(df['high'].max() * (1 - 0.004))
    cumulate_lv.append(df['low'].min() * (1 + 0.004))

    return cumulate_lv

def interval_previous_price_frequency_calc(market_id, interval):
    interval_previous_price_frequency = []
    df=klines(market_id=market_id, interval=interval, limit=2)
    if df['close'].count() > 1 :
        low = df['low'][-2]
        high = df['high'][-2]
    else:
        low = df['low'][-1]
        high = df['high'][-1]
    if (high - low) == 0 :
        previous_price_frequency = 0
    else:
        previous_price_frequency = (high/low - 1)*100
    interval_previous_price_frequency.append(previous_price_frequency)
    return interval_previous_price_frequency

def intervals_previous_price_frequency_calc(intervals):
    intervals_previous_price_frequency = []
    for interval in intervals:
        df=klines(market_id=market_id, interval=interval, limit=2)
        if df['close'].count() > 1 :
            low = df['low'][-2]
            high = df['high'][-2]
        else:
            low = df['low'][-1]
            high = df['high'][-1]
        if (high - low) == 0 :
            previous_price_frequency = 0
        else:
            previous_price_frequency = (high/low - 1)*100
        intervals_previous_price_frequency.append(previous_price_frequency)
    return intervals_previous_price_frequency

def intervals_current_price_frequency_calc():
    intervals_current_price_frequency = []
    for interval in intervals:
        df=klines(market_id=market_id, interval=interval, limit=2)
        low = df['low'][-1]
        high = df['high'][-1]
        if (high - low) == 0 :
            current_price_frequency = 0
        else:
            current_price_frequency = (high/low - 1)*100
        intervals_current_price_frequency.append(current_price_frequency)
    return intervals_current_price_frequency

def intervals_price_frequency_diff_calc():
    intervals_previous_price_frequency = []
    intervals_current_price_frequency = []
    for interval in intervals:
        df=klines(market_id=market_id, interval=interval, limit=2)
        low = df['low'][-2]
        high = df['high'][-2]
        if (high - low) == 0 :
            previous_price_frequency = 0
        else:
            previous_price_frequency = (high/low - 1)*100
        intervals_previous_price_frequency.append(previous_price_frequency)
    for interval in intervals:
        df=klines(market_id=market_id, interval=interval, limit=2)
        low = df['low'][-1]
        high = df['high'][-1]
        if (high - low) == 0 :
            current_price_frequency = 0
        else:
            current_price_frequency = (high/low - 1)*100
        intervals_current_price_frequency.append(current_price_frequency)
    intervals_previous_price_frequency_np = np.array(intervals_previous_price_frequency)
    intervals_current_price_frequency_np = np.array(intervals_current_price_frequency)
    intervals_price_frequency_diff = intervals_previous_price_frequency_np - intervals_current_price_frequency_np
    intervals_price_frequency_diff = intervals_price_frequency_diff.tolist()
    return intervals_price_frequency_diff

def greedy_percentage_calc(interval_peak):
    if interval_peak == '':
        interval_peak = '15m'
    previous_price_frequency = interval_previous_price_frequency_calc(market_id, interval_peak)[0]
    previous_price_frequency_safe_value = previous_price_frequency*70/100
    if previous_price_frequency_safe_value > (3*taker_fee): # fee 0.08 + 0.08 = 0.16
        greedy_percentage = previous_price_frequency_safe_value
    else:
        greedy_percentage = 3*taker_fee
    return previous_price_frequency, greedy_percentage, interval_peak

def volume_frequency_calc():
    intervals_volume_frequency = []
    for interval in intervals:
        df=klines(market_id=market_id, interval=interval, limit=2)
        previous_volume = df['volume'][-2]
        current_volume = df['volume'][-1]
        volume_frequency = previous_volume - current_volume
        intervals_volume_frequency.append(volume_frequency)
    return intervals_volume_frequency

def balance_calc(exchange_id, balance_currency):
    if (exchange_id == 'binanceusdm') or (exchange_id == 'ftx'):
        # wallet_balance = exchange.fetch_balance()[balance_currency]['total'] #binance, bitmex
        wallet_balance = float(exchange.fetch_balance()['info']['totalMarginBalance'])
    elif exchange_id == 'bybit':
        wallet_balance = exchange.fetch_balance()['total'][balance_currency] #bybit
    elif exchange_id == 'bitmex':
        wallet_balance = exchange.fetch_balance()[balance_currency]['total'] #binance, bitmex
    elif exchange_id == 'huobi':
        wallet_balance = float(exchange.fetch_balance()['info']['data'][0]['margin_balance']) # huobi
    return wallet_balance

# def balance_calc(exchange_id, balance_currency):
#     if exchange_id == 'binanceusdm':
#         wallet_balance = float(exchange.fetch_balance()['info']['totalWalletBalance']) #binance, bitmex        
#     elif exchange_id == 'ftx':
#         wallet_balance = float(exchange.fetch_balance()[balance_currency]['total']) #binance, bitmex        
#     elif exchange_id == 'bybit':
#         wallet_balance = float(exchange.fetch_balance()['total'][balance_currency]) #bybit        
#     elif exchange_id == 'bitmex':
#         wallet_balance = float(exchange.fetch_balance()[balance_currency]['total']) #binance, bitmex
#     elif exchange_id == 'huobi':
#         wallet_balance = float(exchange.fetch_balance()['info']['data'][0]['margin_balance']) # huobi
#     return wallet_balance

def position_calc(exchange_id, market_id):
    position_size = 0
    position_side = ''
    position_value = 0
    position_entry_price = ''
    liquidation_price = ''
    unrealised_pnl = 0
    roe_pcnt = 0
    position_entry_time = 0
    if (exchange_id == 'binanceusdm') or (exchange_id == 'bybit') or (exchange_id == 'ftx') or (exchange_id == 'bitmex'):
        position_checker_ = exchange.fetch_positions([market_id])
        if position_checker_:
            position_checker = position_checker_[0]
            position_size = position_checker["contracts"] # always positive number
            if position_size != 0:
                position_side = position_checker["side"]
                position_value = float(position_checker["notional"])
                position_entry_price = position_checker["entryPrice"]
                position_entry_time = position_checker["timestamp"]/1000
                liquidation_price = position_checker["liquidationPrice"]
                unrealised_pnl = position_checker["unrealizedPnl"]
                roe_pcnt = position_checker["percentage"]
                if (position_side == 'short') and position_value > 0:
                    position_value = position_value * -1
                if exchange_id == 'bitmex':
                    if (position_size == 0) or (position_size == None):
                        position_size = 0
                        position_side = ''
                        position_value = 0
                        position_entry_price = ''
                        liquidation_price = ''
                    if position_size > 0:
                        position_side = 'long'
                    elif position_size < 0:
                        position_size = position_size * -1
                        position_side = 'short'






    #elif exchange_id == 'bybit':
    #    position_checker = exchange.fetch_positions([market_id])[0]
    #    position_size = float(position_checker["size"])
    #    if position_size != 0:
    #        if position_checker["side"] == 'Buy':
    #            position_side = 'long'
    #        if position_checker["side"] == 'Sell':
    #            position_side = 'short'        
    #        position_value = float(position_checker["position_value"])
    #        position_entry_price = float(position_checker["entry_price"])
    #        liquidation_price = float(position_checker["liq_price"])
    #        unrealised_pnl = float(position_checker["unrealised_pnl"])
    #        roe_pcnt = unrealised_pnl / (position_entry_price / float(position_checker["leverage"]) * position_size) * 100

    elif (exchange_id == 'huobi'):
        if len(exchange.fetch_positions([market_id])) > 0:
            position_checker = exchange.fetch_positions([market_id])[0]
            position_size = float(position_checker["contractSize"] * position_checker["contracts"])
            if position_size != 0:
                position_side = position_checker["side"]
                position_value = position_checker["notional"]
                position_entry_price = position_checker["entryPrice"]
                liquidation_price = position_checker["liquidationPrice"] # none
                unrealised_pnl = position_checker["unrealizedProfit"]
                roe_pcnt = position_checker["percentage"]
        else:
            position_size = 0
            position_side = ''
            position_value = 0
            position_entry_price = ''
            liquidation_price = ''
            unrealised_pnl = 0
            roe_pcnt = 0

    #elif exchange_id == 'bitmex':
    #    position_checker = exchange.fetch_positions([market_id])[0]
    #    position_size = float(position_checker['info']['currentQty'])
    #    if position_size != 0:
    #        if position_size > 0:
    #            position_side = 'long'
    #            position_value = float(position_checker['info']['homeNotional'])
    #        elif position_size < 0:
    #            position_size = position_size*-1
    #            position_side = 'short'
    #            position_value = float(position_checker['info']['homeNotional'])*-1
    #        position_entry_price = float(position_checker['info']['avgEntryPrice'])
    #        liquidation_price= float(position_checker['info']['liquidationPrice'])
    #        unrealised_pnl= float(position_checker['info']['unrealisedPnl'])/100000000
    #        roe_pcnt= float(position_checker['info']['unrealisedRoePcnt'])*100


    if position_value is None:
        position_value = 0
    elif (isinstance(position_value, int)) or (isinstance(position_value, float)):
        position_value = position_value
    if position_entry_price is None:
        position_entry_price = '-'
    elif (isinstance(position_entry_price, int)) or (isinstance(position_entry_price, float)):
        position_entry_price = position_entry_price
    if liquidation_price is None:
        liquidation_price = '-'
    elif (isinstance(liquidation_price, int)) or (isinstance(liquidation_price, float)):
        liquidation_price = liquidation_price
    if unrealised_pnl is None:
        unrealised_pnl = 0
    elif (isinstance(unrealised_pnl, int)) or (isinstance(unrealised_pnl, float)):
        unrealised_pnl = unrealised_pnl
    if roe_pcnt is None:
        roe_pcnt = 0
    elif (isinstance(roe_pcnt, int)) or (isinstance(roe_pcnt, float)):
        roe_pcnt = roe_pcnt



    return position_side, position_size, position_value, position_entry_price, position_entry_time, liquidation_price, unrealised_pnl, roe_pcnt

def open_order_calc(exchange_id, market_id):
    open_order_side = ''
    open_order_size = ''
    open_order_price = ''
    open_order_type = ''
    stop_market_counter = 0
    open_order_counter = 0
    open_order_checker = exchange.fetch_open_orders(market_id)
    total_open_order_counter = len(open_order_checker)
    if total_open_order_counter > 0:
        for i in open_order_checker:
            if ((i['type'] == 'limit' ) and (i['stopPrice'] == None )):
                open_order_counter += 1
                if (exchange_id == 'binanceusdm') or (exchange_id == 'ftx'):
                    open_order_side = i['side']
                    open_order_size = i['amount']  # always positive number
                    open_order_price = i['price']
                    open_order_type = i['type'] 
                elif exchange_id == 'bybit':
                    open_order_side = i['side']
                    open_order_size = i['amount']
                    open_order_price = i['price']
                    open_order_type = i['type']
                elif exchange_id == 'huobi':
                    open_order_side = i['side']
                    open_order_size = (i['amount']/1000)
                    open_order_price = i['price']
                    open_order_type = i['type']   
                elif exchange_id == 'bitmex':
                    open_order_side = i['side']
                    open_order_size = i['amount']
                    open_order_price = i['price']
                    open_order_type = i['type']

        for i in open_order_checker:
                if ((i['type'] == 'stop_market' ) or ((i['type'] == 'limit' ) and (i['stopPrice'] != None ))):
                    stop_market_counter += 1
    return open_order_counter, open_order_side, open_order_size, open_order_price, open_order_type, stop_market_counter

def ticker_calc(market_id):
    if (exchange_id == 'ftx'):
        symbol_ticker = exchange.fetch_ticker(market_id)
        symbol_ticker_timestamp = 0
        symbol_ticker_last = symbol_ticker['last']
        last_24hr_volatility = 0
    else:
        symbol_ticker = exchange.fetch_ticker(market_id)
        symbol_ticker_timestamp = symbol_ticker['timestamp']
        symbol_ticker_last = symbol_ticker['last']
        last_24hr_volatility = symbol_ticker['change']
    return symbol_ticker_timestamp, symbol_ticker_last, last_24hr_volatility

def stric_exit_price_calc(position_side, position_entry_price, scalping_direction_pick):
    stric_exit_price = 0
    # interval_peak = globals()['atr_pick']

    # if globals()['volatility_macro_state'] == 1:
    #     interval_peak = globals()['volatility_micro_interval_pick']
    # else:
    #     interval_peak = cpu['info']['itv']

    if scalping_direction_pick == 'neutral':
        # interval_peak = '5m'

        if 'df_5m' in globals():
            interval_peak = '5m'
        else:
            interval_peak = '1m'

    elif position_side == 'long' :
        interval_peak = globals()['long_trend_micro_interval_pick']
    elif position_side == 'short':
        interval_peak = globals()['short_trend_micro_interval_pick']
    else:
        interval_peak = ''

    if interval_peak == '':
        # interval_peak = '5m'

        if 'df_5m' in globals():
            interval_peak = '5m'
        else:
            interval_peak = '1m'

    ########### 주석처리 필요 ############
    # interval_peak = '15m'
     ########### 주석처리 필요 ############

    greedy_percentage_calc_values = greedy_percentage_calc(interval_peak)
    previous_price_frequency = greedy_percentage_calc_values[0]
    greedy_percentage = greedy_percentage_calc_values[1]
    picked_interval = greedy_percentage_calc_values[2]
    if position_side == 'long' :
        stric_exit_price = position_entry_price*(1 + (taker_fee + greedy_percentage) / 100)
        if stric_exit_price != 0:
            stric_exit_price = float(exchange.price_to_precision(market_id, stric_exit_price))
    elif position_side == 'short':
        stric_exit_price = position_entry_price*(1 - (taker_fee + greedy_percentage) / 100)
        if stric_exit_price != 0:
            stric_exit_price = float(exchange.price_to_precision(market_id, stric_exit_price))
    return stric_exit_price, previous_price_frequency, greedy_percentage, picked_interval

# def stric_exit_price_calc_min(position_side, position_entry_price):
#     stric_exit_price = 0
#     greedy_percentage_calc_values = greedy_percentage_calc(cpu['info']['itv'])
#     greedy_percentage = greedy_percentage_calc_values[1]
#     if position_side == 'long' :
#         stric_exit_price = position_entry_price*(1 + (taker_fee + greedy_percentage) / 100)
#         stric_exit_price = float(exchange.price_to_precision(market_id, stric_exit_price))
#     elif position_side == 'short':
#         stric_exit_price = position_entry_price*(1 - (taker_fee + greedy_percentage) / 100)
#         stric_exit_price = float(exchange.price_to_precision(market_id, stric_exit_price))
#     return stric_exit_price

def stric_exit_price_calc_min(position_side, position_entry_price): #수수료만 뺀 본전가 산출
    stric_exit_price = 0
    if position_side == 'long' :
        stric_exit_price = position_entry_price*(1 + ((3*taker_fee) / 100))
        if position_entry_price != 0:
            stric_exit_price = float(exchange.price_to_precision(market_id, stric_exit_price))
    elif position_side == 'short':
        stric_exit_price = position_entry_price*(1 - ((3*taker_fee) / 100))
        if position_entry_price != 0:        
            stric_exit_price = float(exchange.price_to_precision(market_id, stric_exit_price))
    return stric_exit_price

def scale_order_position_amount_calc(min_order_amount, wallet_balance, max_leverage, position_size, r, scale_order_max_limit):
    symbol_ticker_timestamp, symbol_ticker_last, last_24hr_volatility = ticker_calc(market_id)

    ########## calc
    # scale_order_position_amount 계사순서
    # 현재시점에서 최대 담을수 있는 수량 -> 최초포지션 수량 -> 물타기수량 산출
    # 현재 수량 -> next_scaled_level_n 산출
    max_available_position_size = wallet_balance * max_leverage / symbol_ticker_last
    if inverse_exchange:
        max_available_position_size = wallet_balance * max_leverage * symbol_ticker_last

    if (r == 0) or (r == 1): # 등차수열 Arithmetical Series
        initial_order_amount = max_available_position_size / scale_order_max_limit # 최소주문수량, unit?, percesion 수수점자릿수 확인필요
        scale_order_position_amount = initial_order_amount # 최소주문수량, unit?, percesion 수수점자릿수 확인필요
        if position_size == 0:
            next_scaled_level_n = 0
        else:
            next_scaled_level_n = position_size/initial_order_amount
    else : # 등비수열 Geometric Series
        initial_order_amount = (max_available_position_size*(r-1)   ) / (pow(r, (scale_order_max_limit + 1)) -1) # max_available_position_size / r ^(scale_order_max_limit-1)
        if (position_size == 0): # or (position_size == initial_order_amount):
            next_scaled_level_n = 0
        else:
            next_scaled_level_n = math.log(((((position_size*(r-1))/initial_order_amount) + 1)), r)
            #next_scaled_level_n = math.log(((1 - ((position_size*(1 - r))/initial_order_amount))), r)
        scale_order_position_amount = initial_order_amount * (pow(r, (next_scaled_level_n)))
    ################################################################################

    ###################### 변동성있을때 수량 최소화 #####################
    #if volatility:
    #    initial_order_amount = initial_order_amount/3
    #    scale_order_position_amount = scale_order_position_amount/5
    ###################################################################
    ##################### 불타기 수량 최소화 #####################
    bul_scale_order_position_amount = scale_order_position_amount/3

    if initial_order_amount < min_order_amount:
        initial_order_amount = min_order_amount
    if scale_order_position_amount < min_order_amount:
        scale_order_position_amount = min_order_amount
    if bul_scale_order_position_amount < min_order_amount:
        bul_scale_order_position_amount = min_order_amount
    if float(next_scaled_level_n) != 0:
        scaled_level_n = round((float(next_scaled_level_n) - 1), 2)
    else:
        scaled_level_n = round((float(next_scaled_level_n)), 2)
    if exchange_id == 'huobi':
        initial_order_amount = float(exchange.amount_to_precision(market_id, initial_order_amount*1000))
        scale_order_position_amount = float(exchange.amount_to_precision(market_id, scale_order_position_amount*1000))
        bul_scale_order_position_amount = float(exchange.amount_to_precision(market_id, bul_scale_order_position_amount*1000))
    else:
        initial_order_amount = float(exchange.amount_to_precision(market_id, initial_order_amount))
        scale_order_position_amount = float(exchange.amount_to_precision(market_id, scale_order_position_amount))
        bul_scale_order_position_amount = float(exchange.amount_to_precision(market_id, bul_scale_order_position_amount))

    return(max_available_position_size, initial_order_amount, scale_order_position_amount, scaled_level_n, bul_scale_order_position_amount)

def calculate_position_size(pa, win_rate, loss_rate, reverse_confirmer, total_played, total_wins, total_losses):
    if win_rate > loss_rate:
        # Increase position amount by 30% if win
        position_size_cus = pa * 1
    elif loss_rate > win_rate:
        # Decrease position amount by 30% if loss
        position_size_cus = pa * 1
    else:
        # Keep position amount unchanged
        position_size_cus = pa
    
    max_available_position_size = wallet_balance_fix * max_leverage / symbol_ticker_last
    position_size_cus_ = min(position_size_cus, max_available_position_size)

    if position_size_cus_ < min_order_amount:
        # position_size_cus_ = min_order_amount
        # return None
        reverse_confirmer = 1 - reverse_confirmer
        total_played = 0
        total_wins = 0
        total_losses = 0
        win_rate = 0
        loss_rate = 0
        position_size_cus_ = initial_order_amount

    position_size_cus_ = float(exchange.amount_to_precision(market_id, position_size_cus_))

    return position_size_cus_, win_rate, loss_rate, reverse_confirmer, total_played, total_wins, total_losses


def check_and_reverse(peaker_side, stg_type, peaker_option, reverse_confirmer):
    # if reverse_confirmer == 1 and stg_type in ['stg1'] and peaker_option == 'forward':
    #     # Switch peaker_side
    #     if peaker_side == 'long':
    #         peaker_side = 'short'
    #     elif peaker_side == 'short':
    #         peaker_side = 'long'
    return peaker_side

def exit_order_position_amount_calc(position_size):
    if position_size == float(0):
        exit_order_position_amount = 0
    else:
        exit_order_position_amount = float(exchange.amount_to_precision(market_id, position_size))
        if exchange_id == 'huobi':
            exit_order_position_amount = float(exchange.amount_to_precision(market_id, (position_size * 1000)))
    return exit_order_position_amount

def stop_loss_price_and_pnl_calc(wallet_balance, position_size, position_side, position_entry_price, stop_lose_limit_percent_range):
    stop_loss_price = 0
    stop_loss_pnl = 0
    if position_size != 0:
        if position_side == 'long':
            if inverse_exchange:
                stop_loss_price = 1/((1/position_entry_price) + ((wallet_balance * stop_lose_limit_percent_range/100)/position_size))
            else:
                stop_loss_price = position_entry_price - ((wallet_balance * stop_lose_limit_percent_range) / (position_size*100))
            stop_loss_price = float(exchange.price_to_precision(market_id, stop_loss_price))
        elif position_side == 'short':
            if inverse_exchange:
                stop_loss_price = 1/((1/position_entry_price) - ((wallet_balance * stop_lose_limit_percent_range/100)/position_size))
            else:
                stop_loss_price = position_entry_price + ((wallet_balance * stop_lose_limit_percent_range) / (position_size*100))
            stop_loss_price = float(exchange.price_to_precision(market_id, stop_loss_price))
        stop_loss_pnl = wallet_balance * stop_lose_limit_percent_range * -1 / 100
        #stop_loss_pnl = float(exchange.price_to_precision(market_id, stop_loss_pnl))
    return stop_loss_price, stop_loss_pnl

def cumulate_lv_calc(market_id, intervals):
    cumulate_lv =[]
    for interval in intervals:
        df=klines(market_id=market_id, interval=interval, limit=limit)
        CustomStrategy = ta.Strategy( # Create your own Custom Strategy
            name="Momo and Volatility",
            description="SMA 22, 50, 100, 200, 999 and EMA 22, 50, 100, 200, 999 BBANDS, RSI, MACD",
            ta=[
                {"kind": "sma", "length": 7},
                {"kind": "sma", "length": 30},
                {"kind": "sma", "length": 50},
                # {"kind": "sma", "length": 100},
                {"kind": "sma", "length": 200},
                {"kind": "ema", "length": 7},
                {"kind": "ema", "length": 30},
                {"kind": "ema", "length": 50},
                # {"kind": "ema", "length": 100},
                {"kind": "ema", "length": 200},
                {"kind": "bbands", "length": 21},
                {"kind": "psar"}
            ]
        )
        df.ta.strategy(CustomStrategy) # To run your "Custom Strategy"
        interval_cumulate_lv(df, cumulate_lv)
    return cumulate_lv

def pick_calc(cumulate_lv, position_side, position_size, position_entry_price, symbol_ticker_last, scalping_direction_pick):
    stric_exit_price_values = stric_exit_price_calc(position_side, position_entry_price, scalping_direction_pick)
    stric_exit_price = stric_exit_price_values[0]
    previous_price_frequency = stric_exit_price_values[1]
    greedy_percentage = stric_exit_price_values[2]
    picked_interval = stric_exit_price_values[3]
    # if position_side == 'long' :
    #     atr_exit_price = (position_entry_price*(1 + (2*taker_fee) / 100)) + (atr_given * 5)
    # elif position_side == 'short':
    #     atr_exit_price = (position_entry_price*(1 - (2*taker_fee) / 100)) - (atr_given * 5)
    if position_size == 0:
        pick_min = max([i for i in cumulate_lv if symbol_ticker_last>i], default=symbol_ticker_last) # current_position 이 short 일때 사용
        pick_max = min([i for i in cumulate_lv if symbol_ticker_last<i], default=symbol_ticker_last) # current_position 이 long 일때 사용
        # pick_min = min([i for i in cumulate_lv if symbol_ticker_last>i], default=symbol_ticker_last) # current_position 이 short 일때 사용
        # pick_max = max([i for i in cumulate_lv if symbol_ticker_last<i], default=symbol_ticker_last) # current_position 이 long 일때 사용        
    else:
        pick_min = max([i for i in cumulate_lv if stric_exit_price>=i], default=stric_exit_price) # current_position 이 short 일때 사용
        pick_max = min([i for i in cumulate_lv if stric_exit_price<=i], default=stric_exit_price) # current_position 이 long 일때 사용
        # pick_min = min([i for i in cumulate_lv if stric_exit_price>=i], default=stric_exit_price) # current_position 이 short 일때 사용
        # pick_max = max([i for i in cumulate_lv if stric_exit_price<=i], default=stric_exit_price) # current_position 이 long 일때 사용        
        # if ((globals()['atr_pick']) and (globals()['atr_pick'] in ['1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w'])):
        #    pick_min = max([i for i in cumulate_lv if atr_exit_price>=i], default=pick_min) # current_position 이 short 일때 사용
        #    pick_max = min([i for i in cumulate_lv if atr_exit_price<=i], default=pick_max) # current_position 이 long 일때 사용
    pick_min = float(exchange.price_to_precision(market_id, pick_min)) # current_position 이 short 일때 사용
    pick_max = float(exchange.price_to_precision(market_id, pick_max)) # current_position 이 long 일때 사용
    return(pick_max, pick_min, previous_price_frequency, greedy_percentage, picked_interval)

def ta_data_frame_calc(market_id, interval):
    df=klines(market_id=market_id, interval=interval, limit=limit)
    CustomStrategy = ta.Strategy( # Create your own Custom Strategy
        name="Momo and Volatility",
        description="SMA 22, 50, 200, EMA 10, 21, 100, BBANDS, RSI, MACD",
        ta=[
            {"kind": "sma", "length": 22},
            {"kind": "sma", "length": 50},
            # {"kind": "sma", "length": 200},
            {"kind": "ema", "length": 10},
            {"kind": "ema", "length": 21},
            # {"kind": "ema", "length": 100},
            {"kind": "bbands", "length": 20},
            {"kind": "rsi"},
            {"kind": "macd"},
            {"kind": "kdj"},
        ]
    )
    df.ta.strategy(CustomStrategy) # To run your "Custom Strategy"
    return df

def maxima_minima_calc(high_1, high_2, rsi_prominence, MACD_12_26_9_prominence, MACD_50_75_35_prominence, dx_prominence, adx_prominence, atr_prominence, obv_prominence, combined_diff_prominence, x, y_open, y_high, y_low, y_close, y_rsi, y_MACD_12_26_9, y_MACD_50_75_35, y_adx, y_dx, y_dmp, y_dmn, y_kdj, y_obv, y_atr, y_atr14, y_atr_p, y_combined_diff, y_combined_diff_filtered, y_second_combined_diff, y_second_combined_diff_filtered):

    distance = 1
    maxima_val_open = find_peaks(y_open, height = high_1, prominence=atr_prominence) #close
    maxima_peak_y_open = maxima_val_open[1]['peak_heights'] #list of the heights of the peaks
    maxima_peak_x_open = x[maxima_val_open[0]] #list of the peaks positions
    y2_open = y_open*-1
    minima_val_open = find_peaks(y2_open, height = high_2, prominence=atr_prominence) #close
    minima_peak_y_open = y2_open[minima_val_open[0]]*-1 #list of the mirrored minima heights
    minima_peak_x_open = x[minima_val_open[0]] #list of the minima positions    

    maxima_val_high = find_peaks(y_high, height = high_1, prominence=atr_prominence) #close
    maxima_peak_y_high = maxima_val_high[1]['peak_heights'] #list of the heights of the peaks
    maxima_peak_x_high = x[maxima_val_high[0]] #list of the peaks positions
    y2_high = y_high*-1
    minima_val_high = find_peaks(y2_high, height = high_2, prominence=atr_prominence) #close
    minima_peak_y_high = y2_high[minima_val_high[0]]*-1 #list of the mirrored minima heights
    minima_peak_x_high = x[minima_val_high[0]] #list of the minima positions    

    maxima_val_low = find_peaks(y_low, height = high_1, prominence=atr_prominence) #close
    maxima_peak_y_low = maxima_val_low[1]['peak_heights'] #list of the heights of the peaks
    maxima_peak_x_low = x[maxima_val_low[0]] #list of the peaks positions
    y2_low = y_low*-1
    minima_val_low = find_peaks(y2_low, height = high_2, prominence=atr_prominence) #close
    minima_peak_y_low = y2_low[minima_val_low[0]]*-1 #list of the mirrored minima heights
    minima_peak_x_low = x[minima_val_low[0]] #list of the minima positions    

    #maxima_peak_close
    #maxima_val_close = find_peaks(y_high, height = 0, prominence=(y_high.max() * 0.003), distance = 30) #close
    maxima_val_close = find_peaks(y_close, height = high_1, prominence=atr_prominence) #close
    maxima_peak_y_close = maxima_val_close[1]['peak_heights'] #list of the heights of the peaks
    maxima_peak_x_close = x[maxima_val_close[0]] #list of the peaks positions
    #minim_peak_close
    y2_close = y_close*-1
    #minima_val_close = find_peaks(y2_close, height = -100000000000000, prominence=(y_high.max() * 0.003), distance = 30) #close
    minima_val_close = find_peaks(y2_close, height = high_2, prominence=atr_prominence) #close
    minima_peak_y_close = y2_close[minima_val_close[0]]*-1 #list of the mirrored minima heights
    minima_peak_x_close = x[minima_val_close[0]] #list of the minima positions

    #maxima_peak_rsi
    maxima_val_rsi = find_peaks(y_rsi, height = high_1, prominence=rsi_prominence) #rsi
    maxima_peak_y_rsi = maxima_val_rsi[1]['peak_heights'] #list of the heights of the peaks
    maxima_peak_x_rsi = x[maxima_val_rsi[0]] #list of the peaks positions
    #minim_peak_rsi
    y2_rsi = y_rsi*-1
    minima_val_rsi = find_peaks(y2_rsi, height = high_2, prominence=rsi_prominence) #rsi, -30
    minima_peak_y_rsi = y2_rsi[minima_val_rsi[0]]*-1 #list of the mirrored minima heights
    minima_peak_x_rsi = x[minima_val_rsi[0]] #list of the minima positions

    #maxima_peak_MACD_12_26_9
    maxima_val_MACD_12_26_9 = find_peaks(y_MACD_12_26_9, height = high_2, prominence=MACD_12_26_9_prominence) #MACD_12_26_9, 0
    maxima_peak_y_MACD_12_26_9 = maxima_val_MACD_12_26_9[1]['peak_heights'] #list of the heights of the peaks
    maxima_peak_x_MACD_12_26_9 = x[maxima_val_MACD_12_26_9[0]] #list of the peaks positions
    #minim_peak_MACD_12_26_9
    y2_MACD_12_26_9 = y_MACD_12_26_9*-1
    minima_val_MACD_12_26_9 = find_peaks(y2_MACD_12_26_9, height = high_2, prominence=MACD_12_26_9_prominence) #MACD_12_26_9, 0
    minima_peak_y_MACD_12_26_9 = y2_MACD_12_26_9[minima_val_MACD_12_26_9[0]]*-1 #list of the mirrored minima heights
    minima_peak_x_MACD_12_26_9 = x[minima_val_MACD_12_26_9[0]] #list of the minima positions

    #maxima_peak_MACD_50_75_35
    maxima_val_MACD_50_75_35 = find_peaks(y_MACD_50_75_35, height = high_2, prominence=MACD_50_75_35_prominence) #MACD_50_75_35, 0
    maxima_peak_y_MACD_50_75_35 = maxima_val_MACD_50_75_35[1]['peak_heights'] #list of the heights of the peaks
    maxima_peak_x_MACD_50_75_35 = x[maxima_val_MACD_50_75_35[0]] #list of the peaks positions
    #minim_peak_MACD_50_75_35
    y2_MACD_50_75_35 = y_MACD_50_75_35*-1
    minima_val_MACD_50_75_35 = find_peaks(y2_MACD_50_75_35, height = high_2, prominence=MACD_50_75_35_prominence) #MACD_50_75_35, 0
    minima_peak_y_MACD_50_75_35 = y2_MACD_50_75_35[minima_val_MACD_50_75_35[0]]*-1 #list of the mirrored minima heights
    minima_peak_x_MACD_50_75_35 = x[minima_val_MACD_50_75_35[0]] #list of the minima positions

    #maxima_peak_adx
    maxima_val_adx = find_peaks(y_adx, height = high_1, prominence=adx_prominence) #adx, 30
    maxima_peak_y_adx = maxima_val_adx[1]['peak_heights'] #list of the heights of the peaks
    maxima_peak_x_adx = x[maxima_val_adx[0]] #list of the peaks positions
    #minim_peak_adx
    y2_adx = y_adx*-1
    minima_val_adx = find_peaks(y2_adx, height = high_2, prominence=adx_prominence) #adx, -30
    minima_peak_y_adx = y2_adx[minima_val_adx[0]]*-1 #list of the mirrored minima heights
    minima_peak_x_adx = x[minima_val_adx[0]] #list of the minima positions

    #maxima_peak_dx
    maxima_val_dx = find_peaks(y_dx, height = high_1, prominence=dx_prominence) #dx, 30
    maxima_peak_y_dx = maxima_val_dx[1]['peak_heights'] #list of the heights of the peaks
    maxima_peak_x_dx = x[maxima_val_dx[0]] #list of the peaks positions
    #minim_peak_dx
    y2_dx = y_dx*-1
    minima_val_dx = find_peaks(y2_dx, height = high_2, prominence=dx_prominence) #dx, -30
    minima_peak_y_dx = y2_dx[minima_val_dx[0]]*-1 #list of the mirrored minima heights
    minima_peak_x_dx = x[minima_val_dx[0]] #list of the minima positions

    #maxima_peak_dmp
    maxima_val_dmp = find_peaks(y_dmp, height = high_1) #dmp
    maxima_peak_y_dmp = maxima_val_dmp[1]['peak_heights'] #list of the heights of the peaks
    maxima_peak_x_dmp = x[maxima_val_dmp[0]] #list of the peaks positions
    #minim_peak_dmp
    y2_dmp = y_dmp*-1
    minima_val_dmp = find_peaks(y2_dmp, height = high_2) #dmp, -10
    minima_peak_y_dmp = y2_dmp[minima_val_dmp[0]]*-1 #list of the mirrored minima heights
    minima_peak_x_dmp = x[minima_val_dmp[0]] #list of the minima positions

    #maxima_peak_dmn
    maxima_val_dmn = find_peaks(y_dmn, height = high_1) #dmn
    maxima_peak_y_dmn = maxima_val_dmn[1]['peak_heights'] #list of the heights of the peaks
    maxima_peak_x_dmn = x[maxima_val_dmn[0]] #list of the peaks positions
    #minim_peak_dmn
    y2_dmn = y_dmn*-1
    minima_val_dmn = find_peaks(y2_dmn, height = high_2) #dmn, -10
    minima_peak_y_dmn = y2_dmn[minima_val_dmn[0]]*-1 #list of the mirrored minima heights
    minima_peak_x_dmn = x[minima_val_dmn[0]] #list of the minima positions

    #maxima_peak_kdj
    maxima_val_kdj = find_peaks(y_kdj, height = high_1) #kdj
    maxima_peak_y_kdj = maxima_val_kdj[1]['peak_heights'] #list of the heights of the peaks
    maxima_peak_x_kdj = x[maxima_val_kdj[0]] #list of the peaks positions
    #minim_peak_kdj
    y2_kdj = y_kdj*-1
    minima_val_kdj = find_peaks(y2_kdj, height = high_2) #kdj, -30
    minima_peak_y_kdj = y2_kdj[minima_val_kdj[0]]*-1 #list of the mirrored minima heights
    minima_peak_x_kdj = x[minima_val_kdj[0]] #list of the minima positions

    #maxima_peak_obv
    maxima_val_obv = find_peaks(y_obv, height = high_2, prominence=obv_prominence) #obv
    maxima_peak_y_obv = maxima_val_obv[1]['peak_heights'] #list of the heights of the peaks
    maxima_peak_x_obv = x[maxima_val_obv[0]] #list of the peaks positions
    #minim_peak_obv
    y2_obv = y_obv*-1
    minima_val_obv = find_peaks(y2_obv, height = high_2, prominence=obv_prominence) #obv
    minima_peak_y_obv = y2_obv[minima_val_obv[0]]*-1 #list of the mirrored minima heights
    minima_peak_x_obv = x[minima_val_obv[0]] #list of the minima positions

    #maxima_peak_atr
    maxima_val_atr = find_peaks(y_atr, height = high_1, prominence=atr_prominence) #rsi
    maxima_peak_y_atr = maxima_val_atr[1]['peak_heights'] #list of the heights of the peaks
    maxima_peak_x_atr = x[maxima_val_atr[0]] #list of the peaks positions
    #minim_peak_atr
    y2_atr = y_atr*-1
    minima_val_atr = find_peaks(y2_atr, height = high_2, prominence=atr_prominence) #rsi, -30
    minima_peak_y_atr = y2_atr[minima_val_atr[0]]*-1 #list of the mirrored minima heights
    minima_peak_x_atr = x[minima_val_atr[0]] #list of the minima positions

    #maxima_peak_atr14
    maxima_val_atr14 = find_peaks(y_atr14, height = high_1, prominence=atr_prominence) #rsi
    maxima_peak_y_atr14 = maxima_val_atr14[1]['peak_heights'] #list of the heights of the peaks
    maxima_peak_x_atr14 = x[maxima_val_atr14[0]] #list of the peaks positions
    #minim_peak_atr14
    y2_atr14 = y_atr14*-1
    minima_val_atr14 = find_peaks(y2_atr14, height = high_2, prominence=atr_prominence) #rsi, -30
    minima_peak_y_atr14 = y2_atr14[minima_val_atr14[0]]*-1 #list of the mirrored minima heights
    minima_peak_x_atr14 = x[minima_val_atr14[0]] #list of the minima positions

    #maxima_peak_atr_p
    maxima_val_atr_p = find_peaks(y_atr_p, height = high_1) #rsi
    maxima_peak_y_atr_p = maxima_val_atr_p[1]['peak_heights'] #list of the heights of the peaks
    maxima_peak_x_atr_p = x[maxima_val_atr_p[0]] #list of the peaks positions
    #minim_peak_atr_p
    y2_atr_p = y_atr_p*-1
    minima_val_atr_p = find_peaks(y2_atr_p, height = high_2) #rsi, -30
    minima_peak_y_atr_p = y2_atr_p[minima_val_atr_p[0]]*-1 #list of the mirrored minima heights
    minima_peak_x_atr_p = x[minima_val_atr_p[0]] #list of the minima positions

    #maxima_peak_combined_diff
    maxima_val_combined_diff = find_peaks(y_combined_diff, height = high_1, prominence=combined_diff_prominence) #combined_diff
    maxima_peak_y_combined_diff = maxima_val_combined_diff[1]['peak_heights'] #list of the heights of the peaks
    maxima_peak_x_combined_diff = x[maxima_val_combined_diff[0]] #list of the peaks positions
    #minim_peak_combined_diff
    y2_combined_diff = y_combined_diff*-1
    minima_val_combined_diff = find_peaks(y2_combined_diff, height = high_2, prominence=combined_diff_prominence) #combined_diff, -30
    minima_peak_y_combined_diff = y2_combined_diff[minima_val_combined_diff[0]]*-1 #list of the mirrored minima heights
    minima_peak_x_combined_diff = x[minima_val_combined_diff[0]] #list of the minima positions

    #maxima_peak_combined_diff_filtered
    maxima_val_combined_diff_filtered = find_peaks(y_combined_diff_filtered, height = high_1, prominence=combined_diff_prominence) #combined_diff_filtered
    maxima_peak_y_combined_diff_filtered = maxima_val_combined_diff_filtered[1]['peak_heights'] #list of the heights of the peaks
    maxima_peak_x_combined_diff_filtered = x[maxima_val_combined_diff_filtered[0]] #list of the peaks positions
    #minim_peak_combined_diff_filtered
    y2_combined_diff_filtered = y_combined_diff_filtered*-1
    minima_val_combined_diff_filtered = find_peaks(y2_combined_diff_filtered, height = high_2, prominence=combined_diff_prominence) #combined_diff_filtered, -30
    minima_peak_y_combined_diff_filtered = y2_combined_diff_filtered[minima_val_combined_diff_filtered[0]]*-1 #list of the mirrored minima heights
    minima_peak_x_combined_diff_filtered = x[minima_val_combined_diff_filtered[0]] #list of the minima positions

    #maxima_peak_second_combined_diff
    maxima_val_second_combined_diff = find_peaks(y_second_combined_diff, height = high_1, prominence=combined_diff_prominence) #second_combined_diff
    maxima_peak_y_second_combined_diff = maxima_val_second_combined_diff[1]['peak_heights'] #list of the heights of the peaks
    maxima_peak_x_second_combined_diff = x[maxima_val_second_combined_diff[0]] #list of the peaks positions
    #minim_peak_second_combined_diff
    y2_second_combined_diff = y_second_combined_diff*-1
    minima_val_second_combined_diff = find_peaks(y2_second_combined_diff, height = high_2, prominence=combined_diff_prominence) #second_combined_diff, -30
    minima_peak_y_second_combined_diff = y2_second_combined_diff[minima_val_second_combined_diff[0]]*-1 #list of the mirrored minima heights
    minima_peak_x_second_combined_diff = x[minima_val_second_combined_diff[0]] #list of the minima positions

    #maxima_peak_second_combined_diff_filtered
    maxima_val_second_combined_diff_filtered = find_peaks(y_second_combined_diff_filtered, height = high_1, prominence=combined_diff_prominence) #second_combined_diff_filtered
    maxima_peak_y_second_combined_diff_filtered = maxima_val_second_combined_diff_filtered[1]['peak_heights'] #list of the heights of the peaks
    maxima_peak_x_second_combined_diff_filtered = x[maxima_val_second_combined_diff_filtered[0]] #list of the peaks positions
    #minim_peak_second_combined_diff_filtered
    y2_second_combined_diff_filtered = y_second_combined_diff_filtered*-1
    minima_val_second_combined_diff_filtered = find_peaks(y2_second_combined_diff_filtered, height = high_2, prominence=combined_diff_prominence) #second_combined_diff_filtered, -30
    minima_peak_y_second_combined_diff_filtered = y2_second_combined_diff_filtered[minima_val_second_combined_diff_filtered[0]]*-1 #list of the mirrored minima heights
    minima_peak_x_second_combined_diff_filtered = x[minima_val_second_combined_diff_filtered[0]] #list of the minima positions

    return maxima_peak_x_open, maxima_peak_y_open, minima_peak_x_open, minima_peak_y_open, maxima_peak_x_high, maxima_peak_y_high, minima_peak_x_high, minima_peak_y_high, maxima_peak_x_low, maxima_peak_y_low, minima_peak_x_low, minima_peak_y_low, maxima_peak_x_close, maxima_peak_y_close, minima_peak_x_close, minima_peak_y_close, maxima_peak_x_rsi, maxima_peak_y_rsi, minima_peak_x_rsi, minima_peak_y_rsi, maxima_peak_x_MACD_12_26_9, maxima_peak_y_MACD_12_26_9, minima_peak_x_MACD_12_26_9, minima_peak_y_MACD_12_26_9, maxima_peak_x_MACD_50_75_35, maxima_peak_y_MACD_50_75_35, minima_peak_x_MACD_50_75_35, minima_peak_y_MACD_50_75_35, maxima_peak_x_adx, maxima_peak_y_adx, minima_peak_x_adx, minima_peak_y_adx, maxima_peak_x_dx, maxima_peak_y_dx, minima_peak_x_dx, minima_peak_y_dx, maxima_peak_x_dmp, maxima_peak_y_dmp, minima_peak_x_dmp, minima_peak_y_dmp, maxima_peak_x_dmn, maxima_peak_y_dmn, minima_peak_x_dmn, minima_peak_y_dmn, maxima_peak_x_kdj, maxima_peak_y_kdj, minima_peak_x_kdj, minima_peak_y_kdj, maxima_peak_x_obv, maxima_peak_y_obv, minima_peak_x_obv, minima_peak_y_obv, maxima_peak_x_atr, maxima_peak_y_atr, minima_peak_x_atr, minima_peak_y_atr, maxima_peak_x_atr14, maxima_peak_y_atr14, minima_peak_x_atr14, minima_peak_y_atr14, maxima_peak_x_atr_p, maxima_peak_y_atr_p, minima_peak_x_atr_p, minima_peak_y_atr_p, maxima_peak_x_combined_diff, maxima_peak_y_combined_diff, minima_peak_x_combined_diff, minima_peak_y_combined_diff, maxima_peak_x_combined_diff_filtered, maxima_peak_y_combined_diff_filtered, minima_peak_x_combined_diff_filtered, minima_peak_y_combined_diff_filtered, maxima_peak_x_second_combined_diff, maxima_peak_y_second_combined_diff, minima_peak_x_second_combined_diff, minima_peak_y_second_combined_diff, maxima_peak_x_second_combined_diff_filtered, maxima_peak_y_second_combined_diff_filtered, minima_peak_x_second_combined_diff_filtered, minima_peak_y_second_combined_diff_filtered



def calculate_slope(point_1, point_2, maxima_or_minima):
    point_1_open_time = point_1[6] # open_time2
    point_2_open_time = point_2[6] # open_time2
    
    if maxima_or_minima in ['R_bear_maxima', 'H_bear_maxima']: # maxima 일때 가격 high로, minima 일때 가격 low로
        point_1_close = point_1[1] # high
        point_2_close = point_2[1] # high
    else:
        point_1_close = point_1[2] # low
        point_2_close = point_2[2] # low

    # point_1_close = point_1[3]
    # point_2_close = point_2[3]

    point_1_rsi = point_1[11] # RSI_14
    point_2_rsi = point_2[11] # RSI_14

    point_1_dx = point_1[16] # DX_14
    point_2_dx = point_2[16] # DX_14


    time_difference = (point_2_open_time - point_1_open_time).total_seconds()

    if time_difference == 0:
        slope_close = 0
        slope_rsi = 0
        slope_dx = 0
    else:
        slope_close = (point_2_close - point_1_close) / time_difference
        slope_rsi = (point_2_rsi - point_1_rsi) / time_difference
        slope_dx = (point_2_dx - point_1_dx) / time_difference

    return slope_close, slope_rsi, slope_dx

def check_points_between(point_1, points_between, maxima_or_minima, slope_close, slope_rsi, slope_dx):

    point_1_open_time = point_1[6] # open_time2
    point_1_close = point_1[1] if maxima_or_minima in ['R_bear_maxima', 'H_bear_maxima'] else point_1[2]
    # point_1_close = point_1[3]
    point_1_rsi = point_1[11] # RSI_14
    point_1_dx = point_1[16] # RSI_14

    point_k_open_time = pd.to_datetime(points_between[:, 6])  # Convert to Pandas Timestamp
    point_k_close = points_between[:, 1] if maxima_or_minima in ['R_bear_maxima', 'H_bear_maxima'] else points_between[:, 2]
    # point_k_close = points_between[:, 3]
    point_k_rsi = points_between[:, 11]
    point_k_dx = points_between[:, 16]

    time_difference = (point_k_open_time - point_1_open_time).total_seconds()
  

    slope_k_close = (point_k_close - point_1_close) / time_difference
    slope_k_rsi = (point_k_rsi - point_1_rsi) / time_difference
    slope_k_dx = (point_k_dx - point_1_dx) / time_difference


    # if maxima_or_minima in ['R_bear_maxima', 'H_bear_maxima']:
    #     if (slope_k_rsi > slope_rsi).any() or (slope_k_close > slope_close).any():
    #         return True
    # else:
    #     if (slope_k_rsi < slope_rsi).any() or (slope_k_close < slope_close).any():
    #         return True


    if maxima_or_minima == 'R_bull_minima':
        if (slope_k_rsi < slope_rsi).any() or (slope_k_close < slope_close).any() or (slope_k_dx < slope_dx).any():
            return True
    elif maxima_or_minima == 'R_bear_maxima':
        if (slope_k_rsi > slope_rsi).any() or (slope_k_close > slope_close).any() or (slope_k_dx > slope_dx).any():
            return True
    elif maxima_or_minima == 'H_bull_minima':
        if (slope_k_rsi < slope_rsi).any() or (slope_k_close < slope_close).any():
            return True
    elif maxima_or_minima == 'H_bear_maxima':
        if (slope_k_rsi > slope_rsi).any() or (slope_k_close > slope_close).any():
            return True

    return False



def handle_divergence(interval, df, maxima_or_minima):
    divergences = []
    # colors = plt.cm.rainbow(np.linspace(0, 1, len(intervals)))  # Generate a color map
    # start = time.time()
    # for interval in intervals:

    # interval = '1m'
    # maxima_or_minima = 'maxima'
    color_mapping = {
        'R_bull_minima': ('green', 2, 1),
        'R_bear_maxima': ('red', 1, -1),
        'H_bull_minima': ('black', 2, 1),
        'H_bear_maxima': ('magenta', 1, -1)
    }



    ####################################################################
    # (maxima_or_minima == 'minima' and slope_close <= 0 and slope_dx >= 0 and slope_rsi >= 0 and slope_close < slope_rsi) or R_bull (DMN > DMP) & (df.maxima_peak_x_dx_last_3 < 0) & (df.MACD_12_26_9 < 0) & (df.MACDh_12_26_9 < 0)
    # (maxima_or_minima == 'maxima' and slope_close >= 0 and slope_dx <= 0 and slope_rsi <= 0 and slope_close > slope_rsi) or R_bear (DMP > DMN) & (df.maxima_peak_x_dx_last_3 < 0) & (df.MACD_12_26_9 > 0) & (df.MACDh_12_26_9 > 0)
    # (maxima_or_minima == 'minima' and slope_close >= 0 and slope_dx <= 0 and slope_rsi <= 0 and slope_close > slope_rsi) or H_bull (DMP > DMN) & (df.minima_peak_x_dx_last_3 > 0) & (df.MACD_12_26_9 > 0) & (df.MACDh_12_26_9 < 0)
    # (maxima_or_minima == 'maxima' and slope_close <= 0 and slope_dx >= 0 and slope_rsi >= 0 and slope_close < slope_rsi) H_bear (DMN > DMP) & (df.minima_peak_x_dx_last_3 > 0) & (df.MACD_12_26_9 < 0) & (df.MACDh_12_26_9 > 0)


    # if maxima_or_minima == 'minima':
    #     points = df[(df.minima_peak_x_rsi > 0) & (df.minima_peak_x_macd > 0) & (df.maxima_peak_x_dx > 0) & (df.RSI_14 < 45) & (df.MACD_12_26_9 < 0)& (df.MACDh_12_26_9 < 0)]
    # else:
    #     points = df[(df.maxima_peak_x_rsi < 0) & (df.maxima_peak_x_macd < 0) & (df.maxima_peak_x_dx > 0) & (df.RSI_14 > 55) & (df.MACD_12_26_9 > 0)& (df.MACDh_12_26_9 > 0)]# ( | (df.minima_peak_x_dmn < 0) | (df.maxima_peak_x_dmp < 0)) & (df.RSI_14 > 55)]
    
    # if maxima_or_minima == 'minima':
    #     points = df[(df.minima_peak_x_rsi_last_3 > 0) & (df.minima_peak_x_macd_last_3 > 0)]# & (df.maxima_peak_x_dx_last_3 < 0)] #& (df.maxima_peak_x_rsi_last_3_RSI_14 < 45) & (df.minima_peak_x_macd_last_3_MACD_12_26_9 < 0)] #& (df.MACDh_12_26_9 < 0)]
    # else:
    #     points = df[(df.maxima_peak_x_rsi_last_3 < 0) & (df.maxima_peak_x_macd_last_3 < 0)]# & (df.maxima_peak_x_dx_last_3 < 0)] #& (df.maxima_peak_x_rsi_last_3_RSI_14 > 55) & (df.minima_peak_x_macd_last_3_MACD_12_26_9 > 0)] #& (df.MACDh_12_26_9 > 0)]# ( | (df.minima_peak_x_dmn < 0) | (df.maxima_peak_x_dmp < 0)) & (df.RSI_14 > 55)]

    if maxima_or_minima == 'R_bull_minima':
        points = df[(df.minima_peak_x_rsi_last_3 > 0) & (df.maxima_peak_x_dx_last_3 < 0)] #& (df.maxima_peak_x_rsi_last_3_RSI_14 < 45) & (df.minima_peak_x_macd_last_3_MACD_12_26_9 < 0)] #& (df.MACDh_12_26_9 < 0)]

    elif maxima_or_minima == 'R_bear_maxima':
        points = df[(df.maxima_peak_x_rsi_last_3 < 0) & (df.maxima_peak_x_dx_last_3 < 0)] #& (df.maxima_peak_x_rsi_last_3_RSI_14 > 55) & (df.minima_peak_x_macd_last_3_MACD_12_26_9 > 0)] #& (df.MACDh_12_26_9 > 0)]# ( | (df.minima_peak_x_dmn < 0) | (df.maxima_peak_x_dmp < 0)) & (df.RSI_14 > 55)]

    elif maxima_or_minima == 'H_bull_minima':
        points = df[(df.minima_peak_x_rsi_last_3 > 0) & (df.minima_peak_x_macd_last_3 > 0) ] #& (df.maxima_peak_x_rsi_last_3_RSI_14 < 45) & (df.minima_peak_x_macd_last_3_MACD_12_26_9 < 0)] #& (df.MACDh_12_26_9 < 0)]

    elif maxima_or_minima == 'H_bear_maxima':
        points = df[(df.maxima_peak_x_rsi_last_3 < 0) & (df.maxima_peak_x_macd_last_3 < 0) ] #& (df.maxima_peak_x_rsi_last_3_RSI_14 > 55) & (df.minima_peak_x_macd_last_3_MACD_12_26_9 > 0)] #& (df.MACDh_12_26_9 > 0)]# ( | (df.minima_peak_x_dmn < 0) | (df.maxima_peak_x_dmp < 0)) & (df.RSI_14 > 55)]


    # df['minima_peak_x_rsi_last_3']





    # if maxima_or_minima == 'minima':
    #     points = df[(df.findpeaks_valley > 0)]
    # else:
    #     points = df[(df.findpeaks_peak < 0)]
    




    # plt.scatter(points['close_time'], points['high'], color=color, label=interval + ' close')
    # plt.scatter(points['close_time'], points['RSI_14'], color=color, marker='x', label=interval + ' RSI')
    ####################################################################


    points_array = points.to_numpy()
    points_between_to_numpy = df.to_numpy()

    num_points = len(points)





    for i in range(num_points):
        for j in range(i + 1, num_points):

            point_1 = points_array[i]
            point_2 = points_array[j]
            slope_close, slope_rsi, slope_dx = calculate_slope(point_1, point_2, maxima_or_minima)
            # points_between = points[(points['open_time2'] > point_1[6]) & (points['open_time2'] < point_2[6])].values

            points_between = points_between_to_numpy[(points_between_to_numpy[:, 6] > point_1[6]) & (points_between_to_numpy[:, 6] < point_2[6])]
            # print(len(points_between))



            if check_points_between(point_1, points_between, maxima_or_minima, slope_close, slope_rsi, slope_dx):
                continue


            if (
                (maxima_or_minima == 'R_bull_minima' and slope_close <= 0 and slope_rsi >= 0 and slope_dx >= 0 and slope_close < slope_rsi) or
                (maxima_or_minima == 'R_bear_maxima' and slope_close >= 0 and slope_rsi <= 0 and slope_dx <= 0 and slope_close > slope_rsi) or
                (maxima_or_minima == 'H_bull_minima' and slope_close >= 0 and slope_rsi <= 0 and slope_close > slope_rsi) or
                (maxima_or_minima == 'H_bear_maxima' and slope_close <= 0 and slope_rsi >= 0 and slope_close < slope_rsi)
            ):
                divergence_name = maxima_or_minima
                color, close_attr, divergence_value = color_mapping.get(divergence_name, ('', '', 0))

                divergence = (1, divergence_name, divergence_value, point_1[6], point_1[close_attr], point_1[10], divergence_value, point_2[6], point_2[close_attr], point_2[10])
                divergences.append(divergence)                
                



#                 df.loc[point_2.name, 'divergence'] = 1
#                 df.loc[point_2.name, 'divergence_name'] = divergence_name

#                 df.loc[point_2.name, 'divergence_point_1'] = divergence_value
#                 df.loc[point_2.name, 'divergence_open_time'] = point_1['open_time2']
#                 df.loc[point_2.name, 'divergence_close'] = point_1[close_attr]
#                 df.loc[point_2.name, 'divergence_rsi'] = point_1['RSI_14']

#                 df.loc[point_2.name, 'divergence_point_2'] = divergence_value
#                 df.loc[point_2.name, 'divergence_open_time_2'] = point_2['open_time2']
#                 df.loc[point_2.name, 'divergence_close_2'] = point_2[close_attr]
#                 df.loc[point_2.name, 'divergence_rsi_2'] = point_2['RSI_14']


                
#                 print('------- df_' + interval + ' -------')
#                 print("Divergence detected between points", i, "and", j)
#                 print("divergence_name:", divergence_type, divergence_direction)
                
#                 print("Open Time 1:", point_1[6])
#                 print("Close 1:", point_1[close_attr])
#                 print("RSI_14 1:", point_1[10])

#                 print("Open Time 2:", point_2[6])
#                 print("Close 2:", point_2[close_attr])
#                 print("RSI_14 2:", point_2[10])


                
                
                # Draw lines connecting RSI points
                # plt.plot([point_1[5], point_2[5]], [point_1[close_attr], point_2[close_attr]], color=color, linestyle='--')
                # # plt.plot([point_1[5], point_2[5]], [point_1[10], point_2[10]], color=color, linestyle='--')
                # plt.scatter(point_2[5], point_2[close_attr], color=color, label=interval + ' close')
                # plt.scatter(point_2[5], point_2[10], color=color, marker='x', label=interval + ' RSI')
    return divergences
                

def db_insert(trader_name, exchange_id, market_id, volatility_macro_state, volatility_micro_interval_pick, trend_macro_state, long_trend_micro_interval_pick, short_trend_micro_interval_pick, atr_pick, balance_currency, wallet_balance, unrealised_pnl, roe_pcnt, loop, scaled_level_n, max_waiting_in_second, exit_status, position_side, position_size, position_value, position_entry_price, liquidation_price, open_order_counter, open_order_side, open_order_size, open_order_price, open_order_type, message):
    db.execute("INSERT INTO crypto_currency (trader_name, exchange_id, market_id, volatility_macro_state, volatility_micro_interval_pick, trend_macro_state, long_trend_micro_interval_pick, short_trend_micro_interval_pick, atr_pick, balance_currency, wallet_balance, unrealised_pnl, roe_pcnt, loop, scaled_level_n, max_waiting_in_second, exit_status, position_side, position_size, position_value, position_entry_price, liquidation_price, open_order_counter, open_order_side, open_order_size, open_order_price, open_order_type, message) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (trader_name, exchange_id, market_id, volatility_macro_state, volatility_micro_interval_pick, trend_macro_state, long_trend_micro_interval_pick, short_trend_micro_interval_pick, atr_pick, balance_currency, wallet_balance, unrealised_pnl, roe_pcnt, loop, scaled_level_n, max_waiting_in_second, exit_status, position_side, position_size, position_value, position_entry_price, liquidation_price, open_order_counter, open_order_side, open_order_size, open_order_price, open_order_type, message))
    #db.close()
    conn.commit()
    return

def check_pointer(trader_name, exchange_id, market_id, balance_currency, loop, max_waiting_in_second, exit_status, message):
    wallet_balance = balance_calc(exchange_id, balance_currency)
    position_side, position_size, position_value, position_entry_price, position_entry_time, liquidation_price, unrealised_pnl, roe_pcnt = position_calc(exchange_id, market_id)
    interval_, side_, last_time_, big_boss_trend_checker = big_boss_trend_re(symbol, globals()['valid_intervals'])

    scaled_level_n = scale_order_position_amount_calc(min_order_amount, wallet_balance, max_leverage, position_size, r, scale_order_max_limit)[3]
    open_order_counter, open_order_side, open_order_size, open_order_price, open_order_type, stop_market_counter = open_order_calc(exchange_id, market_id)
    if position_value is None:
        position_value = 0
    elif (isinstance(position_value, int)) or (isinstance(position_value, float)):
        position_value = position_value
    if position_entry_price is None:
        position_entry_price = '-'
    elif (isinstance(position_entry_price, int)) or (isinstance(position_entry_price, float)):
        position_entry_price = position_entry_price
    if liquidation_price is None:
        liquidation_price = '-'
    elif (isinstance(liquidation_price, int)) or (isinstance(liquidation_price, float)):
        liquidation_price = liquidation_price
    if unrealised_pnl is None:
        unrealised_pnl = 0
    elif (isinstance(unrealised_pnl, int)) or (isinstance(unrealised_pnl, float)):
        unrealised_pnl = unrealised_pnl
    if roe_pcnt is None:
        roe_pcnt = 0
    elif (isinstance(roe_pcnt, int)) or (isinstance(roe_pcnt, float)):
        roe_pcnt = roe_pcnt


    if big_boss_trend_checker == 'long':
        db_insert(trader_name, exchange_id, market_id, globals()['volatility_macro_state'], globals()['volatility_micro_interval_pick'], big_boss_trend_checker, interval_, '-', globals()['atr_pick'], balance_currency, wallet_balance, unrealised_pnl, roe_pcnt, loop, scaled_level_n, max_waiting_in_second, exit_status, position_side, position_size, position_value, position_entry_price, liquidation_price, open_order_counter, open_order_side, open_order_size, open_order_price, open_order_type, message)
    elif big_boss_trend_checker == 'short':
        db_insert(trader_name, exchange_id, market_id, globals()['volatility_macro_state'], globals()['volatility_micro_interval_pick'], big_boss_trend_checker, '-', interval_, globals()['atr_pick'], balance_currency, wallet_balance, unrealised_pnl, roe_pcnt, loop, scaled_level_n, max_waiting_in_second, exit_status, position_side, position_size, position_value, position_entry_price, liquidation_price, open_order_counter, open_order_side, open_order_size, open_order_price, open_order_type, message)
    else:
        db_insert(trader_name, exchange_id, market_id, globals()['volatility_macro_state'], globals()['volatility_micro_interval_pick'], '-', '-', '-', globals()['atr_pick'], balance_currency, wallet_balance, unrealised_pnl, roe_pcnt, loop, scaled_level_n, max_waiting_in_second, exit_status, position_side, position_size, position_value, position_entry_price, liquidation_price, open_order_counter, open_order_side, open_order_size, open_order_price, open_order_type, message)

    
    # db_insert(trader_name, exchange_id, market_id, globals()['volatility_macro_state'], globals()['volatility_micro_interval_pick'], globals()['trend_macro_state'], globals()['long_trend_micro_interval_pick'], globals()['short_trend_micro_interval_pick'], globals()['atr_pick'], balance_currency, wallet_balance, unrealised_pnl, roe_pcnt, loop, scaled_level_n, max_waiting_in_second, exit_status, position_side, position_size, position_value, position_entry_price, liquidation_price, open_order_counter, open_order_side, open_order_size, open_order_price, open_order_type, message)
    return





def optimize_parameters(df, column='close', lamb_range=(100, 10000), alpha_range=(0.1, 0.5)):
    """
    Grid search for optimal lamb and alpha parameters using F1 score as the metric.

    Args:
        df (pd.DataFrame): Time series data.
        column (str, optional): Name of the column representing the time series data. Defaults to 'close'.
        lamb_range (tuple, optional): Range for lamb parameter search. Defaults to (100, 10000).
        alpha_range (tuple, optional): Range for alpha parameter search. Defaults to (0.1, 0.5).

    Returns:
        tuple: (optimal lamb, optimal alpha, best F1 score).
    """

    best_f1 = 0
    best_lamb = None
    best_alpha = None

    for lamb in np.arange(*lamb_range, step=100):
        for alpha in np.arange(*alpha_range, step=0.01):
            anomalies_idx = AnomalyDetection(df, column=column, alpha=alpha, lamb=lamb)

            # Create a DataFrame with anomalies marked
            df_with_anomalies = df.copy()
            df_with_anomalies['anomaly'] = False
            df_with_anomalies.loc[anomalies_idx, 'anomaly'] = True

            # Calculate F1 score
            y_true = df_with_anomalies['anomaly']
            y_pred = df_with_anomalies['anomaly'].astype(int)
            f1 = f1_score(y_true, y_pred)

            if f1 > best_f1:
                best_f1 = f1
                best_lamb = lamb
                best_alpha = alpha

    return best_lamb, best_alpha, best_f1

# Hodrick Prescott filter 1
def hp_filter(x, lamb):
    w = len(x)
    b = [[1]*w, [-2]*w, [1]*w]
    D = sparse.spdiags(b, [0, 1, 2], w-2, w)
    I = sparse.eye(w)
    B = (I + lamb*(D.transpose()*D))
    return sparse.linalg.dsolve.spsolve(B, x)


def mad(data, axis=None):
    return np.mean(np.abs(data - np.mean(data, axis)), axis)

def AnomalyDetection(df, column, alpha, lamb):
    """
    df        : pd.DataFrame
                DataFrame containing time series data
    column    : str
                Column name representing the time series data
    alpha     : float
                The level of statistical significance with which to
                accept or reject anomalies (exponential distribution)
    lamb      : int
                Penalize parameter for the hp filter
    return    : pd.Index
                Index of anomalies
    """
    # extract time series data
    x = df[column]

    # calculate residual
    xhat = hp_filter(x, lamb)
    resid = x - xhat

    # Remove the seasonal and trend component,
    # and the median of the data to create the univariate remainder
    md = np.median(x)
    data = resid - md

    # process data, using median filter
    ares = (data - data.median()).abs()
    data_sigma = mad(data) + 1e-12
    ares = ares / data_sigma

    # compute significance
    p = 1. - alpha
    R = stats.expon.interval(p, loc=ares.mean(), scale=ares.std())
    threshold = R[1]

    # extract index
    anomalies_idx = ares.index[ares > threshold]

    return anomalies_idx

def min_interval_finder(intervals, point_frame):
    # 각 인터벌의 시간을 계산하고 가장 작은 것을 찾기
    min_interval = None
    min_value = float('inf')

    for interval in intervals:
        if interval[-1] in point_frame:
            # '1m', '5m' -> 'm' -> point_frame['m'], '1h' -> 'h', '2w' -> 'w'
            time_value = point_frame[interval[-1]] * int(interval[:-1])  # 수치와 단위 분리
            if time_value < min_value:
                min_value = time_value
                min_interval = interval
    return min_interval

def multiple_frame_stg(intervals):
    # 각 데이터프레임에서 stgQ_long > 0인 close_time 추출
    times_long = {}
    times_short = {}

    for interval in intervals:
        df = globals()[f'df_{interval}']

        if interval == '1m':
            filtered_long = df[df['combined_diff_filtered_diff'] > 0]['close_time']
        else:
            filtered_long = df[df['stgQ_long'] > 0]['close_time']
        
        times_long[interval] = set(filtered_long)

        if interval == '1m':
            filtered_short = df[df['combined_diff_filtered_diff'] > 0]['close_time']
        else:
            filtered_short = df[df['stgQ_short'] < 0]['close_time']
        times_short[interval] = set(filtered_short)

    common_times_long = set.intersection(*times_long.values())
    common_times_short = set.intersection(*times_short.values())

    return common_times_long, common_times_short

def update_stg_values(common_long, common_short, smallest_interval):

    df_smallest = globals()[f'df_{smallest_interval}']

    for common_time in common_long:
        # smallest_interval 데이터에서 해당 common_time에 해당하는 stg2_long 업데이트
        df_smallest.loc[df_smallest['close_time'] == common_time, 'stg2_long'] = 2

    for common_time in common_short:
        # smallest_interval 데이터에서 해당 common_time에 해당하는 'stg2_long', 'stg2_short' 찾아서 값 업데이트
        df_smallest.loc[df_smallest['close_time'] == common_time, 'stg2_short'] = -2  # 예시: 값을 -1로 업데이트


















def peak_calc(market_id, intervals): # df_1m, df_3m, df_5m, df_15m, df_30m, df_1h, df_2h, df_4h, df_6h, df_8h, df_12h, df_1d, df_3d 생성
    for interval in intervals:
        df=klines(market_id=market_id, interval=interval, limit=limit)
        if(df['close'].count() > 250):
            if interval not in globals()['valid_intervals']:
                globals()['valid_intervals'].append(interval)
            CustomStrategy = ta.Strategy( # Create your own Custom Strategy
                name="Momo and Volatility",
                description="RSI, MACD, adx, dmp, dmn, obv, atr",
                ta=[
                    {"kind": "rsi", "length": 5},
                    {"kind": "rsi"},
                    {"kind": "mfi"},
                    {"kind": "rsi", "length": 6},
                    {"kind": "rsi", "length": 12},
                    {"kind": "rsi", "length": 24},
                    {"kind": "rsi", "length": 200},
                    {"kind": "macd"},
                    {"kind": "macd", "fast": 12, "slow": 200},
                    {"kind": "macd", "fast": 50, "slow": 200},
                    {"kind": "macd", "fast": 50, "slow": 75, "signal":35},
                    # {"kind": "macd", "fast": 200, "slow": 700},
                    {"kind": "adx"},
                    {"kind": "adx", "length": 5},
                    {"kind": "adx", "length": 50},
                    {"kind": "adx", "length": 200, "lensig": 14},
                    {"kind": "obv"},
                    {"kind": "kdj"},
                    {"kind": "stochrsi"},
                    {"kind": "willr"},
                    {"kind": "bbands", "length": 21},
                    {"kind": "atr", "length": 1},
                    {"kind": "atr", "length": 3},
                    {"kind": "atr", "length": 5},
                    {"kind": "atr", "length": 14},
                    {"kind": "atr", "length": 22},
                    {"kind": "atr", "length": 50},
                    {"kind": "atr", "length": 200},
                    {"kind": "wma", "length": 7},
                    {"kind": "vwma", "length": 7},
                    {"kind": "sma", "length": 7},
                    {"kind": "sma", "length": 21},
                    {"kind": "sma", "length": 30},
                    {"kind": "sma", "length": 50},
                    {"kind": "sma", "length": 100},
                    {"kind": "sma", "length": 200},
                    {"kind": "ema", "length": 7},
                    {"kind": "ema", "length": 21},
                    {"kind": "ema", "length": 30},
                    {"kind": "ema", "length": 50},
                    {"kind": "ema", "length": 100},
                    {"kind": "ema", "length": 200},
                ]
            )
            df.ta.strategy(CustomStrategy) # To run your "Custom Strategy"
            if ('RSI_5' in df.columns) and ('RSI_14' in df.columns) and ('MACDs_12_26_9' in df.columns) and ('ADX_14' in df.columns) and ('DX_14' in df.columns) and ('DMP_14' in df.columns) and ('DMN_14' in df.columns) and ('OBV' in df.columns) and ('BBU_21_2.0' in df.columns) and ('BBL_21_2.0' in df.columns) and ('ATRr_1' in df.columns) and ('ATRr_5' in df.columns) and ('ATRr_14' in df.columns) and ('ATRr_22' in df.columns) and ('SMA_7' in df.columns) and ('SMA_7' in df.columns) and ('SMA_30' in df.columns) and ('SMA_50' in df.columns) and ('EMA_7' in df.columns) and ('EMA_30' in df.columns) and ('EMA_50' in df.columns):
                df.ta.amat(close=df.RSI_14, append=True, col_names = ("rsi_AMATe_LR_8_21_2", "rsi_AMATe_SR_8_21_2")) #  rsi length 14 에대한 amat를 구해서 가격이 횡보인구간 산출, 가격그래프에 scatter
                df.ta.amat(close=df.MACD_12_26_9, append=True, col_names = ("macd_AMATe_LR_8_21_2", "macd_AMATe_SR_8_21_2")) #  macd length 26 에대한 amat를 구해서 가격이 횡보인구간 산출, 가격그래프에 scatter
                df.ta.amat(close=df.MACDh_12_26_9, append=True, col_names = ("macdh_AMATe_LR_8_21_2", "macdh_AMATe_SR_8_21_2")) #  macd length 26 에대한 amat를 구해서 가격이 횡보인구간 산출, 가격그래프에 scatter
                df.ta.sma(close=df['OBV'], length=9, append=True, col_names = ("obv_sma_9")) # obv 의 moving average 산출
                df.ta.amat(close=df.OBV, append=True, col_names = ("obv_AMATe_LR_8_21_2", "obv_AMATe_SR_8_21_2")) #  obv length 21 에대한 amat를 구해서 가격이 횡보인구간 산출, 가격그래프에 scatter
                df.ta.amat(close=df.ATRr_14, append=True, col_names = ("atr_AMATe_LR_8_21_2", "atr_AMATe_SR_8_21_2")) #  atr length 14 에대한 amat를 구해서 가격이 횡보인구간 산출, 가격그래프에 scatter

                df.ta.amat(close=df.high, append=True, col_names = ("high_AMATe_LR_8_21_2", "high_AMATe_SR_8_21_2")) #  atr length 14 에대한 amat를 구해서 가격이 횡보인구간 산출, 가격그래프에 scatter
                df.ta.amat(close=df.low, append=True, col_names = ("low_AMATe_LR_8_21_2", "low_AMATe_SR_8_21_2")) #  atr length 14 에대한 amat를 구해서 가격이 횡보인구간 산출, 가격그래프에 scatter                

                df.ta.amat(close=df.ADX_14, append=True, col_names = ("adx_AMATe_LR_8_21_2", "adx_AMATe_SR_8_21_2")) #  adx length 14 에대한 amat를 구해서 가격이 횡보인구간 산출, 가격그래프에 scatter
                df.ta.amat(close=df.DX_14, append=True, col_names = ("dx_AMATe_LR_8_21_2", "dx_AMATe_SR_8_21_2")) #  dx length 14 에대한 amat를 구해서 가격이 횡보인구간 산출, 가격그래프에 scatter
                df.ta.amat(close=df.DMP_14, append=True, col_names = ("dmp_AMATe_LR_8_21_2", "dmp_AMATe_SR_8_21_2")) #  dmp length 14 에대한 amat를 구해서 가격이 횡보인구간 산출, 가격그래프에 scatter
                df.ta.amat(close=df.DMN_14, append=True, col_names = ("dmn_AMATe_LR_8_21_2", "dmn_AMATe_SR_8_21_2")) #  dmn length 14 에대한 amat를 구해서 가격이 횡보인구간 산출, 가격그래프에 scatter
                df.ta.amat(close=df.SMA_7, append=True, col_names = ("close_AMATe_LR_8_21_2", "close_AMATe_SR_8_21_2")) #  sma length 21 에대한 amat를 구해서 가격이 횡보인구간 산출, 가격그래프에 scatter
                df.ta.amat(close=df.J_9_3, append=True, col_names = ("kdj_AMATe_LR_8_21_2", "kdj_AMATe_SR_8_21_2")) #  kdj 중 j 에대한 amat를 구해서 가격이 횡보인구간 산출, 가격그래프에 scatter
                df['volume_diff'] = df['volume'].diff() # 앞 20 캔들 전의 볼륨과 물량 비교, 단순 subtract
                df['volume_pct_change'] = df['volume'].pct_change(periods=1) # 앞 20 캔들 전의 볼륨과 물량 비교, ratio
                
                df['EMA_200_close_diff'] = (df['close'] - df['EMA_200'])
                df['EMA_200_BBU_21_2_diff'] = (df['BBU_21_2.0'] - df['EMA_200'])
                df['EMA_200_BBL_21_2_diff'] = (df['BBL_21_2.0'] - df['EMA_200'])
                df['sma7_diff'] = df['SMA_7'].diff()
                df['sma21_diff'] = df['SMA_21'].diff()
                df['sma50_diff'] = df['SMA_50'].diff()
                df['ema7_diff'] = df['EMA_7'].diff()
                df['ema21_diff'] = df['EMA_21'].diff()            
                df['wma7_diff'] = df['WMA_7'].diff()
                df['vwma7_diff'] = df['VWMA_7'].diff()

                df['adx_diff'] = df['ADX_14'].diff()
                df['dx_diff'] = df['DX_14'].diff()
                df['dmp_diff'] = df['DMP_14'].diff()
                df['dmn_diff'] = df['DMN_14'].diff()

                df['adx_5_diff'] = df['ADX_5'].diff()
                df['dx_5_diff'] = df['DX_5'].diff()
                df['dmp_5_diff'] = df['DMP_5'].diff()
                df['dmn_5_diff'] = df['DMN_5'].diff()

                df['adx_200_diff'] = df['ADX_200'].diff()
                df['dx_200_diff'] = df['DX_200'].diff()
                df['dmp_200_diff'] = df['DMP_200'].diff()
                df['dmn_200_diff'] = df['DMN_200'].diff()
                df['dmp_dmn_200_diff'] = df['DMP_200'] - df['DMN_200']

                df['rsi_diff'] = df['RSI_14'].diff()

                df['macd_diff'] = df['MACD_12_26_9'].diff()
                df['macds_diff'] = df['MACDs_12_26_9'].diff()
                df['macdh_diff'] = df['MACDh_12_26_9'].diff()

                df['macd_diff_35'] = df['MACD_50_75_35'].diff()
                df['macds_diff_35'] = df['MACDs_50_75_35'].diff()
                df['macdh_diff_35'] = df['MACDh_50_75_35'].diff()

                df['macd_diff_12'] = df['MACD_12_200_9'].diff()
                df['macds_diff_12'] = df['MACDs_12_200_9'].diff()
                df['macdh_diff_12'] = df['MACDh_12_200_9'].diff()

                df['macd_diff_200'] = df['MACD_50_200_9'].diff()
                df['macds_diff_200'] = df['MACDs_50_200_9'].diff()
                df['macdh_diff_200'] = df['MACDh_50_200_9'].diff()


                # df['macd_diff_700'] = df['MACD_200_700_9'].diff()
                # df['macds_diff_700'] = df['MACDs_200_700_9'].diff()
                # df['macdh_diff_700'] = df['MACDh_200_700_9'].diff()

                df['j_diff'] = df['J_9_3'].diff()
                df['k_diff'] = df['K_9_3'].diff()
                df['d_diff'] = df['D_9_3'].diff()

                df['rsi_6_diff'] = df['RSI_6'].diff()
                df['rsi_12_diff'] = df['RSI_12'].diff()
                df['rsi_24_diff'] = df['RSI_24'].diff()

                df['atr_diff'] = df['ATRr_14'].diff()
                df['atr_p'] = df['ATRr_14']/df['close']*100
                df['atr_p_diff'] = df['atr_p'].diff()

                df['atr5_diff'] = df['ATRr_5'].diff()
                df['atr5_p'] = df['ATRr_5']/df['close']*100
                df['atr5_p_diff'] = df['atr5_p'].diff()

                df['atr200_diff'] = df['ATRr_200'].diff()
                df['atr200_p'] = df['ATRr_200']/df['close']*100
                df['atr200_p_diff'] = df['atr200_p'].diff()






                ########################################################################################
                
                df['abs_EMA_200_close_diff'] = abs(df['close'] - df['EMA_200'])
                df['abs_EMA_200_BBU_21_2_diff'] = abs(df['BBU_21_2.0'] - df['EMA_200'])
                df['abs_EMA_200_BBL_21_2_diff'] = abs(df['BBL_21_2.0'] - df['EMA_200'])
                df['abs_dmp_dmn_200_diff'] = abs(df['DMP_200'] - df['DMN_200'])
                df['abs_MACD_50_75_35'] = abs(df['MACD_50_75_35'])
                df['abs_MACD_12_26_9'] = abs(df['MACD_12_26_9'])
                df['abs_RSI_14'] = abs(df['RSI_14'] - 50)
                df['abs_RSI_200'] = abs(df['RSI_200'] - 50)
                mid_value_J = 50  # 중간값
                df['abs_J_9_3'] = abs(df['J_9_3'] - mid_value_J)  # 중간값 기준 절대값 처리

                # 1. 데이터 정규화 (Min-Max Scaling)
                # scaler = MinMaxScaler()
                scaler = MinMaxScaler(feature_range=(0, 1))  # 최대값을 1로 설정

                # 정규화할 열들
                scaling_columns = [
                    'abs_EMA_200_close_diff', 
                    'abs_EMA_200_BBU_21_2_diff', 
                    'abs_EMA_200_BBL_21_2_diff', 
                    'abs_dmp_dmn_200_diff', 
                    'abs_MACD_50_75_35',
                    'atr_p'
                    # 'abs_MACD_12_26_9'
                ]

                # 2. 원본 데이터는 유지하고, 정규화된 데이터를 새로운 열에 저장
                df[[f'scaled_{col}' for col in scaling_columns]] = scaler.fit_transform(df[scaling_columns])

                # df['scaled_abs_dmp_dmn_200_diff'] = df['abs_dmp_dmn_200_diff'] / 100.0 * 5.0 # 0 ~ 1로 정규화
                # df['scaled_atr_p'] = df['atr_p'] / 100.0 * 100.0 # 0 ~ 1로 정규화
                # df['scaled_RSI_14'] = df['abs_RSI_14']  / 50.0  # 0 ~ 1로 정규화
                df['scaled_RSI_200'] = df['abs_RSI_200']  / 50.0 # * 5.0 # 0 ~ 1로 정규화
                df['scaled_J_9_3'] = df['abs_J_9_3'] / 250.0  # 0 ~ 1로 정규화

                # 3. 세 가지 비슷한 특성의 비중 조정 (1로 합쳐서 계산)
                df['combined_EMA'] = (
                    df['scaled_abs_EMA_200_close_diff'] + 
                    df['scaled_abs_EMA_200_BBU_21_2_diff'] + 
                    df['scaled_abs_EMA_200_BBL_21_2_diff']
                ) / 3  # 평균으로 비중 조정

                # 4. 나머지 특성들과 combined_EMA를 합쳐서 combined_diff 생성
                df['combined_diff'] = (
                    df['combined_EMA'] +  # 조정된 combined_EMA
                    df['scaled_abs_dmp_dmn_200_diff'] +
                    df['scaled_abs_MACD_50_75_35'] +
                    # df['scaled_abs_MACD_12_26_9'] +

                    df['scaled_atr_p'] +
                    # df['scaled_RSI_14'] +
                    df['scaled_RSI_200']
                    # df['scaled_J_9_3']
                    
                ) / 5  # 평균으로 비중 조정



                window_length = 51  # 윈도우 크기는 홀수여야 합니다.
                polyorder = 2
                df['combined_diff_filtered'] = savgol_filter(df['combined_diff'], window_length=window_length, polyorder=polyorder)
                
                
                df['combined_diff_diff'] = df['combined_diff'].diff()
                df['combined_diff_filtered_diff'] = df['combined_diff_filtered'].diff()


                ########################################################################################


                ############## 2. ##########################################################################

                df['second_EMA_200_close_diff'] = df['close'] - df['EMA_200']
                df['second_MACD_50_75_35'] = df['MACD_50_75_35']
                df['second_dmp_dmn_200_diff'] = df['DMP_200'] - df['DMN_200']
                df['second_RSI_200'] = (df['RSI_200'] - 50)/50
                
                # 1. 데이터 정규화 (Min-Max Scaling)
                # scaler = MinMaxScaler()
                scaler = MinMaxScaler(feature_range=(-1, 1))  # 최대값을 1로 설정

                # 정규화할 열들
                scaling_columns = [
                    'second_EMA_200_close_diff',
                    'second_MACD_50_75_35',
                    'second_dmp_dmn_200_diff'
                ]

                # 2. 원본 데이터는 유지하고, 정규화된 데이터를 새로운 열에 저장
                df[[f'scaled_{col}' for col in scaling_columns]] = scaler.fit_transform(df[scaling_columns])

                # df['scaled_second_dmp_dmn_200_diff'] = df['second_dmp_dmn_200_diff'] / 100.0 * 4.0  # 0 ~ 1로 정규화
                df['scaled_second_RSI_200'] = df['second_RSI_200']


                # 4. 나머지 특성들과 combined_EMA를 합쳐서 combined_diff 생성
                df['second_combined_diff'] = (
                    df['scaled_second_EMA_200_close_diff'] +  # 조정된 combined_EMA
                    df['scaled_second_MACD_50_75_35'] +
                    df['scaled_second_dmp_dmn_200_diff'] +
                    df['scaled_second_RSI_200']
                ) / 4  # 평균으로 비중 조정

                df['second_combined_diff_filtered'] = savgol_filter(df['second_combined_diff'], window_length=window_length, polyorder=polyorder)
                
                df['second_combined_diff_diff'] = df['second_combined_diff'].diff()
                df['second_combined_diff_filtered_diff'] = df['second_combined_diff_filtered'].diff()

                ########################################################################################

                df['MACDh_12_26_9_filtered'] = savgol_filter(df['MACDh_12_26_9'], window_length=window_length, polyorder=polyorder)
                df['MACDh_50_75_35_filtered'] = savgol_filter(df['MACDh_50_75_35'], window_length=window_length, polyorder=polyorder)
                





                df['obv_diff'] = df['OBV'].diff()
                df['open_diff'] = df['open'].diff()
                df['high_diff'] = df['high'].diff()
                df['low_diff'] = df['low'].diff()
                df['close_diff'] = df['close'].diff()

                df['sma50_diff'] = df['SMA_50'].diff()
                df['sma200_diff'] = df['SMA_200'].diff()
                df['ema50_diff'] = df['EMA_50'].diff()
                df['ema200_diff'] = df['EMA_200'].diff()
                df['ema100_diff'] = df['EMA_100'].diff()

                df['bbb_diff'] = df['BBB_21_2.0'].diff()
                df['bbu_diff'] = df['BBU_21_2.0'].diff()
                df['bbl_diff'] = df['BBL_21_2.0'].diff()
                dmn_set_1 = df['DMN_14']
                dmp_set_1 = df['DMP_14']
                dmn_set_2 = df['DMN_14'].shift(1)
                dmp_set_2 = df['DMP_14'].shift(1)

                df['bbb_diff'] = df['BBB_21_2.0'].diff()



                df['intersect'] = np.where((dmn_set_2 > dmp_set_2) & (dmn_set_1 > dmp_set_1), 0,
                                np.where((dmn_set_2 > dmp_set_2) & (dmn_set_1 < dmp_set_1), 1,
                                np.where((dmn_set_2 < dmp_set_2) & (dmn_set_1 > dmp_set_1), 1,
                                np.where((dmn_set_2 < dmp_set_2) & (dmn_set_1 < dmp_set_1), 0, np.nan))))



                df['open_change'] = np.where(df['open'].abs().shift() != 0, (df['open'].diff() / df['open'].abs().shift()) * 100, 0)
                df['high_change'] = np.where(df['high'].abs().shift() != 0, (df['high'].diff() / df['high'].abs().shift()) * 100, 0)
                df['low_change'] = np.where(df['low'].abs().shift() != 0, (df['low'].diff() / df['low'].abs().shift()) * 100, 0)
                df['close_change'] = np.where(df['close'].abs().shift() != 0, (df['close'].diff() / df['close'].abs().shift()) * 100, 0)
                df['macd_percentage_change'] = np.where(df['MACD_12_26_9'].abs().shift() != 0, (df['MACD_12_26_9'].diff() / df['MACD_12_26_9'].abs().shift()) * 100, 0)
                df['macds_percentage_change'] = np.where(df['MACDs_12_26_9'].abs().shift() != 0, (df['MACDs_12_26_9'].diff() / df['MACDs_12_26_9'].abs().shift()) * 100, 0)

                df['rsi_percentage_change'] = np.where(df['RSI_14'].abs().shift() != 0, (df['RSI_14'].diff() / df['RSI_14'].abs().shift()) * 100, 0)
                df['sma7_percentage_change'] = np.where(df['SMA_7'].abs().shift() != 0, (df['SMA_7'].diff() / df['SMA_7'].abs().shift()) * 100, 0)
                df['sma21_percentage_change'] = np.where(df['SMA_21'].abs().shift() != 0, (df['SMA_21'].diff() / df['SMA_21'].abs().shift()) * 100, 0)
                df['sma50_percentage_change'] = np.where(df['SMA_50'].abs().shift() != 0, (df['SMA_50'].diff() / df['SMA_50'].abs().shift()) * 100, 0)
                df['sma200_percentage_change'] = np.where(df['SMA_200'].abs().shift() != 0, (df['SMA_200'].diff() / df['SMA_200'].abs().shift()) * 100, 0)
                df['ema7_percentage_change'] = np.where(df['EMA_7'].abs().shift() != 0, (df['EMA_7'].diff() / df['EMA_7'].abs().shift()) * 100, 0)
                df['ema21_percentage_change'] = np.where(df['EMA_21'].abs().shift() != 0, (df['EMA_21'].diff() / df['EMA_21'].abs().shift()) * 100, 0)
                df['ema50_percentage_change'] = np.where(df['EMA_50'].abs().shift() != 0, (df['EMA_50'].diff() / df['EMA_50'].abs().shift()) * 100, 0)
                df['ema200_percentage_change'] = np.where(df['EMA_200'].abs().shift() != 0, (df['EMA_200'].diff() / df['EMA_200'].abs().shift()) * 100, 0)

                df['atr_percentage_change'] = np.where(df['ATRr_14'].abs().shift() != 0, (df['ATRr_14'].diff() / df['ATRr_14'].abs().shift()) * 100, 0)
                df['obv_percentage_change'] = np.where(df['OBV'].abs().shift() != 0, (df['OBV'].diff() / df['OBV'].abs().shift()) * 100, 0)
                df['adx_percentage_change'] = np.where(df['ADX_14'].abs().shift() != 0, (df['ADX_14'].diff() / df['ADX_14'].abs().shift()) * 100, 0)
                df['dx_percentage_change'] = np.where(df['DX_14'].abs().shift() != 0, (df['DX_14'].diff() / df['DX_14'].abs().shift()) * 100, 0)

                price_move = df['close'] - df['open']
                df['price_move'] = price_move
                df['high_low'] = df['high'] - df['low']
                df['volume2'] = np.where(price_move >= 0, df['volume'].abs(), -df['volume'].abs())
                df['volume_change'] = np.where( df['volume2'].abs().shift() != 0,  (df['volume2'].diff() / df['volume2'].abs().shift()) * 100, 0)

                atr_period = 14
                chandelier_info = TA.CHANDELIER(df, short_period=atr_period, long_period=atr_period, k=3) # 목표가 달성 후, chandelier_exit 에 자동 익절
                df = pd.concat([df, chandelier_info], axis=1, ignore_index=False)
                
                length_close = len(df.close) # Calculate the length of df_1m.close
                high_1 = np.zeros(length_close)  # Create an array of zeros with the same length
                high_2 = np.full(length_close, -100000000000)[:length_close]

                ########################################################################################################################
                ########################################################################################################################


                df['savgol'] = savgol_filter(df['close'], window_length=11, polyorder=2)
                df['lowess'] = sm.nonparametric.lowess(df['close'], df.index, frac=0.1)[:, 1]
                df['lowess_1'] = sm.nonparametric.lowess(df['close'], df.index, frac=0.03)[:, 1]

                # df['lowess_MACD_12_26_9'] = sm.nonparametric.lowess(df['MACD_12_26_9'], df.index, frac=1)[:, 1]
                # df['lowess_MACD_50_75_35'] = sm.nonparametric.lowess(df['MACD_50_75_35'], df.index, frac=1)[:, 1]


                # 1. Apply LOWESS to 'MACD_12_26_9'
                lowess_result_MACD_12_26_9 = sm.nonparametric.lowess(df['MACD_12_26_9'], df.index, frac=1)

                # Check if the lengths match
                if len(lowess_result_MACD_12_26_9) == len(df):
                    df['lowess_MACD_12_26_9'] = lowess_result_MACD_12_26_9[:, 1]
                else:
                    # Create a series with the same index and fill with NaN
                    lowess_series = pd.Series(np.nan, index=df.index)
                    # Replace the corresponding values with the lowess results starting from the end
                    lowess_series.iloc[-len(lowess_result_MACD_12_26_9):] = lowess_result_MACD_12_26_9[:, 1]
                    df['lowess_MACD_12_26_9'] = lowess_series


                # 2. Apply LOWESS to 'MACD_50_75_35'
                lowess_result_MACD_50_75_35 = sm.nonparametric.lowess(df['MACD_50_75_35'], df.index, frac=1)

                # Check if the lengths match
                if len(lowess_result_MACD_50_75_35) == len(df):
                    df['lowess_MACD_50_75_35'] = lowess_result_MACD_50_75_35[:, 1]
                else:
                    # Create a series with the same index and fill with NaN
                    lowess_series = pd.Series(np.nan, index=df.index)
                    # Replace the corresponding values with the lowess results starting from the end
                    lowess_series.iloc[-len(lowess_result_MACD_50_75_35):] = lowess_result_MACD_50_75_35[:, 1]
                    df['lowess_MACD_50_75_35'] = lowess_series



                df['savgol_diff'] = df['savgol'].diff()
                df['savgol_gradient_1'] = np.gradient(df['savgol'])
                df['savgol_gradient_1_diff'] = df['savgol_gradient_1'].diff()
                df['savgol_gradient_2'] = np.gradient(df['savgol_gradient_1'])

                df['lowess_diff'] = df['lowess'].diff()
                df['lowess_gradient_1'] = np.gradient(df['lowess'])
                df['lowess_gradient_1_diff'] = df['lowess_gradient_1'].diff()
                df['lowess_gradient_2'] = np.gradient(df['lowess_gradient_1'])

                df['lowess_1_diff'] = df['lowess_1'].diff()
                df['lowess_MACD_12_26_9_diff'] = df['lowess_MACD_12_26_9'].diff()
                df['lowess_MACD_50_75_35_diff'] = df['lowess_MACD_50_75_35'].diff()

                df['sma7_gradient_1'] = np.gradient(df['SMA_7'])
                df['sma7_gradient_1_diff'] = df['sma7_gradient_1'].diff()
                df['sma7_gradient_2'] = np.gradient(df['sma7_gradient_1'])


                df['ema7_gradient_1'] = np.gradient(df['EMA_7'])
                df['ema7_gradient_1_diff'] = df['ema7_gradient_1'].diff()
                df['ema7_gradient_2'] = np.gradient(df['ema7_gradient_1'])

                
                df['vwma7_gradient_1'] = np.gradient(df['EMA_7'])
                df['vwma7_gradient_1_diff'] = df['vwma7_gradient_1'].diff()
                df['vwma7_gradient_2'] = np.gradient(df['vwma7_gradient_1'])

                df['wma7_gradient_1'] = np.gradient(df['EMA_7'])
                df['wma7_gradient_1_diff'] = df['wma7_gradient_1'].diff()
                df['wma7_gradient_2'] = np.gradient(df['wma7_gradient_1'])

                
                df['ema_50_gradient_1'] = np.gradient(df['EMA_50'])
                df['ema_50_gradient_1_diff'] = df['ema_50_gradient_1'].diff()
                df['ema_50_gradient_2'] = np.gradient(df['ema_50_gradient_1'])

                df['ema_200_gradient_1'] = np.gradient(df['EMA_200'])
                df['ema_200_gradient_1_diff'] = df['ema_200_gradient_1'].diff()
                df['ema_200_gradient_2'] = np.gradient(df['ema_200_gradient_1'])
                
                ########################################################################################################################
                ########################################################################################################################

                ##############################################################################
                # # threshold = df.SMA_7 + (df.ATRr_14*3)
                # # threshold = df['BBU_20_2.0'] + (df.ATRr_14*0.5)
                # threshold = df['BBM_20_2.0'] + (df.ATRr_14*4)
                # threshold = threshold.fillna(100000000000)
                # threshold_array = threshold.values
                # high_1 = threshold_array

                # # threshold2 = (df.SMA_7 - (df.ATRr_14*3)) * -1
                # # threshold2 = (df['BBL_20_2.0'] - (df.ATRr_14*0.5)) * -1
                # threshold2 = (df['BBM_20_2.0'] - (df.ATRr_14*4)) * -1
                # threshold2 = threshold2.fillna(100000000000)
                # threshold_array2 = threshold2.values
                # high_2 = threshold_array2
                ##############################################################################


                # close_prominence = np.zeros(len(df.close))
                # close_prominence[:len(df.ATRr_1)] = df.ATRr_1
                # rsi_prominence = np.zeros(len(df.close))
                # rsi_prominence[:len(df['RSI_14'].diff())] = df['RSI_14'].diff()
                # macd_prominence = np.zeros(len(df.close))
                # macd_prominence[:len(df['MACD_12_26_9'].diff())] = df['MACD_12_26_9'].diff()

                ##############################################################################

                # close_prominence = np.zeros(len(df.close))
                # # close_prominence[:len(df.close)] = abs(df.close*3/100)
                # close_prominence[:len(df.ATRr_1)] = abs(df.ATRr_1*130/100)

                # rsi_prominence = np.zeros(len(df.close))
                # rsi_prominence[:len(df['RSI_14'])] = abs(df['RSI_14']*20/100)
                # # rsi_prominence[:len(df['RSI_14'].diff())] = abs(df['RSI_14'].diff()*1500/100)

                # macd_prominence = np.zeros(len(df.close))
                # macd_prominence[:len(df['MACD_12_26_9'])] = abs(df['MACD_12_26_9']*80/100)
                # # macd_prominence[:len(df['MACD_12_26_9'].diff())] = abs(df['MACD_12_26_9'].diff()*1000/100)


                # close_prominence = np.zeros(len(df.close))
                # # close_prominence[:len(df.close)] = abs(df.close*3/100)
                # close_prominence[:len(df.ATRr_1)] = abs(df.ATRr_1*70/100)
                ##############################################################################

                # rsi_prominence = np.zeros(len(df.close))
                # rsi_prominence[:len(df['RSI_14'])] = abs(df['RSI_14'].shift(1) * 3/100)


                # macd_prominence = np.zeros(len(df.close))
                # macd_prominence[:len(df['MACD_12_26_9'])] = abs(df['MACD_12_26_9'].shift(1) * 3/100)


                # dx_prominence = np.zeros(len(df.close))
                # dx_prominence[:len(df['DX_14'])] = abs(df['DX_14'].shift(1) * 3/100)


                # adx_prominence = np.zeros(len(df.close))
                # adx_prominence[:len(df['ADX_14'])] = abs(df['ADX_14'].shift(1) * 3/100)
                
                # atr_prominence = np.zeros(len(df.close))
                # atr_prominence[:len(df['ATRr_1'])] = abs(df['ATRr_1'].shift(1) * 2/100)
                
                # atr_prominence[:len(df['ATRr_14'])] = abs(df['ATRr_14'].shift(1) * 2/100)


                rsi_prominence = np.zeros(len(df.close))
                rsi_prominence[:len(df['RSI_14'])] = abs(df['RSI_14'].shift(1) * 1/100)

                MACD_12_26_9_prominence = np.zeros(len(df.close))
                MACD_12_26_9_prominence[:len(df['MACD_12_26_9'])] = abs(df['MACD_12_26_9'].shift(1) * 3/100)

                MACD_50_75_35_prominence = np.zeros(len(df.close))
                MACD_50_75_35_prominence[:len(df['MACD_50_75_35'])] = abs(df['MACD_50_75_35'].shift(1) * 3/100)

                dx_prominence = np.zeros(len(df.close))
                dx_prominence[:len(df['DX_200'])] = abs(df['DX_200'].shift(1) * 3/100)


                adx_prominence = np.zeros(len(df.close))
                adx_prominence[:len(df['ADX_14'])] = abs(df['ADX_14'].shift(1) * 1/100)
                
                atr_prominence = np.zeros(len(df.close))
                # atr_prominence[:len(df['ATRr_1'])] = abs(df['ATRr_1'].shift(1) * 1/100)
                atr_prominence[:len(df['ATRr_14'])] = abs(df['ATRr_14'].shift(1) * 1/100)
                
                obv_prominence = np.zeros(len(df.close))
                # obv_prominence[:len(df['ATRr_1'])] = abs(df['ATRr_1'].shift(1) * 1/100)
                obv_prominence[:len(df['OBV'])] = abs(df['OBV'].shift(1) * 2/100)

                combined_diff_prominence = np.zeros(len(df.close))
                # combined_diff_prominence[:len(df['ATRr_1'])] = abs(df['ATRr_1'].shift(1) * 1/100)
                combined_diff_prominence[:len(df['combined_diff'])] = abs(df['combined_diff'].shift(1) * 1/100)

                ##############################################################################

                maxima_peak_x_open, maxima_peak_y_open, minima_peak_x_open, minima_peak_y_open, maxima_peak_x_high, maxima_peak_y_high, minima_peak_x_high, minima_peak_y_high, maxima_peak_x_low, maxima_peak_y_low, minima_peak_x_low, minima_peak_y_low, maxima_peak_x_close, maxima_peak_y_close, minima_peak_x_close, minima_peak_y_close, maxima_peak_x_rsi, maxima_peak_y_rsi, minima_peak_x_rsi, minima_peak_y_rsi, maxima_peak_x_MACD_12_26_9, maxima_peak_y_MACD_12_26_9, minima_peak_x_MACD_12_26_9, minima_peak_y_MACD_12_26_9, maxima_peak_x_MACD_50_75_35, maxima_peak_y_MACD_50_75_35, minima_peak_x_MACD_50_75_35, minima_peak_y_MACD_50_75_35, maxima_peak_x_adx, maxima_peak_y_adx, minima_peak_x_adx, minima_peak_y_adx, maxima_peak_x_dx, maxima_peak_y_dx, minima_peak_x_dx, minima_peak_y_dx, maxima_peak_x_dmp, maxima_peak_y_dmp, minima_peak_x_dmp, minima_peak_y_dmp, maxima_peak_x_dmn, maxima_peak_y_dmn, minima_peak_x_dmn, minima_peak_y_dmn, maxima_peak_x_kdj, maxima_peak_y_kdj, minima_peak_x_kdj, minima_peak_y_kdj, maxima_peak_x_obv, maxima_peak_y_obv, minima_peak_x_obv, minima_peak_y_obv, maxima_peak_x_atr, maxima_peak_y_atr, minima_peak_x_atr, minima_peak_y_atr, maxima_peak_x_atr14, maxima_peak_y_atr14, minima_peak_x_atr14, minima_peak_y_atr14, maxima_peak_x_atr_p, maxima_peak_y_atr_p, minima_peak_x_atr_p, minima_peak_y_atr_p, maxima_peak_x_combined_diff, maxima_peak_y_combined_diff, minima_peak_x_combined_diff, minima_peak_y_combined_diff, maxima_peak_x_combined_diff_filtered, maxima_peak_y_combined_diff_filtered, minima_peak_x_combined_diff_filtered, minima_peak_y_combined_diff_filtered, maxima_peak_x_second_combined_diff, maxima_peak_y_second_combined_diff, minima_peak_x_second_combined_diff, minima_peak_y_second_combined_diff, maxima_peak_x_second_combined_diff_filtered, maxima_peak_y_second_combined_diff_filtered, minima_peak_x_second_combined_diff_filtered, minima_peak_y_second_combined_diff_filtered = maxima_minima_calc(high_1, high_2, rsi_prominence, MACD_12_26_9_prominence, MACD_50_75_35_prominence, dx_prominence, adx_prominence, atr_prominence, obv_prominence, combined_diff_prominence, df['open_time2'], df.open, df.high, df.low, df.close, df.RSI_14, df.MACD_12_26_9, df.MACD_50_75_35, df.ADX_14, df.DX_200, df.DMP_14, df.DMN_14, df.J_9_3, df.OBV, df.ATRr_1, df.ATRr_14, df.atr_p, df.combined_diff, df.combined_diff_filtered, df.second_combined_diff, df.second_combined_diff_filtered)


                # ##############################################################################
                # # Create a findpeaks object
                # df['findpeaks_peak'] = 0
                # df['findpeaks_valley'] = 0

                # # tqdm.tqdm = findpeaks.no_tqdm
                
                # fp = findpeaks(method='caerus', params_caerus={'minperc':0.2, 'window':50}, verbose=0)
                # # tqdm_notebook().pandas(disable=True)


                # # Fit the data and find peaks
                # results = fp.fit(df.close)
                # # print(interval, 'maxima_divergences: ', maxima_divergences)
                # #print(interval, 'results: ', results)

                # findpeaks_results = results['df']
                # peaks = findpeaks_results[findpeaks_results['peak']]['x']
                # valleys = findpeaks_results[findpeaks_results['valley']]['x']
                # # print(interval, results)


                # # print(peaks)
                # # print(valleys)

                # # from result
                # # findpeaks_results.loc[peaks, 'y']
                # # findpeaks_results.loc[valleys, 'y']

                # # from original df
                # # df_1m.iloc[peaks]
                # # df_1m.iloc[valleys]
                # # df_1m['open_time2'].iloc[peaks] # x, open_time2
                # # df_1m['open_time2'].iloc[valleys] # x, open_time2
                # # df['close'].iloc[peaks] # y, close
                # # df['close'].iloc[valleys] # y, close

                # # df.loc[(df.iloc[peaks]), "findpeaks_peak"] = -2 # without atr
                # # df.loc[(df.iloc[valleys]), "findpeaks_valley"] = 2 # without atr

                # df.loc[df.index[peaks], 'findpeaks_peak'] = -1
                # df.loc[df.index[valleys], 'findpeaks_valley'] = 1




                # #############################################################################################################################################

                # # Create a findpeaks object
                # df['findpeaks_peak_rsi'] = 0
                # df['findpeaks_valley_rsi'] = 0
                
                # fp_rsi = findpeaks(method='caerus', params_caerus={'minperc':0.2, 'window':50}, verbose=0)

                # # Fit the data and find peaks
                # results_rsi = fp_rsi.fit(df.RSI_14)
                # findpeaks_results_rsi = results_rsi['df']
                # peaks_rsi = findpeaks_results_rsi[findpeaks_results_rsi['peak']]['x']
                # valleys_rsi = findpeaks_results_rsi[findpeaks_results_rsi['valley']]['x']

                # df.loc[df.index[peaks_rsi], 'findpeaks_peak_rsi'] = -1
                # df.loc[df.index[valleys_rsi], 'findpeaks_valley_rsi'] = 1


                # #############################################################################################################################################


                # # Create a findpeaks object
                # df['findpeaks_peak_macd'] = 0
                # df['findpeaks_valley_macd'] = 0
                
                # fp_macd = findpeaks(method='caerus', params_caerus={'minperc':0.2, 'window':50}, verbose=0)

                # # Fit the data and find peaks
                # results_macd = fp_macd.fit(df.MACD_12_26_9)
                # findpeaks_results_macd = results_macd['df']
                # peaks_macd = findpeaks_results_macd[findpeaks_results_macd['peak']]['x']
                # valleys_macd = findpeaks_results_macd[findpeaks_results_macd['valley']]['x']

                # df.loc[df.index[peaks_macd], 'findpeaks_peak_macd'] = -1
                # df.loc[df.index[valleys_macd], 'findpeaks_valley_macd'] = 1


                # #############################################################################################################################################


                # # Create a findpeaks object
                # df['findpeaks_peak_kdj'] = 0
                # df['findpeaks_valley_kdj'] = 0
                
                # fp_kdj = findpeaks(method='caerus', params_caerus={'minperc':0.2, 'window':50}, verbose=0)

                # # Fit the data and find peaks
                # results_kdj = fp_kdj.fit(df.J_9_3)
                # findpeaks_results_kdj = results_kdj['df']
                # peaks_kdj = findpeaks_results_kdj[findpeaks_results_kdj['peak']]['x']
                # valleys_kdj = findpeaks_results_kdj[findpeaks_results_kdj['valley']]['x']

                # df.loc[df.index[peaks_kdj], 'findpeaks_peak_kdj'] = -1
                # df.loc[df.index[valleys_kdj], 'findpeaks_valley_kdj'] = 1


                #############################################################################################################################################


                # # Create a findpeaks object
                # df['findpeaks_peak_atr'] = 0
                # df['findpeaks_valley_atr'] = 0
                
                # fp_atr = findpeaks(method='caerus', params_caerus={'minperc':0.2, 'window':50}, verbose=0)

                # # Fit the data and find peaks
                # results_atr = fp_atr.fit(df.ATRr_14)
                # findpeaks_results_atr = results_atr['df']
                # peaks_atr = findpeaks_results_atr[findpeaks_results_atr['peak']]['x']
                # valleys_atr = findpeaks_results_atr[findpeaks_results_atr['valley']]['x']

                # df.loc[df.index[peaks_atr], 'findpeaks_peak_atr'] = -1
                # df.loc[df.index[valleys_atr], 'findpeaks_valley_atr'] = 1

                #############################################################################################################################################

                # # Create a findpeaks object
                # df['findpeaks_peak_adx'] = 0
                # df['findpeaks_valley_adx'] = 0
                
                # fp_adx = findpeaks(method='caerus', params_caerus={'minperc':0.2, 'window':50}, verbose=0)

                # # Fit the data and find peaks
                # results_adx = fp_adx.fit(df.ADX_14)
                # findpeaks_results_adx = results_adx['df']
                # peaks_adx = findpeaks_results_adx[findpeaks_results_adx['peak']]['x']
                # valleys_adx = findpeaks_results_adx[findpeaks_results_adx['valley']]['x']

                # df.loc[df.index[peaks_adx], 'findpeaks_peak_adx'] = -1
                # df.loc[df.index[valleys_adx], 'findpeaks_valley_adx'] = 1


                #############################################################################################################################################
                
                # # Create a findpeaks object
                # df['findpeaks_peak_dx'] = 0
                # df['findpeaks_valley_dx'] = 0
                
                # fp_dx = findpeaks(method='caerus', params_caerus={'minperc':0.2, 'window':50}, verbose=0)

                # # Fit the data and find peaks
                # results_dx = fp_dx.fit(df.DX_14)
                # findpeaks_results_dx = results_dx['df']
                # peaks_dx = findpeaks_results_dx[findpeaks_results_dx['peak']]['x']
                # valleys_dx = findpeaks_results_dx[findpeaks_results_dx['valley']]['x']

                # df.loc[df.index[peaks_dx], 'findpeaks_peak_dx'] = -1
                # df.loc[df.index[valleys_dx], 'findpeaks_valley_dx'] = 1

                #############################################################################################################################################


                # # Create a findpeaks object
                # df['findpeaks_peak_dmp'] = 0
                # df['findpeaks_valley_dmp'] = 0
                
                # fp_dmp = findpeaks(method='caerus', params_caerus={'minperc':0.2, 'window':50}, verbose=0)

                # # Fit the data and find peaks
                # results_dmp = fp_dmp.fit(df.DMP_14)
                # findpeaks_results_dmp = results_dmp['df']
                # peaks_dmp = findpeaks_results_dmp[findpeaks_results_dmp['peak']]['x']
                # valleys_dmp = findpeaks_results_dmp[findpeaks_results_dmp['valley']]['x']

                # df.loc[df.index[peaks_dmp], 'findpeaks_peak_dmp'] = -1
                # df.loc[df.index[valleys_dmp], 'findpeaks_valley_dmp'] = 1

                # #############################################################################################################################################


                # # Create a findpeaks object
                # df['findpeaks_peak_dmn'] = 0
                # df['findpeaks_valley_dmn'] = 0
                
                # fp_dmn = findpeaks(method='caerus', params_caerus={'minperc':0.2, 'window':50}, verbose=0)

                # # Fit the data and find peaks
                # results_dmn = fp_dmn.fit(df.DMN_14)
                # findpeaks_results_dmn = results_dmn['df']
                # peaks_dmn = findpeaks_results_dmn[findpeaks_results_dmn['peak']]['x']
                # valleys_dmn = findpeaks_results_dmn[findpeaks_results_dmn['valley']]['x']

                # df.loc[df.index[peaks_dmn], 'findpeaks_peak_dmn'] = -1
                # df.loc[df.index[valleys_dmn], 'findpeaks_valley_dmn'] = 1
                # #############################################################################################################################################




                # df.loc[(df.findpeaks_peak_dmp < 0) | (df.findpeaks_valley_dmn < 0), "findpeaks_peak_dmp_with_dmn"] = -1
                # df.loc[(df.findpeaks_valley_dmp > 0) | (df.findpeaks_peak_dmn > 0), "findpeaks_valley_dmp_with_dmn"] = 1




                # # end = time.time()
                # # total_time = end - start
                # # print(total_time)


                # #############################################################################################################################################


                df['maxima_peak_x_open'] = 0
                df['minima_peak_x_open'] = 0
                df['maxima_peak_x_high'] = 0
                df['minima_peak_x_high'] = 0
                df['maxima_peak_x_low'] = 0
                df['minima_peak_x_low'] = 0
                df['maxima_peak_x_close'] = 0
                df['minima_peak_x_close'] = 0
                df['maxima_peak_x_rsi'] = 0
                df['minima_peak_x_rsi'] = 0
                df['maxima_peak_x_MACD_12_26_9'] = 0
                df['minima_peak_x_MACD_12_26_9'] = 0
                df['maxima_peak_x_MACD_50_75_35'] = 0
                df['minima_peak_x_MACD_50_75_35'] = 0
                df['maxima_peak_x_adx'] = 0
                df['minima_peak_x_adx'] = 0
                df['maxima_peak_x_dx'] = 0
                df['minima_peak_x_dx'] = 0                
                df['maxima_peak_x_dmp'] = 0
                df['minima_peak_x_dmp'] = 0
                df['maxima_peak_x_dmn'] = 0
                df['minima_peak_x_dmn'] = 0
                df['maxima_peak_x_kdj'] = 0
                df['minima_peak_x_kdj'] = 0
                df['maxima_peak_x_obv'] = 0
                df['minima_peak_x_obv'] = 0
                df['maxima_peak_x_atr'] = 0
                df['minima_peak_x_atr'] = 0
                df['maxima_peak_x_atr14'] = 0
                df['minima_peak_x_atr14'] = 0
                df['maxima_peak_x_atr_p'] = 0
                df['minima_peak_x_atr_p'] = 0
                df['maxima_peak_x_dmp_with_dmn'] = 0
                df['minima_peak_x_dmp_with_dmn'] = 0
                df['maxima_peak_x_combined_diff'] = 0
                df['minima_peak_x_combined_diff'] = 0
                df['maxima_peak_x_combined_diff_filtered'] = 0
                df['minima_peak_x_combined_diff_filtered'] = 0
                df['maxima_peak_x_second_combined_diff'] = 0
                df['minima_peak_x_second_combined_diff'] = 0
                df['maxima_peak_x_second_second_combined_diff_filtered'] = 0
                df['minima_peak_x_second_second_combined_diff_filtered'] = 0
                #df['total_short'] = 0
                #df['total_long'] = 0
                df['stg1_short'] = 0
                df['stg1_long'] = 0
                df['stg2_short'] = 0
                df['stg2_long'] = 0
                df['stg3_short'] = 0
                df['stg3_long'] = 0
                df['stg4_short'] = 0
                df['stg4_long'] = 0
                df['stg5_short'] = 0
                df['stg5_long'] = 0
                df['stg6_short'] = 0
                df['stg6_long'] = 0
                df['stg7_short'] = 0
                df['stg7_long'] = 0
                df['stg8_short'] = 0
                df['stg8_long'] = 0
                df['stg9_short'] = 0
                df['stg9_long'] = 0                
                df['stg10_short'] = 0
                df['stg10_long'] = 0
                df['stg100_short'] = 0
                df['stg100_long'] = 0
                df['stg110_short'] = 0
                df['stg110_long'] = 0
                df['stgT_short'] = 0
                df['stgT_long'] = 0
                df['stgU_short'] = 0
                df['stgU_long'] = 0
                df['stgQ_short'] = 0
                df['stgQ_long'] = 0
                df['maxima_peak_x_macd_with_condition'] = 0
                df['minima_peak_x_macd_with_condition'] = 0
                df['maxima_peak_x_macd_short_only_when_greater_than_zero'] = 0 # short only when 'df.MACD_12_26_9 > 0'
                df['minima_peak_x_macd_long_only_when_less_than_zero'] = 0 # long only when 'df.MACD_12_26_9 < 0'
                df.loc[maxima_peak_x_open, "maxima_peak_x_open"] = -1
                df.loc[minima_peak_x_open, "minima_peak_x_open"] = 1
                df.loc[maxima_peak_x_high, "maxima_peak_x_high"] = -1
                df.loc[minima_peak_x_high, "minima_peak_x_high"] = 1
                df.loc[maxima_peak_x_low, "maxima_peak_x_low"] = -1
                df.loc[minima_peak_x_low, "minima_peak_x_low"] = 1
                df.loc[maxima_peak_x_close, "maxima_peak_x_close"] = -1
                df.loc[minima_peak_x_close, "minima_peak_x_close"] = 1
                df.loc[maxima_peak_x_rsi, "maxima_peak_x_rsi"] = -1
                df.loc[minima_peak_x_rsi, "minima_peak_x_rsi"] = 1
                df.loc[maxima_peak_x_MACD_12_26_9, "maxima_peak_x_MACD_12_26_9"] = -1
                df.loc[minima_peak_x_MACD_12_26_9, "minima_peak_x_MACD_12_26_9"] = 1
                df.loc[maxima_peak_x_MACD_50_75_35, "maxima_peak_x_MACD_50_75_35"] = -1
                df.loc[minima_peak_x_MACD_50_75_35, "minima_peak_x_MACD_50_75_35"] = 1
                df.loc[maxima_peak_x_adx, "maxima_peak_x_adx"] = -1
                df.loc[minima_peak_x_adx, "minima_peak_x_adx"] = 1
                df.loc[maxima_peak_x_dx, "maxima_peak_x_dx"] = -1
                df.loc[minima_peak_x_dx, "minima_peak_x_dx"] = 1                
                df.loc[maxima_peak_x_dmp, "maxima_peak_x_dmp"] = -1
                df.loc[minima_peak_x_dmp, "minima_peak_x_dmp"] = 1
                df.loc[maxima_peak_x_dmn, "maxima_peak_x_dmn"] = 1
                df.loc[minima_peak_x_dmn, "minima_peak_x_dmn"] = -1
                df.loc[maxima_peak_x_kdj, "maxima_peak_x_kdj"] = -1
                df.loc[minima_peak_x_kdj, "minima_peak_x_kdj"] = 1
                df.loc[maxima_peak_x_obv, "maxima_peak_x_obv"] = -1
                df.loc[minima_peak_x_obv, "minima_peak_x_obv"] = 1
                df.loc[maxima_peak_x_atr, "maxima_peak_x_atr"] = -1 # 횡보시작
                df.loc[minima_peak_x_atr, "minima_peak_x_atr"] = 1 # 변동시작
                df.loc[maxima_peak_x_atr14, "maxima_peak_x_atr14"] = -1 # 횡보시작
                df.loc[minima_peak_x_atr14, "minima_peak_x_atr14"] = 1 # 변동시작
                df.loc[maxima_peak_x_atr14, "maxima_peak_x_atr_p"] = -1 # 횡보시작
                df.loc[minima_peak_x_atr14, "minima_peak_x_atr_p"] = 1 # 변동시작
                df.loc[(df.maxima_peak_x_dmp < 0) | (df.minima_peak_x_dmn < 0), "maxima_peak_x_dmp_with_dmn"] = -1
                df.loc[(df.minima_peak_x_dmp > 0) | (df.maxima_peak_x_dmn > 0), "minima_peak_x_dmp_with_dmn"] = 1


                df.loc[maxima_peak_x_combined_diff, "maxima_peak_x_combined_diff"] = -1 # 횡보시작
                df.loc[minima_peak_x_combined_diff, "minima_peak_x_combined_diff"] = 1 # 변동시작
                df.loc[maxima_peak_x_combined_diff_filtered, "maxima_peak_x_combined_diff_filtered"] = -1 # 횡보시작
                df.loc[minima_peak_x_combined_diff_filtered, "minima_peak_x_combined_diff_filtered"] = 1 # 변동시작
                df.loc[maxima_peak_x_second_combined_diff, "maxima_peak_x_second_combined_diff"] = -1 # 횡보시작
                df.loc[minima_peak_x_second_combined_diff, "minima_peak_x_second_combined_diff"] = 1 # 변동시작
                df.loc[maxima_peak_x_second_combined_diff_filtered, "maxima_peak_x_second_combined_diff_filtered"] = -1 # 횡보시작
                df.loc[minima_peak_x_second_combined_diff_filtered, "minima_peak_x_second_combined_diff_filtered"] = 1 # 변동시작


                df['anomalies_open'] = 0
                df['anomalies_high'] = 0
                df['anomalies_low'] = 0
                df['anomalies_close'] = 0
                df['anomalies_DX_14'] = 0 # dx
                df['anomalies_RSI_14'] = 0 # rsi
                df['anomalies_ATRr_14'] = 0 # dx
                df['anomalies_ATRr_1'] = 0 # rsi

                # maxima_peak_x_atr 최대 3번전까지 카운트 하기위해 df['maxima_peak_x_atr_last_3'] column 신규 생성 + ATRr_1 값 df['maxima_peak_x_atr_last_3_ATRr_1'] column 으로 복사
                mask = (df['maxima_peak_x_atr'] == -1) & (df['ATRr_1'] > df.close*0.7/100)
                df['maxima_peak_x_atr_last_3'] = df['maxima_peak_x_atr'].where(mask).fillna(0).rolling(window=3, min_periods=1).apply(lambda x: any(x == -1) and -1 or 0).astype(int) # Create the 'b2' column using the rolling window
                df['maxima_peak_x_atr_last_3_ATRr_1'] = df['ATRr_1'].where(mask).fillna(0).rolling(window=3, min_periods=1).max() # Create the 'c2' column based on the rolling maximum of 'c' for rows where 'b' is -1
                df['maxima_peak_x_atr_last_3_ATRr_1'] = df['maxima_peak_x_atr_last_3_ATRr_1'].fillna(0) # Fill NaN values in 'c2' with 0

                mask2 = (df['minima_peak_x_atr'] == 1) # & (df['ATRr_1'] > df.close*1/100)
                df['minima_peak_x_atr_last_3'] = df['minima_peak_x_atr'].where(mask2).fillna(0).rolling(window=3, min_periods=1).apply(lambda x: any(x == 1) and 1 or 0).astype(int) # Create the 'b2' column using the rolling window
                df['minima_peak_x_atr_last_3_ATRr_1'] = df['ATRr_1'].where(mask2).fillna(0).rolling(window=3, min_periods=1).max() # Create the 'c2' column based on the rolling maximum of 'c' for rows where 'b' is -1
                df['minima_peak_x_atr_last_3_ATRr_1'] = df['minima_peak_x_atr_last_3_ATRr_1'].fillna(0) # Fill NaN values in 'c2' with 0

                mask3 = (df['maxima_peak_x_rsi'] == -1) #& (df['RSI_14'] > 60)
                df['maxima_peak_x_rsi_last_3'] = df['maxima_peak_x_rsi'].where(mask3).fillna(0).rolling(window=3, min_periods=1).apply(lambda x: any(x == -1) and -1 or 0).astype(int) # Create the 'b2' column using the rolling window
                df['maxima_peak_x_rsi_last_3_RSI_14'] = df['RSI_14'].where(mask3).fillna(0).rolling(window=3, min_periods=1).max() # Create the 'c2' column based on the rolling maximum of 'c' for rows where 'b' is -1
                df['maxima_peak_x_rsi_last_3_RSI_14'] = df['maxima_peak_x_rsi_last_3_RSI_14'].fillna(0) # Fill NaN values in 'c2' with 0

                mask4 = (df['minima_peak_x_rsi'] == 1)  #& (df['RSI_14'] > df.close*1/100)
                df['minima_peak_x_rsi_last_3'] = df['minima_peak_x_rsi'].where(mask4).fillna(0).rolling(window=3, min_periods=1).apply(lambda x: any(x == 1) and 1 or 0).astype(int) # Create the 'b2' column using the rolling window
                df['minima_peak_x_rsi_last_3_RSI_14'] = df['RSI_14'].where(mask4).fillna(0).rolling(window=3, min_periods=1).max() # Create the 'c2' column based on the rolling maximum of 'c' for rows where 'b' is -1
                df['minima_peak_x_rsi_last_3_RSI_14'] = df['minima_peak_x_rsi_last_3_RSI_14'].fillna(0) # Fill NaN values in 'c2' with 0



                mask5 = (df['maxima_peak_x_MACD_12_26_9'] == -1) #& (df['MACD_12_26_9'] > 60)
                df['maxima_peak_x_macd_last_3'] = df['maxima_peak_x_MACD_12_26_9'].where(mask5).fillna(0).rolling(window=3, min_periods=1).apply(lambda x: any(x == -1) and -1 or 0).astype(int) # Create the 'b2' column using the rolling window
                df['maxima_peak_x_macd_last_3_MACD_12_26_9'] = df['MACD_12_26_9'].where(mask5).fillna(0).rolling(window=3, min_periods=1).max() # Create the 'c2' column based on the rolling maximum of 'c' for rows where 'b' is -1
                df['maxima_peak_x_macd_last_3_MACD_12_26_9'] = df['maxima_peak_x_macd_last_3_MACD_12_26_9'].fillna(0) # Fill NaN values in 'c2' with 0

                mask6 = (df['minima_peak_x_MACD_12_26_9'] == 1)  #& (df['MACD_12_26_9'] > df.close*1/100)
                df['minima_peak_x_macd_last_3'] = df['minima_peak_x_MACD_12_26_9'].where(mask6).fillna(0).rolling(window=3, min_periods=1).apply(lambda x: any(x == 1) and 1 or 0).astype(int) # Create the 'b2' column using the rolling window
                df['minima_peak_x_macd_last_3_MACD_12_26_9'] = df['MACD_12_26_9'].where(mask6).fillna(0).rolling(window=3, min_periods=1).max() # Create the 'c2' column based on the rolling maximum of 'c' for rows where 'b' is -1
                df['minima_peak_x_macd_last_3_MACD_12_26_9'] = df['minima_peak_x_macd_last_3_MACD_12_26_9'].fillna(0) # Fill NaN values in 'c2' with 0


                mask7 = (df['maxima_peak_x_dx'] == -1) & (df['DX_14'] > 40)
                df['maxima_peak_x_dx_last_3'] = df['maxima_peak_x_dx'].where(mask7).fillna(0).rolling(window=3, min_periods=1).apply(lambda x: any(x == -1) and -1 or 0).astype(int) # Create the 'b2' column using the rolling window
                df['maxima_peak_x_dx_last_3_DX_14'] = df['DX_14'].where(mask7).fillna(0).rolling(window=3, min_periods=1).max() # Create the 'c2' column based on the rolling maximum of 'c' for rows where 'b' is -1
                df['maxima_peak_x_dx_last_3_DX_14'] = df['maxima_peak_x_dx_last_3_DX_14'].fillna(0) # Fill NaN values in 'c2' with 0

                mask8 = (df['minima_peak_x_dx'] == 1)  #& (df['DX_14'] > df.close*1/100)
                df['minima_peak_x_dx_last_3'] = df['minima_peak_x_dx'].where(mask8).fillna(0).rolling(window=3, min_periods=1).apply(lambda x: any(x == 1) and 1 or 0).astype(int) # Create the 'b2' column using the rolling window
                df['minima_peak_x_dx_last_3_DX_14'] = df['DX_14'].where(mask8).fillna(0).rolling(window=3, min_periods=1).max() # Create the 'c2' column based on the rolling maximum of 'c' for rows where 'b' is -1
                df['minima_peak_x_dx_last_3_DX_14'] = df['minima_peak_x_dx_last_3_DX_14'].fillna(0) # Fill NaN values in 'c2' with 0









                ###################################################################################################
                ###################################################################################################
                ###################################################################################################
                ###################################################################################################
                ###################################################################################################
                # divergence 산출
                df['divergence_point_2'] = 0
                df['divergence_name'] = 'none'
                df['divergence_confirmer'] = 'none'
                df['divergence_point_1'] = 0
                df['divergence_open_time_1'] = 'none'
                df['divergence_close_1'] = 'none'
                df['divergence_rsi_1'] = 'none'
                df['divergence_open_time_2'] = 'none'
                df['divergence_close_2'] = 'none'
                df['divergence_rsi_2'] = 'none'

                ######################^^^^^^ 다이버 이용시 주석 해제  필요 ^^^^^^#############################################################################
                ######################^^^^^^ 다이버 이용시 주석 해제  필요 ^^^^^^#############################################################################
                ######################^^^^^^ 다이버 이용시 주석 해제  필요 ^^^^^^#############################################################################
                ######################^^^^^^ 다이버 이용시 주석 해제  필요 ^^^^^^#############################################################################
                ######################^^^^^^ 다이버 이용시 주석 해제  필요 ^^^^^^#############################################################################
                ######################^^^^^^ 다이버 이용시 주석 해제  필요 ^^^^^^#############################################################################
                
                # R_bull_minima_divergences = handle_divergence(interval, df, 'R_bull_minima')
                # R_bear_maxima_divergences = handle_divergence(interval, df, 'R_bear_maxima')
                # # H_bull_minima_divergences = handle_divergence(interval, df, 'H_bull_minima')
                # # H_bear_maxima_divergences = handle_divergence(interval, df, 'H_bear_maxima')



                # for divergence in R_bull_minima_divergences:
                #     divergence_confirmer, divergence_name, divergence_point_1, divergence_open_time_1, divergence_close_1, divergence_rsi_1, divergence_point_2, divergence_open_time_2, divergence_close_2, divergence_rsi_2 = divergence
                #     df.loc[df['open_time2'] == divergence_open_time_2, ['divergence_confirmer', 'divergence_name', 'divergence_point_1', 'divergence_open_time_1', 'divergence_close_1', 'divergence_rsi_1', 'divergence_point_2', 'divergence_open_time_2', 'divergence_close_2', 'divergence_rsi_2']] = divergence_confirmer, divergence_name, divergence_point_1, divergence_open_time_1, divergence_close_1, divergence_rsi_1, divergence_point_2, divergence_open_time_2, divergence_close_2, divergence_rsi_2

                # for divergence in R_bear_maxima_divergences:
                #     divergence_confirmer, divergence_name, divergence_point_1, divergence_open_time_1, divergence_close_1, divergence_rsi_1, divergence_point_2, divergence_open_time_2, divergence_close_2, divergence_rsi_2 = divergence
                #     df.loc[df['open_time2'] == divergence_open_time_2, ['divergence_confirmer', 'divergence_name', 'divergence_point_1', 'divergence_open_time_1', 'divergence_close_1', 'divergence_rsi_1', 'divergence_point_2', 'divergence_open_time_2', 'divergence_close_2', 'divergence_rsi_2']] = divergence_confirmer, divergence_name, divergence_point_1, divergence_open_time_1, divergence_close_1, divergence_rsi_1, divergence_point_2, divergence_open_time_2, divergence_close_2, divergence_rsi_2
                ######################^^^^^^ 다이버 이용시 주석 해제  필요 ^^^^^^#############################################################################
                ######################^^^^^^ 다이버 이용시 주석 해제  필요 ^^^^^^#############################################################################
                ######################^^^^^^ 다이버 이용시 주석 해제  필요 ^^^^^^#############################################################################
                ######################^^^^^^ 다이버 이용시 주석 해제  필요 ^^^^^^#############################################################################
                ######################^^^^^^ 다이버 이용시 주석 해제  필요 ^^^^^^#############################################################################
                ######################^^^^^^ 다이버 이용시 주석 해제  필요 ^^^^^^#############################################################################

                # for divergence in H_bull_minima_divergences:
                #     divergence_confirmer, divergence_name, divergence_point_1, divergence_open_time_1, divergence_close_1, divergence_rsi_1, divergence_point_2, divergence_open_time_2, divergence_close_2, divergence_rsi_2 = divergence
                #     df.loc[df['open_time2'] == divergence_open_time_2, ['divergence_confirmer', 'divergence_name', 'divergence_point_1', 'divergence_open_time_1', 'divergence_close_1', 'divergence_rsi_1', 'divergence_point_2', 'divergence_open_time_2', 'divergence_close_2', 'divergence_rsi_2']] = divergence_confirmer, divergence_name, divergence_point_1, divergence_open_time_1, divergence_close_1, divergence_rsi_1, divergence_point_2, divergence_open_time_2, divergence_close_2, divergence_rsi_2

                # for divergence in H_bear_maxima_divergences:
                #     divergence_confirmer, divergence_name, divergence_point_1, divergence_open_time_1, divergence_close_1, divergence_rsi_1, divergence_point_2, divergence_open_time_2, divergence_close_2, divergence_rsi_2 = divergence
                #     df.loc[df['open_time2'] == divergence_open_time_2, ['divergence_confirmer', 'divergence_name', 'divergence_point_1', 'divergence_open_time_1', 'divergence_close_1', 'divergence_rsi_1', 'divergence_point_2', 'divergence_open_time_2', 'divergence_close_2', 'divergence_rsi_2']] = divergence_confirmer, divergence_name, divergence_point_1, divergence_open_time_1, divergence_close_1, divergence_rsi_1, divergence_point_2, divergence_open_time_2, divergence_close_2, divergence_rsi_2
















                ########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
                ###################################################################################################
                ###################################################################################################



                iqr_close = None
                iqr_feature1 = None
                iqr_feature2 = None

                df['feature1'] = 0
                df['feature2'] = 0

                df['anomalies_close'] = False
                df['anomalies_feature1'] = False
                df['anomalies_feature2'] = False

                df['maxima_peak_x_anomalies_close'] = 0
                df['minima_peak_x_anomalies_close'] = 0
                df['maxima_peak_x_anomalies_feature1'] = 0
                df['minima_peak_x_anomalies_feature1'] = 0
                df['maxima_peak_x_anomalies_feature2'] = 0
                df['minima_peak_x_anomalies_feature2'] = 0



                df_cleaned = df[[ \


                    'open', \
                    'high', \
                    'low', \
                    'close', \
                    'volume', \
                    'volume2', \
                    'RSI_14', \
                    'MACD_12_26_9', \
                    'MACDh_12_26_9', \
                    'DMP_14', \
                    'DMN_14', \
                    'DX_14', \
                    'ADX_14', \
                    'OBV', \
                    'J_9_3', \
                    'BBL_21_2.0', \
                    'BBM_21_2.0', \
                    'BBU_21_2.0', \
                    'BBB_21_2.0', \
                    'BBP_21_2.0', \
                    'ATRr_1', \
                    'ATRr_14', \
                    'WMA_7', \
                    'VWMA_7', \
                    'SMA_7', \
                    'EMA_7', \
                    'volume_diff', \
                    'STOCHRSIk_14_14_3_3', \
                    'WILLR_14', \
                    'close_change', \
                    'obv_percentage_change', \
                    'atr_percentage_change', \
                    'volume_change', \
                    'macd_percentage_change', \
                    'rsi_percentage_change', \
                    'adx_percentage_change', \
                    'dx_percentage_change', \





                    
                ]].select_dtypes(include=np.number).dropna()


                # # 데이터 준비
                # columns = df_cleaned.columns
                # X = df_cleaned[columns].values

                # 데이터 준비
                columns = df_cleaned.columns
                # X = df_cleaned[columns].values
                desired_columns = ['BBB_21_2.0', 'volume', 'ATRr_1']
                # desired_columns = ['ATRr_1', 'ATRr_14']
                X = df_cleaned[desired_columns].values
                




                # 데이터 전처리
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(X)

                # 저장된 모델 불러오기
                # print(interval)
                loaded_autoencoder = load_model("/aws/engin1/model_1.h5")
                
                # loaded_autoencoder.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
                # Compile the model (assuming you have defined loss and optimizer during training)


                # 학습된 오토인코더를 사용하여 데이터를 변환
                # 수정된 부분: scaled_data 대신에 새로운 데이터를 사용하여 변환
                encoded_data_mlp = loaded_autoencoder.predict(scaled_data)  # 수정된 부분: 저장된 모델을 사용하여 데이터 변환

                # PCA를 사용하여 데이터를 2차원으로 축소
                pca = PCA(n_components=2)
                encoded_data_2d = pca.fit_transform(encoded_data_mlp)

                df.loc[df_cleaned.index, 'feature1'] = encoded_data_2d[:, 0]  # df_cleaned의 인덱스를 기준으로 값 할당
                df.loc[df_cleaned.index, 'feature2'] = encoded_data_2d[:, 1]

                df['feature1_diff'] = df['feature1'].diff()
                df['feature2_diff'] = df['feature2'].diff()
                df['feature1_percentage_change'] = np.where(df['feature1_diff'].abs().shift() != 0, (df['feature1_diff'].diff() / df['feature1_diff'].abs().shift()) * 100, 0)
                df['feature2_percentage_change'] = np.where(df['feature2_diff'].abs().shift() != 0, (df['feature2_diff'].diff() / df['feature2_diff'].abs().shift()) * 100, 0)

                ########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
                ###################################################################################################
                ###################################################################################################




                # 데이터프레임 선택 및 유효성 검사
                # df = validate_series(df_cleaned[['close', 'feature1', 'feature2']])



                # MinClusterDetector 생성 및 학습
                # min_cluster_detector = MinClusterDetector(KMeans(n_clusters=3))
                # anomalies = min_cluster_detector.fit_detect(df)

                # outlier_detector = OutlierDetector(LocalOutlierFactor(contamination=0.05))
                # anomalies = outlier_detector.fit_detect(df)

                # regression_ad = RegressionAD(regressor=LinearRegression(), target="close", c=3.0)
                # anomalies = regression_ad.fit_detect(df)

                # pca_ad = PcaAD(k=1)
                # anomalies = pca_ad.fit_detect(df)


                # from adtk.detector import VolatilityShiftAD
                # volatility_shift_ad = VolatilityShiftAD(c=6.0, side='positive', window=30)
                # anomalies = volatility_shift_ad.fit_detect(df['feature2'])


                # from adtk.detector import LevelShiftAD
                # level_shift_ad = LevelShiftAD(c=6.0, side='both', window=5)
                # anomalies1 = level_shift_ad.fit_detect(df['close'])
                # anomalies2 = level_shift_ad.fit_detect(df['feature1'])
                # anomalies3 = level_shift_ad.fit_detect(df['feature2'])



                # from adtk.detector import PersistAD
                # persist_ad = PersistAD(c=3.0, side='negative')
                # anomalies1 = persist_ad.fit_detect(df['close'])
                # anomalies2 = persist_ad.fit_detect(df['feature1'])
                # anomalies3 = persist_ad.fit_detect(df['feature2'])

                # from adtk.detector import GeneralizedESDTestAD
                # esd_ad = GeneralizedESDTestAD(alpha=0.3)
                # anomalies1 = esd_ad.fit_detect(df['close'])
                # anomalies2 = esd_ad.fit_detect(df['feature1'])
                # anomalies3 = esd_ad.fit_detect(df['feature2'])







                # Load your data into a DataFrame

                # Create and fit a QuantileAD detector for anomaly detection
                # quantile_ad = QuantileAD(high=0.99, low=0.01)
                # anomalies_close = quantile_ad.fit_detect(df['close'])
                # anomalies_feature1 = quantile_ad.fit_detect(df['feature1'])
                # anomalies_feature2 = quantile_ad.fit_detect(df['feature2'])


                quantile_ad = QuantileAD(high=0.99, low=0.01)
                anomalies_open = quantile_ad.fit_detect(df['open'])
                anomalies_high = quantile_ad.fit_detect(df['high'])
                anomalies_low = quantile_ad.fit_detect(df['low'])
                anomalies_close = quantile_ad.fit_detect(df['close'])

                anomalies_dmp_dmn_200_diff = quantile_ad.fit_detect(df['dmp_dmn_200_diff'])
                anomalies_EMA_200_close_diff = quantile_ad.fit_detect(df['EMA_200_close_diff'])
                anomalies_EMA_200_BBU_21_2_diff = quantile_ad.fit_detect(df['EMA_200_BBU_21_2_diff'])
                anomalies_EMA_200_BBL_21_2_diff = quantile_ad.fit_detect(df['EMA_200_BBL_21_2_diff'])
                anomalies_combined_diff = quantile_ad.fit_detect(df['combined_diff'])
                anomalies_combined_diff_filtered = quantile_ad.fit_detect(df['combined_diff_filtered'])
                anomalies_second_combined_diff = quantile_ad.fit_detect(df['second_combined_diff'])
                anomalies_second_combined_diff_filtered = quantile_ad.fit_detect(df['second_combined_diff_filtered'])

                anomalies_DX_14 = quantile_ad.fit_detect(df['DX_14'])
                anomalies_RSI_14 = quantile_ad.fit_detect(df['RSI_14'])

                anomalies_ATRr_14 = quantile_ad.fit_detect(df['ATRr_14'])
                anomalies_ATRr_1 = quantile_ad.fit_detect(df['ATRr_1'])
                anomalies_MACD_12_26_9 = quantile_ad.fit_detect(df['MACD_12_26_9'])
                anomalies_MACD_50_75_35 = quantile_ad.fit_detect(df['MACD_50_75_35'])
                anomalies_MACDh_12_26_9 = quantile_ad.fit_detect(df['MACDh_12_26_9'])
                anomalies_MACDh_12_26_9_filtered = quantile_ad.fit_detect(df['MACDh_12_26_9_filtered'])
                anomalies_MACDh_50_75_35 = quantile_ad.fit_detect(df['MACDh_50_75_35'])
                anomalies_MACDh_50_75_35_filtered = quantile_ad.fit_detect(df['MACDh_50_75_35_filtered'])

                # anomalies_close = anomalies_close.fillna(False)  # 또는 다른 값으로 대체
                # anomalies_feature1 = anomalies_feature1.fillna(False)  # 또는 다른 값으로 대체
                # anomalies_feature2 = anomalies_feature2.fillna(False)  # 또는 다른 값으로 대체

                df['anomalies_open'] = anomalies_open
                df['anomalies_high'] = anomalies_high
                df['anomalies_low'] = anomalies_low
                df['anomalies_close'] = anomalies_close

                df['anomalies_dmp_dmn_200_diff'] = anomalies_dmp_dmn_200_diff
                df['anomalies_EMA_200_close_diff'] = anomalies_EMA_200_close_diff
                df['anomalies_EMA_200_BBU_21_2_diff'] = anomalies_EMA_200_BBU_21_2_diff
                df['anomalies_EMA_200_BBL_21_2_diff'] = anomalies_EMA_200_BBL_21_2_diff
                df['anomalies_combined_diff'] = anomalies_combined_diff
                df['anomalies_combined_diff_filtered'] = anomalies_combined_diff_filtered
                df['anomalies_second_combined_diff'] = anomalies_second_combined_diff
                df['anomalies_second_combined_diff_filtered'] = anomalies_second_combined_diff_filtered

                df['anomalies_DX_14'] = anomalies_DX_14 # dx
                df['anomalies_RSI_14'] = anomalies_RSI_14 # rsi
                
                df['anomalies_MACD_12_26_9'] = anomalies_MACD_12_26_9 # macd
                df['anomalies_MACD_50_75_35'] = anomalies_MACD_50_75_35 # macd
                
                df['anomalies_MACDh_12_26_9'] = anomalies_MACDh_12_26_9 # macd
                df['anomalies_MACDh_12_26_9_filtered'] = anomalies_MACDh_12_26_9_filtered # macd

                df['anomalies_MACDh_50_75_35'] = anomalies_MACDh_50_75_35 # macd
                df['anomalies_MACDh_50_75_35_filtered'] = anomalies_MACDh_50_75_35_filtered # macd

                df['anomalies_ATRr_14'] = anomalies_ATRr_14 # dx
                df['anomalies_ATRr_1'] = anomalies_ATRr_1 # rsi



                # anomalies_close = anomalies_close.fillna(False)  # 또는 다른 값으로 대체
                # anomalies_feature1 = anomalies_feature1.fillna(False)  # 또는 다른 값으로 대체
                # anomalies_feature2 = anomalies_feature2.fillna(False)  # 또는 다른 값으로 대체


                # df['anomalies_close'] = anomalies_close 
                # df['anomalies_feature1'] = anomalies_feature1 
                # df['anomalies_feature2'] = anomalies_feature2 


                # Extract the anomalies DataFrame
                # anomalies = df.loc[anomalies_close].fillna(False)

                # Create columns for peak and valley anomalies


                # Calculate IQR (optional)
                iqr_close = df.loc[anomalies_close]['close'].quantile(0.51) -df.loc[anomalies_close]['close'].quantile(0.49)
                # iqr_feature1 = df.loc[anomalies_feature1]['feature1'].quantile(0.51) - df.loc[anomalies_feature1]['feature1'].quantile(0.49)
                # iqr_feature2 = df.loc[anomalies_feature2]['feature2'].quantile(0.51) - df.loc[anomalies_feature2]['feature2'].quantile(0.49)


                upper_bound_close = df.loc[anomalies_close]['close'].quantile(0.51) #+ (1.5 * iqr_close)
                lower_bound_close = df.loc[anomalies_close]['close'].quantile(0.49) #- (1.5 * iqr_close)
                # upper_bound_feature1 =  df.loc[anomalies_close]['feature1'].quantile(0.51) #+ (1.5 * iqr_feature1)
                # lower_bound_feature1 =  df.loc[anomalies_close]['feature1'].quantile(0.49) #- (1.5 * iqr_feature1)
                # upper_bound_feature2 =  df.loc[anomalies_close]['feature2'].quantile(0.51) #+ (1.5 * iqr_feature2)
                # lower_bound_feature2 =  df.loc[anomalies_close]['feature2'].quantile(0.49) #- (1.5 * iqr_feature2)
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################

                    
                # # Using boolean indexing to assign values
                df.loc[((df['anomalies_close'] > 0) & (df.close > upper_bound_close)), "maxima_peak_x_anomalies_close"] = -2 # without atr
                df.loc[((df['anomalies_close'] > 0) & (df.close < lower_bound_close)), "minima_peak_x_anomalies_close"] = 2 # without atr
                # df.loc[((df['anomalies_feature1'] > 0) & (df.feature1 > upper_bound_feature1)), "maxima_peak_x_anomalies_feature1"] = -2 # without atr
                # df.loc[((df['anomalies_feature1'] > 0) & (df.feature1 < lower_bound_feature1)), "minima_peak_x_anomalies_feature1"] = 2 # without atr
                # df.loc[((df['anomalies_feature2'] > 0) & (df.feature2 > upper_bound_feature2)), "maxima_peak_x_anomalies_feature2"] = -2 # without atr
                # df.loc[((df['anomalies_feature2'] > 0) & (df.feature2 < lower_bound_feature2)), "minima_peak_x_anomalies_feature2"] = 2 # without atr

                # Using boolean indexing to assign values
                # df.loc[((df['anomalies_close'] > 0) & (df.close > upper_bound_close)), "stg1_short"] = -2 # without atr
                # df.loc[((df['anomalies_close'] > 0) & (df.close < lower_bound_close)), "stg1_long"] = 2 # without atr
                # df.loc[((df['anomalies_feature1'] > 0) & (df.feature1 > upper_bound_feature1)), "stg2_short"] = -2 # without atr
                # df.loc[((df['anomalies_feature1'] > 0) & (df.feature1 < lower_bound_feature1)), "stg2_long"] = 2 # without atr
                # df.loc[((df['anomalies_feature2'] > 0) & (df.feature2 > upper_bound_feature2)), "stg3_short"] = -2 # without atr
                # df.loc[((df['anomalies_feature2'] > 0) & (df.feature2 < lower_bound_feature2)), "stg3_long"] = 2 # without atr

                ########################################################################################################################################################



                # from adtk.detector import PersistAD

                # persist_ad_positive = PersistAD(c=3.0, side='positive')
                # persist_ad_positive.window = 24
                # anomalies_close_positive = persist_ad_positive.fit_detect(df['close'])

                # persist_ad_negative = PersistAD(c=3.0, side='negative')
                # persist_ad_negative.window = 24
                # anomalies_close_negative = persist_ad_negative.fit_detect(df['close'])


                # df['anomalies_close_positive'] = anomalies_close_positive
                # df['anomalies_close_negative'] = anomalies_close_negative

                # df.loc[df['anomalies_close_positive'] == True, 'stg1_long'] = 2
                # df.loc[df['anomalies_close_negative'] == True, 'stg1_short'] = -2


                ########################################################################################################################################################


                # from adtk.detector import LevelShiftAD
                # level_shift_ad_positive = LevelShiftAD(c=6.0, side='positive', window=5)
                # anomalies_close_positive = level_shift_ad_positive.fit_detect(df['close'])

                # level_shift_ad_negative = LevelShiftAD(c=6.0, side='negative', window=5)
                # anomalies_close_negative = level_shift_ad_negative.fit_detect(df['close'])

                # df['anomalies_close_positive'] = anomalies_close_positive
                # df['anomalies_close_negative'] = anomalies_close_negative

                # df.loc[df['anomalies_close_positive'] == True, 'stg1_long'] = 2
                # df.loc[df['anomalies_close_negative'] == True, 'stg1_short'] = -2
                

                ########################################################################################################################################################
                
                # from adtk.detector import VolatilityShiftAD
                # volatility_shift_ad_positive = VolatilityShiftAD(c=6.0, side='positive', window=30)
                # anomalies_close_positive = volatility_shift_ad_positive.fit_detect(df['close'])

                # volatility_shift_ad_negative = VolatilityShiftAD(c=6.0, side='negative', window=30)
                # anomalies_close_negative = volatility_shift_ad_negative.fit_detect(df['close'])


                # df['anomalies_close_positive'] = anomalies_close_positive
                # df['anomalies_close_negative'] = anomalies_close_negative

                # # df.loc[df['anomalies_close_positive'] == True, 'stg1_long'] = 2
                # # df.loc[df['anomalies_close_negative'] == True, 'stg1_short'] = -2
                









                ###################################################################################################
                ###################################################################################################











                ############### AnomalyDetection ####################################################################################
                ############### AnomalyDetection ####################################################################################
                ############### AnomalyDetection ####################################################################################
                ############### AnomalyDetection ####################################################################################
                ############### AnomalyDetection ####################################################################################
                ############### AnomalyDetection ####################################################################################


                df['AnomalyDetection_close'] = 0
                df['AnomalyDetection_RSI_14'] = 0
                df['AnomalyDetection_MACD_12_26_9'] = 0
                df['AnomalyDetection_MACD_50_75_35'] = 0
                df['AnomalyDetection_OBV'] = 0
                df['AnomalyDetection_feature1'] = 0
                df['AnomalyDetection_feature2'] = 0



                df['anomaly_type_close'] = ''
                df['anomaly_type_RSI_14'] = ''
                df['anomaly_type_MACD_12_26_9'] = ''
                df['anomaly_type_MACD_50_75_35'] = ''
                df['anomaly_type_OBV'] = ''
                df['anomaly_type_feature1'] = ''
                df['anomaly_type_feature2'] = ''
                df['anomaly_type_dmp_dmn_200_diff'] = 0
                df['anomaly_type_EMA_200_close_diff'] = 0
                df['anomaly_type_EMA_200_BBU_21_2_diff'] = 0
                df['anomaly_type_EMA_200_BBL_21_2_diff'] = 0
                df['anomaly_type_combined_diff'] = 0
                df['anomaly_type_combined_diff_filtered'] = 0
                df['anomaly_type_second_combined_diff'] = 0
                df['anomaly_type_second_combined_diff_filtered'] = 0
                df['anomaly_type_ATRr_14'] = 0




                df['anomaly_type_close_n'] = 0
                df['anomaly_type_RSI_14_n'] = 0
                df['anomaly_type_MACD_12_26_9_n'] = 0
                df['anomaly_type_MACD_50_75_35_n'] = 0
                df['anomaly_type_OBV_n'] = 0
                df['anomaly_type_feature1_n'] = 0
                df['anomaly_type_feature2_n'] = 0
                df['anomaly_type_dmp_dmn_200_diff_n'] = 0
                df['anomaly_type_EMA_200_close_diff_n'] = 0
                df['anomaly_type_EMA_200_BBU_21_2_diff_n'] = 0
                df['anomaly_type_EMA_200_BBL_21_2_diff_n'] = 0
                df['anomaly_type_combined_diff_n'] = 0
                df['anomaly_type_combined_diff_filtered_n'] = 0
                df['anomaly_type_second_combined_diff'] = 0
                df['anomaly_type_second_combined_diff_filtered'] = 0
                df['anomaly_type_ATRr_14_n'] = 0


                prev_values_close = [np.nan] * len(df)  # 초기값으로 NaN 할당
                prev_values_RSI_14 = [np.nan] * len(df)  # 초기값으로 NaN 할당
                prev_values_MACD_12_26_9 = [np.nan] * len(df)  # 초기값으로 NaN 할당
                prev_values_MACD_50_75_35 = [np.nan] * len(df)  # 초기값으로 NaN 할당
                prev_values_OBV = [np.nan] * len(df)  # 초기값으로 NaN 할당
                prev_values_feature1 = [np.nan] * len(df)  # 초기값으로 NaN 할당
                prev_values_feature2 = [np.nan] * len(df)  # 초기값으로 NaN 할당
                prev_values_dmp_dmn_200_diff = [np.nan] * len(df)  # 초기값으로 NaN 할당
                prev_values_EMA_200_close_diff = [np.nan] * len(df)  # 초기값으로 NaN 할당
                prev_values_EMA_200_BBU_21_2_diff = [np.nan] * len(df)  # 초기값으로 NaN 할당
                prev_values_EMA_200_BBL_21_2_diff = [np.nan] * len(df)  # 초기값으로 NaN 할당
                prev_values_combined_diff = [np.nan] * len(df)  # 초기값으로 NaN 할당
                prev_values_combined_diff_filtered = [np.nan] * len(df)  # 초기값으로 NaN 할당
                prev_values_second_combined_diff = [np.nan] * len(df)  # 초기값으로 NaN 할당
                prev_values_second_combined_diff_filtered = [np.nan] * len(df)  # 초기값으로 NaN 할당
                prev_values_ATRr_14 = [np.nan] * len(df)  # 초기값으로 NaN 할당

                best_alpha = 0.1
                best_lamb = 100
                df['RSI_14'] = df['RSI_14'].fillna(0)
                df['MACD_12_26_9'] = df['MACD_12_26_9'].fillna(0)
                df['MACD_50_75_35'] = df['MACD_50_75_35'].fillna(0)
                df['OBV'] = df['OBV'].fillna(0)
                df['dmp_dmn_200_diff'] = df['dmp_dmn_200_diff'].fillna(0)
                df['EMA_200_close_diff'] = df['EMA_200_close_diff'].fillna(0)
                df['EMA_200_BBU_21_2_diff'] = df['EMA_200_BBU_21_2_diff'].fillna(0)
                df['EMA_200_BBL_21_2_diff'] = df['EMA_200_BBL_21_2_diff'].fillna(0)
                df['combined_diff'] = df['combined_diff'].fillna(0)
                df['combined_diff_filtered'] = df['combined_diff_filtered'].fillna(0)
                df['second_combined_diff'] = df['second_combined_diff'].fillna(0)
                df['second_combined_diff_filtered'] = df['second_combined_diff_filtered'].fillna(0)
                df['ATRr_14'] = df['ATRr_14'].fillna(0)

                AnomalyDetection_close = AnomalyDetection(df, 'close', best_alpha, best_lamb)
                AnomalyDetection_RSI_14 = AnomalyDetection(df, 'RSI_14', best_alpha, best_lamb)
                AnomalyDetection_MACD_12_26_9 = AnomalyDetection(df, 'MACD_12_26_9', best_alpha, best_lamb)
                AnomalyDetection_MACD_50_75_35 = AnomalyDetection(df, 'MACD_50_75_35', best_alpha, best_lamb)
                AnomalyDetection_OBV = AnomalyDetection(df, 'OBV', best_alpha, best_lamb)
                AnomalyDetection_feature1 = AnomalyDetection(df, 'feature1', best_alpha, best_lamb)
                AnomalyDetection_feature2 = AnomalyDetection(df, 'feature2', best_alpha, best_lamb)
                AnomalyDetection_dmp_dmn_200_diff = AnomalyDetection(df, 'dmp_dmn_200_diff', best_alpha, best_lamb)
                AnomalyDetection_EMA_200_close_diff = AnomalyDetection(df, 'EMA_200_close_diff', best_alpha, best_lamb)
                AnomalyDetection_EMA_200_BBU_21_2_diff = AnomalyDetection(df, 'EMA_200_BBU_21_2_diff', best_alpha, best_lamb)
                AnomalyDetection_EMA_200_BBL_21_2_diff = AnomalyDetection(df, 'EMA_200_BBL_21_2_diff', best_alpha, best_lamb)
                AnomalyDetection_combined_diff = AnomalyDetection(df, 'combined_diff', best_alpha, best_lamb)
                AnomalyDetection_combined_diff_filtered = AnomalyDetection(df, 'combined_diff_filtered', best_alpha, best_lamb)
                AnomalyDetection_second_combined_diff = AnomalyDetection(df, 'second_combined_diff', best_alpha, best_lamb)
                AnomalyDetection_second_combined_diff_filtered = AnomalyDetection(df, 'second_combined_diff_filtered', best_alpha, best_lamb)
                AnomalyDetection_ATRr_14 = AnomalyDetection(df, 'ATRr_14', best_alpha, best_lamb)

                df.loc[AnomalyDetection_close, 'AnomalyDetection_close'] = 2
                df.loc[AnomalyDetection_close, 'anomaly_type_close'] = 'anomaly'
                df.loc[AnomalyDetection_RSI_14, 'AnomalyDetection_RSI_14'] = 2
                df.loc[AnomalyDetection_RSI_14, 'anomaly_type_RSI_14'] = 'anomaly'
                df.loc[AnomalyDetection_MACD_12_26_9, 'AnomalyDetection_MACD_12_26_9'] = 2
                df.loc[AnomalyDetection_MACD_12_26_9, 'anomaly_type_MACD_12_26_9'] = 'anomaly'
                df.loc[AnomalyDetection_MACD_50_75_35, 'AnomalyDetection_MACD_50_75_35'] = 2
                df.loc[AnomalyDetection_MACD_50_75_35, 'anomaly_type_MACD_50_75_35'] = 'anomaly'
                df.loc[AnomalyDetection_OBV, 'AnomalyDetection_OBV'] = 2
                df.loc[AnomalyDetection_OBV, 'anomaly_type_OBV'] = 'anomaly'
                df.loc[AnomalyDetection_feature1, 'AnomalyDetection_feature1'] = 2
                df.loc[AnomalyDetection_feature1, 'anomaly_type_feature1'] = 'anomaly'
                df.loc[AnomalyDetection_feature2, 'AnomalyDetection_feature2'] = 2
                df.loc[AnomalyDetection_feature2, 'anomaly_type_feature2'] = 'anomaly'
                df.loc[AnomalyDetection_dmp_dmn_200_diff, 'AnomalyDetection_dmp_dmn_200_diff'] = 2
                df.loc[AnomalyDetection_dmp_dmn_200_diff, 'anomaly_type_dmp_dmn_200_diff'] = 'anomaly'
                df.loc[AnomalyDetection_EMA_200_close_diff, 'AnomalyDetection_EMA_200_close_diff'] = 2
                df.loc[AnomalyDetection_EMA_200_close_diff, 'anomaly_type_EMA_200_close_diff'] = 'anomaly'
                df.loc[AnomalyDetection_EMA_200_BBU_21_2_diff, 'AnomalyDetection_EMA_200_BBU_21_2_diff'] = 2
                df.loc[AnomalyDetection_EMA_200_BBU_21_2_diff, 'anomaly_type_EMA_200_BBU_21_2_diff'] = 'anomaly'
                df.loc[AnomalyDetection_EMA_200_BBL_21_2_diff, 'AnomalyDetection_EMA_200_BBL_21_2_diff'] = 2
                df.loc[AnomalyDetection_EMA_200_BBL_21_2_diff, 'anomaly_type_EMA_200_BBL_21_2_diff'] = 'anomaly'
                df.loc[AnomalyDetection_combined_diff, 'AnomalyDetection_combined_diff'] = 2
                df.loc[AnomalyDetection_combined_diff, 'anomaly_type_combined_diff'] = 'anomaly'
                df.loc[AnomalyDetection_combined_diff_filtered, 'AnomalyDetection_combined_diff_filtered'] = 2
                df.loc[AnomalyDetection_combined_diff_filtered, 'anomaly_type_combined_diff_filtered'] = 'anomaly'            
                df.loc[AnomalyDetection_second_combined_diff, 'AnomalyDetection_second_combined_diff'] = 2
                df.loc[AnomalyDetection_second_combined_diff, 'anomaly_type_second_combined_diff'] = 'anomaly'
                df.loc[AnomalyDetection_second_combined_diff_filtered, 'AnomalyDetection_second_combined_diff_filtered'] = 2
                df.loc[AnomalyDetection_second_combined_diff_filtered, 'anomaly_type_second_combined_diff_filtered'] = 'anomaly'

                df.loc[AnomalyDetection_ATRr_14, 'AnomalyDetection_ATRr_14'] = 2
                df.loc[AnomalyDetection_ATRr_14, 'anomaly_type_ATRr_14'] = 'anomaly'

                for idx in AnomalyDetection_close:
                    idx_num = df.index.get_loc(idx)
                    if 0 < idx_num < len(df) - 1:  # 첫 번째와 마지막 인덱스는 비교에서 제외
                        
                        prev_value_close = df['close'].iloc[idx_num - 1]
                        prev_values_close[idx_num] = prev_value_close
                        
                        if df['close'].iloc[idx_num] > df['close'].iloc[idx_num - 1] and df['close'].iloc[idx_num] > df['close'].iloc[idx_num + 1]:
                            df.at[idx, 'anomaly_type_close'] = 'peak'
                            df.at[idx, 'anomaly_type_close_n'] = -2
                        elif df['close'].iloc[idx_num] < df['close'].iloc[idx_num - 1] and df['close'].iloc[idx_num] < df['close'].iloc[idx_num + 1]:
                            df.at[idx, 'anomaly_type_close'] = 'valley'
                            df.at[idx, 'anomaly_type_close_n'] = 2

                for idx in AnomalyDetection_RSI_14:
                    idx_num = df.index.get_loc(idx)
                    if 0 < idx_num < len(df) - 1:  # 첫 번째와 마지막 인덱스는 비교에서 제외
                        
                        prev_value_RSI_14 = df['RSI_14'].iloc[idx_num - 1]
                        prev_values_RSI_14[idx_num] = prev_value_RSI_14
                        
                        if df['RSI_14'].iloc[idx_num] > df['RSI_14'].iloc[idx_num - 1] and df['RSI_14'].iloc[idx_num] > df['RSI_14'].iloc[idx_num + 1]:
                            df.at[idx, 'anomaly_type_RSI_14'] = 'peak'
                            df.at[idx, 'anomaly_type_RSI_14_n'] = -2
                        elif df['RSI_14'].iloc[idx_num] < df['RSI_14'].iloc[idx_num - 1] and df['RSI_14'].iloc[idx_num] < df['RSI_14'].iloc[idx_num + 1]:
                            df.at[idx, 'anomaly_type_RSI_14'] = 'valley'
                            df.at[idx, 'anomaly_type_RSI_14_n'] = 2

                for idx in AnomalyDetection_MACD_12_26_9:
                    idx_num = df.index.get_loc(idx)
                    if 0 < idx_num < len(df) - 1:  # 첫 번째와 마지막 인덱스는 비교에서 제외
                        
                        prev_value_MACD_12_26_9 = df['MACD_12_26_9'].iloc[idx_num - 1]
                        prev_values_MACD_12_26_9[idx_num] = prev_value_MACD_12_26_9
                        
                        if df['MACD_12_26_9'].iloc[idx_num] > df['MACD_12_26_9'].iloc[idx_num - 1] and df['MACD_12_26_9'].iloc[idx_num] > df['MACD_12_26_9'].iloc[idx_num + 1]:
                            df.at[idx, 'anomaly_type_MACD_12_26_9'] = 'peak'
                            df.at[idx, 'anomaly_type_MACD_12_26_9_n'] = -2
                        elif df['MACD_12_26_9'].iloc[idx_num] < df['MACD_12_26_9'].iloc[idx_num - 1] and df['MACD_12_26_9'].iloc[idx_num] < df['MACD_12_26_9'].iloc[idx_num + 1]:
                            df.at[idx, 'anomaly_type_MACD_12_26_9'] = 'valley'
                            df.at[idx, 'anomaly_type_MACD_12_26_9_n'] = 2

                for idx in AnomalyDetection_MACD_50_75_35:
                    idx_num = df.index.get_loc(idx)
                    if 0 < idx_num < len(df) - 1:  # 첫 번째와 마지막 인덱스는 비교에서 제외
                        
                        prev_value_MACD_50_75_35 = df['MACD_50_75_35'].iloc[idx_num - 1]
                        prev_values_MACD_50_75_35[idx_num] = prev_value_MACD_50_75_35
                        
                        if df['MACD_50_75_35'].iloc[idx_num] > df['MACD_50_75_35'].iloc[idx_num - 1] and df['MACD_50_75_35'].iloc[idx_num] > df['MACD_50_75_35'].iloc[idx_num + 1]:
                            df.at[idx, 'anomaly_type_MACD_50_75_35'] = 'peak'
                            df.at[idx, 'anomaly_type_MACD_50_75_35_n'] = -2
                        elif df['MACD_50_75_35'].iloc[idx_num] < df['MACD_50_75_35'].iloc[idx_num - 1] and df['MACD_50_75_35'].iloc[idx_num] < df['MACD_50_75_35'].iloc[idx_num + 1]:
                            df.at[idx, 'anomaly_type_MACD_50_75_35'] = 'valley'
                            df.at[idx, 'anomaly_type_MACD_50_75_35_n'] = 2

                for idx in AnomalyDetection_OBV:
                    idx_num = df.index.get_loc(idx)
                    if 0 < idx_num < len(df) - 1:  # 첫 번째와 마지막 인덱스는 비교에서 제외
                        
                        prev_value_OBV = df['OBV'].iloc[idx_num - 1]
                        prev_values_OBV[idx_num] = prev_value_OBV
                        
                        if df['OBV'].iloc[idx_num] > df['OBV'].iloc[idx_num - 1] and df['OBV'].iloc[idx_num] > df['OBV'].iloc[idx_num + 1]:
                            df.at[idx, 'anomaly_type_OBV'] = 'peak'
                            df.at[idx, 'anomaly_type_OBV_n'] = -2
                        elif df['OBV'].iloc[idx_num] < df['OBV'].iloc[idx_num - 1] and df['OBV'].iloc[idx_num] < df['OBV'].iloc[idx_num + 1]:
                            df.at[idx, 'anomaly_type_OBV'] = 'valley'
                            df.at[idx, 'anomaly_type_OBV_n'] = 2

                for idx in AnomalyDetection_feature1:
                    idx_num = df.index.get_loc(idx)
                    if 0 < idx_num < len(df) - 1:  # 첫 번째와 마지막 인덱스는 비교에서 제외
                        
                        prev_value_feature1 = df['feature1'].iloc[idx_num - 1]
                        prev_values_feature1[idx_num] = prev_value_feature1
                        
                        if df['feature1'].iloc[idx_num] > df['feature1'].iloc[idx_num - 1] and df['feature1'].iloc[idx_num] > df['feature1'].iloc[idx_num + 1]:
                            df.at[idx, 'anomaly_type_feature1'] = 'peak'
                            df.at[idx, 'anomaly_type_feature1_n'] = -2
                        elif df['feature1'].iloc[idx_num] < df['feature1'].iloc[idx_num - 1] and df['feature1'].iloc[idx_num] < df['feature1'].iloc[idx_num + 1]:
                            df.at[idx, 'anomaly_type_feature1'] = 'valley'
                            df.at[idx, 'anomaly_type_feature1_n'] = 2

                for idx in AnomalyDetection_feature2:
                    idx_num = df.index.get_loc(idx)
                    if 0 < idx_num < len(df) - 1:  # 첫 번째와 마지막 인덱스는 비교에서 제외
                        
                        prev_value_feature2 = df['feature2'].iloc[idx_num - 1]
                        prev_values_feature2[idx_num] = prev_value_feature2
                        
                        if df['feature2'].iloc[idx_num] > df['feature2'].iloc[idx_num - 1] and df['feature2'].iloc[idx_num] > df['feature2'].iloc[idx_num + 1]:
                            df.at[idx, 'anomaly_type_feature2'] = 'peak'
                            df.at[idx, 'anomaly_type_feature2_n'] = -2
                        elif df['feature2'].iloc[idx_num] < df['feature2'].iloc[idx_num - 1] and df['feature2'].iloc[idx_num] < df['feature2'].iloc[idx_num + 1]:
                            df.at[idx, 'anomaly_type_feature2'] = 'valley'
                            df.at[idx, 'anomaly_type_feature2_n'] = 2

                for idx in AnomalyDetection_dmp_dmn_200_diff:
                    idx_num = df.index.get_loc(idx)
                    if 0 < idx_num < len(df) - 1:  # 첫 번째와 마지막 인덱스는 비교에서 제외
                        
                        prev_value_dmp_dmn_200_diff = df['dmp_dmn_200_diff'].iloc[idx_num - 1]
                        prev_values_dmp_dmn_200_diff[idx_num] = prev_value_dmp_dmn_200_diff
                        
                        if df['dmp_dmn_200_diff'].iloc[idx_num] > df['dmp_dmn_200_diff'].iloc[idx_num - 1] and df['dmp_dmn_200_diff'].iloc[idx_num] > df['dmp_dmn_200_diff'].iloc[idx_num + 1]:
                            df.at[idx, 'anomaly_type_dmp_dmn_200_diff'] = 'peak'
                            df.at[idx, 'anomaly_type_dmp_dmn_200_diff_n'] = -2
                        elif df['dmp_dmn_200_diff'].iloc[idx_num] < df['dmp_dmn_200_diff'].iloc[idx_num - 1] and df['dmp_dmn_200_diff'].iloc[idx_num] < df['dmp_dmn_200_diff'].iloc[idx_num + 1]:
                            df.at[idx, 'anomaly_type_dmp_dmn_200_diff'] = 'valley'
                            df.at[idx, 'anomaly_type_dmp_dmn_200_diff_n'] = 2

                for idx in AnomalyDetection_EMA_200_close_diff:
                    idx_num = df.index.get_loc(idx)
                    if 0 < idx_num < len(df) - 1:  # 첫 번째와 마지막 인덱스는 비교에서 제외
                        
                        prev_value_EMA_200_close_diff = df['EMA_200_close_diff'].iloc[idx_num - 1]
                        prev_values_EMA_200_close_diff[idx_num] = prev_value_EMA_200_close_diff
                        
                        if df['EMA_200_close_diff'].iloc[idx_num] > df['EMA_200_close_diff'].iloc[idx_num - 1] and df['EMA_200_close_diff'].iloc[idx_num] > df['EMA_200_close_diff'].iloc[idx_num + 1]:
                            df.at[idx, 'anomaly_type_EMA_200_close_diff'] = 'peak'
                            df.at[idx, 'anomaly_type_EMA_200_close_diff_n'] = -2
                        elif df['EMA_200_close_diff'].iloc[idx_num] < df['EMA_200_close_diff'].iloc[idx_num - 1] and df['EMA_200_close_diff'].iloc[idx_num] < df['EMA_200_close_diff'].iloc[idx_num + 1]:
                            df.at[idx, 'anomaly_type_EMA_200_close_diff'] = 'valley'
                            df.at[idx, 'anomaly_type_EMA_200_close_diff_n'] = 2

                for idx in AnomalyDetection_EMA_200_BBU_21_2_diff:
                    idx_num = df.index.get_loc(idx)
                    if 0 < idx_num < len(df) - 1:  # 첫 번째와 마지막 인덱스는 비교에서 제외
                        
                        prev_value_EMA_200_BBU_21_2_diff = df['EMA_200_BBU_21_2_diff'].iloc[idx_num - 1]
                        prev_values_EMA_200_BBU_21_2_diff[idx_num] = prev_value_EMA_200_BBU_21_2_diff
                        
                        if df['EMA_200_BBU_21_2_diff'].iloc[idx_num] > df['EMA_200_BBU_21_2_diff'].iloc[idx_num - 1] and df['EMA_200_BBU_21_2_diff'].iloc[idx_num] > df['EMA_200_BBU_21_2_diff'].iloc[idx_num + 1]:
                            df.at[idx, 'anomaly_type_EMA_200_BBU_21_2_diff'] = 'peak'
                            df.at[idx, 'anomaly_type_EMA_200_BBU_21_2_diff_n'] = -2
                        elif df['EMA_200_BBU_21_2_diff'].iloc[idx_num] < df['EMA_200_BBU_21_2_diff'].iloc[idx_num - 1] and df['EMA_200_BBU_21_2_diff'].iloc[idx_num] < df['EMA_200_BBU_21_2_diff'].iloc[idx_num + 1]:
                            df.at[idx, 'anomaly_type_EMA_200_BBU_21_2_diff'] = 'valley'
                            df.at[idx, 'anomaly_type_EMA_200_BBU_21_2_diff_n'] = 2

                for idx in AnomalyDetection_EMA_200_BBL_21_2_diff:
                    idx_num = df.index.get_loc(idx)
                    if 0 < idx_num < len(df) - 1:  # 첫 번째와 마지막 인덱스는 비교에서 제외
                        
                        prev_value_EMA_200_BBL_21_2_diff = df['EMA_200_BBL_21_2_diff'].iloc[idx_num - 1]
                        prev_values_EMA_200_BBL_21_2_diff[idx_num] = prev_value_EMA_200_BBL_21_2_diff
                        
                        if df['EMA_200_BBL_21_2_diff'].iloc[idx_num] > df['EMA_200_BBL_21_2_diff'].iloc[idx_num - 1] and df['EMA_200_BBL_21_2_diff'].iloc[idx_num] > df['EMA_200_BBL_21_2_diff'].iloc[idx_num + 1]:
                            df.at[idx, 'anomaly_type_EMA_200_BBL_21_2_diff'] = 'peak'
                            df.at[idx, 'anomaly_type_EMA_200_BBL_21_2_diff_n'] = -2
                        elif df['EMA_200_BBL_21_2_diff'].iloc[idx_num] < df['EMA_200_BBL_21_2_diff'].iloc[idx_num - 1] and df['EMA_200_BBL_21_2_diff'].iloc[idx_num] < df['EMA_200_BBL_21_2_diff'].iloc[idx_num + 1]:
                            df.at[idx, 'anomaly_type_EMA_200_BBL_21_2_diff'] = 'valley'
                            df.at[idx, 'anomaly_type_EMA_200_BBL_21_2_diff_n'] = 2

                for idx in AnomalyDetection_combined_diff:
                    idx_num = df.index.get_loc(idx)
                    if 0 < idx_num < len(df) - 1:  # 첫 번째와 마지막 인덱스는 비교에서 제외
                        
                        prev_value_combined_diff = df['combined_diff'].iloc[idx_num - 1]
                        prev_values_combined_diff[idx_num] = prev_value_combined_diff
                        
                        if df['combined_diff'].iloc[idx_num] > df['combined_diff'].iloc[idx_num - 1] and df['combined_diff'].iloc[idx_num] > df['combined_diff'].iloc[idx_num + 1]:
                            df.at[idx, 'anomaly_type_combined_diff'] = 'peak'
                            df.at[idx, 'anomaly_type_combined_diff_n'] = -2
                        elif df['combined_diff'].iloc[idx_num] < df['combined_diff'].iloc[idx_num - 1] and df['combined_diff'].iloc[idx_num] < df['combined_diff'].iloc[idx_num + 1]:
                            df.at[idx, 'anomaly_type_combined_diff'] = 'valley'
                            df.at[idx, 'anomaly_type_combined_diff_n'] = 2

                for idx in AnomalyDetection_combined_diff_filtered:
                    idx_num = df.index.get_loc(idx)
                    if 0 < idx_num < len(df) - 1:  # 첫 번째와 마지막 인덱스는 비교에서 제외
                        
                        prev_value_combined_diff_filtered = df['combined_diff_filtered'].iloc[idx_num - 1]
                        prev_values_combined_diff_filtered[idx_num] = prev_value_combined_diff_filtered
                        
                        if df['combined_diff_filtered'].iloc[idx_num] > df['combined_diff_filtered'].iloc[idx_num - 1] and df['combined_diff_filtered'].iloc[idx_num] > df['combined_diff_filtered'].iloc[idx_num + 1]:
                            df.at[idx, 'anomaly_type_combined_diff_filtered'] = 'peak'
                            df.at[idx, 'anomaly_type_combined_diff_filtered_n'] = -2
                        elif df['combined_diff_filtered'].iloc[idx_num] < df['combined_diff_filtered'].iloc[idx_num - 1] and df['combined_diff_filtered'].iloc[idx_num] < df['combined_diff_filtered'].iloc[idx_num + 1]:
                            df.at[idx, 'anomaly_type_combined_diff_filtered'] = 'valley'
                            df.at[idx, 'anomaly_type_combined_diff_filtered_n'] = 2

                for idx in AnomalyDetection_second_combined_diff:
                    idx_num = df.index.get_loc(idx)
                    if 0 < idx_num < len(df) - 1:  # 첫 번째와 마지막 인덱스는 비교에서 제외
                        
                        prev_value_second_combined_diff = df['second_combined_diff'].iloc[idx_num - 1]
                        prev_values_second_combined_diff[idx_num] = prev_value_second_combined_diff
                        
                        if df['second_combined_diff'].iloc[idx_num] > df['second_combined_diff'].iloc[idx_num - 1] and df['second_combined_diff'].iloc[idx_num] > df['second_combined_diff'].iloc[idx_num + 1]:
                            df.at[idx, 'anomaly_type_second_combined_diff'] = 'peak'
                            df.at[idx, 'anomaly_type_second_combined_diff_n'] = -2
                        elif df['second_combined_diff'].iloc[idx_num] < df['second_combined_diff'].iloc[idx_num - 1] and df['second_combined_diff'].iloc[idx_num] < df['second_combined_diff'].iloc[idx_num + 1]:
                            df.at[idx, 'anomaly_type_second_combined_diff'] = 'valley'
                            df.at[idx, 'anomaly_type_second_combined_diff_n'] = 2

                for idx in AnomalyDetection_second_combined_diff_filtered:
                    idx_num = df.index.get_loc(idx)
                    if 0 < idx_num < len(df) - 1:  # 첫 번째와 마지막 인덱스는 비교에서 제외
                        
                        prev_value_second_combined_diff_filtered = df['second_combined_diff_filtered'].iloc[idx_num - 1]
                        prev_values_second_combined_diff_filtered[idx_num] = prev_value_second_combined_diff_filtered
                        
                        if df['second_combined_diff_filtered'].iloc[idx_num] > df['second_combined_diff_filtered'].iloc[idx_num - 1] and df['second_combined_diff_filtered'].iloc[idx_num] > df['second_combined_diff_filtered'].iloc[idx_num + 1]:
                            df.at[idx, 'anomaly_type_second_combined_diff_filtered'] = 'peak'
                            df.at[idx, 'anomaly_type_second_combined_diff_filtered_n'] = -2
                        elif df['second_combined_diff_filtered'].iloc[idx_num] < df['second_combined_diff_filtered'].iloc[idx_num - 1] and df['second_combined_diff_filtered'].iloc[idx_num] < df['second_combined_diff_filtered'].iloc[idx_num + 1]:
                            df.at[idx, 'anomaly_type_second_combined_diff_filtered'] = 'valley'
                            df.at[idx, 'anomaly_type_second_combined_diff_filtered_n'] = 2

                for idx in AnomalyDetection_ATRr_14:
                    idx_num = df.index.get_loc(idx)
                    if 0 < idx_num < len(df) - 1:  # 첫 번째와 마지막 인덱스는 비교에서 제외
                        
                        prev_value_ATRr_14 = df['ATRr_14'].iloc[idx_num - 1]
                        prev_values_ATRr_14[idx_num] = prev_value_ATRr_14
                        
                        if df['ATRr_14'].iloc[idx_num] > df['ATRr_14'].iloc[idx_num - 1] and df['ATRr_14'].iloc[idx_num] > df['ATRr_14'].iloc[idx_num + 1]:
                            df.at[idx, 'anomaly_type_ATRr_14'] = 'peak'
                            df.at[idx, 'anomaly_type_ATRr_14_n'] = -2
                        elif df['ATRr_14'].iloc[idx_num] < df['ATRr_14'].iloc[idx_num - 1] and df['ATRr_14'].iloc[idx_num] < df['ATRr_14'].iloc[idx_num + 1]:
                            df.at[idx, 'anomaly_type_ATRr_14'] = 'valley'
                            df.at[idx, 'anomaly_type_ATRr_14_n'] = 2


                # # 원본 데이터 시각화
                # plt.figure(figsize=(12, 6))

                # plt.plot(df.index, df['close'], label='Original Data', color='blue')
                # plt.plot(df.index, prev_values_close, linestyle='--', color='orange', label='Previous Value')

                # # plt.plot(df.index, df['feature1'], label='Original Data', color='blue')
                # # plt.plot(df.index, prev_values_feature1, linestyle='--', color='orange', label='Previous Value')

                # # plt.plot(df.index, df['feature2'], label='Original Data', color='blue')
                # # plt.plot(df.index, prev_values_feature2, linestyle='--', color='orange', label='Previous Value')

                # plt.scatter(df.index[df['anomaly_type_close_n'] < 0], df.loc[df['anomaly_type_close_n'] < 0, 'close'], color='red', label='Peak Anomalies')
                # plt.scatter(df.index[df['anomaly_type_close_n'] > 0], df.loc[df['anomaly_type_close_n'] > 0, 'close'], color='green', label='Valley Anomalies')
                # plt.scatter(df.index[df['anomaly_type_close'] == 'anomaly'], df.loc[df['anomaly_type_close'] == 'anomaly', 'close'], color='orange', label='Other Anomalies')

                # plt.scatter(df.index[df['anomaly_type_feature1_n'] < 0], df.loc[df['anomaly_type_feature1_n'] < 0, 'close'], color='red', label='Peak Anomalies')
                # plt.scatter(df.index[df['anomaly_type_feature1_n'] > 0], df.loc[df['anomaly_type_feature1_n'] > 0, 'close'], color='green', label='Valley Anomalies')
                # plt.scatter(df.index[df['anomaly_type_feature1'] == 'anomaly'], df.loc[df['anomaly_type_feature1'] == 'anomaly', 'close'], color='orange', label='Other Anomalies')

                # plt.scatter(df.index[df['anomaly_type_feature2_n'] < 0], df.loc[df['anomaly_type_feature2_n'] < 0, 'close'], color='red', label='Peak Anomalies')
                # plt.scatter(df.index[df['anomaly_type_feature2_n'] > 0], df.loc[df['anomaly_type_feature2_n'] > 0, 'close'], color='green', label='Valley Anomalies')
                # plt.scatter(df.index[df['anomaly_type_feature2'] == 'anomaly'], df.loc[df['anomaly_type_feature2'] == 'anomaly', 'close'], color='orange', label='Other Anomalies')


                # plt.title('Anomaly Detection Results')
                # plt.xlabel('Time')
                # plt.ylabel('Close Price')
                # plt.legend()
                # plt.grid(True)
                # plt.show()




                ############### AnomalyDetection ####################################################################################
                ############### AnomalyDetection ####################################################################################
                ############### AnomalyDetection ####################################################################################
                ############### AnomalyDetection ####################################################################################
                ############### AnomalyDetection ####################################################################################
                ############### AnomalyDetection ####################################################################################



                if interval == '1m':
                    shift_value = 2
                else:
                    shift_value = 1






                # maxima_divergences = handle_divergence(interval, df, 'maxima')
                # minima_divergences = handle_divergence(interval, df, 'minima')


                # for divergence in maxima_divergences:
                #     divergence_confirmer, divergence_name, divergence_point_1, divergence_open_time_1, divergence_close_1, divergence_rsi_1, divergence_point_2, divergence_open_time_2, divergence_close_2, divergence_rsi_2 = divergence
                #     df.loc[df['open_time2'] == divergence_open_time_2, ['divergence_confirmer', 'divergence_name', 'divergence_point_1', 'divergence_open_time_1', 'divergence_close_1', 'divergence_rsi_1', 'divergence_point_2', 'divergence_open_time_2', 'divergence_close_2', 'divergence_rsi_2']] = divergence_confirmer, divergence_name, divergence_point_1, divergence_open_time_1, divergence_close_1, divergence_rsi_1, divergence_point_2, divergence_open_time_2, divergence_close_2, divergence_rsi_2

                # for divergence in minima_divergences:
                #     divergence_confirmer, divergence_name, divergence_point_1, divergence_open_time_1, divergence_close_1, divergence_rsi_1, divergence_point_2, divergence_open_time_2, divergence_close_2, divergence_rsi_2 = divergence
                #     df.loc[df['open_time2'] == divergence_open_time_2, ['divergence_confirmer', 'divergence_name', 'divergence_point_1', 'divergence_open_time_1', 'divergence_close_1', 'divergence_rsi_1', 'divergence_point_2', 'divergence_open_time_2', 'divergence_close_2', 'divergence_rsi_2']] = divergence_confirmer, divergence_name, divergence_point_1, divergence_open_time_1, divergence_close_1, divergence_rsi_1, divergence_point_2, divergence_open_time_2, divergence_close_2, divergence_rsi_2






                # for divergence in maxima_divergences:
                #     divergence_confirmer, divergence_name, divergence_point_1, divergence_close_time_1, divergence_close_1, divergence_rsi_1, divergence_point_2, divergence_close_time_2, divergence_close_2, divergence_rsi_2 = divergence
                #     df.loc[df['close_time'] == divergence_close_time_2, ['divergence_confirmer', 'divergence_name', 'divergence_point_1', 'divergence_close_time_1', 'divergence_close_1', 'divergence_rsi_1', 'divergence_point_2', 'divergence_close_time_2', 'divergence_close_2', 'divergence_rsi_2']] = divergence_confirmer, divergence_name, divergence_point_1, divergence_close_time_1, divergence_close_1, divergence_rsi_1, divergence_point_2, divergence_close_time_2, divergence_close_2, divergence_rsi_2

                # for divergence in minima_divergences:
                #     divergence_confirmer, divergence_name, divergence_point_1, divergence_close_time_1, divergence_close_1, divergence_rsi_1, divergence_point_2, divergence_close_time_2, divergence_close_2, divergence_rsi_2 = divergence
                #     df.loc[df['close_time'] == divergence_close_time_2, ['divergence_confirmer', 'divergence_name', 'divergence_point_1', 'divergence_close_time_1', 'divergence_close_1', 'divergence_rsi_1', 'divergence_point_2', 'divergence_close_time_2', 'divergence_close_2', 'divergence_rsi_2']] = divergence_confirmer, divergence_name, divergence_point_1, divergence_close_time_1, divergence_close_1, divergence_rsi_1, divergence_point_2, divergence_close_time_2, divergence_close_2, divergence_rsi_2


                ####################################################################################################
                # mask9 = (df['divergence_point_2'] > 0)  #& (df['divergence_name'] > df.close*1/100)
                # df['divergence_point_2_long_last_3'] = df['divergence_point_2'].where(mask9).fillna(0).rolling(window=3, min_periods=1).apply(lambda x: any(x == 1) and 1 or 0).astype(int) # Create the 'b2' column using the rolling window
                # # df['divergence_point_2_long_last_3_divergence_name'] = df['divergence_name'].where(mask9).fillna(0).rolling(window=3, min_periods=1).max() # Create the 'c2' column based on the rolling maximum of 'c' for rows where 'b' is -1
                # # df['divergence_point_2_long_last_3_divergence_name'] = df['divergence_point_2_long_last_3_divergence_name'].fillna(0) # Fill NaN values in 'c2' with 0

                # mask10 = (df['divergence_point_2'] < 0)
                # df['divergence_point_2_short_last_3'] = df['divergence_point_2'].where(mask10).fillna(0).rolling(window=3, min_periods=1).apply(lambda x: any(x == -1) and -1 or 0).astype(int) # Create the 'b2' column using the rolling window
                # # df['divergence_point_2_short_last_3_divergence_name'] = df['divergence_name'].where(mask10).fillna(0).rolling(window=3, min_periods=1).max() # Create the 'c2' column based on the rolling maximum of 'c' for rows where 'b' is -1
                # # df['divergence_point_2_short_last_3_divergence_name'] = df['divergence_point_2_short_last_3_divergence_name'].fillna(0) # Fill NaN values in 'c2' with 0
                ####################################################################################################



            
                #df['maxima_peak_x_close'][maxima_peak_x_close] = -1
                #df['minima_peak_x_close'][minima_peak_x_close] = 1
                #df['maxima_peak_x_rsi'][maxima_peak_x_rsi] = -1
                #df['minima_peak_x_rsi'][minima_peak_x_rsi] = 1
                #df['maxima_peak_x_macd'][maxima_peak_x_macd] = -1
                #df['minima_peak_x_macd'][minima_peak_x_macd] = 1
                #df['maxima_peak_x_adx'][maxima_peak_x_adx] = 0
                #df['minima_peak_x_adx'][minima_peak_x_adx] = 1
                #df['maxima_peak_x_dmp'][maxima_peak_x_dmp] = -1
                #df['minima_peak_x_dmp'][minima_peak_x_dmp] = 1
                #df['maxima_peak_x_dmn'][maxima_peak_x_dmn] = 1
                #df['minima_peak_x_dmn'][minima_peak_x_dmn] = -1
                #df['maxima_peak_x_obv'][maxima_peak_x_obv] = -1
                #df['minima_peak_x_obv'][minima_peak_x_obv] = 1
                df.loc[((((df.maxima_peak_x_MACD_12_26_9 < 0) & (df.MACD_12_26_9 < 10)) & (df.maxima_peak_x_rsi < 0)) ) | \
                    ((((df.maxima_peak_x_MACD_12_26_9 < 0) & (df.MACD_12_26_9 < 10)) & (df.maxima_peak_x_close < 0)) )\
                    , "maxima_peak_x_macd_with_condition"] = -2
                df.loc[((((df.minima_peak_x_MACD_12_26_9 > 0) & (df.MACD_12_26_9 > -10)) & (df.minima_peak_x_rsi > 0)) ) | \
                    ((((df.minima_peak_x_MACD_12_26_9 > 0) & (df.MACD_12_26_9 > -10)) & (df.minima_peak_x_close > 0)) )\
                    , "minima_peak_x_macd_with_condition"] = 2
        #        df.loc[((((df.maxima_peak_x_macd < 0) & (df.MACD_12_26_9 < 10)) & (df.maxima_peak_x_rsi < 0))&((df.minima_peak_x_adx > 0 ) | (df.maxima_peak_x_adx < 0 ) | (df.maxima_peak_x_dmp < 0 )   |   (df.minima_peak_x_dmn < 0 )    ) ) | \
        #               ((((df.maxima_peak_x_macd < 0) & (df.MACD_12_26_9 < 10)) & (df.maxima_peak_x_close < 0))&((df.minima_peak_x_adx > 0 ) | (df.maxima_peak_x_adx < 0 )  | (df.maxima_peak_x_dmp < 0 )   |   (df.minima_peak_x_dmn < 0 )   ) )\
        #               , "maxima_peak_x_macd_with_condition"] = -2
        #    
        #        df.loc[((((df.minima_peak_x_macd > 0) & (df.MACD_12_26_9 > -10)) & (df.minima_peak_x_rsi > 0))&((df.minima_peak_x_adx > 0 )| (df.maxima_peak_x_adx < 0 ) | (df.minima_peak_x_dmp > 0 )   |   (df.maxima_peak_x_dmn > 0 )    ) ) | \
        #               ((((df.minima_peak_x_macd > 0) & (df.MACD_12_26_9 > -10)) & (df.minima_peak_x_close > 0))&((df.minima_peak_x_adx > 0 )| (df.maxima_peak_x_adx < 0 ) | (df.minima_peak_x_dmp > 0 )   |   (df.maxima_peak_x_dmn > 0 )    ) )\
        #               , "minima_peak_x_macd_with_condition"] = 2
        #
        #        df.loc[(df.maxima_peak_x_macd < 0 ) & (df.MACD_12_26_9 < 5), "maxima_peak_x_macd_with_condition"] = -1
        #        df.loc[(df.minima_peak_x_macd > 0 ) & (df.MACD_12_26_9 > -5), "minima_peak_x_macd_with_condition"] = 1
        #
        #        df['total_short'] = df.maxima_peak_x_macd_with_condition + df.maxima_peak_x_rsi
        #        df['total_long'] = df.minima_peak_x_macd_with_condition + df.minima_peak_x_rsi

                ##########################################################################################
                df.loc[((df.maxima_peak_x_MACD_12_26_9 < 0) & (df.MACD_12_26_9 > 0)), "maxima_peak_x_macd_short_only_when_greater_than_zero"] = -1 # short only when 'df.MACD_12_26_9 > 0'
                df.loc[((df.minima_peak_x_MACD_12_26_9 > 0) & (df.MACD_12_26_9 < 0)), "minima_peak_x_macd_long_only_when_less_than_zero"] = 1 # long only when 'df.MACD_12_26_9 < 0'
                ##########################################################################################
        #######################################################################################################################################
        #######################################################################################################################################
        #######################################################################################################################################
        #######################################################################################################################################
        #######################################################################################################################################
                #df['total_short'] = df.maxima_peak_x_close + df.maxima_peak_x_rsi + df.maxima_peak_x_macd_short_only_when_greater_than_zero + df.maxima_peak_x_dmp_with_dmn # short only when 'df.MACD_12_26_9 > 0'
                #df['total_long'] = df.minima_peak_x_close + df.minima_peak_x_rsi + df.minima_peak_x_macd_long_only_when_less_than_zero + df.minima_peak_x_dmp_with_dmn # long only when 'df.MACD_12_26_9 < 0'
                #df['total_short'] = df.maxima_peak_x_macd_short_only_when_greater_than_zero # short only when 'df.MACD_12_26_9 > 0'
                #df['total_long'] = df.minima_peak_x_macd_long_only_when_less_than_zero # long only when 'df.MACD_12_26_9 < 0'

                df['total_short_base'] = df.maxima_peak_x_close + df.maxima_peak_x_rsi + df.maxima_peak_x_MACD_12_26_9 + df.maxima_peak_x_kdj # + df.maxima_peak_x_dmp_with_dmn
                df['total_long_base'] = df.minima_peak_x_close + df.minima_peak_x_rsi + df.minima_peak_x_MACD_12_26_9 + df.minima_peak_x_kdj # + df.minima_peak_x_dmp_with_dmn
                # df['total_short_base'] = df.findpeaks_peak + df.findpeaks_peak_rsi + df.findpeaks_peak_macd + df.findpeaks_peak_kdj + df.findpeaks_peak_dmp_with_dmn
                # df['total_long_base'] = df.findpeaks_valley + df.findpeaks_valley_rsi + df.findpeaks_valley_macd + df.findpeaks_valley_kdj + df.findpeaks_valley_dmp_with_dmn
                # df['total_short_base'] = df.findpeaks_peak + df.findpeaks_peak_rsi + df.findpeaks_peak_macd + df.findpeaks_peak_kdj
                # df['total_long_base'] = df.findpeaks_valley + df.findpeaks_valley_rsi + df.findpeaks_valley_macd + df.findpeaks_valley_kdj


########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
                # if ('SMA_50' in df.columns): # SMA_200 있을 시, 사용
                #     #df.loc[(df['atr_AMATe_LR_8_21_2'] > 0) & (df.close < df.SMA_200) & (df.total_short_base < -1) , "total_short"] = df.total_short_base # with atr
                #     #df.loc[(df['atr_AMATe_LR_8_21_2'] > 0) & (df.close > df.SMA_200) & (df.total_long_base > 1) , "total_long"] = df.total_long_base # with atr
                #     #df.loc[(df.close < df.SMA_200) & (df.total_short_base < -1) , "total_short"] = df.total_short_base # without atr
                #     #df.loc[(df.close > df.SMA_200) & (df.total_long_base > 1) , "total_long"] = df.total_long_base # without atr
                #     # df.loc[((df['RSI_14'] > 70) & (df['rsi_AMATe_LR_8_21_2'] > 0) & (df['macd_AMATe_LR_8_21_2'] > 0)  & (df['atr_AMATe_LR_8_21_2'] > 0)) & (df.close < df.SMA_200) & (df.total_short_base < -1) , "stg5_short"] = df.total_short_base # without atr
                #     # df.loc[((df['RSI_14'] < 30) & (df['rsi_AMATe_SR_8_21_2'] > 0) & (df['macd_AMATe_SR_8_21_2'] > 0)  & (df['atr_AMATe_LR_8_21_2'] > 0)) & (df.close > df.SMA_200) & (df.total_long_base > 1) , "stg5_long"] = df.total_long_base # without atr
                #     # df.loc[((((df['RSI_14'] > 70) & df['close_AMATe_LR_8_21_2'] > 0) & (df['obv_AMATe_LR_8_21_2'] > 0) & (df['rsi_AMATe_LR_8_21_2'] > 0) & (df['macd_AMATe_LR_8_21_2'] > 0) ) & (df.close < df.SMA_200) & (df.total_short_base < -1)), "stg5_short"] = df.total_short_base # without atr
                #     # df.loc[((((df['RSI_14'] < 30) & df['close_AMATe_SR_8_21_2'] > 0) & (df['obv_AMATe_SR_8_21_2'] > 0) & (df['rsi_AMATe_SR_8_21_2'] > 0) & (df['macd_AMATe_SR_8_21_2'] > 0) ) & (df.close > df.SMA_200) & (df.total_long_base > 1)), "stg5_long"] = df.total_long_base # without atr
                #     df.loc[( (df['close'] > df['BBU_20_2.0']) & ((df.close < df.SMA_50) ) & (df.total_short_base < -1)) | (((df['RSI_14'] > 70) & (df['rsi_AMATe_LR_8_21_2'] > 0) & (df['macd_AMATe_LR_8_21_2'] > 0) & (df['obv_AMATe_LR_8_21_2'] > 0) & (df['close_AMATe_LR_8_21_2'] > 0) & (df.close > df.SMA_50)) & (df.total_short_base < -1)), "stg5_short"] = df.total_short_base # without atr
                #     df.loc[( (df['close'] < df['BBL_20_2.0']) & ((df.close > df.SMA_50) ) & (df.total_long_base > 1)) | (((df['RSI_14'] < 30) & (df['rsi_AMATe_SR_8_21_2'] > 0) & (df['macd_AMATe_SR_8_21_2'] > 0) & (df['obv_AMATe_SR_8_21_2'] > 0) & (df['close_AMATe_SR_8_21_2'] > 0) & (df.close < df.SMA_50)) & (df.total_long_base > 1)), "stg5_long"] = df.total_long_base # without atr
                #     df.loc[(((df['RSI_14'] > 70) & (df['rsi_AMATe_LR_8_21_2'] > 0) & (df['macd_AMATe_LR_8_21_2'] > 0) & (df['obv_AMATe_LR_8_21_2'] > 0) & (df['close_AMATe_LR_8_21_2'] > 0) & (df.close > df.SMA_50)) & (df.total_short_base < -1)), "total_short"] = df.total_short_base # without atr
                #     df.loc[(((df['RSI_14'] < 30) & (df['rsi_AMATe_SR_8_21_2'] > 0) & (df['macd_AMATe_SR_8_21_2'] > 0) & (df['obv_AMATe_SR_8_21_2'] > 0) & (df['close_AMATe_SR_8_21_2'] > 0) & (df.close < df.SMA_50)) & (df.total_long_base > 1)), "total_long"] = df.total_long_base # without atr

                # # else: # SMA_200 없을 시 기존 total_stg
                # #     df['total_short'] = df.total_short_base
                # #     df['total_long'] = df.total_long_base
                # #df.loc[((df['volume_pct_change'] > 8) & ((df['RSI_14'] > 70)) & (df['close'] > df['BBU_20_2.0'])) & (df.total_short_base < -1), "total_short"] = df.total_short_base # without atr
                # #df.loc[((df['volume_pct_change'] > 8) & ((df['RSI_14'] < 30)) & (df['close'] < df['BBL_20_2.0'])) & (df.total_long_base > 1), "total_long"] = df.total_long_base # without atr
                # #df.loc[((df['volume_pct_change'] > 8) & ((df['RSI_14'] > 70)) & (df['close'] > df['BBU_20_2.0'])) & (df.total_short_base < -1), "total_short"] = df.total_short_base # without atr
                # #df.loc[((df['volume_pct_change'] > 8) & ((df['RSI_14'] < 30)) & (df['close'] < df['BBL_20_2.0'])) & (df.total_long_base > 1), "total_long"] = df.total_long_base # without atr
                # # df.loc[((df['RSI_14'] > 70) & (df['rsi_AMATe_LR_8_21_2'] > 0) & (df['macd_AMATe_LR_8_21_2'] > 0)  & (df['atr_AMATe_LR_8_21_2'] > 0)) & (df.total_short_base < -1), "total_short"] = df.total_short_base # without atr
                # # df.loc[((df['RSI_14'] < 30) & (df['rsi_AMATe_SR_8_21_2'] > 0) & (df['macd_AMATe_SR_8_21_2'] > 0)  & (df['atr_AMATe_LR_8_21_2'] > 0)) & (df.total_long_base > 1), "total_long"] = df.total_long_base # without atr
                # #df.loc[((((df['RSI_14'] > 70) & df['close_AMATe_LR_8_21_2'] > 0) & (df['obv_AMATe_LR_8_21_2'] > 0) & (df['rsi_AMATe_LR_8_21_2'] > 0) & (df['macd_AMATe_LR_8_21_2'] > 0) ) & (df.total_short_base < -1)), "total_short"] = df.total_short_base # without atr
                # #df.loc[((((df['RSI_14'] < 30) & df['close_AMATe_SR_8_21_2'] > 0) & (df['obv_AMATe_SR_8_21_2'] > 0) & (df['rsi_AMATe_SR_8_21_2'] > 0) & (df['macd_AMATe_SR_8_21_2'] > 0) ) & (df.total_long_base > 1)), "total_long"] = df.total_long_base # without atr
                # # df.loc[((((df['RSI_14'] > 70) & df['close_AMATe_LR_8_21_2'] > 0) & (df['obv_AMATe_LR_8_21_2'] > 0) & (df['rsi_AMATe_LR_8_21_2'] > 0) & (df['macd_AMATe_LR_8_21_2'] > 0) ) & (df.total_short_base < -1)), "total_short"] = df.total_short_base # without atr
                # # df.loc[((((df['RSI_14'] < 30) & df['close_AMATe_SR_8_21_2'] > 0) & (df['obv_AMATe_SR_8_21_2'] > 0) & (df['rsi_AMATe_SR_8_21_2'] > 0) & (df['macd_AMATe_SR_8_21_2'] > 0) ) & (df.total_long_base > 1)), "total_long"] = df.total_long_base # without atr
                # else:
                #     df.loc[(((df['RSI_14'] > 70) & (df['rsi_AMATe_LR_8_21_2'] > 0) & (df['macd_AMATe_LR_8_21_2'] > 0) & (df['obv_AMATe_LR_8_21_2'] > 0) & (df['close_AMATe_LR_8_21_2'] > 0)) & (df.total_short_base < -1)), "stg5_short"] = df.total_short_base # without atr
                #     df.loc[(((df['RSI_14'] < 30) & (df['rsi_AMATe_SR_8_21_2'] > 0) & (df['macd_AMATe_SR_8_21_2'] > 0) & (df['obv_AMATe_SR_8_21_2'] > 0) & (df['close_AMATe_SR_8_21_2'] > 0)) & (df.total_long_base > 1)), "stg5_long"] = df.total_long_base # without atr
                #     df.loc[(((df['RSI_14'] > 70) & (df['rsi_AMATe_LR_8_21_2'] > 0) & (df['macd_AMATe_LR_8_21_2'] > 0) & (df['obv_AMATe_LR_8_21_2'] > 0) & (df['close_AMATe_LR_8_21_2'] > 0)) & (df.total_short_base < -1)), "total_short"] = df.total_short_base # without atr
                #     df.loc[(((df['RSI_14'] < 30) & (df['rsi_AMATe_SR_8_21_2'] > 0) & (df['macd_AMATe_SR_8_21_2'] > 0) & (df['obv_AMATe_SR_8_21_2'] > 0) & (df['close_AMATe_SR_8_21_2'] > 0)) & (df.total_long_base > 1)), "total_long"] = df.total_long_base # without atr

                # df.loc[(((df['RSI_14'] > 70) & (df['rsi_AMATe_LR_8_21_2'] > 0) & (df['macd_AMATe_LR_8_21_2'] > 0) & (df['obv_AMATe_LR_8_21_2'] > 0) & (df['close_AMATe_LR_8_21_2'] > 0)) & (df.total_short_base < -1)), "stg5_short"] = df.total_short_base # without atr
                # df.loc[(((df['RSI_14'] < 30) & (df['rsi_AMATe_SR_8_21_2'] > 0) & (df['macd_AMATe_SR_8_21_2'] > 0) & (df['obv_AMATe_SR_8_21_2'] > 0) & (df['close_AMATe_SR_8_21_2'] > 0)) & (df.total_long_base > 1)), "stg5_long"] = df.total_long_base # without atr

                #df.loc[((df['volume_pct_change'] >= 1) & (df.total_short_base < -2)), "stg5_short"] = df.total_short_base # without atr
                #df.loc[((df['volume_pct_change'] >= 1) & (df.total_long_base > 2)), "stg5_long"] = df.total_long_base # without atr
                # df.loc[((df['RSI_14'] > 70) & (df['close_AMATe_LR_8_21_2'] > 0) & (df['kdj_AMATe_LR_8_21_2'] > 0) & (df['rsi_AMATe_LR_8_21_2'] > 0) & (df['macd_AMATe_LR_8_21_2'] > 0) & (df.total_short_base < -1)), "stg5_short"] = df.total_short_base # without atr
                # df.loc[((df['RSI_14'] < 30) & (df['close_AMATe_SR_8_21_2'] > 0) & (df['kdj_AMATe_SR_8_21_2'] > 0) & (df['rsi_AMATe_SR_8_21_2'] > 0) & (df['macd_AMATe_SR_8_21_2'] > 0) & (df.total_long_base > 1)), "stg5_short"] = df.total_long_base # without atr
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################

                # # trend 결정 용도
                # df.loc[((df['RSI_14'] > 30) & (df['close_AMATe_SR_8_21_2'] > 0) & (df['kdj_AMATe_SR_8_21_2'] > 0) & (df['rsi_AMATe_SR_8_21_2'] > 0) & (df['macd_AMATe_SR_8_21_2'] > 0)) & ((df['close_AMATe_LR_8_21_2'] == 0) & (df['kdj_AMATe_LR_8_21_2'] == 0) & (df['rsi_AMATe_LR_8_21_2'] == 0) & (df['macd_AMATe_LR_8_21_2'] == 0)), "stg_scalping_0_short"] = -2 # without atr
                # df.loc[((df['RSI_14'] < 70) & (df['close_AMATe_LR_8_21_2'] > 0) & (df['kdj_AMATe_LR_8_21_2'] > 0) & (df['rsi_AMATe_LR_8_21_2'] > 0) & (df['macd_AMATe_LR_8_21_2'] > 0)) & ((df['close_AMATe_SR_8_21_2'] == 0) & (df['kdj_AMATe_SR_8_21_2'] == 0) & (df['rsi_AMATe_SR_8_21_2'] == 0) & (df['macd_AMATe_SR_8_21_2'] == 0)), "stg_scalping_0_long"] = 2 # without atr

                # # stg1, scalping 용도
                # df.loc[(df.maxima_peak_x_macd < 0), "stg1_short"] = 2# df.maxima_peak_x_macd # without atr
                # df.loc[(df.minima_peak_x_macd > 0), "stg1_long"] = 2# df.minima_peak_x_macd # without atr

                # # stg2, stg4
                # df.loc[((df.total_short_base < -3)), "stg2_short"] = df.total_short_base # without atr
                # df.loc[((df.total_long_base > 3)), "stg2_long"] = df.total_long_base # without atr

                # # stg3, stg9
                # df.loc[((df['RSI_14'] > 70) & (df['close_AMATe_LR_8_21_2'] > 0) & (df['kdj_AMATe_LR_8_21_2'] > 0) & (df['rsi_AMATe_LR_8_21_2'] > 0) & (df['macd_AMATe_LR_8_21_2'] > 0) & (df.total_short_base < -1)), "stg3_short"] = df.total_short_base # without atr
                # df.loc[((df['RSI_14'] < 30) & (df['close_AMATe_SR_8_21_2'] > 0) & (df['kdj_AMATe_SR_8_21_2'] > 0) & (df['rsi_AMATe_SR_8_21_2'] > 0) & (df['macd_AMATe_SR_8_21_2'] > 0) & (df.total_long_base > 1)), "stg3_long"] = df.total_long_base # without atr
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################

                # trend 결정 용도
                # df.loc[(((df['RSI_14'] > 20)) & (df.close < df.SMA_50) & (df['close_AMATe_SR_8_21_2'] > 0) & (df['kdj_AMATe_SR_8_21_2'] > 0) & (df['rsi_AMATe_SR_8_21_2'] > 0) & (df['macd_AMATe_SR_8_21_2'] > 0) & ((df['atr_AMATe_LR_8_21_2'] > 0) | (df['atr_AMATe_SR_8_21_2'] > 0))), "stg_scalping_0_short"] = -2  # without atr
                # df.loc[(((df['RSI_14'] < 80)) & (df.close > df.SMA_50) & (df['close_AMATe_LR_8_21_2'] > 0) & (df['kdj_AMATe_LR_8_21_2'] > 0) & (df['rsi_AMATe_LR_8_21_2'] > 0) & (df['macd_AMATe_LR_8_21_2'] > 0) & ((df['atr_AMATe_LR_8_21_2'] > 0) | (df['atr_AMATe_SR_8_21_2'] > 0))), "stg_scalping_0_long"] = 2  # without atr

                df.loc[((df.DX_14 > df.DMN_14) & (df.dx_diff > 0) & (df.dmn_diff > 0) & (df.DMN_14 > df.DMP_14)), "stg_scalping_0_short"] = -2  # without atr
                df.loc[((df.DX_14 > df.DMP_14) & (df.dx_diff > 0) & (df.dmp_diff > 0) & (df.DMP_14 > df.DMN_14)), "stg_scalping_0_long"] = 2  # without atr

                # # stg1, scalping 용도
                # df.loc[(((df.maxima_peak_x_macd_short_only_when_greater_than_zero < 0)) ), "stg1_short"] = -2 # without atr
                # df.loc[(((df.minima_peak_x_macd_long_only_when_less_than_zero > 0)) ), "stg1_long"] = 2 # without atr
                
                # # # stg2, scalping 용도
                # df.loc[((df.total_short_base < -3)), "stg2_short"] = df.total_short_base # without atr
                # df.loc[((df.total_long_base > 3)), "stg2_long"] = df.total_long_base # without atr

                # stg3, stg9, Absolute pick - old
                # df.loc[((df['RSI_14'] > 70) & (df['close_AMATe_LR_8_21_2'] > 0) & (df['kdj_AMATe_LR_8_21_2'] > 0) & (df['rsi_AMATe_LR_8_21_2'] > 0) & (df['macd_AMATe_LR_8_21_2'] > 0) & (df.total_short_base < -1)) | ((df['RSI_14'] > 60) & (df.close > df.SMA_50) & (df['close_AMATe_LR_8_21_2'] > 0) & (df['kdj_AMATe_LR_8_21_2'] > 0) & (df['rsi_AMATe_LR_8_21_2'] > 0) & (df['macd_AMATe_LR_8_21_2'] > 0) & (df.total_short_base < -2)), "stg3_short"] = df.total_short_base # without atr
                # df.loc[((df['RSI_14'] < 30) & (df['close_AMATe_SR_8_21_2'] > 0) & (df['kdj_AMATe_SR_8_21_2'] > 0) & (df['rsi_AMATe_SR_8_21_2'] > 0) & (df['macd_AMATe_SR_8_21_2'] > 0) & (df.total_long_base > 1)) | ((df['RSI_14'] < 40) & (df.close < df.SMA_50) & (df['close_AMATe_SR_8_21_2'] > 0) & (df['kdj_AMATe_SR_8_21_2'] > 0) & (df['rsi_AMATe_SR_8_21_2'] > 0) & (df['macd_AMATe_SR_8_21_2'] > 0) & (df.total_long_base > 2)), "stg3_long"] = df.total_long_base # without atr

                # stg4, stg16, Absolute pick - old
                # df.loc[((df.maxima_peak_x_close < 0) & (df['RSI_14'] > 30) & (df['RSI_14'] < 75) & (df.MACDh_12_26_9 > 0) & (df.MACD_12_26_9 > 0) & (df['close_AMATe_LR_8_21_2'] > 0) & (df['kdj_AMATe_SR_8_21_2'] > 0) & (df['rsi_AMATe_SR_8_21_2'] > 0) & (df['macd_AMATe_LR_8_21_2'] > 0) & (df['macdh_AMATe_SR_8_21_2'] > 0)) | ((df.maxima_peak_x_macd < 0) & (df.MACDh_12_26_9 > 0) & (df.MACD_12_26_9 < 0) & (df['RSI_14'] > 30) & (df.close < df['SMA_50']) ), "stg4_short"] = -2 # without atr
                # df.loc[((df.minima_peak_x_close > 0) & (df['RSI_14'] < 70) & (df['RSI_14'] > 25) & (df.MACDh_12_26_9 < 0) & (df.MACD_12_26_9 < 0) & (df['close_AMATe_SR_8_21_2'] > 0) & (df['kdj_AMATe_LR_8_21_2'] > 0) & (df['rsi_AMATe_LR_8_21_2'] > 0) & (df['macd_AMATe_SR_8_21_2'] > 0) & (df['macdh_AMATe_LR_8_21_2'] > 0)) | ((df.minima_peak_x_macd > 0) & (df.MACDh_12_26_9 < 0) & (df.MACD_12_26_9 > 0) & (df['RSI_14'] < 70) & (df.close > df['SMA_50']) ), "stg4_long"] = 2 # without atr

                ####################################################################################################################################
                ####################################################################################################################################
                ####################################################################################################################################
                ####################################################################################################################################



                # # stg3, stg9, Absolute pick # 1. 트랜드 방향에 따른, 포지션 최초 진입에 사용
                # df.loc[ \
                # # ((df.total_short_base < -1) & (df.MACDh_12_26_9 > 0)) \
                # # ((df.total_short_base < -2) & (df.MACDh_12_26_9 > 0) & ((df['J_9_3'] > df['K_9_3']) & (df['K_9_3'] > df['D_9_3'])) ) \
                # (df.maxima_peak_x_macd < 0) & (df.MACD_12_26_9 > 0) & (df.ADX_14 >= 25) & (df.ADX_14 <= 45) # & (df.close < df['SMA_200'] ) \
                # # ((df.findpeaks_peak < 0) & (df.MACDh_12_26_9 > 0)) \
                # # ((df.total_short_base < -3)) | \
                # # ((df.findpeaks_peak < 0) & (df['RSI_14'] > 30) & (df['RSI_14'] < 75) & (df.MACDh_12_26_9 > 0) & (df.MACD_12_26_9 > 0)) \
                # # ((df.findpeaks_peak < 0) & (df.findpeaks_peak_rsi < 0) & (df['RSI_14'] > 70)) | \
                # # ((df.findpeaks_peak < 0) & (df.minima_peak_x_dmn < 0)) | \
                # # ((df.findpeaks_peak_macd < 0) & (df.MACDh_12_26_9 > 0) & (df.MACD_12_26_9 < 0) & (df['RSI_14'] > 30) & (df.close < df['SMA_50'])) \
                # , "stg1_short"] = -2 # without atr

                # df.loc[ \
                # # ((df.total_long_base > 1) & (df.MACDh_12_26_9 < 0)) \
                # # ((df.total_long_base > 2) & (df.MACDh_12_26_9 < 0) & ((df['J_9_3'] < df['K_9_3']) & (df['K_9_3'] < df['D_9_3'])) ) \
                # (df.minima_peak_x_macd > 0) & (df.MACD_12_26_9 < 0) & (df.ADX_14 >= 25) & (df.ADX_14 <= 45) # & (df.close > df['SMA_200'] ) \
                # # ((df.findpeaks_valley > 0) & (df.MACDh_12_26_9 < 0)) \
                # # ((df.total_long_base > 3)) | \
                # # ((df.findpeaks_valley > 0) & (df['RSI_14'] < 70) & (df['RSI_14'] > 25) & (df.MACDh_12_26_9 < 0) & (df.MACD_12_26_9 < 0)) \
                # # ((df.findpeaks_valley > 0) & (df.findpeaks_valley_rsi > 0) & (df['RSI_14'] < 30)) | \
                # # ((df.findpeaks_valley > 0) & (df.minima_peak_x_dmp > 0)) | \
                # # ((df.findpeaks_valley_macd > 0) & (df.MACDh_12_26_9 < 0) & (df.MACD_12_26_9 > 0) & (df['RSI_14'] < 70) & (df.close > df['SMA_50'])) \
                # , "stg1_long"] = 2 # without atr


                df.loc[
                    # (
                    #     (df.minima_peak_x_macd > 0)
                    # )
                    # |
                    ((df.DMN_14 > df.DMP_14) & (df.ATRr_1 > df.close*6/100) & (df.ATRr_1 > (df.ATRr_14 * 2)))
                , "stg110_long"] = 2 # without atr
            
                df.loc[
                    # (
                    #     (df.maxima_peak_x_macd < 0)
                    # )
                    # |
                    ((df.DMP_14 > df.DMN_14) & (df.ATRr_1 > df.close*6/100) & (df.ATRr_1 > (df.ATRr_14 * 2)))
                , "stg110_short"] = -2 # without atr





                df.loc[
                    (
                        (   #절대픽 '1m', '5m', '15m',
                            (
                                (
                                    (df['anomalies_second_combined_diff'] > 0) 
                                    & (df['second_combined_diff'] < df['second_combined_diff'].quantile(0.5))
                                    & (df.MACDh_12_26_9 > 0)
                                    & (df['second_combined_diff_filtered_diff'] > 0)
                                    & (df['second_combined_diff_diff'] < 0)
                                    # & (df.minima_peak_x_close > 0)
                                )
                            )
                            & 
                            (
                                (
                                    # (df['combined_diff_filtered_diff'] < 0)
                                    # & 
                                    (df['combined_diff_filtered'] > 0.5)
                                    & 
                                    (df['second_combined_diff_filtered'] < -0.6)
                                    # & 
                                    # (df.RSI_14 < 30)
                                    # & (df.MACDh_12_26_9 > 0)
                                )
                                #     | 
                                # (
                                #     (df['combined_diff_filtered_diff'] > 0)
                                #     & (df['combined_diff_filtered'] < 0.3)
                                # )
                                # # (
                                # #     (df['combined_diff_filtered_diff'] > 0)
                                # #     & (df['combined_diff_filtered'] < 0.2)
                                # # )
                            )
                        )
                            |
                        (
                            # (df['anomalies_MACDh_50_75_35'] > 0) 
                            # & (df['MACDh_50_75_35'] < df['MACDh_50_75_35'].quantile(0.5))
                            # & 
                            (df['anomalies_MACD_50_75_35'] > 0) 
                            & (df['MACD_50_75_35'] < df['MACD_50_75_35'].quantile(0.5))
                            & (df['second_combined_diff_filtered_diff'] > 0)
                            & (df['second_combined_diff_diff'] < 0)
                            # & (df.MACDh_12_26_9 > 0)
                            # & (df.minima_peak_x_close > 0)
                        )
                    )
                    , "stg10_long"] = 2 # without atr

                df.loc[
                    (
                        (   #절대픽 '1m', '5m', '15m',
                            (
                                (
                                    (df['anomalies_second_combined_diff'] > 0) 
                                    & (df['second_combined_diff'] > df['second_combined_diff'].quantile(0.5))
                                    & (df.MACDh_12_26_9 < 0)
                                    & (df['second_combined_diff_filtered_diff'] < 0)
                                    & (df['second_combined_diff_diff'] > 0)
                                    # & (df.maxima_peak_x_close < 0)
                                )
                            )
                            & 
                            (
                                (
                                    # (df['combined_diff_filtered_diff'] < 0)
                                    # & 
                                    (df['combined_diff_filtered'] > 0.5)
                                    & 
                                    (df['second_combined_diff_filtered'] > 0.6)
                                    # & 
                                    # (df.RSI_14 > 70)
                                    # & (df.MACDh_12_26_9 < 0)
                                )
                                #     | 
                                # (
                                #     (df['combined_diff_filtered_diff'] > 0)
                                #     & (df['combined_diff_filtered'] < 0.3)
                                # )
                                # # (
                                # #     (df['combined_diff_filtered_diff'] > 0)
                                # #     & (df['combined_diff_filtered'] < 0.2)
                                # # )
                            )
                        )
                        |
                        (
                            # (df['anomalies_MACDh_50_75_35'] > 0) 
                            # & (df['MACDh_50_75_35'] > df['MACDh_50_75_35'].quantile(0.5))
                            # & 
                            (df['anomalies_MACD_50_75_35'] > 0) 
                            & (df['MACD_50_75_35'] > df['MACD_50_75_35'].quantile(0.5))
                            & (df['second_combined_diff_filtered_diff'] < 0)
                            & (df['second_combined_diff_diff'] > 0)
                            # & (df.MACDh_12_26_9 < 0)
                            # & (df.maxima_peak_x_close < 0)
                        )
                    )
                    , "stg10_short"] = -2 # without atr




                df.loc[
                    (
                        (
                            # (
                            #     (
                            #         # (df['combined_diff_filtered_diff'] > 0)
                            #         # &
                            #         # (df['combined_diff_filtered'] < 0.4)
                            #         # & 


                            #         (df['second_combined_diff_filtered'] < -0.6)
                            #         & 
                            #         (df.RSI_14 < 30)
                            #         # &
                            #         # (df.DX_14 > 40)
                            #         # & (df.MACD_12_26_9 < 0)
                            #         # & (df.MACD_50_75_35 < 0)
                            #     )
                            # )

                            (
                                # (
                                #     # (df['combined_diff_filtered_diff'] < 0)
                                #     # & 
                                #     (df['combined_diff_filtered'] > 0.5)
                                #     & 
                                #     (df['second_combined_diff_filtered'] < -0.6)
                                #     & 
                                #     (df.RSI_14 < 30)
                                #     & (df.MACDh_12_26_9 > 0)
                                # )
                            #         | 
                                (
                                    # (df['combined_diff_filtered_diff'] < 0)
                                    # & 
                                    (df['combined_diff_filtered'] < 0.3)
                                    & 
                                    (df['second_combined_diff_filtered'] < -0.6)
                                    # & 
                                    # (df.RSI_14 < 30)
                                    & 
                                    (df.MACDh_12_26_9 > 0)
                                )
                            )
                            &
                            (
                                (df['minima_peak_x_second_combined_diff_filtered'] > 0)
                                #     | 
                                # (df['minima_peak_x_macd'] > 0)
                            )
                        )
                            |
                        (
                            (df.minima_peak_x_MACD_50_75_35 > 0)
                                &
                            (df['combined_diff_filtered_diff'] > 0)
                                &
                            (df['combined_diff_filtered'] < 0.3)
                            # (df.abs_dmp_dmn_200_diff < 5)
                                &
                            (df['second_combined_diff_filtered_diff'] > 0)
                                &
                            (df['second_combined_diff_filtered'] < -0.5)
                        )
                    )
                    , "stg3_long"] = 2 # without atr

                df.loc[
                    (
                        (
                            # (
                            #     (
                            #         # (df['combined_diff_filtered_diff'] > 0)
                            #         # &
                            #         # (df['combined_diff_filtered'] < 0.4)
                            #         # & 


                            #         (df['second_combined_diff_filtered'] > 0.6)
                            #         & 
                            #         (df.RSI_14 > 70)
                            #         # &
                            #         # (df.DX_14 > 40)
                            #         # & (df.MACD_12_26_9 > 0)
                            #         # & (df.MACD_50_75_35 > 0)
                            #     )
                            # )


                            (
                                # (
                                #     # (df['combined_diff_filtered_diff'] < 0)
                                #     # & 
                                #     (df['combined_diff_filtered'] > 0.5)
                                #     & 
                                #     (df['second_combined_diff_filtered'] > 0.6)
                                #     & 
                                #     (df.RSI_14 > 70)
                                #     & (df.MACDh_12_26_9 < 0)
                                # )
                            #         | 
                                (
                                    # (df['combined_diff_filtered_diff'] < 0)
                                    # & 
                                    (df['combined_diff_filtered'] < 0.3)
                                    & 
                                    (df['second_combined_diff_filtered'] > 0.6)
                                    # & 
                                    # (df.RSI_14 > 70)
                                    & 
                                    (df.MACDh_12_26_9 < 0)
                                )
                            )

                            & 
                            (
                                (df['maxima_peak_x_second_combined_diff_filtered'] < 0)
                                #     | 
                                # (df['maxima_peak_x_macd'] < 0)
                            )
                        )
                        |
                        (
                            (df.maxima_peak_x_MACD_50_75_35 < 0)
                                &
                            (df['combined_diff_filtered_diff'] > 0)
                                &
                            (df['combined_diff_filtered'] < 0.3)
                            # (df.abs_dmp_dmn_200_diff < 5)
                                &
                            (df['second_combined_diff_filtered_diff'] < 0)
                                &
                            (df['second_combined_diff_filtered'] > 0.5)
                        )
                    )
                    , "stg3_short"] = -2 # without atr
                

                # df.loc[
                #     (
                #         # (df.minima_peak_x_MACD_12_26_9 > 0) # | 
                #         (
                #             (df.minima_peak_x_MACD_50_75_35 > 0)
                #                 &
                #             (df['combined_diff_filtered_diff'] > 0)
                #                 &
                #             # (df['combined_diff_filtered'] < 0.5)
                #             (df.abs_dmp_dmn_200_diff < 5)
                #                 &
                #             (df['second_combined_diff_filtered_diff'] > 0)
                #                 &
                #             (df['second_combined_diff_filtered'] < -0.5)
                #         )
                #         # &diff_filtered'] > 0)
                #         #         &
                #         #     (df['combined_diff_filtered'] < 0.3)
                #         # )
                #         # &
                #         # (
                #         #     (df['second_combined_diff_filtered'] > -0.3)
                #         #         &
                #         #     (df['second_combined_diff_filtered'] < 0.3)
                #         # )
                #     )
                #     , "stg1_long"] = 2 # without atr

                # df.loc[
                #     (
                #         # (df.maxima_peak_x_MACD_12_26_9 < 0) # | 
                #         (
                #             (df.maxima_peak_x_MACD_50_75_35 < 0)
                #                 &
                #             (df['combined_diff_filtered_diff'] > 0)
                #                 &
                #             # (df['combined_diff_filtered'] < 0.5)
                #             (df.abs_dmp_dmn_200_diff < 5)
                #                 &
                #             (df['second_combined_diff_filtered_diff'] < 0)
                #                 &
                #             (df['second_combined_diff_filtered'] > 0.5)
                #         )
                #         # &
                #         # (
                #         #     (df['combined_diff_filtered'] > 0)
                #         #         &
                #         #     (df['combined_diff_filtered'] < 0.3)
                #         # )
                #         # &
                #         # (
                #         #     (df['second_combined_diff_filtered'] > -0.3)
                #         #         &
                #         #     (df['second_combined_diff_filtered'] < 0.3)
                #         # )
                #     )
                #     , "stg1_short"] = -2 # without atr




                # df.loc[
                #     (
                #         (
                #             (
                #                 (df['combined_diff_filtered_diff'] > 0)
                #                 &
                #                 (df['combined_diff_filtered'] < 0.2)
                #             )
                #                 |
                #             (
                #                 (df['combined_diff_filtered_diff'] < 0)
                #                 &
                #                 (df['combined_diff_filtered'] > 0.9)
                #             )



                #         )
                #             &
                #         (df['second_combined_diff_filtered_diff'] > 0)
                #     )
                #     &~(
                #         # (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_macd.shift(1) < 0) 
                #         # | (df.maxima_peak_x_rsi.shift(1) < 0)
                #         (df.maxima_peak_x_close < 0) | (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) 
                #         | (df.maxima_peak_x_dmp < 0) | (df.minima_peak_x_dmn < 0)
                #     )


                #     , "stgQ_long"] = 2 # without atr

                # df.loc[
                #     (
                #         (
                #             (
                #                 (df['combined_diff_filtered_diff'] > 0)
                #                 &
                #                 (df['combined_diff_filtered'] < 0.2)
                #             )
                #                 |
                #             (
                #                 (df['combined_diff_filtered_diff'] < 0)
                #                 &
                #                 (df['combined_diff_filtered'] > 0.9)
                #             )
                #         )
                #             &
                #             (df['second_combined_diff_filtered_diff'] < 0)
                #     )
                #     &~(
                #         # (df.minima_peak_x_macd > 0) | (df.minima_peak_x_rsi > 0) | (df.minima_peak_x_macd.shift(1) > 0) 
                #         # | (df.minima_peak_x_rsi.shift(1) > 0)
                #         (df.minima_peak_x_close > 0) | (df.minima_peak_x_macd > 0) | (df.minima_peak_x_rsi > 0) 
                #         | (df.minima_peak_x_dmp > 0) | (df.maxima_peak_x_dmn > 0)
                #     )
                #     , "stgQ_short"] = -2 # without atr


                df.loc[
                    (
                        (
                            (
                                (df['combined_diff_filtered_diff'] > 0)
                                &
                                (df['combined_diff_diff'] > 0)
                                &
                                (
                                    (df['combined_diff_filtered'] < 0.2)
                                    &
                                    (df['combined_diff'] < 0.2)
                                    #     |
                                    # (df['combined_diff_filtered'] > 0.35)
                                )
                                &
                                (df['second_combined_diff_filtered'] < 0.5)
                                &
                                (df['second_combined_diff'] < 0.5)
                                # &
                                # (df['combined_diff_filtered'] < 0.45)
                                # &
                                # (df.abs_dmp_dmn_200_diff > 5)
                                # &
                                # (df.RSI_14 < 70)
                                # &
                                # (df.DX_14 < 40)
                                # & 
                                # (df.MACDh_50_75_35 > 0)
                                # &
                                # (df['combined_diff_filtered'] < 0.7)
                                # &
                                # (df['second_combined_diff_filtered'] > 0.35)
                                # &
                                # (df['second_combined_diff_filtered'] < 0)
                            )
                            #     |
                            # (
                            #     (df['combined_diff_filtered_diff'] < 0)
                            #     &
                            #     (df['combined_diff_filtered'] > 0.5)
                            # )
                        )
                            &
                        (df['second_combined_diff_filtered_diff'] > 0)
                        #     &
                        # (df['second_combined_diff_diff'] > 0)
                        # (df.minima_peak_x_close > 0)
                    )
                    # |
                    # (
                    #     (df['RSI_14'] < 25)
                    # )
                    # &~(
                    #     # (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_macd.shift(1) < 0) 
                    #     # | (df.maxima_peak_x_rsi.shift(1) < 0)
                    #     (df.maxima_peak_x_close < 0) | (df.maxima_peak_x_MACD_12_26_9 < 0) | (df.maxima_peak_x_MACD_50_75_35 < 0) | (df.maxima_peak_x_rsi < 0) 
                    #     | (df.maxima_peak_x_dmp < 0) | (df.minima_peak_x_dmn < 0)
                    # )
                    , "stgQ_long"] = 2 # without atr

                df.loc[
                    (
                        (
                            (
                                (df['combined_diff_filtered_diff'] > 0)
                                &
                                (df['combined_diff_diff'] > 0)
                                &
                                (
                                    (df['combined_diff_filtered'] < 0.2)
                                    &
                                    (df['combined_diff'] < 0.2)
                                    #     |
                                    # (df['combined_diff_filtered'] > 0.35)
                                )
                                &
                                (df['second_combined_diff_filtered'] > -0.5)
                                &
                                (df['second_combined_diff'] > -0.5)
                                # &
                                # (df['combined_diff_filtered'] < 0.45)
                                # &
                                # (df.abs_dmp_dmn_200_diff > 5)
                                # &
                                # (df.RSI_14 > 30)
                                # &
                                # (df.DX_14 < 40)
                                # & 
                                # (df.MACDh_50_75_35 < 0)
                                # &
                                # (df['combined_diff_filtered'] < 0.7)
                                # # &
                                # # (df['second_combined_diff_filtered'] > 0.35)
                                # &
                                # (df['second_combined_diff_filtered'] > 0)
                            )
                            #     |
                            # (
                            #     (df['combined_diff_filtered_diff'] < 0)
                            #     &
                            #     (df['combined_diff_filtered'] > 0.5)
                            # )
                        )
                                &
                            (df['second_combined_diff_filtered_diff'] < 0)
                            #     &
                            # (df['second_combined_diff_diff'] < 0)
                            # (df.maxima_peak_x_close < 0)
                    )
                    # |
                    # (
                    #     (df['RSI_14'] > 75)
                    # )
                    # &~(
                    #     # (df.minima_peak_x_macd > 0) | (df.minima_peak_x_rsi > 0) | (df.minima_peak_x_macd.shift(1) > 0) 
                    #     # | (df.minima_peak_x_rsi.shift(1) > 0)
                    #     (df.minima_peak_x_close > 0) | (df.minima_peak_x_MACD_12_26_9 > 0) | (df.minima_peak_x_MACD_50_75_35 > 0) | (df.minima_peak_x_rsi > 0) 
                    #     | (df.minima_peak_x_dmp > 0) | (df.maxima_peak_x_dmn > 0)
                    # )
                    , "stgQ_short"] = -2 # without atr







                # df.loc[
                #     (
                #         # (df.minima_peak_x_MACD_12_26_9 > 0) # | 
                #         ((df.minima_peak_x_MACD_50_75_35 > 0) & (df['combined_diff_filtered_diff'] < 0))
                #         &~
                #         (
                #             (df['combined_diff_filtered_diff'] > 0)
                #             &
                #             (df['combined_diff_filtered'] > 0.35)
                #             &
                #             (df['combined_diff_filtered'] < 0.8)
                #             &
                #             (df['second_combined_diff_filtered'] > -0.35)
                #             &
                #             (df['second_combined_diff_filtered'] < 0.35)
                #         )
                #     )
                #     , "stg1_long"] = 2 # without atr

                # df.loc[
                #     (
                #         # (df.maxima_peak_x_MACD_12_26_9 < 0) # | 
                #         ((df.maxima_peak_x_MACD_50_75_35 < 0) & (df['combined_diff_filtered_diff'] < 0))
                #         &~
                #         (
                #             (df['combined_diff_filtered_diff'] > 0)
                #             &
                #             (df['combined_diff_filtered'] > 0.35)
                #             &
                #             (df['combined_diff_filtered'] < 0.8)
                #             &
                #             (df['second_combined_diff_filtered'] > -0.35)
                #             &
                #             (df['second_combined_diff_filtered'] < 0.35)
                #         )
                #     )
                #     , "stg1_short"] = -2 # without atr







                df.loc[
                    (
                        (
                            # (df.minima_peak_x_MACD_12_26_9 > 0) # | 
                            (df.minima_peak_x_MACD_50_75_35 > 0)
                            &
                            (df['combined_diff_diff'] > 0)
                            # &
                            # (df['combined_diff_filtered'] < 0.2)
                            # &
                            # (df['combined_diff_filtered'] > 0.15)
                            # &
                            # # (df['second_combined_diff_filtered'] < 0)
                            &
                            (df['second_combined_diff_diff'] > 0)
                            # &
                            # (df['second_combined_diff_diff'] < 0)
                            # &
                            # (df['second_combined_diff_filtered'] < 0.3)
                        )
                        |
                        (
                            (
                                (
                                    (df['combined_diff_filtered_diff'] > 0)
                                    &
                                    (df['combined_diff_diff'] > 0)
                                    &
                                    (
                                        (df['combined_diff_filtered'].shift(1) < 0.2)
                                        &
                                        (df['combined_diff_filtered'] > 0.2)
                                    )
                                    &
                                    (df['second_combined_diff'] < 0.4)
                                    &
                                    (df['second_combined_diff_filtered'] < 0.4)
                                )
                            )
                                &
                            (df['second_combined_diff_filtered_diff'] > 0)
                                &
                            (df['second_combined_diff_diff'] > 0)
                        )
                    )
                    , "stg1_long"] = 2 # without atr

                df.loc[
                    (
                        (
                            # (df.maxima_peak_x_MACD_12_26_9 < 0) # | 
                            (df.maxima_peak_x_MACD_50_75_35 < 0)
                            &
                            (df['combined_diff_diff'] > 0)
                            # &
                            # (df['combined_diff_filtered'] < 0.2)
                            # &
                            # (df['combined_diff_filtered'] > 0.15)
                            # &
                            # # (df['second_combined_diff_filtered'] > 0)
                            &
                            (df['second_combined_diff_diff'] < 0)
                            # &
                            # (df['second_combined_diff_diff'] > 0)
                            # &
                            # (df['second_combined_diff_filtered'] > -0.3)
                        )
                        |
                        (
                            (
                                (
                                    (df['combined_diff_filtered_diff'] > 0)
                                    &
                                    (df['combined_diff_diff'] > 0)
                                    &
                                    (
                                        (df['combined_diff_filtered'].shift(1) < 0.2)
                                        &
                                        (df['combined_diff_filtered'] > 0.2)
                                    )
                                    &
                                    (df['second_combined_diff'] > -0.4)
                                    &
                                    (df['second_combined_diff_filtered'] > -0.4)
                                )
                            )
                                    &
                                (df['second_combined_diff_filtered_diff'] < 0)
                                    &
                                (df['second_combined_diff_diff'] < 0)
                        )
                    )
                    , "stg1_short"] = -2 # without atr







                # if True:



                #     ####################################################################################################################################################################################
                #     ####################################################################################################################################################################################
                #     ####################################################################################################################################################################################


                #     # df.loc[
                #     #     # 추세 픽
                #     #     ( # 진입전 추세 재확인 필요, atr14 증가하는지 확인 필요
                #     #       (df.rsi_diff > 0)
                #     #     )

                #     # , "stg2_long"] = 2 # without atr

                #     # df.loc[
                #     #     # 추세 픽
                #     #     ( # 진입전 추세 재확인 필요, atr14 증가하는지 확인 필요
                #     #       (df.rsi_diff < 0) 
                #     #     )

                #     # , "stg2_short"] = -2 # without atr



                #     # df.loc[
                #     #     # 추세 픽
                #     #     (
                #     #         ( # 진입전 추세 재확인 필요, atr14 증가하는지 확인 필요
                #     #             (df['minima_peak_x_macd'].shift(2) > 0) # & (df['macd_diff_200'] > df['macds_diff_200']) & (df['macd_diff_200'] > 0) & (df['macds_diff_200'] > 0) & (df['macd_diff_700'] > 0) & (df['macds_diff_700'] > 0) & (df['MACD_12_200_9'].shift(2) < df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] > 0) & (df['k_diff'] > 0) & (df['d_diff'] > 0)
                #     #             # (df['minima_peak_x_macd'].shift(2) > 0) & (df['macd_diff_200'] > df['macds_diff_200']) & (df['macd_diff_200'].shift(2) > 0) & (df['macds_diff_200'].shift(2) > 0) & (df['macd_diff_700'].shift(2) > 0) & (df['macds_diff_700'].shift(2) > 0) & (df['MACD_12_200_9'].shift(2) < df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] > 0) & (df['k_diff'] > 0) & (df['d_diff'] > 0)
                #     #         )
                #     #         & (
                #     #             (
                #     #                 ((df.dx_diff < 0) & (df.DMN_14 > df.DMP_14) & (df.intersect == 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.high_diff > 0)           & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff > 0)
                #     #             )
                #     #                 |
                #     #             (
                #     #                 ((df.dx_diff < 0) & (dmn_set_2 > dmp_set_2) & (df.intersect > 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.high_diff > 0)           & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff > 0)
                #     #             )
                #     #                 |
                #     #             (
                #     #                 (df.dx_diff > 0) & (df.DMP_14 > df.DMN_14) #& (df.atr_p_diff < 0)# & (df.high_diff > 0) & (df.rsi_diff > 0) # & (df.obv_diff > 0) & (df.j_diff > 0) & (df['atr_AMATe_LR_8_21_2'] > 0)
                #     #             )
                #     #         )
                #     #         & (
                #     #             (df.RSI_14 > 50) & (df.RSI_14 < 80)
                #     #         )
                #     #     )
                #     #     &~(
                #     #         (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_macd.shift(1) < 0) | (df.maxima_peak_x_rsi.shift(1) < 0)
                #     #     )
                #     # , "stg2_long"] = 2 # without atr

                #     # df.loc[
                #     #     # 추세 픽
                #     #     (
                #     #         ( # 진입전 추세 재확인 필요, atr14 증가하는지 확인 필요
                #     #             (df['maxima_peak_x_macd'].shift(2) < 0) # & (df['macd_diff_200'] < df['macds_diff_200']) & (df['macd_diff_200'] < 0) & (df['macds_diff_200'] < 0) & (df['macd_diff_700'] < 0) & (df['macds_diff_700'] < 0) & (df['MACD_12_200_9'].shift(2) > df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] < 0) & (df['k_diff'] < 0) & (df['d_diff'] < 0)
                #     #             # (df['maxima_peak_x_macd'].shift(2) < 0) & (df['macd_diff_200'] < df['macds_diff_200']) & (df['macd_diff_200'].shift(2) < 0) & (df['macds_diff_200'].shift(2) < 0) & (df['macd_diff_700'].shift(2) < 0) & (df['macds_diff_700'].shift(2) < 0) & (df['MACD_12_200_9'].shift(2) > df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] < 0) & (df['k_diff'] < 0) & (df['d_diff'] < 0)
                #     #         )
                #     #         & (
                #     #             (
                #     #                 ((df.dx_diff < 0) & (df.DMP_14 > df.DMN_14) & (df.intersect == 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.low_diff < 0)          & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff < 0)
                #     #             )
                #     #                 |
                #     #             (
                #     #                 ((df.dx_diff < 0) & (dmp_set_2 > dmn_set_2) & (df.intersect > 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.low_diff < 0)          & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff < 0)
                #     #             )
                #     #                 |
                #     #             (
                #     #                 (df.dx_diff > 0) & (df.DMN_14 > df.DMP_14) #& (df.atr_p_diff < 0)#& (df.intersect == 0) & (df.low_diff < 0)  # & (df.obv_diff < 0) & (df.j_diff < 0) & (df['atr_AMATe_LR_8_21_2'] > 0)
                #     #             )
                #     #         )
                #     #         & (
                #     #             (df.RSI_14 < 50) & (df.RSI_14 > 20)
                #     #         )
                #     #     )
                #     #     &~(
                #     #         (df.minima_peak_x_macd > 0) | (df.minima_peak_x_rsi > 0) | (df.minima_peak_x_macd.shift(1) > 0) | (df.minima_peak_x_rsi.shift(1) > 0)
                #     #     )
                #     # , "stg2_short"] = -2 # without atr



                #     # df.loc[
                #     #     # 추세 픽
                #     #     ( # 진입전 추세 재확인 필요, atr14 증가하는지 확인 필요
                #     #         (df.minima_peak_x_macd > 0) & (df['macd_diff'] > 0) & (df['macds_diff'] > 0) & (df['macd_diff_12'] > 0) & (df['macds_diff_12'] > 0) & (df.atr_diff < 0) & (df.adx_diff < 0) & (df['j_diff'] > 0)
                #     #     )

                #     # , "stg3_long"] = 2 # without atr

                #     # df.loc[
                #     #     # 추세 픽
                #     #     ( # 진입전 추세 재확인 필요, atr14 증가하는지 확인 필요
                #     #         (df.maxima_peak_x_macd < 0) & (df['macd_diff'] < 0) & (df['macds_diff'] < 0) & (df['macd_diff_12'] < 0) & (df['macds_diff_12'] < 0) & (df.atr_diff < 0) & (df.adx_diff < 0) & (df['j_diff'] < 0)
                #     #     )

                #     # , "stg3_short"] = -2 # without atr






                #     # df.loc[
                #     #     # 추세 픽
                #     #     ( # 진입전 추세 재확인 필요, atr14 증가하는지 확인 필요
                #     #         (df.minima_peak_x_macd > 0) & (df['macd_diff'] > 0) & (df['macds_diff'] > 0) & (df['macd_diff_12'] > 0) & (df['macds_diff_12'] > 0) & (df['j_diff'] > 0) #& (df['k_diff'] > 0) & (df['d_diff'] > 0)
                #     #     )

                #     # , "stg3_long"] = 2 # without atr

                #     # df.loc[
                #     #     # 추세 픽
                #     #     ( # 진입전 추세 재확인 필요, atr14 증가하는지 확인 필요
                #     #         (df.maxima_peak_x_macd < 0) & (df['macd_diff'] < 0) & (df['macds_diff'] < 0) & (df['macd_diff_12'] < 0) & (df['macds_diff_12'] < 0) & (df['j_diff'] < 0) #& (df['k_diff'] < 0) & (df['d_diff'] < 0)
                #     #     )

                #     # , "stg3_short"] = -2 # without atr


                #     # df.loc[
                #     #     # 추세 픽
                #     #     ( # 진입전 추세 재확인 필요, atr14 증가하는지 확인 필요
                #     #         (df['minima_peak_x_macd'].shift(1) > 0) & (df['macd_diff_200'].shift(1) > 0) & (df['macds_diff_200'].shift(1) > 0) & (df['macd_diff_700'].shift(1) > 0) & (df['macds_diff_700'].shift(1) > 0) & (df['MACD_12_200_9'].shift(2) < df['MACD_12_200_9']) #& (df['adx_200_diff'] < 0) & (df['atr200_diff'] < 0) #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] > 0) & (df['k_diff'] > 0) & (df['d_diff'] > 0)
                #     #     )
                #     #     &~(
                #     #         (df.maxima_peak_x_close < 0) | (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_dmp < 0) | (df.minima_peak_x_dmn < 0)
                #     #     )
                #     # , "stg3_long"] = 2 # without atr

                #     # df.loc[
                #     #     # 추세 픽
                #     #     ( # 진입전 추세 재확인 필요, atr14 증가하는지 확인 필요
                #     #         (df['maxima_peak_x_macd'].shift(1) < 0) & (df['macd_diff_200'].shift(1) < 0) & (df['macds_diff_200'].shift(1) < 0) & (df['macd_diff_700'].shift(1) < 0) & (df['macds_diff_700'].shift(1) < 0) & (df['MACD_12_200_9'].shift(2) > df['MACD_12_200_9']) #& (df['adx_200_diff'] < 0) & (df['atr200_diff'] < 0) #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] < 0) & (df['k_diff'] < 0) & (df['d_diff'] < 0)
                #     #     )
                #     #     &~(
                #     #         (df.minima_peak_x_close > 0) | (df.minima_peak_x_macd > 0) | (df.minima_peak_x_rsi > 0) | (df.minima_peak_x_dmp > 0) | (df.maxima_peak_x_dmn > 0)
                #     #     )
                #     # , "stg3_short"] = -2 # without atr






                #     # df.loc[
                #     #     # 추세 픽
                #     #     ( # 진입전 추세 재확인 필요, atr14 증가하는지 확인 필요
                #     #         (df['minima_peak_x_macd'].shift(2) > 0) & (df['macd_diff_200'] > df['macds_diff_200']) & (df['macd_diff_200'] > 0) & (df['macds_diff_200'] > 0) & (df['macd_diff_700'] > 0) & (df['macds_diff_700'] > 0) & (df['MACD_12_200_9'].shift(2) < df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] > 0) & (df['k_diff'] > 0) & (df['d_diff'] > 0)
                #     #         # (df['minima_peak_x_macd'].shift(2) > 0) & (df['macd_diff_200'] > df['macds_diff_200']) & (df['macd_diff_200'].shift(2) > 0) & (df['macds_diff_200'].shift(2) > 0) & (df['macd_diff_700'].shift(2) > 0) & (df['macds_diff_700'].shift(2) > 0) & (df['MACD_12_200_9'].shift(2) < df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] > 0) & (df['k_diff'] > 0) & (df['d_diff'] > 0)
                            
                #     #     )
                #     #     &~(
                #     #         (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_macd.shift(1) < 0) | (df.maxima_peak_x_rsi.shift(1) < 0)
                #     #     )
                #     # , "stg3_long"] = 2 # without atr

                #     # df.loc[
                #     #     # 추세 픽
                #     #     ( # 진입전 추세 재확인 필요, atr14 증가하는지 확인 필요
                #     #         (df['maxima_peak_x_macd'].shift(2) < 0) & (df['macd_diff_200'] < df['macds_diff_200']) & (df['macd_diff_200'] < 0) & (df['macds_diff_200'] < 0) & (df['macd_diff_700'] < 0) & (df['macds_diff_700'] < 0) & (df['MACD_12_200_9'].shift(2) > df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] < 0) & (df['k_diff'] < 0) & (df['d_diff'] < 0)
                #     #         # (df['maxima_peak_x_macd'].shift(2) < 0) & (df['macd_diff_200'] < df['macds_diff_200']) & (df['macd_diff_200'].shift(2) < 0) & (df['macds_diff_200'].shift(2) < 0) & (df['macd_diff_700'].shift(2) < 0) & (df['macds_diff_700'].shift(2) < 0) & (df['MACD_12_200_9'].shift(2) > df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] < 0) & (df['k_diff'] < 0) & (df['d_diff'] < 0)
                #     #     )
                #     #     &~(
                #     #         (df.minima_peak_x_macd > 0) | (df.minima_peak_x_rsi > 0) | (df.minima_peak_x_macd.shift(1) > 0) | (df.minima_peak_x_rsi.shift(1) > 0)
                #     #     )
                #     # , "stg3_short"] = -2 # without atr






                #     ################################################################################################################################################

                #     # df.loc[
                #     #     # 추세 픽
                #     #     (
                #     #         ( # 진입전 추세 재확인 필요, atr14 증가하는지 확인 필요
                #     #             (df['minima_peak_x_macd'] > 0) #& (df['percentage_change'].abs() > 8) # & (df['macd_diff_200'] > df['macds_diff_200']) & (df['macd_diff_200'] > 0) & (df['macds_diff_200'] > 0) & (df['macd_diff_700'] > 0) & (df['macds_diff_700'] > 0) & (df['MACD_12_200_9'].shift(2) < df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] > 0) & (df['k_diff'] > 0) & (df['d_diff'] > 0)

                #     #             # (df['minima_peak_x_macd'].shift(2) > 0) & (df['macd_diff_200'] > df['macds_diff_200']) & (df['macd_diff_200'].shift(2) > 0) & (df['macds_diff_200'].shift(2) > 0) & (df['macd_diff_700'].shift(2) > 0) & (df['macds_diff_700'].shift(2) > 0) & (df['MACD_12_200_9'].shift(2) < df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] > 0) & (df['k_diff'] > 0) & (df['d_diff'] > 0)
                #     #         )
                #     #         & (
                #     #             (
                #     #                 ((df.dx_diff < 0) & (df.DMN_14 > df.DMP_14) & (df.intersect == 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.high_diff > 0)           & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff > 0)
                #     #             )
                #     #                 |
                #     #             (
                #     #                 ((df.dx_diff < 0) & (dmn_set_2 > dmp_set_2) & (df.intersect > 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.high_diff > 0)           & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff > 0)
                #     #             )
                #     #                 |
                #     #             (
                #     #                 (df.dx_diff > 0) & (df.DMP_14 > df.DMN_14) #& (df.atr_p_diff < 0)# & (df.high_diff > 0) & (df.rsi_diff > 0) # & (df.obv_diff > 0) & (df.j_diff > 0) & (df['atr_AMATe_LR_8_21_2'] > 0)
                #     #             )
                #     #         )
                #     #         # & (
                #     #         #     (df.RSI_14 > 50) & (df.RSI_14 < 80)
                #     #         # )
                #     #     )
                #     #     &~(
                #     #         # (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_macd.shift(1) < 0) | (df.maxima_peak_x_rsi.shift(1) < 0)
                #     #         (df.maxima_peak_x_close < 0) | (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_dmp < 0) | (df.minima_peak_x_dmn < 0)
                #     #     )
                #     # , "stg3_long"] = 2 # without atr

                #     # df.loc[
                #     #     # 추세 픽
                #     #     (
                #     #         ( # 진입전 추세 재확인 필요, atr14 증가하는지 확인 필요
                #     #             (df['maxima_peak_x_macd'] < 0) #& (df['percentage_change'].abs() > 8) # & (df['macd_diff_200'] < df['macds_diff_200']) & (df['macd_diff_200'] < 0) & (df['macds_diff_200'] < 0) & (df['macd_diff_700'] < 0) & (df['macds_diff_700'] < 0) & (df['MACD_12_200_9'].shift(2) > df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] < 0) & (df['k_diff'] < 0) & (df['d_diff'] < 0)
                #     #             # (df['maxima_peak_x_macd'].shift(2) < 0) & (df['macd_diff_200'] < df['macds_diff_200']) & (df['macd_diff_200'].shift(2) < 0) & (df['macds_diff_200'].shift(2) < 0) & (df['macd_diff_700'].shift(2) < 0) & (df['macds_diff_700'].shift(2) < 0) & (df['MACD_12_200_9'].shift(2) > df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] < 0) & (df['k_diff'] < 0) & (df['d_diff'] < 0)
                #     #         )
                #     #         & (
                #     #             (
                #     #                 ((df.dx_diff < 0) & (df.DMP_14 > df.DMN_14) & (df.intersect == 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.low_diff < 0)          & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff < 0)
                #     #             )
                #     #                 |
                #     #             (
                #     #                 ((df.dx_diff < 0) & (dmp_set_2 > dmn_set_2) & (df.intersect > 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.low_diff < 0)          & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff < 0)
                #     #             )
                #     #                 |
                #     #             (
                #     #                 (df.dx_diff > 0) & (df.DMN_14 > df.DMP_14) #& (df.atr_p_diff < 0)#& (df.intersect == 0) & (df.low_diff < 0)  # & (df.obv_diff < 0) & (df.j_diff < 0) & (df['atr_AMATe_LR_8_21_2'] > 0)
                #     #             )
                #     #         )
                #     #         # & (
                #     #         #     (df.RSI_14 < 50) & (df.RSI_14 > 20)
                #     #         # )
                #     #     )
                #     #     &~(
                #     #         # (df.minima_peak_x_macd > 0) | (df.minima_peak_x_rsi > 0) | (df.minima_peak_x_macd.shift(1) > 0) | (df.minima_peak_x_rsi.shift(1) > 0)
                #     #         (df.minima_peak_x_close > 0) | (df.minima_peak_x_macd > 0) | (df.minima_peak_x_rsi > 0) | (df.minima_peak_x_dmp > 0) | (df.maxima_peak_x_dmn > 0)
                #     #     )
                #     # , "stg3_short"] = -2 # without atr



























                #     # df.loc[
                #     #     # 추세 픽
                #     #     (
                #     #         ( # 진입전 추세 재확인 필요, atr14 증가하는지 확인 필요
                #     #             # (df['minima_peak_x_rsi'] > 0) #& (df['percentage_change2'] > 3) # & (df['macd_diff_200'] > df['macds_diff_200']) & (df['macd_diff_200'] > 0) & (df['macds_diff_200'] > 0) & (df['macd_diff_700'] > 0) & (df['macds_diff_700'] > 0) & (df['MACD_12_200_9'].shift(2) < df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] > 0) & (df['k_diff'] > 0) & (df['d_diff'] > 0)
                #     #             # (df['rsi_percentage_change'] > 5) & (df['atr_percentage_change'] > 4) & (df['macd_percentage_change'] > 50) & (df['obv_percentage_change'] > 100) #& (df['atr_percentage_change'] > 4)
                #     #             # & (df['obv_percentage_change'] > 150)
                #     #             (df['obv_percentage_change'].abs() < 50)
                #     #             & (((df['sma50_diff'] > 0) & (df['sma200_diff'] > 0)))
                #     #             # & (((df['sma50_diff'] < 0) & (df['sma200_diff'] < 0)))

                                
                #     #             # (df['minima_peak_x_macd'].shift(2) > 0) & (df['macd_diff_200'] > df['macds_diff_200']) & (df['macd_diff_200'].shift(2) > 0) & (df['macds_diff_200'].shift(2) > 0) & (df['macd_diff_700'].shift(2) > 0) & (df['macds_diff_700'].shift(2) > 0) & (df['MACD_12_200_9'].shift(2) < df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] > 0) & (df['k_diff'] > 0) & (df['d_diff'] > 0)
                #     #         )
                #     #         & (
                #     #             (
                #     #                 ((df.dx_diff < 0) & (df.DMN_14 > df.DMP_14) & (df.intersect == 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.high_diff > 0)           & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff > 0)
                #     #             )
                #     #                 |
                #     #             (
                #     #                 ((df.dx_diff < 0) & (dmn_set_2 > dmp_set_2) & (df.intersect > 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.high_diff > 0)           & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff > 0)
                #     #             )
                #     #                 |
                #     #             (
                #     #                 (df.dx_diff > 0) & (df.DMP_14 > df.DMN_14) #& (df.atr_p_diff < 0)# & (df.high_diff > 0) & (df.rsi_diff > 0) # & (df.obv_diff > 0) & (df.j_diff > 0) & (df['atr_AMATe_LR_8_21_2'] > 0)
                #     #             )
                #     #         )
                #     #         & (
                #     #             (df.vwma7_diff > 0)
                #     #         )
                #     #     )
                #     #     &~(
                #     #         # (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_macd.shift(1) < 0) | (df.maxima_peak_x_rsi.shift(1) < 0)
                #     #         (df.maxima_peak_x_close < 0) | (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_dmp < 0) | (df.minima_peak_x_dmn < 0)
                #     #     )
                #     # , "stg2_long"] = 2 # without atr

                #     # df.loc[
                #     #     # 추세 픽
                #     #     (
                #     #         ( # 진입전 추세 재확인 필요, atr14 증가하는지 확인 필요
                #     #             # (df['maxima_peak_x_rsi'] < 0) #& (df['percentage_change2'] < -3) # & (df['macd_diff_200'] < df['macds_diff_200']) & (df['macd_diff_200'] < 0) & (df['macds_diff_200'] < 0) & (df['macd_diff_700'] < 0) & (df['macds_diff_700'] < 0) & (df['MACD_12_200_9'].shift(2) > df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] < 0) & (df['k_diff'] < 0) & (df['d_diff'] < 0)
                #     #             # (df['rsi_percentage_change'] < -5) & (df['atr_percentage_change'] > 4) & (df['macd_percentage_change'] < -50) & (df['obv_percentage_change'] < -100) #& (df['atr_percentage_change'] > 4)
                #     #             # (df['obv_percentage_change'] < -150)
                #     #             (df['obv_percentage_change'].abs() < 50)
                #     #             # & (((df['sma50_diff'] > 0) & (df['sma200_diff'] > 0)))
                #     #             & (((df['sma50_diff'] < 0) & (df['sma200_diff'] < 0)))







                                
                #     #             # (df['maxima_peak_x_macd'].shift(2) < 0) & (df['macd_diff_200'] < df['macds_diff_200']) & (df['macd_diff_200'].shift(2) < 0) & (df['macds_diff_200'].shift(2) < 0) & (df['macd_diff_700'].shift(2) < 0) & (df['macds_diff_700'].shift(2) < 0) & (df['MACD_12_200_9'].shift(2) > df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] < 0) & (df['k_diff'] < 0) & (df['d_diff'] < 0)
                #     #         )
                #     #         & (
                #     #             (
                #     #                 ((df.dx_diff < 0) & (df.DMP_14 > df.DMN_14) & (df.intersect == 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.low_diff < 0)          & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff < 0)
                #     #             )
                #     #                 |
                #     #             (
                #     #                 ((df.dx_diff < 0) & (dmp_set_2 > dmn_set_2) & (df.intersect > 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.low_diff < 0)          & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff < 0)
                #     #             )
                #     #                 |
                #     #             (
                #     #                 (df.dx_diff > 0) & (df.DMN_14 > df.DMP_14) #& (df.atr_p_diff < 0)#& (df.intersect == 0) & (df.low_diff < 0)  # & (df.obv_diff < 0) & (df.j_diff < 0) & (df['atr_AMATe_LR_8_21_2'] > 0)
                #     #             )
                #     #         )
                #     #         & (
                #     #             (df.vwma7_diff < 0)
                #     #         )
                #     #     )
                #     #     &~(
                #     #         # (df.minima_peak_x_macd > 0) | (df.minima_peak_x_rsi > 0) | (df.minima_peak_x_macd.shift(1) > 0) | (df.minima_peak_x_rsi.shift(1) > 0)
                #     #         (df.minima_peak_x_close > 0) | (df.minima_peak_x_macd > 0) | (df.minima_peak_x_rsi > 0) | (df.minima_peak_x_dmp > 0) | (df.maxima_peak_x_dmn > 0)
                #     #     )
                #     # , "stg2_short"] = -2 # without atr



















































                #     # df.loc[
                #     #     # 추세 픽
                #     #     (
                #     #         ( # 진입전 추세 재확인 필요, atr14 증가하는지 확인 필요
                #     #             # (df['minima_peak_x_macd'] > 0) #& (df['percentage_change2'] > 3) # & (df['macd_diff_200'] > df['macds_diff_200']) & (df['macd_diff_200'] > 0) & (df['macds_diff_200'] > 0) & (df['macd_diff_700'] > 0) & (df['macds_diff_700'] > 0) & (df['MACD_12_200_9'].shift(2) < df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] > 0) & (df['k_diff'] > 0) & (df['d_diff'] > 0)
                #     #             # (df['rsi_percentage_change'] > 5) & (df['atr_percentage_change'] > 4) & (df['macd_percentage_change'] > 50) & (df['obv_percentage_change'] > 100) #& (df['atr_percentage_change'] > 4)
                #     #             (df['obv_percentage_change'] > 50)
                #     #             # (df['minima_peak_x_macd'].shift(2) > 0) & (df['macd_diff_200'] > df['macds_diff_200']) & (df['macd_diff_200'].shift(2) > 0) & (df['macds_diff_200'].shift(2) > 0) & (df['macd_diff_700'].shift(2) > 0) & (df['macds_diff_700'].shift(2) > 0) & (df['MACD_12_200_9'].shift(2) < df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] > 0) & (df['k_diff'] > 0) & (df['d_diff'] > 0)
                #     #         )
                #     #         & (
                #     #             (
                #     #                 ((df.dx_diff < 0) & (df.DMN_14 > df.DMP_14) & (df.intersect == 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.high_diff > 0)           & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff > 0)
                #     #             )
                #     #                 |
                #     #             (
                #     #                 ((df.dx_diff < 0) & (dmn_set_2 > dmp_set_2) & (df.intersect > 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.high_diff > 0)           & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff > 0)
                #     #             )
                #     #                 |
                #     #             (
                #     #                 (df.dx_diff > 0) & (df.DMP_14 > df.DMN_14) #& (df.atr_p_diff < 0)# & (df.high_diff > 0) & (df.rsi_diff > 0) # & (df.obv_diff > 0) & (df.j_diff > 0) & (df['atr_AMATe_LR_8_21_2'] > 0)
                #     #             )
                #     #         )
                #     #         # & (
                #     #         #     (df.vwma7_diff > 0)
                #     #         # )
                #     #     )
                #     #     &~(
                #     #         # (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_macd.shift(1) < 0) | (df.maxima_peak_x_rsi.shift(1) < 0)
                #     #         (df.maxima_peak_x_close < 0) | (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_dmp < 0) | (df.minima_peak_x_dmn < 0)
                #     #     )
                #     # , "stg2_long"] = 2 # without atr

                #     # df.loc[
                #     #     # 추세 픽
                #     #     (
                #     #         ( # 진입전 추세 재확인 필요, atr14 증가하는지 확인 필요
                #     #             # (df['maxima_peak_x_macd'] < 0) #& (df['percentage_change2'] < -3) # & (df['macd_diff_200'] < df['macds_diff_200']) & (df['macd_diff_200'] < 0) & (df['macds_diff_200'] < 0) & (df['macd_diff_700'] < 0) & (df['macds_diff_700'] < 0) & (df['MACD_12_200_9'].shift(2) > df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] < 0) & (df['k_diff'] < 0) & (df['d_diff'] < 0)
                #     #             # (df['rsi_percentage_change'] < -5) & (df['atr_percentage_change'] > 4) & (df['macd_percentage_change'] < -50) & (df['obv_percentage_change'] < -100) #& (df['atr_percentage_change'] > 4)
                #     #             (df['obv_percentage_change'] < -50)
                #     #             # (df['maxima_peak_x_macd'].shift(2) < 0) & (df['macd_diff_200'] < df['macds_diff_200']) & (df['macd_diff_200'].shift(2) < 0) & (df['macds_diff_200'].shift(2) < 0) & (df['macd_diff_700'].shift(2) < 0) & (df['macds_diff_700'].shift(2) < 0) & (df['MACD_12_200_9'].shift(2) > df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] < 0) & (df['k_diff'] < 0) & (df['d_diff'] < 0)
                #     #         )
                #     #         & (
                #     #             (
                #     #                 ((df.dx_diff < 0) & (df.DMP_14 > df.DMN_14) & (df.intersect == 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.low_diff < 0)          & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff < 0)
                #     #             )
                #     #                 |
                #     #             (
                #     #                 ((df.dx_diff < 0) & (dmp_set_2 > dmn_set_2) & (df.intersect > 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.low_diff < 0)          & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff < 0)
                #     #             )
                #     #                 |
                #     #             (
                #     #                 (df.dx_diff > 0) & (df.DMN_14 > df.DMP_14) #& (df.atr_p_diff < 0)#& (df.intersect == 0) & (df.low_diff < 0)  # & (df.obv_diff < 0) & (df.j_diff < 0) & (df['atr_AMATe_LR_8_21_2'] > 0)
                #     #             )
                #     #         )
                #     #         # & (
                #     #         #     (df.vwma7_diff < 0)
                #     #         # )
                #     #     )
                #     #     &~(
                #     #         # (df.minima_peak_x_macd > 0) | (df.minima_peak_x_rsi > 0) | (df.minima_peak_x_macd.shift(1) > 0) | (df.minima_peak_x_rsi.shift(1) > 0)
                #     #         (df.minima_peak_x_close > 0) | (df.minima_peak_x_macd > 0) | (df.minima_peak_x_rsi > 0) | (df.minima_peak_x_dmp > 0) | (df.maxima_peak_x_dmn > 0)
                #     #     )
                #     # , "stg2_short"] = -2 # without atr







                #     # df.loc[
                #     #     (
                #     #         (df.stg2_long.shift(1) > 0)
                #     #     )

                #     # , "stg3_long"] = 2 # without atr
                #     # df.loc[
                #     #     (
                #     #         (df.stg2_short.shift(1) < 0)
                #     #     )

                #     # , "stg3_short"] = -2 # without atr











                #     # df.loc[
                #     #     # 추세 픽
                #     #     (
                #     #         ( # 진입전 추세 재확인 필요, atr14 증가하는지 확인 필요
                #     #             (df['minima_peak_x_macd'].shift(2) > 0) # & (df['macd_diff_200'] > df['macds_diff_200']) & (df['macd_diff_200'] > 0) & (df['macds_diff_200'] > 0) & (df['macd_diff_700'] > 0) & (df['macds_diff_700'] > 0) & (df['MACD_12_200_9'].shift(2) < df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] > 0) & (df['k_diff'] > 0) & (df['d_diff'] > 0)
                #     #             # (df['minima_peak_x_macd'].shift(2) > 0) & (df['macd_diff_200'] > df['macds_diff_200']) & (df['macd_diff_200'].shift(2) > 0) & (df['macds_diff_200'].shift(2) > 0) & (df['macd_diff_700'].shift(2) > 0) & (df['macds_diff_700'].shift(2) > 0) & (df['MACD_12_200_9'].shift(2) < df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] > 0) & (df['k_diff'] > 0) & (df['d_diff'] > 0)
                #     #         )
                #     #         & (
                #     #             (
                #     #                 ((df.dx_diff < 0) & (df.DMN_14 > df.DMP_14) & (df.intersect == 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.high_diff > 0)           & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff > 0)
                #     #             )
                #     #                 |
                #     #             (
                #     #                 ((df.dx_diff < 0) & (dmn_set_2 > dmp_set_2) & (df.intersect > 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.high_diff > 0)           & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff > 0)
                #     #             )
                #     #                 |
                #     #             (
                #     #                 (df.dx_diff > 0) & (df.DMP_14 > df.DMN_14) #& (df.atr_p_diff < 0)# & (df.high_diff > 0) & (df.rsi_diff > 0) # & (df.obv_diff > 0) & (df.j_diff > 0) & (df['atr_AMATe_LR_8_21_2'] > 0)
                #     #             )
                #     #         )
                #     #         & (
                #     #             (df.RSI_14 > 50) & (df.RSI_14 < 80)
                #     #         )
                #     #     )
                #     #     &~(
                #     #         (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_macd.shift(1) < 0) | (df.maxima_peak_x_rsi.shift(1) < 0)
                #     #     )
                #     # , "stg3_long"] = 2 # without atr

                #     # df.loc[
                #     #     # 추세 픽
                #     #     (
                #     #         ( # 진입전 추세 재확인 필요, atr14 증가하는지 확인 필요
                #     #             (df['maxima_peak_x_macd'].shift(2) < 0) # & (df['macd_diff_200'] < df['macds_diff_200']) & (df['macd_diff_200'] < 0) & (df['macds_diff_200'] < 0) & (df['macd_diff_700'] < 0) & (df['macds_diff_700'] < 0) & (df['MACD_12_200_9'].shift(2) > df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] < 0) & (df['k_diff'] < 0) & (df['d_diff'] < 0)
                #     #             # (df['maxima_peak_x_macd'].shift(2) < 0) & (df['macd_diff_200'] < df['macds_diff_200']) & (df['macd_diff_200'].shift(2) < 0) & (df['macds_diff_200'].shift(2) < 0) & (df['macd_diff_700'].shift(2) < 0) & (df['macds_diff_700'].shift(2) < 0) & (df['MACD_12_200_9'].shift(2) > df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] < 0) & (df['k_diff'] < 0) & (df['d_diff'] < 0)
                #     #         )
                #     #         & (
                #     #             (
                #     #                 ((df.dx_diff < 0) & (df.DMP_14 > df.DMN_14) & (df.intersect == 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.low_diff < 0)          & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff < 0)
                #     #             )
                #     #                 |
                #     #             (
                #     #                 ((df.dx_diff < 0) & (dmp_set_2 > dmn_set_2) & (df.intersect > 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.low_diff < 0)          & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff < 0)
                #     #             )
                #     #                 |
                #     #             (
                #     #                 (df.dx_diff > 0) & (df.DMN_14 > df.DMP_14) #& (df.atr_p_diff < 0)#& (df.intersect == 0) & (df.low_diff < 0)  # & (df.obv_diff < 0) & (df.j_diff < 0) & (df['atr_AMATe_LR_8_21_2'] > 0)
                #     #             )
                #     #         )
                #     #         & (
                #     #             (df.RSI_14 < 50) & (df.RSI_14 > 20)
                #     #         )
                #     #     )
                #     #     &~(
                #     #         (df.minima_peak_x_macd > 0) | (df.minima_peak_x_rsi > 0) | (df.minima_peak_x_macd.shift(1) > 0) | (df.minima_peak_x_rsi.shift(1) > 0)
                #     #     )
                #     # , "stg3_short"] = -2 # without atr











                #     # 1#################&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&###############################################################################################################################
                #     # 1#################&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&###############################################################################################################################
                #     # 1#################&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&###############################################################################################################################







                #     # df.loc[
                #     #     # 추세 픽
                #     #     (
                #     #         ( # 진입전 추세 재확인 필요, atr14 증가하는지 확인 필요
                #     #             # (df['minima_peak_x_rsi'] > 0) #& (df['percentage_change2'] > 3) # & (df['macd_diff_200'] > df['macds_diff_200']) & (df['macd_diff_200'] > 0) & (df['macds_diff_200'] > 0) & (df['macd_diff_700'] > 0) & (df['macds_diff_700'] > 0) & (df['MACD_12_200_9'].shift(2) < df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] > 0) & (df['k_diff'] > 0) & (df['d_diff'] > 0)
                #     #             # (df['rsi_percentage_change'] > 5) & (df['atr_percentage_change'] > 4) & (df['macd_percentage_change'] > 50) & (df['obv_percentage_change'] > 100) #& (df['atr_percentage_change'] > 4)
                #     #             # & (df['obv_percentage_change'] > 150)
                #     #             (df['obv_percentage_change'].abs() < 50)
                #     #             & (((df['sma50_diff'] > 0) & (df['sma200_diff'] > 0)))
                #     #             # & (((df['sma50_diff'] < 0) & (df['sma200_diff'] < 0)))

                                
                #     #             # (df['minima_peak_x_macd'].shift(2) > 0) & (df['macd_diff_200'] > df['macds_diff_200']) & (df['macd_diff_200'].shift(2) > 0) & (df['macds_diff_200'].shift(2) > 0) & (df['macd_diff_700'].shift(2) > 0) & (df['macds_diff_700'].shift(2) > 0) & (df['MACD_12_200_9'].shift(2) < df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] > 0) & (df['k_diff'] > 0) & (df['d_diff'] > 0)
                #     #         )
                #     #         & (
                #     #             (
                #     #                 ((df.dx_diff < 0) & (df.DMN_14 > df.DMP_14) & (df.intersect == 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.high_diff > 0)           & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff > 0)
                #     #             )
                #     #                 |
                #     #             (
                #     #                 ((df.dx_diff < 0) & (dmn_set_2 > dmp_set_2) & (df.intersect > 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.high_diff > 0)           & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff > 0)
                #     #             )
                #     #                 |
                #     #             (
                #     #                 (df.dx_diff > 0) & (df.DMP_14 > df.DMN_14) #& (df.atr_p_diff < 0)# & (df.high_diff > 0) & (df.rsi_diff > 0) # & (df.obv_diff > 0) & (df.j_diff > 0) & (df['atr_AMATe_LR_8_21_2'] > 0)
                #     #             )
                #     #         )
                #     #         & (
                #     #             (df.vwma7_diff > 0)
                #     #         )
                #     #     )
                #     #     &~(
                #     #         # (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_macd.shift(1) < 0) | (df.maxima_peak_x_rsi.shift(1) < 0)
                #     #         (df.maxima_peak_x_close < 0) | (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_dmp < 0) | (df.minima_peak_x_dmn < 0)
                #     #     )
                #     # , "stg5_long"] = 2 # without atr

                #     # df.loc[
                #     #     # 추세 픽
                #     #     (
                #     #         ( # 진입전 추세 재확인 필요, atr14 증가하는지 확인 필요
                #     #             # (df['maxima_peak_x_rsi'] < 0) #& (df['percentage_change2'] < -3) # & (df['macd_diff_200'] < df['macds_diff_200']) & (df['macd_diff_200'] < 0) & (df['macds_diff_200'] < 0) & (df['macd_diff_700'] < 0) & (df['macds_diff_700'] < 0) & (df['MACD_12_200_9'].shift(2) > df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] < 0) & (df['k_diff'] < 0) & (df['d_diff'] < 0)
                #     #             # (df['rsi_percentage_change'] < -5) & (df['atr_percentage_change'] > 4) & (df['macd_percentage_change'] < -50) & (df['obv_percentage_change'] < -100) #& (df['atr_percentage_change'] > 4)
                #     #             # (df['obv_percentage_change'] < -150)
                #     #             (df['obv_percentage_change'].abs() < 50)
                #     #             # & (((df['sma50_diff'] > 0) & (df['sma200_diff'] > 0)))
                #     #             & (((df['sma50_diff'] < 0) & (df['sma200_diff'] < 0)))







                                
                #     #             # (df['maxima_peak_x_macd'].shift(2) < 0) & (df['macd_diff_200'] < df['macds_diff_200']) & (df['macd_diff_200'].shift(2) < 0) & (df['macds_diff_200'].shift(2) < 0) & (df['macd_diff_700'].shift(2) < 0) & (df['macds_diff_700'].shift(2) < 0) & (df['MACD_12_200_9'].shift(2) > df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] < 0) & (df['k_diff'] < 0) & (df['d_diff'] < 0)
                #     #         )
                #     #         & (
                #     #             (
                #     #                 ((df.dx_diff < 0) & (df.DMP_14 > df.DMN_14) & (df.intersect == 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.low_diff < 0)          & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff < 0)
                #     #             )
                #     #                 |
                #     #             (
                #     #                 ((df.dx_diff < 0) & (dmp_set_2 > dmn_set_2) & (df.intersect > 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.low_diff < 0)          & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff < 0)
                #     #             )
                #     #                 |
                #     #             (
                #     #                 (df.dx_diff > 0) & (df.DMN_14 > df.DMP_14) #& (df.atr_p_diff < 0)#& (df.intersect == 0) & (df.low_diff < 0)  # & (df.obv_diff < 0) & (df.j_diff < 0) & (df['atr_AMATe_LR_8_21_2'] > 0)
                #     #             )
                #     #         )
                #     #         & (
                #     #             (df.vwma7_diff < 0)
                #     #         )
                #     #     )
                #     #     &~(
                #     #         # (df.minima_peak_x_macd > 0) | (df.minima_peak_x_rsi > 0) | (df.minima_peak_x_macd.shift(1) > 0) | (df.minima_peak_x_rsi.shift(1) > 0)
                #     #         (df.minima_peak_x_close > 0) | (df.minima_peak_x_macd > 0) | (df.minima_peak_x_rsi > 0) | (df.minima_peak_x_dmp > 0) | (df.maxima_peak_x_dmn > 0)
                #     #     )
                #     # , "stg5_short"] = -2 # without atr

                #     # 1 #################&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&###############################################################################################################################
                #     # 1 #################&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&###############################################################################################################################
                #     # 1 #################&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&###############################################################################################################################




                #     # 2 #################&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&###############################################################################################################################
                #     # 2 #################&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&###############################################################################################################################
                #     # 2 #################&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&###############################################################################################################################


                #     # df.loc[
                #     #     # 추세 픽
                #     #     (
                #     #         ( # 진입전 추세 재확인 필요, atr14 증가하는지 확인 필요
                #     #             ((df['minima_peak_x_rsi'] > 0) | (df['minima_peak_x_macd'] > 0))
                #     #             & (df['volume_change'].abs() > 400)
                #     #             & (df['close'] < df['SMA_50'] )
                #     #             & (df['SMA_50'] < df['SMA_200'] )
                #     #             & (((df['sma50_diff'] < 0) & (df['sma200_diff'] < 0)))

                                
                #     #             # (df['minima_peak_x_macd'].shift(2) > 0) & (df['macd_diff_200'] > df['macds_diff_200']) & (df['macd_diff_200'].shift(2) > 0) & (df['macds_diff_200'].shift(2) > 0) & (df['macd_diff_700'].shift(2) > 0) & (df['macds_diff_700'].shift(2) > 0) & (df['MACD_12_200_9'].shift(2) < df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] > 0) & (df['k_diff'] > 0) & (df['d_diff'] > 0)
                #     #         )
                #     #         # & (
                #     #         #     (
                #     #         #         ((df.dx_diff < 0) & (df.DMN_14 > df.DMP_14) & (df.intersect == 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.high_diff > 0)           & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff > 0)
                #     #         #     )
                #     #         #         |
                #     #         #     (
                #     #         #         ((df.dx_diff < 0) & (dmn_set_2 > dmp_set_2) & (df.intersect > 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.high_diff > 0)           & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff > 0)
                #     #         #     )
                #     #         #         |
                #     #         #     (
                #     #         #         (df.dx_diff > 0) & (df.DMP_14 > df.DMN_14) #& (df.atr_p_diff < 0)# & (df.high_diff > 0) & (df.rsi_diff > 0) # & (df.obv_diff > 0) & (df.j_diff > 0) & (df['atr_AMATe_LR_8_21_2'] > 0)
                #     #         #     )
                #     #         # )
                #     #         # & (
                #     #         #     (df.vwma7_diff > 0)
                #     #         # )
                #     #     )
                #     #     &~(
                #     #         # (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_macd.shift(1) < 0) | (df.maxima_peak_x_rsi.shift(1) < 0)
                #     #         (df.maxima_peak_x_close < 0) | (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_dmp < 0) | (df.minima_peak_x_dmn < 0)
                #     #     )
                #     # , "stg3_long"] = 2 # without atr

                #     # df.loc[
                #     #     # 추세 픽
                #     #     (
                #     #         ( # 진입전 추세 재확인 필요, atr14 증가하는지 확인 필요
                #     #             ((df['maxima_peak_x_rsi'] < 0) | (df['maxima_peak_x_macd'] < 0))
                #     #             & (df['volume_change'].abs() > 400)
                #     #             & (df['close'] > df['SMA_50'] )
                #     #             & (df['SMA_50'] > df['SMA_200'] )
                #     #             & (((df['sma50_diff'] > 0) & (df['sma200_diff'] > 0)))

                                
                #     #             # (df['maxima_peak_x_macd'].shift(2) < 0) & (df['macd_diff_200'] < df['macds_diff_200']) & (df['macd_diff_200'].shift(2) < 0) & (df['macds_diff_200'].shift(2) < 0) & (df['macd_diff_700'].shift(2) < 0) & (df['macds_diff_700'].shift(2) < 0) & (df['MACD_12_200_9'].shift(2) > df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] < 0) & (df['k_diff'] < 0) & (df['d_diff'] < 0)
                #     #         )
                #     #         # & (
                #     #         #     (
                #     #         #         ((df.dx_diff < 0) & (df.DMP_14 > df.DMN_14) & (df.intersect == 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.low_diff < 0)          & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff < 0)
                #     #         #     )
                #     #         #         |
                #     #         #     (
                #     #         #         ((df.dx_diff < 0) & (dmp_set_2 > dmn_set_2) & (df.intersect > 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.low_diff < 0)          & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff < 0)
                #     #         #     )
                #     #         #         |
                #     #         #     (
                #     #         #         (df.dx_diff > 0) & (df.DMN_14 > df.DMP_14) #& (df.atr_p_diff < 0)#& (df.intersect == 0) & (df.low_diff < 0)  # & (df.obv_diff < 0) & (df.j_diff < 0) & (df['atr_AMATe_LR_8_21_2'] > 0)
                #     #         #     )
                #     #         # )
                #     #         # & (
                #     #         #     (df.vwma7_diff < 0)
                #     #         # )
                #     #     )
                #     #     &~(
                #     #         # (df.minima_peak_x_macd > 0) | (df.minima_peak_x_rsi > 0) | (df.minima_peak_x_macd.shift(1) > 0) | (df.minima_peak_x_rsi.shift(1) > 0)
                #     #         (df.minima_peak_x_close > 0) | (df.minima_peak_x_macd > 0) | (df.minima_peak_x_rsi > 0) | (df.minima_peak_x_dmp > 0) | (df.maxima_peak_x_dmn > 0)
                #     #     )
                #     # , "stg3_short"] = -2 # without atr



                #     # 2 #################&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&###############################################################################################################################
                #     # 2 #################&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&###############################################################################################################################
                #     # 2 #################&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&###############################################################################################################################




                #     # 3 #################&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&###############################################################################################################################
                #     # 3 #################&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&###############################################################################################################################
                #     # 3 #################&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&###############################################################################################################################






                #     # if interval == '1m':
                #     #     df.loc[
                #     #         # 추세 픽
                #     #         (
                #     #             ( # 진입전 추세 재확인 필요, atr14 증가하는지 확인 필요
                                    
                #     #                 ((df['AnomalyDetection_close'] > 0) & ( (df.minima_peak_x_close > 0)) & (df['atr_p'] >= 0.7))
                #     #                 # |
                #     #                 # ((df['AnomalyDetection_close'] > 0) & (df.close_diff > 0) &~ ((df.minima_peak_x_rsi > 0) | (df.maxima_peak_x_rsi < 0) | (df.minima_peak_x_close > 0) | (df.maxima_peak_x_close < 0)))
                #     #             )
                #     #             # & (
                #     #             #     (
                #     #             #         ((df.dx_diff < 0) & (df.DMN_14 > df.DMP_14) & (df.intersect == 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.high_diff > 0)           & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff > 0)
                #     #             #     )
                #     #             #         |
                #     #             #     (
                #     #             #         ((df.dx_diff < 0) & (dmn_set_2 > dmp_set_2) & (df.intersect > 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.high_diff > 0)           & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff > 0)
                #     #             #     )
                #     #             #         |
                #     #             #     (
                #     #             #         (df.dx_diff > 0) & (df.DMP_14 > df.DMN_14) #& (df.atr_p_diff < 0)# & (df.high_diff > 0) & (df.rsi_diff > 0) # & (df.obv_diff > 0) & (df.j_diff > 0) & (df['atr_AMATe_LR_8_21_2'] > 0)
                #     #             #     )
                #     #             # )
                #     #             # & (
                #     #             #     (df.vwma7_diff > 0)
                #     #             # )
                #     #         )
                #     #         # &~(
                #     #         #     # (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_macd.shift(1) < 0) | (df.maxima_peak_x_rsi.shift(1) < 0)
                #     #         #     (df.maxima_peak_x_close < 0) | (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_dmp < 0) | (df.minima_peak_x_dmn < 0)
                #     #         # )
                #     #     , "stg3_long"] = 2 # without atr

                #     #     df.loc[
                #     #         # 추세 픽
                #     #         (
                #     #             ( # 진입전 추세 재확인 필요, atr14 증가하는지 확인 필요
                #     #                 ((df['AnomalyDetection_close'] > 0) & ( (df.maxima_peak_x_close < 0)) & (df['atr_p'] >= 0.7))
                #     #                 # |
                #     #                 # ((df['AnomalyDetection_close'] > 0) & (df.close_diff < 0) &~ ((df.minima_peak_x_rsi > 0) | (df.maxima_peak_x_rsi < 0) | (df.minima_peak_x_close > 0) | (df.maxima_peak_x_close < 0)))
                #     #             )
                #     #             # & (
                #     #             #     (
                #     #             #         ((df.dx_diff < 0) & (df.DMP_14 > df.DMN_14) & (df.intersect == 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.low_diff < 0)          & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff < 0)
                #     #             #     )
                #     #             #         |
                #     #             #     (
                #     #             #         ((df.dx_diff < 0) & (dmp_set_2 > dmn_set_2) & (df.intersect > 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.low_diff < 0)          & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff < 0)
                #     #             #     )
                #     #             #         |
                #     #             #     (
                #     #             #         (df.dx_diff > 0) & (df.DMN_14 > df.DMP_14) #& (df.atr_p_diff < 0)#& (df.intersect == 0) & (df.low_diff < 0)  # & (df.obv_diff < 0) & (df.j_diff < 0) & (df['atr_AMATe_LR_8_21_2'] > 0)
                #     #             #     )
                #     #             # )
                #     #             # & (
                #     #             #     (df.vwma7_diff < 0)
                #     #             # )
                #     #         )
                #     #         # &~(
                #     #         #     # (df.minima_peak_x_macd > 0) | (df.minima_peak_x_rsi > 0) | (df.minima_peak_x_macd.shift(1) > 0) | (df.minima_peak_x_rsi.shift(1) > 0)
                #     #         #     (df.minima_peak_x_close > 0) | (df.minima_peak_x_macd > 0) | (df.minima_peak_x_rsi > 0) | (df.minima_peak_x_dmp > 0) | (df.maxima_peak_x_dmn > 0)
                #     #         # )
                #     #     , "stg3_short"] = -2 # without atr
                #     # else:
                #     #     df.loc[
                #     #         # 추세 픽
                #     #         (
                #     #             ( # 진입전 추세 재확인 필요, atr14 증가하는지 확인 필요
                                    
                #     #                 ((df['AnomalyDetection_close'] > 0) & ( (df.minima_peak_x_close > 0)))
                #     #                 # |
                #     #                 # ((df['AnomalyDetection_close'] > 0) & (df.close_diff > 0) &~ ((df.minima_peak_x_rsi > 0) | (df.maxima_peak_x_rsi < 0) | (df.minima_peak_x_close > 0) | (df.maxima_peak_x_close < 0)))
                #     #             )
                #     #             # & (
                #     #             #     (
                #     #             #         ((df.dx_diff < 0) & (df.DMN_14 > df.DMP_14) & (df.intersect == 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.high_diff > 0)           & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff > 0)
                #     #             #     )
                #     #             #         |
                #     #             #     (
                #     #             #         ((df.dx_diff < 0) & (dmn_set_2 > dmp_set_2) & (df.intersect > 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.high_diff > 0)           & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff > 0)
                #     #             #     )
                #     #             #         |
                #     #             #     (
                #     #             #         (df.dx_diff > 0) & (df.DMP_14 > df.DMN_14) #& (df.atr_p_diff < 0)# & (df.high_diff > 0) & (df.rsi_diff > 0) # & (df.obv_diff > 0) & (df.j_diff > 0) & (df['atr_AMATe_LR_8_21_2'] > 0)
                #     #             #     )
                #     #             # )
                #     #             # & (
                #     #             #     (df.vwma7_diff > 0)
                #     #             # )
                #     #         )
                #     #         # &~(
                #     #         #     # (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_macd.shift(1) < 0) | (df.maxima_peak_x_rsi.shift(1) < 0)
                #     #         #     (df.maxima_peak_x_close < 0) | (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_dmp < 0) | (df.minima_peak_x_dmn < 0)
                #     #         # )
                #     #     , "stg3_long"] = 2 # without atr

                #     #     df.loc[
                #     #         # 추세 픽
                #     #         (
                #     #             ( # 진입전 추세 재확인 필요, atr14 증가하는지 확인 필요
                #     #                 ((df['AnomalyDetection_close'] > 0) & ( (df.maxima_peak_x_close < 0)))
                #     #                 # |
                #     #                 # ((df['AnomalyDetection_close'] > 0) & (df.close_diff < 0) &~ ((df.minima_peak_x_rsi > 0) | (df.maxima_peak_x_rsi < 0) | (df.minima_peak_x_close > 0) | (df.maxima_peak_x_close < 0)))
                #     #             )
                #     #             # & (
                #     #             #     (
                #     #             #         ((df.dx_diff < 0) & (df.DMP_14 > df.DMN_14) & (df.intersect == 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.low_diff < 0)          & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff < 0)
                #     #             #     )
                #     #             #         |
                #     #             #     (
                #     #             #         ((df.dx_diff < 0) & (dmp_set_2 > dmn_set_2) & (df.intersect > 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.low_diff < 0)          & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff < 0)
                #     #             #     )
                #     #             #         |
                #     #             #     (
                #     #             #         (df.dx_diff > 0) & (df.DMN_14 > df.DMP_14) #& (df.atr_p_diff < 0)#& (df.intersect == 0) & (df.low_diff < 0)  # & (df.obv_diff < 0) & (df.j_diff < 0) & (df['atr_AMATe_LR_8_21_2'] > 0)
                #     #             #     )
                #     #             # )
                #     #             # & (
                #     #             #     (df.vwma7_diff < 0)
                #     #             # )
                #     #         )
                #     #         # &~(
                #     #         #     # (df.minima_peak_x_macd > 0) | (df.minima_peak_x_rsi > 0) | (df.minima_peak_x_macd.shift(1) > 0) | (df.minima_peak_x_rsi.shift(1) > 0)
                #     #         #     (df.minima_peak_x_close > 0) | (df.minima_peak_x_macd > 0) | (df.minima_peak_x_rsi > 0) | (df.minima_peak_x_dmp > 0) | (df.maxima_peak_x_dmn > 0)
                #     #         # )
                #     #     , "stg3_short"] = -2 # without atr



                #     # 3 #################&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&###############################################################################################################################
                #     # 3 #################&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&###############################################################################################################################
                #     # 3 #################&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&###############################################################################################################################



                #     # 4 #################&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&###############################################################################################################################
                #     # 4 #################&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&###############################################################################################################################
                #     # 4 #################&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&###############################################################################################################################






                #     # # df.loc[
                #     # #     # 추세 픽
                #     # #     (

                #     # #         (
                #     # #             (
                #     # #                 ((df.dx_diff < 0) & (df.DMN_14 > df.DMP_14) & (df.intersect == 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.high_diff > 0)           & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff > 0)
                #     # #             )
                #     # #                 |
                #     # #             (
                #     # #                 ((df.dx_diff < 0) & (dmn_set_2 > dmp_set_2) & (df.intersect > 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.high_diff > 0)           & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff > 0)
                #     # #             )
                #     # #                 |
                #     # #             (
                #     # #                 (df.dx_diff > 0) & (df.DMP_14 > df.DMN_14) #& (df.atr_p_diff < 0)# & (df.high_diff > 0) & (df.rsi_diff > 0) # & (df.obv_diff > 0) & (df.j_diff > 0) & (df['atr_AMATe_LR_8_21_2'] > 0)
                #     # #             )
                #     # #         )
                #     # #         & 
                #     # #         (
                #     # #             (df.RSI_14 > 50) & (df.RSI_14 < 80)
                #     # #         )
                #     # #     )
                #     # #     &~(
                #     # #         (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_macd.shift(1) < 0) | (df.maxima_peak_x_rsi.shift(1) < 0)
                #     # #     )
                #     # # , "stg5_long"] = 2 # without atr

                #     # # df.loc[
                #     # #     # 추세 픽
                #     # #     (

                #     # #         (
                #     # #             (
                #     # #                 ((df.dx_diff < 0) & (df.DMP_14 > df.DMN_14) & (df.intersect == 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.low_diff < 0)          & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff < 0)
                #     # #             )
                #     # #                 |
                #     # #             (
                #     # #                 ((df.dx_diff < 0) & (dmp_set_2 > dmn_set_2) & (df.intersect > 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.low_diff < 0)          & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff < 0)
                #     # #             )
                #     # #                 |
                #     # #             (
                #     # #                 (df.dx_diff > 0) & (df.DMN_14 > df.DMP_14) #& (df.atr_p_diff < 0)#& (df.intersect == 0) & (df.low_diff < 0)  # & (df.obv_diff < 0) & (df.j_diff < 0) & (df['atr_AMATe_LR_8_21_2'] > 0)
                #     # #             )
                #     # #         )
                #     # #         & 
                #     # #         (
                #     # #             (df.RSI_14 < 50) & (df.RSI_14 > 20)
                #     # #         )
                #     # #     )
                #     # #     &~(
                #     # #         (df.minima_peak_x_macd > 0) | (df.minima_peak_x_rsi > 0) | (df.minima_peak_x_macd.shift(1) > 0) | (df.minima_peak_x_rsi.shift(1) > 0)
                #     # #     )
                #     # # , "stg5_short"] = -2 # without atr

















                #     # df.loc[
                #     #     # 추세 픽
                #     #     (
                #     #         ( # 진입전 추세 재확인 필요, atr14 증가하는지 확인 필요
                #     #             # (df['minima_peak_x_rsi'] > 0) #& (df['percentage_change2'] > 3) # & (df['macd_diff_200'] > df['macds_diff_200']) & (df['macd_diff_200'] > 0) & (df['macds_diff_200'] > 0) & (df['macd_diff_700'] > 0) & (df['macds_diff_700'] > 0) & (df['MACD_12_200_9'].shift(2) < df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] > 0) & (df['k_diff'] > 0) & (df['d_diff'] > 0)
                #     #             # (df['rsi_percentage_change'] > 5) & (df['atr_percentage_change'] > 4) & (df['macd_percentage_change'] > 50) & (df['obv_percentage_change'] > 100) #& (df['atr_percentage_change'] > 4)
                #     #             # & (df['obv_percentage_change'] > 150)
                #     #             (df['obv_percentage_change'].abs() < 50) & (df.atr_p_diff < 0)
                #     #             & (((df['sma50_diff'] > 0) & (df['sma200_diff'] > 0)))
                #     #             # & (((df['sma50_diff'] < 0) & (df['sma200_diff'] < 0)))

                                
                #     #             # (df['minima_peak_x_macd'].shift(2) > 0) & (df['macd_diff_200'] > df['macds_diff_200']) & (df['macd_diff_200'].shift(2) > 0) & (df['macds_diff_200'].shift(2) > 0) & (df['macd_diff_700'].shift(2) > 0) & (df['macds_diff_700'].shift(2) > 0) & (df['MACD_12_200_9'].shift(2) < df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] > 0) & (df['k_diff'] > 0) & (df['d_diff'] > 0)
                #     #         )
                #     #         & (
                #     #             (
                #     #                 ((df.dx_diff < 0) & (df.DMN_14 > df.DMP_14) & (df.intersect == 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.high_diff > 0)           & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff > 0)
                #     #             )
                #     #                 |
                #     #             (
                #     #                 ((df.dx_diff < 0) & (dmn_set_2 > dmp_set_2) & (df.intersect > 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.high_diff > 0)           & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff > 0)
                #     #             )
                #     #                 |
                #     #             (
                #     #                 (df.dx_diff > 0) & (df.DMP_14 > df.DMN_14) #& (df.atr_p_diff < 0)# & (df.high_diff > 0) & (df.rsi_diff > 0) # & (df.obv_diff > 0) & (df.j_diff > 0) & (df['atr_AMATe_LR_8_21_2'] > 0)
                #     #             )
                #     #         )
                #     #         & (
                #     #             (df.vwma7_diff > 0)
                #     #         )
                #     #     )
                #     #     &~(
                #     #         # (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_macd.shift(1) < 0) | (df.maxima_peak_x_rsi.shift(1) < 0)
                #     #         (df.maxima_peak_x_close < 0) | (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_dmp < 0) | (df.minima_peak_x_dmn < 0)
                #     #     )
                #     # , "stg5_long"] = 2 # without atr

                #     # df.loc[
                #     #     # 추세 픽
                #     #     (
                #     #         ( # 진입전 추세 재확인 필요, atr14 증가하는지 확인 필요
                #     #             # (df['maxima_peak_x_rsi'] < 0) #& (df['percentage_change2'] < -3) # & (df['macd_diff_200'] < df['macds_diff_200']) & (df['macd_diff_200'] < 0) & (df['macds_diff_200'] < 0) & (df['macd_diff_700'] < 0) & (df['macds_diff_700'] < 0) & (df['MACD_12_200_9'].shift(2) > df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] < 0) & (df['k_diff'] < 0) & (df['d_diff'] < 0)
                #     #             # (df['rsi_percentage_change'] < -5) & (df['atr_percentage_change'] > 4) & (df['macd_percentage_change'] < -50) & (df['obv_percentage_change'] < -100) #& (df['atr_percentage_change'] > 4)
                #     #             # (df['obv_percentage_change'] < -150)
                #     #             (df['obv_percentage_change'].abs() < 50) & (df.atr_p_diff < 0)
                #     #             # & (((df['sma50_diff'] > 0) & (df['sma200_diff'] > 0)))
                #     #             & (((df['sma50_diff'] < 0) & (df['sma200_diff'] < 0)))







                                
                #     #             # (df['maxima_peak_x_macd'].shift(2) < 0) & (df['macd_diff_200'] < df['macds_diff_200']) & (df['macd_diff_200'].shift(2) < 0) & (df['macds_diff_200'].shift(2) < 0) & (df['macd_diff_700'].shift(2) < 0) & (df['macds_diff_700'].shift(2) < 0) & (df['MACD_12_200_9'].shift(2) > df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] < 0) & (df['k_diff'] < 0) & (df['d_diff'] < 0)
                #     #         )
                #     #         & (
                #     #             (
                #     #                 ((df.dx_diff < 0) & (df.DMP_14 > df.DMN_14) & (df.intersect == 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.low_diff < 0)          & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff < 0)
                #     #             )
                #     #                 |
                #     #             (
                #     #                 ((df.dx_diff < 0) & (dmp_set_2 > dmn_set_2) & (df.intersect > 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.low_diff < 0)          & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff < 0)
                #     #             )
                #     #                 |
                #     #             (
                #     #                 (df.dx_diff > 0) & (df.DMN_14 > df.DMP_14) #& (df.atr_p_diff < 0)#& (df.intersect == 0) & (df.low_diff < 0)  # & (df.obv_diff < 0) & (df.j_diff < 0) & (df['atr_AMATe_LR_8_21_2'] > 0)
                #     #             )
                #     #         )
                #     #         & (
                #     #             (df.vwma7_diff < 0)
                #     #         )
                #     #     )
                #     #     &~(
                #     #         # (df.minima_peak_x_macd > 0) | (df.minima_peak_x_rsi > 0) | (df.minima_peak_x_macd.shift(1) > 0) | (df.minima_peak_x_rsi.shift(1) > 0)
                #     #         (df.minima_peak_x_close > 0) | (df.minima_peak_x_macd > 0) | (df.minima_peak_x_rsi > 0) | (df.minima_peak_x_dmp > 0) | (df.maxima_peak_x_dmn > 0)
                #     #     )
                #     # , "stg5_short"] = -2 # without atr








                #     # from adtk.detector import VolatilityShiftAD
                #     # volatility_shift_ad_positive = VolatilityShiftAD(c=6.0, side='positive', window=30)
                #     # anomalies_close_positive = volatility_shift_ad_positive.fit_detect(df['close'])

                #     # volatility_shift_ad_negative = VolatilityShiftAD(c=6.0, side='negative', window=30)
                #     # anomalies_close_negative = volatility_shift_ad_negative.fit_detect(df['close'])

                #     # df['anomalies_close_positive'] = anomalies_close_positive
                #     # df['anomalies_close_negative'] = anomalies_close_negative

                #     # # df.loc[df['anomalies_close_positive'] == True, 'stg7_long'] = 2
                #     # # df.loc[df['anomalies_close_negative'] == True, 'stg7_short'] = -2
                    
                #     # df.loc[(df['anomalies_close_positive'] == True) & (df['stg5_long'] > 0), 'stg3_long'] = 2
                #     # df.loc[(df['anomalies_close_positive'] == True) & (df['stg5_short'] < 0), 'stg3_short'] = -2
                    






                #     # 4 #################&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&###############################################################################################################################
                #     # 4 #################&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&###############################################################################################################################
                #     # 4 #################&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&###############################################################################################################################
















                #     # df.loc[
                #     #     # 추세 픽
                #     #     (
                #     #         ( # 진입전 추세 재확인 필요, atr14 증가하는지 확인 필요
                                
                #     #             ((df['AnomalyDetection_close'] > 0) & ((df.minima_peak_x_rsi > 0) | (df.minima_peak_x_close > 0)))
                #     #             # |
                #     #             # ((df['AnomalyDetection_close'] > 0) & (df.close_diff > 0) &~ ((df.minima_peak_x_rsi > 0) | (df.maxima_peak_x_rsi < 0) | (df.minima_peak_x_close > 0) | (df.maxima_peak_x_close < 0)))
                #     #         )
                #     #         # & (
                #     #         #     (
                #     #         #         ((df.dx_diff < 0) & (df.DMN_14 > df.DMP_14) & (df.intersect == 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.high_diff > 0)           & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff > 0)
                #     #         #     )
                #     #         #         |
                #     #         #     (
                #     #         #         ((df.dx_diff < 0) & (dmn_set_2 > dmp_set_2) & (df.intersect > 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.high_diff > 0)           & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff > 0)
                #     #         #     )
                #     #         #         |
                #     #         #     (
                #     #         #         (df.dx_diff > 0) & (df.DMP_14 > df.DMN_14) #& (df.atr_p_diff < 0)# & (df.high_diff > 0) & (df.rsi_diff > 0) # & (df.obv_diff > 0) & (df.j_diff > 0) & (df['atr_AMATe_LR_8_21_2'] > 0)
                #     #         #     )
                #     #         # )
                #     #         # & (
                #     #         #     (df.vwma7_diff > 0)
                #     #         # )
                #     #     )
                #     #     # &~(
                #     #     #     # (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_macd.shift(1) < 0) | (df.maxima_peak_x_rsi.shift(1) < 0)
                #     #     #     (df.maxima_peak_x_close < 0) | (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_dmp < 0) | (df.minima_peak_x_dmn < 0)
                #     #     # )
                #     # , "stg3_long"] = 2 # without atr

                #     # df.loc[
                #     #     # 추세 픽
                #     #     (
                #     #         ( # 진입전 추세 재확인 필요, atr14 증가하는지 확인 필요
                #     #             ((df['AnomalyDetection_close'] > 0) & ((df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_close < 0)))
                #     #             # |
                #     #             # ((df['AnomalyDetection_close'] > 0) & (df.close_diff < 0) &~ ((df.minima_peak_x_rsi > 0) | (df.maxima_peak_x_rsi < 0) | (df.minima_peak_x_close > 0) | (df.maxima_peak_x_close < 0)))
                #     #         )
                #     #         # & (
                #     #         #     (
                #     #         #         ((df.dx_diff < 0) & (df.DMP_14 > df.DMN_14) & (df.intersect == 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.low_diff < 0)          & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff < 0)
                #     #         #     )
                #     #         #         |
                #     #         #     (
                #     #         #         ((df.dx_diff < 0) & (dmp_set_2 > dmn_set_2) & (df.intersect > 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.low_diff < 0)          & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff < 0)
                #     #         #     )
                #     #         #         |
                #     #         #     (
                #     #         #         (df.dx_diff > 0) & (df.DMN_14 > df.DMP_14) #& (df.atr_p_diff < 0)#& (df.intersect == 0) & (df.low_diff < 0)  # & (df.obv_diff < 0) & (df.j_diff < 0) & (df['atr_AMATe_LR_8_21_2'] > 0)
                #     #         #     )
                #     #         # )
                #     #         # & (
                #     #         #     (df.vwma7_diff < 0)
                #     #         # )
                #     #     )
                #     #     # &~(
                #     #     #     # (df.minima_peak_x_macd > 0) | (df.minima_peak_x_rsi > 0) | (df.minima_peak_x_macd.shift(1) > 0) | (df.minima_peak_x_rsi.shift(1) > 0)
                #     #     #     (df.minima_peak_x_close > 0) | (df.minima_peak_x_macd > 0) | (df.minima_peak_x_rsi > 0) | (df.minima_peak_x_dmp > 0) | (df.maxima_peak_x_dmn > 0)
                #     #     # )
                #     # , "stg3_short"] = -2 # without atr






                #     # df.loc[
                #     #     # (
                #     #     #     (df.minima_peak_x_macd > 0)
                #     #     # )
                #     #     # |
                #     #     ((df.DMN_14 > df.DMP_14) & (df.ATRr_1 > df.close*8/100) & (df.ATRr_1 > (df.ATRr_14 * 2)))
                #     # , "stg1_long"] = 2 # without atr
                #     # df.loc[
                #     #     # (
                #     #     #     (df.maxima_peak_x_macd < 0)
                #     #     # )
                #     #     # |
                #     #     ((df.DMP_14 > df.DMN_14) & (df.ATRr_1 > df.close*8/100) & (df.ATRr_1 > (df.ATRr_14 * 2)))
                #     # , "stg1_short"] = -2 # without atr

                #     # df.loc[
                #     #     (
                #     #         (df.stg1_long.shift(1) > 0)
                #     #     )

                #     # , "stg10_long"] = 2 # without atr
                #     # df.loc[
                #     #     (
                #     #         (df.stg1_short.shift(1) < 0)
                #     #     )

                #     # , "stg10_short"] = -2 # without atr




                #     # # 5 #################&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&###############################################################################################################################
                #     # # 5 #################&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&###############################################################################################################################
                #     # # 5 #################&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&###############################################################################################################################

















































































                #     # # # 5 #################&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&###############################################################################################################################
                #     # # # 5 #################&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&###############################################################################################################################
                #     # # # 5 #################&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&###############################################################################################################################





                #     # # if ((interval == '1m') and (df['atr_p'] >= 0.7).all()) or (interval != '1m'):




                #     # df.loc[
                #     #     # 추세 픽
                #     #     (
                #     #         ( # 진입전 추세 재확인 필요, atr14 증가하는지 확인 필요
                #     #             # (df['minima_peak_x_rsi'] > 0) #& (df['percentage_change2'] > 3) # & (df['macd_diff_200'] > df['macds_diff_200']) & (df['macd_diff_200'] > 0) & (df['macds_diff_200'] > 0) & (df['macd_diff_700'] > 0) & (df['macds_diff_700'] > 0) & (df['MACD_12_200_9'].shift(2) < df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] > 0) & (df['k_diff'] > 0) & (df['d_diff'] > 0)
                #     #             # (df['rsi_percentage_change'] > 5) & (df['atr_percentage_change'] > 4) & (df['macd_percentage_change'] > 50) & (df['obv_percentage_change'] > 100) #& (df['atr_percentage_change'] > 4)
                #     #             # & (df['obv_percentage_change'] > 150)
                #     #             # (df['obv_percentage_change'].abs() < 50)
                #     #             (((df['sma50_diff'] > 0) & (df['sma200_diff'] > 0)))
                #     #             & (df.SMA_50 > df.SMA_200)
                #     #             & (df.sma50_percentage_change > df.sma200_percentage_change)
                #     #             # & (df.RSI_14 < 75)
                #     #             # & (df.macd_percentage_change > df.macds_percentage_change)
                #     #             # & (df.DX_14 < 70)
                #     #             # & (df.adx_diff > 0)
                #     #             # # & (df.atr_diff < 0)
                #     #             # &~ ((df.adx_diff < 0) & (df.atr_diff < 0) & (df.atr_p < 0.5))
                #     #             # & (df.atr_p < 0.7)







                #     #             # & (((df['sma50_diff'] > df['sma200_diff'])))

                #     #             # & (df.adx_diff > 0)

                #     #             # & (df.bbb_diff > 0)

                #     #             # & (df.atr_diff > 0)
                #     #             # & ((df.MACD_12_26_9 > 0) & (df.MACDh_12_26_9 > 0) & (df.macd_diff > 0) & (df.macdh_diff > 0))
                #     #             & (df.obv_diff > 0)


                #     #             # (df['minima_peak_x_macd'].shift(2) > 0) & (df['macd_diff_200'] > df['macds_diff_200']) & (df['macd_diff_200'].shift(2) > 0) & (df['macds_diff_200'].shift(2) > 0) & (df['macd_diff_700'].shift(2) > 0) & (df['macds_diff_700'].shift(2) > 0) & (df['MACD_12_200_9'].shift(2) < df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] > 0) & (df['k_diff'] > 0) & (df['d_diff'] > 0)
                #     #         )
                #     #         # & 

                #     #         # (
                #     #         #     (
                #     #         #         ((df.dx_diff < 0) & (df.DMN_14 > df.DMP_14) & (df.intersect == 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.high_diff > 0)           & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff > 0)
                #     #         #     )
                #     #         #         |
                #     #         #     (
                #     #         #         ((df.dx_diff < 0) & (dmn_set_2 > dmp_set_2) & (df.intersect > 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.high_diff > 0)           & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff > 0)
                #     #         #     )
                #     #         #         |
                #     #         #     (
                #     #         #         (df.dx_diff > 0) & (df.DMP_14 > df.DMN_14) #& (df.atr_p_diff < 0)# & (df.high_diff > 0) & (df.rsi_diff > 0) # & (df.obv_diff > 0) & (df.j_diff > 0) & (df['atr_AMATe_LR_8_21_2'] > 0)
                #     #         #     )
                #     #         # )
                #     #         & (
                #     #             (df.vwma7_diff > 0)
                #     #         )
                #     #     )
                #     #     # &~(
                #     #     #     # ((df.bbb_diff > 0) & (df.atr_diff > 0) & (df.adx_diff > 0) & ((df.MACD_12_26_9 > 0) & (df.MACDh_12_26_9 > 0) & (df.macd_diff > 0) & (df.macdh_diff > 0))) |
                #     #     #     # (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_macd.shift(1) < 0) | (df.maxima_peak_x_rsi.shift(1) < 0)
                #     #     #     (df.maxima_peak_x_close < 0) | (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_dmp < 0) | (df.minima_peak_x_dmn < 0)
                #     #     # )
                #     #     # &~(
                #     #     #     # ((df.bbb_diff > 0) & (df.atr_diff > 0) & (df.adx_diff > 0) & ((df.MACD_12_26_9 > 0) & (df.MACDh_12_26_9 > 0) & (df.macd_diff > 0) & (df.macdh_diff > 0))) |
                #     #     #     # (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_macd.shift(1) < 0) | (df.maxima_peak_x_rsi.shift(1) < 0)
                #     #     #     (df.adx_diff < 0) & (df.atr_diff < 0) #& (df.atr_p < 0.7)
                #     #     # )
                #     # , "stg1_long"] = 2 # without atr

                #     # df.loc[
                #     #     # 추세 픽
                #     #     (
                #     #         ( # 진입전 추세 재확인 필요, atr14 증가하는지 확인 필요
                #     #             # (df['maxima_peak_x_rsi'] < 0) #& (df['percentage_change2'] < -3) # & (df['macd_diff_200'] < df['macds_diff_200']) & (df['macd_diff_200'] < 0) & (df['macds_diff_200'] < 0) & (df['macd_diff_700'] < 0) & (df['macds_diff_700'] < 0) & (df['MACD_12_200_9'].shift(2) > df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] < 0) & (df['k_diff'] < 0) & (df['d_diff'] < 0)
                #     #             # (df['rsi_percentage_change'] < -5) & (df['atr_percentage_change'] > 4) & (df['macd_percentage_change'] < -50) & (df['obv_percentage_change'] < -100) #& (df['atr_percentage_change'] > 4)
                #     #             # (df['obv_percentage_change'] < -150)
                #     #             # (df['obv_percentage_change'].abs() < 50)
                #     #             # & (((df['sma50_diff'] > 0) & (df['sma200_diff'] > 0)))
                #     #             (((df['sma50_diff'] < 0) & (df['sma200_diff'] < 0)))
                #     #             & (df.SMA_50 < df.SMA_200)
                #     #             & (df.sma50_percentage_change < df.sma200_percentage_change)
                #     #             # & (df.RSI_14 > 25)
                #     #             # & (df.macd_percentage_change < df.macds_percentage_change)
                #     #             # & (df.DX_14 < 70)
                #     #             # & (df.adx_diff > 0)
                #     #             # # & (df.atr_diff < 0)
                #     #             # &~ ((df.adx_diff < 0) & (df.atr_diff < 0) & (df.atr_p < 0.5))                       
                #     #             # & (df.atr_p > 1.2)

                #     #             # & (((df['sma50_diff'] < df['sma200_diff'])))
                #     #             # & (df.adx_diff > 0)

                #     #             # & (df.bbb_diff > 0)
                                
                #     #             # & (df.atr_diff > 0)
                #     #             # & ((df.MACD_12_26_9 < 0) & (df.MACDh_12_26_9 < 0) & (df.macd_diff < 0) & (df.macdh_diff < 0))
                #     #             & (df.obv_diff < 0)
                #     #             # & (df.atr_p < 0.7)

                                
                #     #             # (df['maxima_peak_x_macd'].shift(2) < 0) & (df['macd_diff_200'] < df['macds_diff_200']) & (df['macd_diff_200'].shift(2) < 0) & (df['macds_diff_200'].shift(2) < 0) & (df['macd_diff_700'].shift(2) < 0) & (df['macds_diff_700'].shift(2) < 0) & (df['MACD_12_200_9'].shift(2) > df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] < 0) & (df['k_diff'] < 0) & (df['d_diff'] < 0)
                #     #         )
                #     #         # & 

                #     #         # (
                #     #         #     (
                #     #         #         ((df.dx_diff < 0) & (df.DMP_14 > df.DMN_14) & (df.intersect == 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.low_diff < 0)          & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff < 0)
                #     #         #     )
                #     #         #         |
                #     #         #     (
                #     #         #         ((df.dx_diff < 0) & (dmp_set_2 > dmn_set_2) & (df.intersect > 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.low_diff < 0)          & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff < 0)
                #     #         #     )
                #     #         #         |
                #     #         #     (
                #     #         #         (df.dx_diff > 0) & (df.DMN_14 > df.DMP_14) #& (df.atr_p_diff < 0)#& (df.intersect == 0) & (df.low_diff < 0)  # & (df.obv_diff < 0) & (df.j_diff < 0) & (df['atr_AMATe_LR_8_21_2'] > 0)
                #     #         #     )
                #     #         # )
                #     #         & (
                #     #             (df.vwma7_diff < 0)
                #     #         )
                #     #     )
                #     #     # &~(
                #     #     #     # ((df.bbb_diff > 0) & (df.atr_diff > 0) & (df.adx_diff > 0) & ((df.MACD_12_26_9 < 0) & (df.MACDh_12_26_9 < 0) & (df.macd_diff < 0) & (df.macdh_diff < 0))) |
                #     #     #     # (df.bbb_diff > 0) | (df.atr_diff > 0) |
                #     #     #     # (df.minima_peak_x_macd > 0) | (df.minima_peak_x_rsi > 0) | (df.minima_peak_x_macd.shift(1) > 0) | (df.minima_peak_x_rsi.shift(1) > 0)
                #     #     #     (df.minima_peak_x_close > 0) | (df.minima_peak_x_macd > 0) | (df.minima_peak_x_rsi > 0) | (df.minima_peak_x_dmp > 0) | (df.maxima_peak_x_dmn > 0)
                #     #     # )
                #     #     # &~(
                #     #     #     # ((df.bbb_diff > 0) & (df.atr_diff > 0) & (df.adx_diff > 0) & ((df.MACD_12_26_9 > 0) & (df.MACDh_12_26_9 > 0) & (df.macd_diff > 0) & (df.macdh_diff > 0))) |
                #     #     #     # (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_macd.shift(1) < 0) | (df.maxima_peak_x_rsi.shift(1) < 0)
                #     #     #     (df.adx_diff < 0) & (df.atr_diff < 0) #& (df.atr_p < 0.7)
                #     #     # )                
                #     # , "stg1_short"] = -2 # without atr





                #     # # df.loc[
                #     # #     (
                #     # #         (df.stg15_long.shift(2) > 0)
                #     # #     )

                #     # # , "stg5_long"] = 2 # without atr
                #     # # df.loc[
                #     # #     (
                #     # #         (df.stg15_short.shift(2) < 0)
                #     # #     )

                #     # # , "stg5_short"] = -2 # without atr











                #     # df.loc[
                #     #     # 추세 픽
                #     #     (
                #     #         ( # 진입전 추세 재확인 필요, atr14 증가하는지 확인 필요
                #     #             # (df['minima_peak_x_rsi'] > 0) #& (df['percentage_change2'] > 3) # & (df['macd_diff_200'] > df['macds_diff_200']) & (df['macd_diff_200'] > 0) & (df['macds_diff_200'] > 0) & (df['macd_diff_700'] > 0) & (df['macds_diff_700'] > 0) & (df['MACD_12_200_9'].shift(2) < df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] > 0) & (df['k_diff'] > 0) & (df['d_diff'] > 0)
                #     #             # (df['rsi_percentage_change'] > 5) & (df['atr_percentage_change'] > 4) & (df['macd_percentage_change'] > 50) & (df['obv_percentage_change'] > 100) #& (df['atr_percentage_change'] > 4)
                #     #             # & (df['obv_percentage_change'] > 150)
                #     #             # (df['obv_percentage_change'].abs() < 50)
                #     #             (((df['sma50_diff'] < 0) & (df['sma200_diff'] < 0)) | ((df['sma50_diff'] > 0) & (df['sma200_diff'] < 0)))
                #     #             & (df.SMA_50 < df.SMA_200)
                #     #             & (df.sma50_percentage_change > df.sma200_percentage_change)
                #     #             & (df.macd_percentage_change > df.macds_percentage_change)
                #     #             # & (df.RSI_14 < 75)
                #     #             # & (df.DX_14 < 70)
                #     #             # &~ ((df.adx_diff < 0) & (df.atr_diff < 0) & (df.atr_p < 0.5))

                #     #             # & (((df['sma50_diff'] > df['sma200_diff'])))
                #     #             # & (df.adx_diff > 0)

                #     #             # & (df.bbb_diff > 0)
                                
                #     #             # & (df.atr_diff > 0)
                #     #             # & ((df.MACD_12_26_9 > 0) & (df.MACDh_12_26_9 > 0) & (df.macd_diff > 0) & (df.macdh_diff > 0))
                #     #             & (df.obv_diff > 0)
                #     #             # & (df.atr_p < 0.7)


                #     #             # (df['minima_peak_x_macd'].shift(2) > 0) & (df['macd_diff_200'] > df['macds_diff_200']) & (df['macd_diff_200'].shift(2) > 0) & (df['macds_diff_200'].shift(2) > 0) & (df['macd_diff_700'].shift(2) > 0) & (df['macds_diff_700'].shift(2) > 0) & (df['MACD_12_200_9'].shift(2) < df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] > 0) & (df['k_diff'] > 0) & (df['d_diff'] > 0)
                #     #         )


                #     #         # & 

                #     #         # (
                #     #         #     (
                #     #         #         ((df.dx_diff < 0) & (df.DMN_14 > df.DMP_14) & (df.intersect == 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.high_diff > 0)           & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff > 0)
                #     #         #     )
                #     #         #         |
                #     #         #     (
                #     #         #         ((df.dx_diff < 0) & (dmn_set_2 > dmp_set_2) & (df.intersect > 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.high_diff > 0)           & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff > 0)
                #     #         #     )
                #     #         #         |
                #     #         #     (
                #     #         #         (df.dx_diff > 0) & (df.DMP_14 > df.DMN_14) #& (df.atr_p_diff < 0)# & (df.high_diff > 0) & (df.rsi_diff > 0) # & (df.obv_diff > 0) & (df.j_diff > 0) & (df['atr_AMATe_LR_8_21_2'] > 0)
                #     #         #     )
                #     #         # )
                #     #         # & (
                #     #         #     (df.vwma7_diff > 0)
                #     #         # )
                #     #     )
                #     #     # &~(
                #     #     #     # ((df.bbb_diff > 0) & (df.atr_diff > 0) & (df.adx_diff > 0) & ((df.MACD_12_26_9 > 0) & (df.MACDh_12_26_9 > 0) & (df.macd_diff > 0) & (df.macdh_diff > 0))) |
                #     #     #     # (df.bbb_diff > 0) | (df.atr_diff > 0) |
                #     #     #     # (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_macd.shift(1) < 0) | (df.maxima_peak_x_rsi.shift(1) < 0)
                #     #     #     (df.maxima_peak_x_close < 0) | (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_dmp < 0) | (df.minima_peak_x_dmn < 0)
                #     #     # )
                #     #     # &~(
                #     #     #     # ((df.bbb_diff > 0) & (df.atr_diff > 0) & (df.adx_diff > 0) & ((df.MACD_12_26_9 > 0) & (df.MACDh_12_26_9 > 0) & (df.macd_diff > 0) & (df.macdh_diff > 0))) |
                #     #     #     # (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_macd.shift(1) < 0) | (df.maxima_peak_x_rsi.shift(1) < 0)
                #     #     #     (df.adx_diff < 0) & (df.atr_diff < 0) #& (df.atr_p < 0.7)
                #     #     # )                
                #     # , "stg2_long"] = 2 # without atr

                #     # df.loc[
                #     #     # 추세 픽
                #     #     (
                #     #         ( # 진입전 추세 재확인 필요, atr14 증가하는지 확인 필요
                #     #             # (df['maxima_peak_x_rsi'] < 0) #& (df['percentage_change2'] < -3) # & (df['macd_diff_200'] < df['macds_diff_200']) & (df['macd_diff_200'] < 0) & (df['macds_diff_200'] < 0) & (df['macd_diff_700'] < 0) & (df['macds_diff_700'] < 0) & (df['MACD_12_200_9'].shift(2) > df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] < 0) & (df['k_diff'] < 0) & (df['d_diff'] < 0)
                #     #             # (df['rsi_percentage_change'] < -5) & (df['atr_percentage_change'] > 4) & (df['macd_percentage_change'] < -50) & (df['obv_percentage_change'] < -100) #& (df['atr_percentage_change'] > 4)
                #     #             # (df['obv_percentage_change'] < -150)
                #     #             # (df['obv_percentage_change'].abs() < 50)
                #     #             # & (((df['sma50_diff'] > 0) & (df['sma200_diff'] > 0)))
                #     #             (((df['sma50_diff'] > 0) & (df['sma200_diff'] > 0)) | ((df['sma50_diff'] < 0) & (df['sma200_diff'] > 0)))
                #     #             & (df.SMA_50 > df.SMA_200)
                #     #             & (df.sma50_percentage_change < df.sma200_percentage_change)
                #     #             & (df.macd_percentage_change < df.macds_percentage_change)
                #     #             # & (df.RSI_14 > 25)
                #     #             # & (df.DX_14 < 70)
                #     #             # &~ ((df.adx_diff < 0) & (df.atr_diff < 0) & (df.atr_p < 0.5))
                #     #             # &~ (df.atr_diff < 0)

                #     #             # & (((df['sma50_diff'] < df['sma200_diff'])))
                #     #             # &~ (df.adx_diff > 0)

                #     #             # & (df.bbb_diff > 0)
                                
                #     #             # & (df.atr_diff > 0)
                #     #             # & ((df.MACD_12_26_9 < 0) & (df.MACDh_12_26_9 < 0) & (df.macd_diff < 0) & (df.macdh_diff < 0))
                #     #             & (df.obv_diff < 0)
                #     #             # & (df.atr_p < 0.7)

                                
                #     #             # (df['maxima_peak_x_macd'].shift(2) < 0) & (df['macd_diff_200'] < df['macds_diff_200']) & (df['macd_diff_200'].shift(2) < 0) & (df['macds_diff_200'].shift(2) < 0) & (df['macd_diff_700'].shift(2) < 0) & (df['macds_diff_700'].shift(2) < 0) & (df['MACD_12_200_9'].shift(2) > df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] < 0) & (df['k_diff'] < 0) & (df['d_diff'] < 0)
                #     #         )


                #     #         # & 

                #     #         # (
                #     #         #     (
                #     #         #         ((df.dx_diff < 0) & (df.DMP_14 > df.DMN_14) & (df.intersect == 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.low_diff < 0)          & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff < 0)
                #     #         #     )
                #     #         #         |
                #     #         #     (
                #     #         #         ((df.dx_diff < 0) & (dmp_set_2 > dmn_set_2) & (df.intersect > 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.low_diff < 0)          & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff < 0)
                #     #         #     )
                #     #         #         |
                #     #         #     (
                #     #         #         (df.dx_diff > 0) & (df.DMN_14 > df.DMP_14) #& (df.atr_p_diff < 0)#& (df.intersect == 0) & (df.low_diff < 0)  # & (df.obv_diff < 0) & (df.j_diff < 0) & (df['atr_AMATe_LR_8_21_2'] > 0)
                #     #         #     )
                #     #         # )
                #     #         # & (
                #     #         #     (df.vwma7_diff < 0)
                #     #         # )
                #     #     )
                #     #     # &~(
                #     #     #     # ((df.bbb_diff > 0) & (df.atr_diff > 0) & (df.adx_diff > 0) & ((df.MACD_12_26_9 < 0) & (df.MACDh_12_26_9 < 0) & (df.macd_diff < 0) & (df.macdh_diff < 0))) |
                #     #     #     # (df.bbb_diff > 0) | (df.atr_diff > 0) |
                #     #     #     # (df.minima_peak_x_macd > 0) | (df.minima_peak_x_rsi > 0) | (df.minima_peak_x_macd.shift(1) > 0) | (df.minima_peak_x_rsi.shift(1) > 0)
                #     #     #     (df.minima_peak_x_close > 0) | (df.minima_peak_x_macd > 0) | (df.minima_peak_x_rsi > 0) | (df.minima_peak_x_dmp > 0) | (df.maxima_peak_x_dmn > 0)
                #     #     # )
                #     #     # &~(
                #     #     #     # ((df.bbb_diff > 0) & (df.atr_diff > 0) & (df.adx_diff > 0) & ((df.MACD_12_26_9 > 0) & (df.MACDh_12_26_9 > 0) & (df.macd_diff > 0) & (df.macdh_diff > 0))) |
                #     #     #     # (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_macd.shift(1) < 0) | (df.maxima_peak_x_rsi.shift(1) < 0)
                #     #     #     (df.adx_diff < 0) & (df.atr_diff < 0) #& (df.atr_p < 0.7)
                #     #     # )
                #     # , "stg2_short"] = -2 # without atr




































                #     # # # 5 #################&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&###############################################################################################################################
                #     # # # 5 #################&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&###############################################################################################################################
                #     # # # 5 #################&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&###############################################################################################################################



                #     # # # 5 -re1 #################&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&###############################################################################################################################
                #     # # # 5 -re1 #################&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&###############################################################################################################################
                #     # # # 5 -re1 #################&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&###############################################################################################################################





                #     # df.loc[
                #     #     # 추세 픽
                #     #     (
                #     #         ( # 진입전 추세 재확인 필요, atr14 증가하는지 확인 필요
                #     #             # (df['minima_peak_x_rsi'] > 0) #& (df['percentage_change2'] > 3) # & (df['macd_diff_200'] > df['macds_diff_200']) & (df['macd_diff_200'] > 0) & (df['macds_diff_200'] > 0) & (df['macd_diff_700'] > 0) & (df['macds_diff_700'] > 0) & (df['MACD_12_200_9'].shift(2) < df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] > 0) & (df['k_diff'] > 0) & (df['d_diff'] > 0)
                #     #             # (df['rsi_percentage_change'] > 5) & (df['atr_percentage_change'] > 4) & (df['macd_percentage_change'] > 50) & (df['obv_percentage_change'] > 100) #& (df['atr_percentage_change'] > 4)
                #     #             # & (df['obv_percentage_change'] > 150)
                #     #             # (df['obv_percentage_change'].abs() < 50)
                #     #             (((df['sma50_diff'] > 0) & (df['sma200_diff'] > 0)))
                #     #             & (df.SMA_50 > df.SMA_200)
                #     #             & (df.sma50_percentage_change > df.sma200_percentage_change)
                #     #             # & (df.RSI_14 < 75)
                #     #             # & (df.macd_percentage_change > df.macds_percentage_change)
                #     #             # & (df.DX_14 < 70)
                #     #             # & (df.adx_diff > 0)
                #     #             # # & (df.atr_diff < 0)
                #     #             # &~ ((df.adx_diff < 0) & (df.atr_diff < 0) & (df.atr_p < 0.5))
                #     #             # & (df.atr_p < 0.7)







                #     #             # & (((df['sma50_diff'] > df['sma200_diff'])))

                #     #             # & (df.adx_diff > 0)

                #     #             # & (df.bbb_diff > 0)

                #     #             # & (df.atr_diff > 0)
                #     #             # & ((df.MACD_12_26_9 > 0) & (df.MACDh_12_26_9 > 0) & (df.macd_diff > 0) & (df.macdh_diff > 0))
                #     #             & (df.obv_diff > 0)


                #     #             # (df['minima_peak_x_macd'].shift(2) > 0) & (df['macd_diff_200'] > df['macds_diff_200']) & (df['macd_diff_200'].shift(2) > 0) & (df['macds_diff_200'].shift(2) > 0) & (df['macd_diff_700'].shift(2) > 0) & (df['macds_diff_700'].shift(2) > 0) & (df['MACD_12_200_9'].shift(2) < df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] > 0) & (df['k_diff'] > 0) & (df['d_diff'] > 0)
                #     #         )
                #     #         & 

                #     #         (
                #     #             # (
                #     #             #     ((df.dx_diff < 0) & (df.DMN_14 > df.DMP_14) & (df.intersect == 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.high_diff > 0)           & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff > 0)
                #     #             # )
                #     #             #     |
                #     #             # (
                #     #             #     ((df.dx_diff < 0) & (dmn_set_2 > dmp_set_2) & (df.intersect > 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.high_diff > 0)           & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff > 0)
                #     #             # )
                #     #                 # |
                #     #             (
                #     #                 (df.adx_diff > 0) & (df.DMP_14 > df.DMN_14) #& (df.atr_p_diff < 0)# & (df.high_diff > 0) & (df.rsi_diff > 0) # & (df.obv_diff > 0) & (df.j_diff > 0) & (df['atr_AMATe_LR_8_21_2'] > 0)
                #     #             )
                #     #         )
                #     #         # & (
                #     #         #     (df.vwma7_diff > 0)
                #     #         # )
                #     #     )
                #     #     # &~(
                #     #     #     # ((df.bbb_diff > 0) & (df.atr_diff > 0) & (df.adx_diff > 0) & ((df.MACD_12_26_9 > 0) & (df.MACDh_12_26_9 > 0) & (df.macd_diff > 0) & (df.macdh_diff > 0))) |
                #     #     #     # (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_macd.shift(1) < 0) | (df.maxima_peak_x_rsi.shift(1) < 0)
                #     #     #     (df.maxima_peak_x_close < 0) | (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_dmp < 0) | (df.minima_peak_x_dmn < 0)
                #     #     # )
                #     #     # &~(
                #     #     #     # ((df.bbb_diff > 0) & (df.atr_diff > 0) & (df.adx_diff > 0) & ((df.MACD_12_26_9 > 0) & (df.MACDh_12_26_9 > 0) & (df.macd_diff > 0) & (df.macdh_diff > 0))) |
                #     #     #     # (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_macd.shift(1) < 0) | (df.maxima_peak_x_rsi.shift(1) < 0)
                #     #     #     (df.adx_diff < 0) & (df.atr_diff < 0) #& (df.atr_p < 0.7)
                #     #     # )
                #     #     &~ (
                #     #         (
                #     #             (
                #     #                 ((df.DX_14 > 40) & (df.adx_diff < 0) & (df.DMP_14 > df.DMN_14) & (df.intersect == 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.low_diff < 0)          & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff < 0)
                #     #             )


                #     #         )
                #     #     )                
                #     # , "stg1_long"] = 2 # without atr

                #     # df.loc[
                #     #     # 추세 픽
                #     #     (
                #     #         ( # 진입전 추세 재확인 필요, atr14 증가하는지 확인 필요
                #     #             # (df['maxima_peak_x_rsi'] < 0) #& (df['percentage_change2'] < -3) # & (df['macd_diff_200'] < df['macds_diff_200']) & (df['macd_diff_200'] < 0) & (df['macds_diff_200'] < 0) & (df['macd_diff_700'] < 0) & (df['macds_diff_700'] < 0) & (df['MACD_12_200_9'].shift(2) > df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] < 0) & (df['k_diff'] < 0) & (df['d_diff'] < 0)
                #     #             # (df['rsi_percentage_change'] < -5) & (df['atr_percentage_change'] > 4) & (df['macd_percentage_change'] < -50) & (df['obv_percentage_change'] < -100) #& (df['atr_percentage_change'] > 4)
                #     #             # (df['obv_percentage_change'] < -150)
                #     #             # (df['obv_percentage_change'].abs() < 50)
                #     #             # & (((df['sma50_diff'] > 0) & (df['sma200_diff'] > 0)))
                #     #             (((df['sma50_diff'] < 0) & (df['sma200_diff'] < 0)))
                #     #             & (df.SMA_50 < df.SMA_200)
                #     #             & (df.sma50_percentage_change < df.sma200_percentage_change)
                #     #             # & (df.RSI_14 > 25)
                #     #             # & (df.macd_percentage_change < df.macds_percentage_change)
                #     #             # & (df.DX_14 < 70)
                #     #             # & (df.adx_diff > 0)
                #     #             # # & (df.atr_diff < 0)
                #     #             # &~ ((df.adx_diff < 0) & (df.atr_diff < 0) & (df.atr_p < 0.5))                       
                #     #             # & (df.atr_p > 1.2)

                #     #             # & (((df['sma50_diff'] < df['sma200_diff'])))
                #     #             # & (df.adx_diff > 0)

                #     #             # & (df.bbb_diff > 0)
                                
                #     #             # & (df.atr_diff > 0)
                #     #             # & ((df.MACD_12_26_9 < 0) & (df.MACDh_12_26_9 < 0) & (df.macd_diff < 0) & (df.macdh_diff < 0))
                #     #             & (df.obv_diff < 0)
                #     #             # & (df.atr_p < 0.7)

                                
                #     #             # (df['maxima_peak_x_macd'].shift(2) < 0) & (df['macd_diff_200'] < df['macds_diff_200']) & (df['macd_diff_200'].shift(2) < 0) & (df['macds_diff_200'].shift(2) < 0) & (df['macd_diff_700'].shift(2) < 0) & (df['macds_diff_700'].shift(2) < 0) & (df['MACD_12_200_9'].shift(2) > df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] < 0) & (df['k_diff'] < 0) & (df['d_diff'] < 0)
                #     #         )
                #     #         & 

                #     #         (
                #     #             # (
                #     #             #     ((df.dx_diff < 0) & (df.DMP_14 > df.DMN_14) & (df.intersect == 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.low_diff < 0)          & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff < 0)
                #     #             # )
                #     #             #     |
                #     #             # (
                #     #             #     ((df.dx_diff < 0) & (dmp_set_2 > dmn_set_2) & (df.intersect > 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.low_diff < 0)          & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff < 0)
                #     #             # )
                #     #             #     |
                #     #             (
                #     #                 (df.adx_diff > 0) & (df.DMN_14 > df.DMP_14) #& (df.atr_p_diff < 0)#& (df.intersect == 0) & (df.low_diff < 0)  # & (df.obv_diff < 0) & (df.j_diff < 0) & (df['atr_AMATe_LR_8_21_2'] > 0)
                #     #             )
                #     #         )
                #     #         # & (
                #     #         #     (df.vwma7_diff < 0)
                #     #         # )
                #     #     )
                #     #     # &~(
                #     #     #     # ((df.bbb_diff > 0) & (df.atr_diff > 0) & (df.adx_diff > 0) & ((df.MACD_12_26_9 < 0) & (df.MACDh_12_26_9 < 0) & (df.macd_diff < 0) & (df.macdh_diff < 0))) |
                #     #     #     # (df.bbb_diff > 0) | (df.atr_diff > 0) |
                #     #     #     # (df.minima_peak_x_macd > 0) | (df.minima_peak_x_rsi > 0) | (df.minima_peak_x_macd.shift(1) > 0) | (df.minima_peak_x_rsi.shift(1) > 0)
                #     #     #     (df.minima_peak_x_close > 0) | (df.minima_peak_x_macd > 0) | (df.minima_peak_x_rsi > 0) | (df.minima_peak_x_dmp > 0) | (df.maxima_peak_x_dmn > 0)
                #     #     # )
                #     #     # &~(
                #     #     #     # ((df.bbb_diff > 0) & (df.atr_diff > 0) & (df.adx_diff > 0) & ((df.MACD_12_26_9 > 0) & (df.MACDh_12_26_9 > 0) & (df.macd_diff > 0) & (df.macdh_diff > 0))) |
                #     #     #     # (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_macd.shift(1) < 0) | (df.maxima_peak_x_rsi.shift(1) < 0)
                #     #     #     (df.adx_diff < 0) & (df.atr_diff < 0) #& (df.atr_p < 0.7)
                #     #     # )
                #     #     &~ (
                            
                #     #         (
                #     #             (
                #     #                 ((df.DX_14 > 40) & (df.adx_diff < 0) & (df.DMN_14 > df.DMP_14) & (df.intersect == 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.high_diff > 0)           & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff > 0)
                #     #             )


                #     #         )
                #     #     )                
                #     # , "stg1_short"] = -2 # without atr





                #     # # df.loc[
                #     # #     (
                #     # #         (df.stg15_long.shift(2) > 0)
                #     # #     )

                #     # # , "stg5_long"] = 2 # without atr
                #     # # df.loc[
                #     # #     (
                #     # #         (df.stg15_short.shift(2) < 0)
                #     # #     )

                #     # # , "stg5_short"] = -2 # without atr






























































































































































































                #     df.loc[
                #         # 추세 픽
                #         (
                #             ( # 진입전 추세 재확인 필요, atr14 증가하는지 확인 필요
                #                 # (df['minima_peak_x_rsi'] > 0) #& (df['percentage_change2'] > 3) # & (df['macd_diff_200'] > df['macds_diff_200']) & (df['macd_diff_200'] > 0) & (df['macds_diff_200'] > 0) & (df['macd_diff_700'] > 0) & (df['macds_diff_700'] > 0) & (df['MACD_12_200_9'].shift(2) < df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] > 0) & (df['k_diff'] > 0) & (df['d_diff'] > 0)
                #                 # (df['rsi_percentage_change'] > 5) & (df['atr_percentage_change'] > 4) & (df['macd_percentage_change'] > 50) & (df['obv_percentage_change'] > 100) #& (df['atr_percentage_change'] > 4)
                #                 # & (df['obv_percentage_change'] > 150)
                #                 # (df['obv_percentage_change'].abs() < 50)







                #                 # (((df['ema50_diff'] > 0) & (df['ema200_diff'] > 0)))
                #                 # & (df.EMA_50 > df.EMA_200)
                #                 # & (df.ema50_percentage_change > df.ema200_percentage_change)
                #                 # & ((df.bbb_diff > 0))







                #                 (df['ema50_diff'] > 0)
                #                 # & (df.EMA_50 > df.EMA_200)
                #                 & (df.ema50_percentage_change > df.ema200_percentage_change)
                #                 & (df.bbb_diff > 0)
                #                 & (df['BBP_21_2.0'] > 0.5)

                #                 # & (df.MACD_50_75_35 > 0)
                #                 # & (df.MACDh_50_75_35 > 0)
                #                 # & (df.macdh_diff_35 > 0)
                #                 & (df['lowess_MACD_12_26_9_diff'] > 0)










                #                 & (df['lowess'] > df['close'])
                #                 # & (df['lowess_diff'] > 0)
                #                 & (df['lowess_1'] > df['close'])
                #                 # & (df['lowess_1_diff'] > 0)

                #                 & ((df.dx_200_diff > 0))
                #                 & (df.DX_200 > df.ADX_200)
                #                 # & ((df.dx_diff > 0))
                #                 # # & ((df.adx_diff > 0))
                #                 # & (df.DX_14 > df.ADX_14)

                #                 # & (abs(df['DMP_14'] - df['DMN_14']) > 5)
                #                 # (((df['close_diff'] > 0) & (df['open_diff'] > 0)))
                #                 # & (df.close > df.open)
                #                 # & (df.close_change > df.open_change)


                #                 # (((df['sma7_diff'] > 0) & (df['sma50_diff'] > 0)))
                #                 # & (df.SMA_50 > df.SMA_200)
                #                 # & (df.SMA_7 > df.SMA_50)
                #                 # & (df.sma7_percentage_change > df.sma50_percentage_change)
                #                 # & (df.high_change < df.low_change)
                #                 # & (df.macdh_diff > 0)


                #                 # (((df['close_diff'] > 0) & (df['sma7_diff'] > 0)))
                #                 # & (df.close > df.SMA_7)
                #                 # & (df.close_change > df.sma7_percentage_change)
                                


                #                 # & (df.RSI_14 < 75)
                #                 # & (df.macd_percentage_change > df.macds_percentage_change)
                #                 # & (df.DX_14 < 70)
                #                 # & (df.atr_p_diff > 0)
                #                 # & (df.adx_diff > 0)
                #                 # # & (df.atr_diff < 0)
                #                 # &~ ((df.adx_diff < 0) & (df.atr_diff < 0) & (df.atr_p < 0.5))
                #                 # & (df.atr_p < 0.7)







                #                 # & (((df['sma50_diff'] > df['sma200_diff'])))

                #                 # & (df.adx_diff > 0)

                #                 # & (df.bbb_diff > 0)

                #                 # & (df.atr_diff > 0)
                #                 # & ((df.MACD_12_26_9 > 0) & (df.MACDh_12_26_9 > 0) & (df.macd_diff > 0) & (df.macdh_diff > 0))
                #                 # & (df.obv_diff > 0)


                #                 # (df['minima_peak_x_macd'].shift(2) > 0) & (df['macd_diff_200'] > df['macds_diff_200']) & (df['macd_diff_200'].shift(2) > 0) & (df['macds_diff_200'].shift(2) > 0) & (df['macd_diff_700'].shift(2) > 0) & (df['macds_diff_700'].shift(2) > 0) & (df['MACD_12_200_9'].shift(2) < df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] > 0) & (df['k_diff'] > 0) & (df['d_diff'] > 0)
                #             )
                #             # & 

                #             # (
                #             #     (
                #             #         ((df.dx_diff < 0) & (df.DMN_14 > df.DMP_14) & (df.intersect == 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.high_diff > 0)           & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff > 0)
                #             #     )
                #             #         |
                #             #     (
                #             #         ((df.dx_diff < 0) & (dmn_set_2 > dmp_set_2) & (df.intersect > 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.high_diff > 0)           & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff > 0)
                #             #     )
                #             #         |
                #             #     (
                #             #         (df.dx_diff > 0) & (df.DMP_14 > df.DMN_14) #& (df.atr_p_diff < 0)# & (df.high_diff > 0) & (df.rsi_diff > 0) # & (df.obv_diff > 0) & (df.j_diff > 0) & (df['atr_AMATe_LR_8_21_2'] > 0)
                #             #     )
                #             # )
                #             # & (
                #             #     (df.vwma7_diff > 0)
                #             # )
                #         )
                #         # &~(
                #         #     # ((df.bbb_diff > 0) & (df.atr_diff > 0) & (df.adx_diff > 0) & ((df.MACD_12_26_9 > 0) & (df.MACDh_12_26_9 > 0) & (df.macd_diff > 0) & (df.macdh_diff > 0))) |
                #         #     # (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_macd.shift(1) < 0) | (df.maxima_peak_x_rsi.shift(1) < 0)
                #         #     (df.maxima_peak_x_close < 0) | (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_dmp < 0) | (df.minima_peak_x_dmn < 0)
                #         # )
                #         # &~(
                #         #     # ((df.bbb_diff > 0) & (df.atr_diff > 0) & (df.adx_diff > 0) & ((df.MACD_12_26_9 > 0) & (df.MACDh_12_26_9 > 0) & (df.macd_diff > 0) & (df.macdh_diff > 0))) |
                #         #     # (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_macd.shift(1) < 0) | (df.maxima_peak_x_rsi.shift(1) < 0)
                #         #     (df.adx_diff < 0) & (df.atr_diff < 0) #& (df.atr_p < 0.7)
                #         # )
                #         &~ (
                #             (
                #                 (
                #                     (((df.DX_14 > 40) & (df.adx_diff < 0) & (df.DMP_14 > df.DMN_14) & (df.intersect == 0) )) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.low_diff < 0)          & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff < 0)
                #                     | ((df.maxima_peak_x_close < 0) | (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_dmp < 0) | (df.minima_peak_x_dmn < 0))
                #                 )


                #             )
                #         )
                #     , "stg11_long"] = 2 # without atr

                #     df.loc[
                #         # 추세 픽
                #         (
                #             ( # 진입전 추세 재확인 필요, atr14 증가하는지 확인 필요
                #                 # (df['maxima_peak_x_rsi'] < 0) #& (df['percentage_change2'] < -3) # & (df['macd_diff_200'] < df['macds_diff_200']) & (df['macd_diff_200'] < 0) & (df['macds_diff_200'] < 0) & (df['macd_diff_700'] < 0) & (df['macds_diff_700'] < 0) & (df['MACD_12_200_9'].shift(2) > df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] < 0) & (df['k_diff'] < 0) & (df['d_diff'] < 0)
                #                 # (df['rsi_percentage_change'] < -5) & (df['atr_percentage_change'] > 4) & (df['macd_percentage_change'] < -50) & (df['obv_percentage_change'] < -100) #& (df['atr_percentage_change'] > 4)
                #                 # (df['obv_percentage_change'] < -150)
                #                 # (df['obv_percentage_change'].abs() < 50)
                #                 # & (((df['sma50_diff'] > 0) & (df['sma200_diff'] > 0)))














                #                 # (((df['ema50_diff'] < 0) & (df['ema200_diff'] < 0)))
                #                 # & (df.EMA_50 < df.EMA_200)
                #                 # & (df.ema50_percentage_change < df.ema200_percentage_change)
                #                 # & ((df.bbb_diff > 0))






                #                 (df['ema50_diff'] < 0)
                #                 # & (df.EMA_50 < df.EMA_200)
                #                 & (df.ema50_percentage_change < df.ema200_percentage_change)
                #                 & (df.bbb_diff > 0)
                #                 & (df['BBP_21_2.0'] < 0.5)


                #                 # & (df.MACD_50_75_35 < 0)
                #                 # & (df.MACDh_50_75_35 < 0)
                #                 # & (df.macdh_diff_35 < 0)
                #                 & (df['lowess_MACD_12_26_9_diff'] < 0)





                #                 & (df['lowess'] < df['close'])
                #                 # & (df['lowess_diff'] < 0)
                #                 & (df['lowess_1'] < df['close'])
                #                 # & (df['lowess_1_diff'] < 0)

                #                 & ((df.dx_200_diff > 0))
                #                 & (df.DX_200 > df.ADX_200)

                #                 # & (df.dx_diff > 0)
                #                 # # & ((df.adx_diff > 0))
                #                 # & (df.DX_14 > df.ADX_14)
                                
                #                 # & (abs(df['DMP_14'] - df['DMN_14']) > 5)

                #                 # (((df['close_diff'] < 0) & (df['open_diff'] < 0)))
                #                 # & (df.close < df.open)
                #                 # & (df.close_change < df.open_change)


                #                 # (((df['sma7_diff'] < 0) & (df['sma50_diff'] < 0)))
                #                 # & (df.SMA_50 < df.SMA_200)
                #                 # & (df.SMA_7 < df.SMA_50)
                #                 # & (df.sma7_percentage_change < df.sma50_percentage_change)
                #                 # & (df.high_change < df.low_change)
                #                 # & (df.macdh_diff < 0)
                                




                #                 # (((df['close_diff'] < 0) & (df['sma7_diff'] < 0)))
                #                 # & (df.close < df.SMA_7)
                #                 # & (df.close_change < df.sma7_percentage_change)
                                


                #                 # & (df.RSI_14 > 25)
                #                 # & (df.macd_percentage_change < df.macds_percentage_change)
                #                 # & (df.DX_14 < 70)
                #                 # & (df.atr_p_diff > 0)
                #                 # & (df.adx_diff > 0)
                #                 # # & (df.atr_diff < 0)
                #                 # &~ ((df.adx_diff < 0) & (df.atr_diff < 0) & (df.atr_p < 0.5))                       
                #                 # & (df.atr_p > 1.2)

                #                 # & (((df['sma50_diff'] < df['sma200_diff'])))
                #                 # & (df.adx_diff > 0)

                #                 # & (df.bbb_diff > 0)
                                
                #                 # & (df.atr_diff > 0)
                #                 # & ((df.MACD_12_26_9 < 0) & (df.MACDh_12_26_9 < 0) & (df.macd_diff < 0) & (df.macdh_diff < 0))
                #                 # & (df.obv_diff < 0)
                #                 # & (df.atr_p < 0.7)

                                
                #                 # (df['maxima_peak_x_macd'].shift(2) < 0) & (df['macd_diff_200'] < df['macds_diff_200']) & (df['macd_diff_200'].shift(2) < 0) & (df['macds_diff_200'].shift(2) < 0) & (df['macd_diff_700'].shift(2) < 0) & (df['macds_diff_700'].shift(2) < 0) & (df['MACD_12_200_9'].shift(2) > df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] < 0) & (df['k_diff'] < 0) & (df['d_diff'] < 0)
                #             )
                #             # & 

                #             # (
                #             #     (
                #             #         ((df.dx_diff < 0) & (df.DMP_14 > df.DMN_14) & (df.intersect == 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.low_diff < 0)          & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff < 0)
                #             #     )
                #             #         |
                #             #     (
                #             #         ((df.dx_diff < 0) & (dmp_set_2 > dmn_set_2) & (df.intersect > 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.low_diff < 0)          & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff < 0)
                #             #     )
                #             #         |
                #             #     (
                #             #         (df.dx_diff > 0) & (df.DMN_14 > df.DMP_14) #& (df.atr_p_diff < 0)#& (df.intersect == 0) & (df.low_diff < 0)  # & (df.obv_diff < 0) & (df.j_diff < 0) & (df['atr_AMATe_LR_8_21_2'] > 0)
                #             #     )
                #             # )
                #             # & (
                #             #     (df.vwma7_diff < 0)
                #             # )
                #         )
                #         # &~(
                #         #     # ((df.bbb_diff > 0) & (df.atr_diff > 0) & (df.adx_diff > 0) & ((df.MACD_12_26_9 < 0) & (df.MACDh_12_26_9 < 0) & (df.macd_diff < 0) & (df.macdh_diff < 0))) |
                #         #     # (df.bbb_diff > 0) | (df.atr_diff > 0) |
                #         #     # (df.minima_peak_x_macd > 0) | (df.minima_peak_x_rsi > 0) | (df.minima_peak_x_macd.shift(1) > 0) | (df.minima_peak_x_rsi.shift(1) > 0)
                #         #     (df.minima_peak_x_close > 0) | (df.minima_peak_x_macd > 0) | (df.minima_peak_x_rsi > 0) | (df.minima_peak_x_dmp > 0) | (df.maxima_peak_x_dmn > 0)
                #         # )
                #         # &~(
                #         #     # ((df.bbb_diff > 0) & (df.atr_diff > 0) & (df.adx_diff > 0) & ((df.MACD_12_26_9 > 0) & (df.MACDh_12_26_9 > 0) & (df.macd_diff > 0) & (df.macdh_diff > 0))) |
                #         #     # (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_macd.shift(1) < 0) | (df.maxima_peak_x_rsi.shift(1) < 0)
                #         #     (df.adx_diff < 0) & (df.atr_diff < 0) #& (df.atr_p < 0.7)
                #         # )
                #         &~ (
                            
                #             (
                #                 (
                #                     (((df.DX_14 > 40) & (df.adx_diff < 0) & (df.DMN_14 > df.DMP_14) & (df.intersect == 0) )) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.high_diff > 0)           & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff > 0)
                #                     | ((df.minima_peak_x_close > 0) | (df.minima_peak_x_macd > 0) | (df.minima_peak_x_rsi > 0) | (df.minima_peak_x_dmp > 0) | (df.maxima_peak_x_dmn > 0))
                #                 )


                #             )
                #         )
                #     , "stg11_short"] = -2 # without atr



                #     df.loc[
                #         # 추세 픽
                #         (
                #             ( 
                                
                #                 (df['lowess'] > df['close'])
                #                 & (df['lowess_diff'] > 0)
                #                 & (df['lowess_1'] > df['close'])
                #                 # & (df['lowess_1_diff'] > 0)
                #                 & (df['lowess_MACD_12_26_9_diff'] > 0)

                #                 # & (abs(df['DMP_14'] - df['DMN_14']) > 5)

                #                 # & ((df.dx_diff > 0))
                #                 # # & ((df.adx_diff > 0))
                #                 # & (df.DX_14 > df.ADX_14)

                #                 # & (df['EMA_200'] < df['EMA_50'])
                #                 & (

                #                 ((df.DX_200.shift(1) < df.ADX_200.shift(1))
                #                 & (df.DX_200 > df.ADX_200)
                #                 & (df.DMP_200 > df.DMN_200)
                #                 & ((df.dx_200_diff > 0)))


                #                 |

                #                 ((df.DX_200.shift(1) > df.ADX_200.shift(1))
                #                 & (df.DX_200 < df.ADX_200)
                #                 & (df.DMP_200 < df.DMN_200)
                #                 & ((df.dx_200_diff < 0)))
                #                 & (abs(df['DX_200'] - df['ADX_200']) > 1)

                #                 )

                #             )

                #         )

                #         # &~ (
                #         #     (
                #         #         (
                #         #             (((df.DX_14 > 40) & (df.adx_diff < 0) & (df.DMP_14 > df.DMN_14) & (df.intersect == 0) )) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.low_diff < 0)          & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff < 0)
                #         #             | 
                #         #             ((df.maxima_peak_x_close < 0) | (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_dmp < 0) | (df.minima_peak_x_dmn < 0))
                #         #         )


                #         #     )
                #         # )
                #     , "stg55_long"] = 2 # without atr

                #     df.loc[
                #         # 추세 픽
                #         (
                #             (

                #                 (df['lowess'] < df['close'])
                #                 & (df['lowess_diff'] < 0)
                #                 & (df['lowess_1'] < df['close'])
                #                 # & (df['lowess_1_diff'] < 0)
                #                 & (df['lowess_MACD_12_26_9_diff'] < 0)

                #                 # & (abs(df['DMP_14'] - df['DMN_14']) > 5)

                #                 # & (df.dx_diff > 0)
                #                 # # & ((df.adx_diff > 0))
                #                 # & (df.DX_14 > df.ADX_14)

                #                 # & (df['EMA_200'] > df['EMA_50'])
                #                 & (

                #                 ((df.DX_200.shift(1) < df.ADX_200.shift(1))
                #                 & (df.DX_200 > df.ADX_200)
                #                 & (df.DMP_200 < df.DMN_200)
                #                 & ((df.dx_200_diff > 0)))


                #                 |

                #                 ((df.DX_200.shift(1) > df.ADX_200.shift(1))
                #                 & (df.DX_200 < df.ADX_200)
                #                 & (df.DMP_200 > df.DMN_200)
                #                 & ((df.dx_200_diff < 0)))
                #                 & (abs(df['DX_200'] - df['ADX_200']) > 1)

                #                 )


                #             )
                #         )

                #         # &~ (
                            
                #         #     (
                #         #         (
                #         #             (((df.DX_14 > 40) & (df.adx_diff < 0) & (df.DMN_14 > df.DMP_14) & (df.intersect == 0) )) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.high_diff > 0)           & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff > 0)
                #         #             | 
                #         #             ((df.minima_peak_x_close > 0) | (df.minima_peak_x_macd > 0) | (df.minima_peak_x_rsi > 0) | (df.minima_peak_x_dmp > 0) | (df.maxima_peak_x_dmn > 0))
                #         #         )


                #         #     )
                #         # )                
                #     , "stg55_short"] = -2 # without atr



                #     # df.loc[
                #     #     (
                #     #         (df.stg15_long.shift(2) > 0)
                #     #     )

                #     # , "stg5_long"] = 2 # without atr
                #     # df.loc[
                #     #     (
                #     #         (df.stg15_short.shift(2) < 0)
                #     #     )

                #     # , "stg5_short"] = -2 # without atr












                #     df.loc[
                #         # 추세 픽
                #         (
                #             ( # 진입전 추세 재확인 필요, atr14 증가하는지 확인 필요
                #                 # (df['minima_peak_x_rsi'] > 0) #& (df['percentage_change2'] > 3) # & (df['macd_diff_200'] > df['macds_diff_200']) & (df['macd_diff_200'] > 0) & (df['macds_diff_200'] > 0) & (df['macd_diff_700'] > 0) & (df['macds_diff_700'] > 0) & (df['MACD_12_200_9'].shift(2) < df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] > 0) & (df['k_diff'] > 0) & (df['d_diff'] > 0)
                #                 # (df['rsi_percentage_change'] > 5) & (df['atr_percentage_change'] > 4) & (df['macd_percentage_change'] > 50) & (df['obv_percentage_change'] > 100) #& (df['atr_percentage_change'] > 4)
                #                 # & (df['obv_percentage_change'] > 150)
                #                 # (df['obv_percentage_change'].abs() < 50)

                #                 (df.wma7_diff > 0)

                #                 & (df['sma200_diff'] < 0)
                #                 & (df.SMA_50 < df.SMA_200)
                #                 # & (df['macdh_diff'] > 0)
                #                 & (df['rsi_diff'] > 0)
                #                 # & (df['close_diff'] > 0)
                #                 # & (df.RSI_14 < 45)
                #                 & (df.obv_diff > 0)
                #                 # & (df.bbl_diff > 0)
                #                 & (df.bbb_diff < 0)
                #                 & (df.atr_p_diff > 0)

                #                 # & (df.MACDh_12_26_9 < 0)




                #                 # & (df.sma50_percentage_change > df.sma200_percentage_change)
                #                 # & (df.macd_percentage_change > df.macds_percentage_change)


                #                 # # (( (df['sma50_diff'] < 0))) # | 
                #                 # & ((((df['sma7_diff'] < 0) & (df['sma50_diff'] < 0))) | (((df['sma7_diff'] > 0) & (df['sma50_diff'] < 0))))
                #                 # & (df.SMA_7 < df.SMA_50)
                #                 # # & (df.SMA_50 < df.SMA_200)
                #                 # & (df.sma7_percentage_change > df.sma50_percentage_change)
                #                 # # & (df.macd_percentage_change > df.macds_percentage_change)
                #                 # & (df.high_change < df.low_change)









                                
                #                 # & (df.DX_14 > 40)
                #                 # &~ ((df.adx_diff < 0) & (df.atr_diff < 0) & (df.atr_p < 0.5))

                #                 # & (((df['sma50_diff'] > df['sma200_diff'])))
                #                 # & (df.adx_diff > 0)

                #                 # & (df.bbb_diff > 0)
                                
                #                 # & (df.atr_diff > 0)
                #                 # & ((df.MACD_12_26_9 > 0) & (df.MACDh_12_26_9 > 0) & (df.macd_diff > 0) & (df.macdh_diff > 0))
                                
                #                 # & (df.atr_p < 0.7)


                #                 # (df['minima_peak_x_macd'].shift(2) > 0) & (df['macd_diff_200'] > df['macds_diff_200']) & (df['macd_diff_200'].shift(2) > 0) & (df['macds_diff_200'].shift(2) > 0) & (df['macd_diff_700'].shift(2) > 0) & (df['macds_diff_700'].shift(2) > 0) & (df['MACD_12_200_9'].shift(2) < df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] > 0) & (df['k_diff'] > 0) & (df['d_diff'] > 0)
                #             )


                #             & 

                #             (
                #                 (
                #                     ((df.dx_diff < 0) & (df.DMN_14 > df.DMP_14) & (df.intersect == 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.high_diff > 0)           & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff > 0)
                #                 )
                #                     |
                #                 (
                #                     ((df.dx_diff < 0) & (dmn_set_2 > dmp_set_2) & (df.intersect > 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.high_diff > 0)           & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff > 0)
                #                 )
                #                     |
                #                 (
                #                     (df.dx_diff > 0) & (df.DMP_14 > df.DMN_14) #& (df.atr_p_diff < 0)# & (df.high_diff > 0) & (df.rsi_diff > 0) # & (df.obv_diff > 0) & (df.j_diff > 0) & (df['atr_AMATe_LR_8_21_2'] > 0)
                #                 )
                #             )
                            
                #         )
                #         &~(
                #             # ((df.bbb_diff > 0) & (df.atr_diff > 0) & (df.adx_diff > 0) & ((df.MACD_12_26_9 > 0) & (df.MACDh_12_26_9 > 0) & (df.macd_diff > 0) & (df.macdh_diff > 0))) |
                #             # (df.bbb_diff > 0) | (df.atr_diff > 0) |
                #             # (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_macd.shift(1) < 0) | (df.maxima_peak_x_rsi.shift(1) < 0)
                #             (df.maxima_peak_x_close < 0) | (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_dmp < 0) | (df.minima_peak_x_dmn < 0)
                #         )
                #         # &~(
                #         #     # ((df.bbb_diff > 0) & (df.atr_diff > 0) & (df.adx_diff > 0) & ((df.MACD_12_26_9 > 0) & (df.MACDh_12_26_9 > 0) & (df.macd_diff > 0) & (df.macdh_diff > 0))) |
                #         #     # (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_macd.shift(1) < 0) | (df.maxima_peak_x_rsi.shift(1) < 0)
                #         #     (df.adx_diff < 0) & (df.atr_diff < 0) #& (df.atr_p < 0.7)
                #         # )
                #         # &~ (
                #         #     (
                #         #         (
                #         #             ((df.DX_14 > 40) & (df.adx_diff < 0) & (df.DMP_14 > df.DMN_14) & (df.intersect == 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.low_diff < 0)          & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff < 0)
                #         #         )


                #         #     )
                #         # )                
                #     , "stg12_long"] = 2 # without atr

                #     df.loc[
                #         # 추세 픽
                #         (
                #             ( # 진입전 추세 재확인 필요, atr14 증가하는지 확인 필요
                #                 # (df['maxima_peak_x_rsi'] < 0) #& (df['percentage_change2'] < -3) # & (df['macd_diff_200'] < df['macds_diff_200']) & (df['macd_diff_200'] < 0) & (df['macds_diff_200'] < 0) & (df['macd_diff_700'] < 0) & (df['macds_diff_700'] < 0) & (df['MACD_12_200_9'].shift(2) > df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] < 0) & (df['k_diff'] < 0) & (df['d_diff'] < 0)
                #                 # (df['rsi_percentage_change'] < -5) & (df['atr_percentage_change'] > 4) & (df['macd_percentage_change'] < -50) & (df['obv_percentage_change'] < -100) #& (df['atr_percentage_change'] > 4)
                #                 # (df['obv_percentage_change'] < -150)
                #                 # (df['obv_percentage_change'].abs() < 50)
                #                 # & (((df['sma50_diff'] > 0) & (df['sma200_diff'] > 0)))

                #                 (df.wma7_diff < 0)

                #                 & (df['sma200_diff'] > 0)
                #                 & (df.SMA_50 > df.SMA_200)
                #                 # & (df['macdh_diff'] < 0)
                #                 & (df['rsi_diff'] < 0)
                #                 # & (df['close_diff'] < 0)
                #                 # & (df.RSI_14 > 55)
                #                 & (df.obv_diff < 0)
                #                 # & (df.bbu_diff < 0)
                #                 & (df.bbb_diff < 0)
                #                 & (df.atr_p_diff > 0)

                #                 # & (df.MACDh_12_26_9 > 0)

                #                 # & (df.sma50_percentage_change < df.sma200_percentage_change)
                #                 # & (df.macd_percentage_change < df.macds_percentage_change)






                #                 # & ((((df['sma7_diff'] > 0) & (df['sma50_diff'] > 0))) | (((df['sma7_diff'] < 0) & (df['sma50_diff'] > 0))))
                #                 # & (df.SMA_7 > df.SMA_50)
                #                 # # & (df.SMA_50 > df.SMA_200)
                #                 # & (df.sma7_percentage_change < df.sma50_percentage_change)
                #                 # # & (df.macd_percentage_change < df.macds_percentage_change)
                #                 # & (df.high_change < df.low_change)









                                
                #                 # & (df.DX_14 > 40)
                #                 # &~ ((df.adx_diff < 0) & (df.atr_diff < 0) & (df.atr_p < 0.5))
                #                 # &~ (df.atr_diff < 0)

                #                 # & (((df['sma50_diff'] < df['sma200_diff'])))
                #                 # &~ (df.adx_diff > 0)

                #                 # & (df.bbb_diff > 0)
                                
                #                 # & (df.atr_diff > 0)
                #                 # & ((df.MACD_12_26_9 < 0) & (df.MACDh_12_26_9 < 0) & (df.macd_diff < 0) & (df.macdh_diff < 0))
                                
                #                 # & (df.atr_p < 0.7)

                                
                #                 # (df['maxima_peak_x_macd'].shift(2) < 0) & (df['macd_diff_200'] < df['macds_diff_200']) & (df['macd_diff_200'].shift(2) < 0) & (df['macds_diff_200'].shift(2) < 0) & (df['macd_diff_700'].shift(2) < 0) & (df['macds_diff_700'].shift(2) < 0) & (df['MACD_12_200_9'].shift(2) > df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] < 0) & (df['k_diff'] < 0) & (df['d_diff'] < 0)
                #             )


                #             & 

                #             (
                #                 (
                #                     ((df.dx_diff < 0) & (df.DMP_14 > df.DMN_14) & (df.intersect == 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.low_diff < 0)          & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff < 0)
                #                 )
                #                     |
                #                 (
                #                     ((df.dx_diff < 0) & (dmp_set_2 > dmn_set_2) & (df.intersect > 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.low_diff < 0)          & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff < 0)
                #                 )
                #                     |
                #                 (
                #                     (df.dx_diff > 0) & (df.DMN_14 > df.DMP_14) #& (df.atr_p_diff < 0)#& (df.intersect == 0) & (df.low_diff < 0)  # & (df.obv_diff < 0) & (df.j_diff < 0) & (df['atr_AMATe_LR_8_21_2'] > 0)
                #                 )
                #             )
                            
                #         )
                #         &~(
                #             # ((df.bbb_diff > 0) & (df.atr_diff > 0) & (df.adx_diff > 0) & ((df.MACD_12_26_9 < 0) & (df.MACDh_12_26_9 < 0) & (df.macd_diff < 0) & (df.macdh_diff < 0))) |
                #             # (df.bbb_diff > 0) | (df.atr_diff > 0) |
                #             # (df.minima_peak_x_macd > 0) | (df.minima_peak_x_rsi > 0) | (df.minima_peak_x_macd.shift(1) > 0) | (df.minima_peak_x_rsi.shift(1) > 0)
                #             (df.minima_peak_x_close > 0) | (df.minima_peak_x_macd > 0) | (df.minima_peak_x_rsi > 0) | (df.minima_peak_x_dmp > 0) | (df.maxima_peak_x_dmn > 0)
                #         )
                #         # &~(
                #         #     # ((df.bbb_diff > 0) & (df.atr_diff > 0) & (df.adx_diff > 0) & ((df.MACD_12_26_9 > 0) & (df.MACDh_12_26_9 > 0) & (df.macd_diff > 0) & (df.macdh_diff > 0))) |
                #         #     # (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_macd.shift(1) < 0) | (df.maxima_peak_x_rsi.shift(1) < 0)
                #         #     (df.adx_diff < 0) & (df.atr_diff < 0) #& (df.atr_p < 0.7)
                #         # )
                #         # &~ (
                            
                #         #     (
                #         #         (
                #         #             ((df.DX_14 > 40) & (df.adx_diff < 0) & (df.DMN_14 > df.DMP_14) & (df.intersect == 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.high_diff > 0)           & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff > 0)
                #         #         )


                #         #     )
                #         # )
                #     , "stg12_short"] = -2 # without atr










                #     df.loc[
                #         # 추세 픽
                #         (

                #             (
                #                 (
                #                     ((df.dx_diff < 0) & (df.DMN_14 > df.DMP_14) & (df.intersect == 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.high_diff > 0)           & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff > 0)
                #                 )
                #                     |
                #                 (
                #                     ((df.dx_diff < 0) & (dmn_set_2 > dmp_set_2) & (df.intersect > 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.high_diff > 0)           & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff > 0)
                #                 )
                #                     |
                #                 (
                #                     (df.dx_diff > 0) & (df.DMP_14 > df.DMN_14) #& (df.atr_p_diff < 0)# & (df.high_diff > 0) & (df.rsi_diff > 0) # & (df.obv_diff > 0) & (df.j_diff > 0) & (df['atr_AMATe_LR_8_21_2'] > 0)
                #                 )
                #             )
                #             & 
                #             (
                #                 (df.RSI_14 > 50) & (df.RSI_14 < 80)
                #             )
                #         )
                #         &~(
                #             (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_macd.shift(1) < 0) | (df.maxima_peak_x_rsi.shift(1) < 0)
                #         )
                #     , "stg13_long"] = 2 # without atr

                #     df.loc[
                #         # 추세 픽
                #         (

                #             (
                #                 (
                #                     ((df.dx_diff < 0) & (df.DMP_14 > df.DMN_14) & (df.intersect == 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.low_diff < 0)          & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff < 0)
                #                 )
                #                     |
                #                 (
                #                     ((df.dx_diff < 0) & (dmp_set_2 > dmn_set_2) & (df.intersect > 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.low_diff < 0)          & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff < 0)
                #                 )
                #                     |
                #                 (
                #                     (df.dx_diff > 0) & (df.DMN_14 > df.DMP_14) #& (df.atr_p_diff < 0)#& (df.intersect == 0) & (df.low_diff < 0)  # & (df.obv_diff < 0) & (df.j_diff < 0) & (df['atr_AMATe_LR_8_21_2'] > 0)
                #                 )
                #             )
                #             & 
                #             (
                #                 (df.RSI_14 < 50) & (df.RSI_14 > 20)
                #             )
                #         )
                #         &~(
                #             (df.minima_peak_x_macd > 0) | (df.minima_peak_x_rsi > 0) | (df.minima_peak_x_macd.shift(1) > 0) | (df.minima_peak_x_rsi.shift(1) > 0)
                #         )
                #     , "stg13_short"] = -2 # without atr





                #     df.loc[
                #         # 추세 픽
                #         (
                #             ( # 진입전 추세 재확인 필요, atr14 증가하는지 확인 필요
                #                 # (df['minima_peak_x_rsi'] > 0) #& (df['percentage_change2'] > 3) # & (df['macd_diff_200'] > df['macds_diff_200']) & (df['macd_diff_200'] > 0) & (df['macds_diff_200'] > 0) & (df['macd_diff_700'] > 0) & (df['macds_diff_700'] > 0) & (df['MACD_12_200_9'].shift(2) < df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] > 0) & (df['k_diff'] > 0) & (df['d_diff'] > 0)
                #                 # (df['rsi_percentage_change'] > 5) & (df['atr_percentage_change'] > 4) & (df['macd_percentage_change'] > 50) & (df['obv_percentage_change'] > 100) #& (df['atr_percentage_change'] > 4)
                #                 # & (df['obv_percentage_change'] > 150)
                #                 # (df['obv_percentage_change'].abs() < 50)
                #                 (df.EMA_7 > df.EMA_21)
                #                 & (((df['ema7_diff'] > 0) & (df['ema21_diff'] > 0)))
                                
                #                 & (df.ema7_percentage_change > df.ema21_percentage_change)
                #                 & ((df.bbb_diff > 0))

                #                 # & (df.close > df.close.shift(1)) & (df.close.shift(-1) > df.close)



                #                 # (((df['close_diff'] > 0) & (df['open_diff'] > 0)))
                #                 # & (df.close > df.open)
                #                 # & (df.close_change > df.open_change)


                #                 # (((df['sma7_diff'] > 0) & (df['sma50_diff'] > 0)))
                #                 # & (df.SMA_50 > df.SMA_200)
                #                 # & (df.SMA_7 > df.SMA_50)
                #                 # & (df.sma7_percentage_change > df.sma50_percentage_change)
                #                 # & (df.high_change < df.low_change)
                #                 # & (df.macdh_diff > 0)


                #                 # (((df['close_diff'] > 0) & (df['sma7_diff'] > 0)))
                #                 # & (df.close > df.SMA_7)
                #                 # & (df.close_change > df.sma7_percentage_change)
                                


                #                 # & (df.RSI_14 < 75)
                #                 # & (df.macd_percentage_change > df.macds_percentage_change)
                #                 # & (df.DX_14 < 70)
                #                 # # & (df.atr_p_diff > 0)
                #                 # & (df.adx_diff > 0)
                #                 # # & (df.atr_diff < 0)
                #                 # &~ ((df.adx_diff < 0) & (df.atr_diff < 0) & (df.atr_p < 0.5))
                #                 # & (df.atr_p < 0.7)







                #                 # & (((df['sma50_diff'] > df['sma200_diff'])))

                #                 # & (df.adx_diff > 0)

                #                 # & (df.bbb_diff > 0)

                #                 # & (df.atr_diff > 0)
                #                 # & ((df.MACD_12_26_9 > 0) & (df.MACDh_12_26_9 > 0) & (df.macd_diff > 0) & (df.macdh_diff > 0))
                #                 # & (df.obv_diff > 0)


                #                 # (df['minima_peak_x_macd'].shift(2) > 0) & (df['macd_diff_200'] > df['macds_diff_200']) & (df['macd_diff_200'].shift(2) > 0) & (df['macds_diff_200'].shift(2) > 0) & (df['macd_diff_700'].shift(2) > 0) & (df['macds_diff_700'].shift(2) > 0) & (df['MACD_12_200_9'].shift(2) < df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] > 0) & (df['k_diff'] > 0) & (df['d_diff'] > 0)
                #             )
                #             & 

                #             (
                #                 (
                #                     ((df.dx_diff < 0) & (df.DMN_14 > df.DMP_14) & (df.intersect == 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.high_diff > 0)           & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff > 0)
                #                 )
                #                     |
                #                 (
                #                     ((df.dx_diff < 0) & (dmn_set_2 > dmp_set_2) & (df.intersect > 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.high_diff > 0)           & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff > 0)
                #                 )
                #                     |
                #                 (
                #                     (df.dx_diff > 0) & (df.DMP_14 > df.DMN_14) #& (df.atr_p_diff < 0)# & (df.high_diff > 0) & (df.rsi_diff > 0) # & (df.obv_diff > 0) & (df.j_diff > 0) & (df['atr_AMATe_LR_8_21_2'] > 0)
                #                 )
                #             )
                #             # & (
                #             #     (df.vwma7_diff > 0)
                #             # )
                #         )
                #         # &~(
                #         #     # ((df.bbb_diff > 0) & (df.atr_diff > 0) & (df.adx_diff > 0) & ((df.MACD_12_26_9 > 0) & (df.MACDh_12_26_9 > 0) & (df.macd_diff > 0) & (df.macdh_diff > 0))) |
                #         #     # (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_macd.shift(1) < 0) | (df.maxima_peak_x_rsi.shift(1) < 0)
                #         #     (df.maxima_peak_x_close < 0) | (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_dmp < 0) | (df.minima_peak_x_dmn < 0)
                #         # )
                #         # &~(
                #         #     # ((df.bbb_diff > 0) & (df.atr_diff > 0) & (df.adx_diff > 0) & ((df.MACD_12_26_9 > 0) & (df.MACDh_12_26_9 > 0) & (df.macd_diff > 0) & (df.macdh_diff > 0))) |
                #         #     # (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_macd.shift(1) < 0) | (df.maxima_peak_x_rsi.shift(1) < 0)
                #         #     (df.adx_diff < 0) & (df.atr_diff < 0) #& (df.atr_p < 0.7)
                #         # )
                #         &~ (
                #             (
                #                 (
                #                     (((df.DX_14 > 40) & (df.adx_diff < 0) & (df.DMP_14 > df.DMN_14) & (df.intersect == 0) )) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.low_diff < 0)          & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff < 0)
                #                     | ((df.maxima_peak_x_close < 0) | (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_dmp < 0) | (df.minima_peak_x_dmn < 0))
                #                 )


                #             )
                #         )
                #     , "stg14_long"] = 2 # without atr

                #     df.loc[
                #         # 추세 픽
                #         (
                #             ( # 진입전 추세 재확인 필요, atr14 증가하는지 확인 필요
                #                 # (df['maxima_peak_x_rsi'] < 0) #& (df['percentage_change2'] < -3) # & (df['macd_diff_200'] < df['macds_diff_200']) & (df['macd_diff_200'] < 0) & (df['macds_diff_200'] < 0) & (df['macd_diff_700'] < 0) & (df['macds_diff_700'] < 0) & (df['MACD_12_200_9'].shift(2) > df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] < 0) & (df['k_diff'] < 0) & (df['d_diff'] < 0)
                #                 # (df['rsi_percentage_change'] < -5) & (df['atr_percentage_change'] > 4) & (df['macd_percentage_change'] < -50) & (df['obv_percentage_change'] < -100) #& (df['atr_percentage_change'] > 4)
                #                 # (df['obv_percentage_change'] < -150)
                #                 # (df['obv_percentage_change'].abs() < 50)
                #                 # & (((df['sma50_diff'] > 0) & (df['sma200_diff'] > 0)))
                #                 (df.EMA_7 < df.EMA_21)
                #                 & (((df['ema7_diff'] < 0) & (df['ema21_diff'] < 0)))
                #                 & (df.ema7_percentage_change < df.ema21_percentage_change)
                #                 & ((df.bbb_diff > 0))

                #                 # & (df.close < df.close.shift(1)) & (df.close.shift(-1) < df.close)


                #                 # (((df['close_diff'] < 0) & (df['open_diff'] < 0)))
                #                 # & (df.close < df.open)
                #                 # & (df.close_change < df.open_change)


                #                 # (((df['sma7_diff'] < 0) & (df['sma50_diff'] < 0)))
                #                 # & (df.SMA_50 < df.SMA_200)
                #                 # & (df.SMA_7 < df.SMA_50)
                #                 # & (df.sma7_percentage_change < df.sma50_percentage_change)
                #                 # & (df.high_change < df.low_change)
                #                 # & (df.macdh_diff < 0)
                                




                #                 # (((df['close_diff'] < 0) & (df['sma7_diff'] < 0)))
                #                 # & (df.close < df.SMA_7)
                #                 # & (df.close_change < df.sma7_percentage_change)
                                


                #                 # & (df.RSI_14 > 25)
                #                 # & (df.macd_percentage_change < df.macds_percentage_change)
                #                 # & (df.DX_14 < 70)
                #                 # & (df.atr_p_diff > 0)
                #                 # & (df.adx_diff > 0)
                #                 # # & (df.atr_diff < 0)
                #                 # &~ ((df.adx_diff < 0) & (df.atr_diff < 0) & (df.atr_p < 0.5))                       
                #                 # & (df.atr_p > 1.2)

                #                 # & (((df['sma50_diff'] < df['sma200_diff'])))
                #                 # & (df.adx_diff > 0)

                #                 # & (df.bbb_diff > 0)
                                
                #                 # & (df.atr_diff > 0)
                #                 # & ((df.MACD_12_26_9 < 0) & (df.MACDh_12_26_9 < 0) & (df.macd_diff < 0) & (df.macdh_diff < 0))
                #                 # & (df.obv_diff < 0)
                #                 # & (df.atr_p < 0.7)

                                
                #                 # (df['maxima_peak_x_macd'].shift(2) < 0) & (df['macd_diff_200'] < df['macds_diff_200']) & (df['macd_diff_200'].shift(2) < 0) & (df['macds_diff_200'].shift(2) < 0) & (df['macd_diff_700'].shift(2) < 0) & (df['macds_diff_700'].shift(2) < 0) & (df['MACD_12_200_9'].shift(2) > df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] < 0) & (df['k_diff'] < 0) & (df['d_diff'] < 0)
                #             )
                #             & 

                #             (
                #                 (
                #                     ((df.dx_diff < 0) & (df.DMP_14 > df.DMN_14) & (df.intersect == 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.low_diff < 0)          & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff < 0)
                #                 )
                #                     |
                #                 (
                #                     ((df.dx_diff < 0) & (dmp_set_2 > dmn_set_2) & (df.intersect > 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.low_diff < 0)          & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff < 0)
                #                 )
                #                     |
                #                 (
                #                     (df.dx_diff > 0) & (df.DMN_14 > df.DMP_14) #& (df.atr_p_diff < 0)#& (df.intersect == 0) & (df.low_diff < 0)  # & (df.obv_diff < 0) & (df.j_diff < 0) & (df['atr_AMATe_LR_8_21_2'] > 0)
                #                 )
                #             )
                #             # & (
                #             #     (df.vwma7_diff < 0)
                #             # )
                #         )
                #         # &~(
                #         #     # ((df.bbb_diff > 0) & (df.atr_diff > 0) & (df.adx_diff > 0) & ((df.MACD_12_26_9 < 0) & (df.MACDh_12_26_9 < 0) & (df.macd_diff < 0) & (df.macdh_diff < 0))) |
                #         #     # (df.bbb_diff > 0) | (df.atr_diff > 0) |
                #         #     # (df.minima_peak_x_macd > 0) | (df.minima_peak_x_rsi > 0) | (df.minima_peak_x_macd.shift(1) > 0) | (df.minima_peak_x_rsi.shift(1) > 0)
                #         #     (df.minima_peak_x_close > 0) | (df.minima_peak_x_macd > 0) | (df.minima_peak_x_rsi > 0) | (df.minima_peak_x_dmp > 0) | (df.maxima_peak_x_dmn > 0)
                #         # )
                #         # &~(
                #         #     # ((df.bbb_diff > 0) & (df.atr_diff > 0) & (df.adx_diff > 0) & ((df.MACD_12_26_9 > 0) & (df.MACDh_12_26_9 > 0) & (df.macd_diff > 0) & (df.macdh_diff > 0))) |
                #         #     # (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_macd.shift(1) < 0) | (df.maxima_peak_x_rsi.shift(1) < 0)
                #         #     (df.adx_diff < 0) & (df.atr_diff < 0) #& (df.atr_p < 0.7)
                #         # )
                #         &~ (
                            
                #             (
                #                 (
                #                     (((df.DX_14 > 40) & (df.adx_diff < 0) & (df.DMN_14 > df.DMP_14) & (df.intersect == 0) )) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.high_diff > 0)           & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff > 0)
                #                     | ((df.minima_peak_x_close > 0) | (df.minima_peak_x_macd > 0) | (df.minima_peak_x_rsi > 0) | (df.minima_peak_x_dmp > 0) | (df.maxima_peak_x_dmn > 0))
                #                 )


                #             )
                #         )                
                #     , "stg14_short"] = -2 # without atr





                #     df.loc[
                #         # 추세 픽
                #             (
                #                 ( # 진입전 추세 재확인 필요, atr14 증가하는지 확인 필요
                #                     (

                #                         ((df.bbb_diff > 0) & (df['ema200_diff'] > 0))
                #                         |
                #                         ((df.bbb_diff < 0) & (df['ema200_diff'] < 0))
                #                     )
                #                 )
                #             )
                #             &~(
                #                 # (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_macd.shift(1) < 0) | (df.maxima_peak_x_rsi.shift(1) < 0)
                #                 (df.maxima_peak_x_close < 0) | (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_dmp < 0) | (df.minima_peak_x_dmn < 0)
                #             )

                #         , "stg22_long"] = 2 # without atr

                #     df.loc[
                #         # 추세 픽
                #         (
                #             ( # 진입전 추세 재확인 필요, atr14 증가하는지 확인 필요
                #                         ((df.bbb_diff > 0) & (df['ema200_diff'] < 0))
                #                         |
                #                         ((df.bbb_diff < 0) & (df['ema200_diff'] > 0))
                #             )
                #         )
                #         &~(
                #             # (df.minima_peak_x_macd > 0) | (df.minima_peak_x_rsi > 0) | (df.minima_peak_x_macd.shift(1) > 0) | (df.minima_peak_x_rsi.shift(1) > 0)
                #             (df.minima_peak_x_close > 0) | (df.minima_peak_x_macd > 0) | (df.minima_peak_x_rsi > 0) | (df.minima_peak_x_dmp > 0) | (df.maxima_peak_x_dmn > 0)
                #         )

                #     , "stg22_short"] = -2 # without atr


                #     # df.loc[
                #     #     # 추세 픽
                #     #         (
                #     #             ( # 진입전 추세 재확인 필요, atr14 증가하는지 확인 필요
                #     #                 (
                #     #                     # ((df['minima_peak_x_macd'] > 0) & (df.MACD_12_26_9 < 0)  & (df.MACD_50_75_35 > 0)) #  & (df.macdh_diff_35 > 0)) # & (df.MACDh_12_26_9 > 0)) # & (df.macdh_diff > 0)) # # & (df.MACD_12_26_9 < 0) & (df.MACDh_12_26_9 < 0))
                #     #                     ((df['minima_peak_x_macd'] > 0)) # & (df.MACD_12_26_9 < 0)  & (df.MACD_50_75_35 > 0)& (df.MACDh_12_26_9 < 0)& (df.MACDh_50_75_35 < 0)) #  & (df.macdh_diff_35 > 0)) # & (df.MACDh_12_26_9 > 0)) # & (df.macdh_diff > 0)) # # & (df.MACD_12_26_9 < 0) & (df.MACDh_12_26_9 < 0))
                                        
                #     #                     # & (df['ema50_diff'] > 0)
                #     #                     # & (df['EMA_200'] < df['EMA_50'])
                #     #                     # & (df.bbb_diff < 0)
                #     #                     # & (df['close'] > df['BBL_21_2.0'])
                #     #                     # & (df['wma7_diff'] < 0)
                #     #                     # & (df.bbb_diff > 0)
                #     #                     & (df['EMA_200'] > df['close'])
                #     #                     & (df['lowess'] > df['close'])
                #     #                     # & (df['lowess_diff'] > 0)
                #     #                     & (df['lowess_1'] > df['close'])
                #     #                     & (df['lowess_1_diff'] > 0)
                #     #                     & (df['lowess_MACD_12_26_9'] > df['close'])
                #     #                     & (df['lowess_MACD_12_26_9_diff'] > 0)
                #     #                     # & (df['lowess_MACD_50_75_35'] > df['close'])
                #     #                     # & (df['lowess_MACD_50_75_35_diff'] > 0)

                #     #                     # & (abs(df['DMP_14'] - df['DMN_14']) > 5)

                #     #                     # & (df['EMA_7'] > df['close'])
                #     #                     # & (df['ema7_diff'] > 0)
                                        
                #     #                     # & ((df.dx_diff > 0))
                #     #                     # & ((df.adx_diff > 0))
                #     #                     # & (df.DX_14 < df.ADX_14)


                #     #                     # & (((df['ema50_diff'] > 0) & (df['ema200_diff'] > 0)))
                #     #                     # & (df.EMA_50 > df.EMA_200)
                #     #                     # & (df.ema50_percentage_change > df.ema200_percentage_change)
                #     #                     # & ((df.bbb_diff > 0))
                
                                                
                #     #                 )
                #     #             )
                #     #         )
                #     #         # &~(
                #     #         #     (df.close < df['BBL_21_2.0']) & (df.bbb_diff > 0)
                #     #         # )
                #     #         &~(
                #     #             (df.close > df.EMA_50) & (df.close > df.EMA_200)
                #     #         )
                #     #         &~(
                #     #             # (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_macd.shift(1) < 0) | (df.maxima_peak_x_rsi.shift(1) < 0)
                #     #             (df.maxima_peak_x_close < 0) | (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_dmp < 0) | (df.minima_peak_x_dmn < 0)
                #     #         )

                #     #     , "stg23_long"] = 2 # without atr

                #     # df.loc[
                #     #     # 추세 픽
                #     #     (
                #     #         ( # 진입전 추세 재확인 필요, atr14 증가하는지 확인 필요
                #     #             (
                #     #                 # ((df['maxima_peak_x_macd'] < 0) & (df.MACD_12_26_9 > 0) & (df.MACD_50_75_35 < 0)) # & (df.macdh_diff_35 < 0)) # & (df.MACDh_12_26_9 < 0)) # & (df.macdh_diff < 0)) # # & (df.MACD_12_26_9 > 0) & (df.MACDh_12_26_9 > 0))
                #     #                 ((df['maxima_peak_x_macd'] < 0)) #  & (df.MACD_12_26_9 > 0) & (df.MACD_50_75_35 < 0)& (df.MACDh_12_26_9 > 0)& (df.MACDh_50_75_35 > 0)) # & (df.macdh_diff_35 < 0)) # & (df.MACDh_12_26_9 < 0)) # & (df.macdh_diff < 0)) # # & (df.MACD_12_26_9 > 0) & (df.MACDh_12_26_9 > 0))
                                    
                #     #                 # & (df['ema50_diff'] < 0)
                #     #                 # & (df['EMA_200'] > df['EMA_50'])
                #     #                 # & (df.bbb_diff < 0)
                #     #                 # & (df['close'] < df['BBU_21_2.0'])
                #     #                 # & (df['wma7_diff'] > 0)
                #     #                 # & (df.bbb_diff > 0)
                #     #                 & (df['EMA_200'] < df['close'])
                #     #                 & (df['lowess'] < df['close'])
                #     #                 # & (df['lowess_diff'] < 0)
                #     #                 & (df['lowess_1'] < df['close'])
                #     #                 & (df['lowess_1_diff'] < 0)
                #     #                 & (df['lowess_MACD_12_26_9'] < df['close'])
                #     #                 & (df['lowess_MACD_12_26_9_diff'] < 0)
                #     #                 # & (df['lowess_MACD_50_75_35'] < df['close'])
                #     #                 # & (df['lowess_MACD_50_75_35_diff'] < 0)

                #     #                 # & (abs(df['DMP_14'] - df['DMN_14']) > 5)

                #     #                 # & (df['EMA_7'] < df['close'])
                #     #                 # & (df['ema7_diff'] < 0)
                                    
                #     #                 # & ((df.dx_diff > 0))
                #     #                 # & ((df.adx_diff > 0))
                #     #                 # & (df.DX_14 < df.ADX_14)


                #     #                 # & (((df['ema50_diff'] < 0) & (df['ema200_diff'] < 0)))
                #     #                 # & (df.EMA_50 < df.EMA_200)
                #     #                 # & (df.ema50_percentage_change < df.ema200_percentage_change)
                #     #                 # & ((df.bbb_diff > 0))



                                    
                #     #             )
                #     #         )
                #     #     )
                #     #     # &~(
                #     #     #     (df.close > df['BBU_21_2.0']) & (df.bbb_diff > 0)
                #     #     # )
                #     #     &~(
                #     #         (df.close < df.EMA_50) & (df.close < df.EMA_200)
                #     #     )
                #     #     &~(
                #     #         # (df.minima_peak_x_macd > 0) | (df.minima_peak_x_rsi > 0) | (df.minima_peak_x_macd.shift(1) > 0) | (df.minima_peak_x_rsi.shift(1) > 0)
                #     #         (df.minima_peak_x_close > 0) | (df.minima_peak_x_macd > 0) | (df.minima_peak_x_rsi > 0) | (df.minima_peak_x_dmp > 0) | (df.maxima_peak_x_dmn > 0)
                #     #     )

                #     # , "stg23_short"] = -2 # without atr






                #     df.loc[
                #         # 추세 픽
                #             (
                #                 ( # 진입전 추세 재확인 필요, atr14 증가하는지 확인 필요
                #                     (
                #                         # ((df['minima_peak_x_macd'] > 0) & (df.MACD_12_26_9 < 0)  & (df.MACD_50_75_35 > 0)) #  & (df.macdh_diff_35 > 0)) # & (df.MACDh_12_26_9 > 0)) # & (df.macdh_diff > 0)) # # & (df.MACD_12_26_9 < 0) & (df.MACDh_12_26_9 < 0))
                #                         (
                #                             (df['minima_peak_x_macd'] > 0)
                #                             # & (df.MACD_12_26_9 < 0)
                #                             # & (df.MACDh_12_26_9 < 0)
                #                             # & (df.MACD_50_75_35 < 0)
                #                             # & (df.MACDh_50_75_35 < 0)

                #                         ) #  & (df.macdh_diff_35 > 0)) # & (df.MACDh_12_26_9 > 0)) # & (df.macdh_diff > 0)) # # & (df.MACD_12_26_9 < 0) & (df.MACDh_12_26_9 < 0))
                                        
                                        
                                                
                #                     )
                #                 )
                #             )
                #             # &~(
                #             #     (df.close < df['BBL_21_2.0']) & (df.bbb_diff > 0)
                #             # )
                #             # &~(
                #             #     (df.close > df.EMA_50) & (df.close > df.EMA_200)
                #             # )
                #             &~(
                #                 # (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_macd.shift(1) < 0) | (df.maxima_peak_x_rsi.shift(1) < 0)
                #                 (df.maxima_peak_x_close < 0) | (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_dmp < 0) | (df.minima_peak_x_dmn < 0)
                #             )

                #         , "stg23_long"] = 2 # without atr

                #     df.loc[
                #         # 추세 픽
                #         (
                #             ( # 진입전 추세 재확인 필요, atr14 증가하는지 확인 필요
                #                 (
                #                     # ((df['maxima_peak_x_macd'] < 0) & (df.MACD_12_26_9 > 0) & (df.MACD_50_75_35 < 0)) # & (df.macdh_diff_35 < 0)) # & (df.MACDh_12_26_9 < 0)) # & (df.macdh_diff < 0)) # # & (df.MACD_12_26_9 > 0) & (df.MACDh_12_26_9 > 0))
                #                     (
                #                         (df['maxima_peak_x_macd'] < 0)
                #                         # & (df.MACD_12_26_9 > 0)
                #                         # & (df.MACDh_12_26_9 > 0)
                #                         # & (df.MACD_50_75_35 > 0)
                #                         # & (df.MACDh_50_75_35 > 0)

                #                     ) # & (df.macdh_diff_35 < 0)) # & (df.MACDh_12_26_9 < 0)) # & (df.macdh_diff < 0)) # # & (df.MACD_12_26_9 > 0) & (df.MACDh_12_26_9 > 0))
                                    
                                    

                                    
                #                 )
                #             )
                #         )
                #         # &~(
                #         #     (df.close > df['BBU_21_2.0']) & (df.bbb_diff > 0)
                #         # )
                #         # &~(
                #         #     (df.close < df.EMA_50) & (df.close < df.EMA_200)
                #         # )
                #         &~(
                #             # (df.minima_peak_x_macd > 0) | (df.minima_peak_x_rsi > 0) | (df.minima_peak_x_macd.shift(1) > 0) | (df.minima_peak_x_rsi.shift(1) > 0)
                #             (df.minima_peak_x_close > 0) | (df.minima_peak_x_macd > 0) | (df.minima_peak_x_rsi > 0) | (df.minima_peak_x_dmp > 0) | (df.maxima_peak_x_dmn > 0)
                #         )

                #     , "stg23_short"] = -2 # without atr















                #     df.loc[
                #         (
                #             (df.stg11_long > 0) | (df.stg55_long > 0) # | (df.stg23_long > 0) # | (df.stg23_long.shift(2) > 0)
                #         )
                #         #& (df['atr_diff'] > 0)
                #         # & (df.feature1 < -0.2)
                #         # & (df.feature1 > 0)
                #         # & (df.feature1_diff > 0)

                #     , "stg1_long"] = 2 # without atr
                #     df.loc[
                #         (
                #             (df.stg11_short < 0) | (df.stg55_short < 0) # | (df.stg23_short < 0) # | (df.stg23_short.shift(2) < 0)
                #         )
                #         #& (df['atr_diff'] > 0)
                #         # & (df.feature1 < -0.2)
                #         # & (df.feature1 > 0)
                #         # & (df.feature1_diff > 0)

                #     , "stg1_short"] = -2 # without atr


                #     # df.loc[
                #     #     (
                #     #         (df.stg23_long > 0) # | (df.stg12_long > 0)
                #     #     )
                #     #     & (df.feature1 < -0.2)
                #     #     # & (df.feature1 > 0)
                #     #     # & (df.feature1 > 0)
                #     #     # & (df.feature1_diff > 0)
                #     #     #& (df['atr_diff'] > 0)

                #     # , "stg1_long"] = 2 # without atr
                #     # df.loc[
                #     #     (
                #     #         (df.stg23_short < 0) # | (df.stg12_short < 0)
                #     #     )
                #     #     & (df.feature1 < -0.2)
                #     #     # & (df.feature1 > 0)
                #     #     # & (df.feature1 > 0)
                #     #     # & (df.feature1_diff > 0)
                #     #     #& (df['atr_diff'] > 0)

                #     # , "stg1_short"] = -2 # without atr





                #     df.loc[
                #         # 추세 픽
                #             (
                #                 ( # 진입전 추세 재확인 필요, atr14 증가하는지 확인 필요
                #                     (
                #                         # ((df['minima_peak_x_macd'] > 0) & (df.MACD_12_26_9 < 0)  & (df.MACD_50_75_35 > 0)) #  & (df.macdh_diff_35 > 0)) # & (df.MACDh_12_26_9 > 0)) # & (df.macdh_diff > 0)) # # & (df.MACD_12_26_9 < 0) & (df.MACDh_12_26_9 < 0))
                #                         # ((df['minima_peak_x_macd'] > 0) & (df.MACD_12_26_9 < 0)  & (df.MACD_50_75_35 > 0)& (df.MACDh_12_26_9 < 0)& (df.MACDh_50_75_35 < 0)) #  & (df.macdh_diff_35 > 0)) # & (df.MACDh_12_26_9 > 0)) # & (df.macdh_diff > 0)) # # & (df.MACD_12_26_9 < 0) & (df.MACDh_12_26_9 < 0))
                                        
                #                         # & (df['ema200_diff'] > 0)
                #                         # (df['EMA_200'] < df['EMA_50'])
                #                         # & (df.bbb_diff < 0)
                #                         # & (df['close'] > df['BBL_21_2.0'])
                #                         # & (df['wma7_diff'] < 0)
                #                         # & (df.bbb_diff > 0)



                #                         # & (((df['ema50_diff'] > 0) & (df['ema200_diff'] > 0)))
                #                         # & (df.EMA_50 > df.EMA_200)
                #                         # & (df.ema50_percentage_change > df.ema200_percentage_change)
                #                         # & ((df.bbb_diff > 0))


                #                         # & (df.close > df['BBU_21_2.0']) & (df.bbb_diff > 0)

                #                         # & (df.MACD_12_26_9 > 0)
                #                         # & (df['macd_diff'] > 0)
                #                         # & (df['macdh_diff'] > 0)
                #                         # & (df['obv_diff'] > 0)
                #                         # & (df['rsi_diff'] > 0)
                #                         # # & (df['dx_diff'] > 0)
                #                         # # & (df['j_diff'] > 0)
                #                         # # & (df['k_diff'] > 0)
                #                         # # & (df['d_diff'] > 0)
                #                         # & (df['vwma7_diff'] > 0)
                #                         # & (df['sma7_diff'] > 0)
                #                         # & (df['ema7_diff'] > 0)
                #                         # # & (df['ema50_diff'] > 0)
                #                         # # & (df['ema200_diff'] > 0)
                #                         # # & (df['atr_diff'] > 0)
                #                         # # & (df['atr_p_diff'] > 0)




                #                         # & (df['ema50_diff'] > 0)
                #                         # & (df['ema_50_gradient_1_diff'].shift(1) < 0)
                #                         # & (df['ema_50_gradient_1_diff'] > 0)
                #                         # & (df['ema_50_gradient_2'].shift(1) < 0)
                #                         # & (df['ema_50_gradient_2'] > 0)






                #                         # & 
                #                         # (
                #                         #     (
                #                         #         (df['savgol_diff'].shift(1) > 0) & (df['savgol_diff'] > 0)
                #                         #         & (df['savgol_gradient_2'].shift(1) < 0) & (df['savgol_gradient_2'] > 0)
                #                         #         # & (df['EMA_200'] < df['EMA_50'])
                #                         #     )

                #                         #     |
            
                #                         #     (
                #                         #         (df['savgol_diff'].shift(1) < 0) & (df['savgol_diff'] > 0)
                #                         #         & (df['savgol_gradient_2'].shift(1) > 0) & (df['savgol_gradient_2'] > 0)
                #                         #         # & (df['EMA_200'] > df['EMA_50'])
                #                         #     )
                #                         # )
                #                         # & (df['lowess'] > df['close'])


                #                         (
                #                             # (
                #                             #     (df['lowess_diff'].shift(1) > 0) & (df['lowess_diff'] > 0)
                #                             #     & (df['lowess_gradient_2'].shift(1) < 0) & (df['lowess_gradient_2'] > 0)
                #                             #     & (df['EMA_200'] < df['EMA_50'])
                #                             # )

                #                             # |
            
                #                             (
                #                                 (df['lowess_diff'].shift(1) < 0) & (df['lowess_diff'] > 0)
                #                                 & (df['lowess_gradient_2'].shift(1) > 0) & (df['lowess_gradient_2'] > 0)
                #                                 & (df['EMA_200'] > df['EMA_50'])
                #                             )
                #                         )
                #                         & (df['lowess'] > df['close'])


                #                     )
                #                 )
                #             )
                #             # &~(
                #             #     (df.close < df['BBL_21_2.0']) & (df.bbb_diff > 0)
                #             # )
                #             # &~(
                #             #     (df.close > df.EMA_50) & (df.close > df.EMA_200)
                #             # )
                #             &~(
                #                 # (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_macd.shift(1) < 0) | (df.maxima_peak_x_rsi.shift(1) < 0)
                #                 (df.maxima_peak_x_close < 0) | (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_dmp < 0) | (df.minima_peak_x_dmn < 0)
                #             )

                #         , "stg25_long"] = 2 # without atr

                #     df.loc[
                #         # 추세 픽
                #         (
                #             ( # 진입전 추세 재확인 필요, atr14 증가하는지 확인 필요
                #                 (
                #                     # ((df['maxima_peak_x_macd'] < 0) & (df.MACD_12_26_9 > 0) & (df.MACD_50_75_35 < 0)) # & (df.macdh_diff_35 < 0)) # & (df.MACDh_12_26_9 < 0)) # & (df.macdh_diff < 0)) # # & (df.MACD_12_26_9 > 0) & (df.MACDh_12_26_9 > 0))
                #                     # ((df['minima_peak_x_macd'] < 0) & (df.MACD_12_26_9 > 0) & (df.MACD_50_75_35 < 0)& (df.MACDh_12_26_9 > 0)& (df.MACDh_50_75_35 > 0)) # & (df.macdh_diff_35 < 0)) # & (df.MACDh_12_26_9 < 0)) # & (df.macdh_diff < 0)) # # & (df.MACD_12_26_9 > 0) & (df.MACDh_12_26_9 > 0))
                                    
                #                     # & (df['ema200_diff'] < 0)
                #                     # (df['EMA_200'] > df['EMA_50'])
                #                     # & (df.bbb_diff < 0)
                #                     # & (df['close'] < df['BBU_21_2.0'])
                #                     # & (df['wma7_diff'] > 0)
                #                     # & (df.bbb_diff > 0)




                #                     # & (((df['ema50_diff'] < 0) & (df['ema200_diff'] < 0)))
                #                     # & (df.EMA_50 < df.EMA_200)
                #                     # & (df.ema50_percentage_change < df.ema200_percentage_change)
                #                     # & ((df.bbb_diff > 0))


                #                     # & (df.close < df['BBL_21_2.0']) & (df.bbb_diff > 0)
                                    
                                                                
                #                     # & (df.MACD_12_26_9 < 0)
                #                     # & (df['macd_diff'] < 0)
                #                     # & (df['macdh_diff'] < 0)
                #                     # & (df['obv_diff'] < 0)
                #                     # & (df['rsi_diff'] < 0)
                #                     # # & (df['dx_diff'] < 0)
                #                     # # & (df['j_diff'] < 0)
                #                     # # & (df['k_diff'] < 0)
                #                     # # & (df['d_diff'] < 0)
                #                     # & (df['vwma7_diff'] < 0)
                #                     # & (df['sma7_diff'] < 0)
                #                     # & (df['ema7_diff'] < 0)
                #                     # # & (df['ema50_diff'] < 0)
                #                     # # & (df['ema200_diff'] < 0)
                #                     # # & (df['atr_diff'] < 0)
                #                     # # & (df['atr_p_diff'] < 0)



                #                     # & (df['ema50_diff'] < 0)
                #                     # & (df['ema_50_gradient_1_diff'].shift(1) > 0)
                #                     # & (df['ema_50_gradient_1_diff'] < 0)
                #                     # & (df['ema_50_gradient_2'].shift(1) > 0)
                #                     # & (df['ema_50_gradient_2'] < 0)


                                    
                                
                #                     # & 

                #                         # savgol
                #                         # lowess

                #                     # (
                #                     #     (
                #                     #         (df['savgol_diff'].shift(1) < 0) & (df['savgol_diff'] < 0)
                #                     #         & (df['savgol_gradient_2'].shift(1) > 0) & (df['savgol_gradient_2'] < 0)
                #                     #         # & (df['EMA_200'] > df['EMA_50'])
                #                     #     )

                #                     #     |

                #                     #     (
                #                     #         (df['savgol_diff'].shift(1) > 0) & (df['savgol_diff'] < 0)
                #                     #         & (df['savgol_gradient_2'].shift(1) < 0) & (df['savgol_gradient_2'] < 0)
                #                     #         # & (df['EMA_200'] < df['EMA_50'])
                #                     #     )
                #                     # )
                #                     # & (df['lowess'] < df['close'])
                                    





                #                     (
                #                         # (
                #                         #     (df['lowess_diff'].shift(1) < 0) & (df['lowess_diff'] < 0)
                #                         #     & (df['lowess_gradient_2'].shift(1) > 0) & (df['lowess_gradient_2'] < 0)
                #                         #     & (df['EMA_200'] > df['EMA_50'])
                #                         # )

                #                         # |

                #                         (
                #                             (df['lowess_diff'].shift(1) > 0) & (df['lowess_diff'] < 0)
                #                             & (df['lowess_gradient_2'].shift(1) < 0) & (df['lowess_gradient_2'] < 0)
                #                             & (df['EMA_200'] < df['EMA_50'])
                #                         )
                #                     )
                #                     & (df['lowess'] < df['close'])





                                    
                #                 )
                #             )
                #         )
                #         # &~(
                #         #     (df.close > df['BBU_21_2.0']) & (df.bbb_diff > 0)
                #         # )
                #         # &~(
                #         #     (df.close < df.EMA_50) & (df.close < df.EMA_200)
                #         # )
                #         &~(
                #             # (df.minima_peak_x_macd > 0) | (df.minima_peak_x_rsi > 0) | (df.minima_peak_x_macd.shift(1) > 0) | (df.minima_peak_x_rsi.shift(1) > 0)
                #             (df.minima_peak_x_close > 0) | (df.minima_peak_x_macd > 0) | (df.minima_peak_x_rsi > 0) | (df.minima_peak_x_dmp > 0) | (df.maxima_peak_x_dmn > 0)
                #         )

                #     , "stg25_short"] = -2 # without atr


                #     df.loc[
                #         (
                #             (df.stg23_long > 0)   #| (df.stg12_long > 0)
                #         )
                #         #& (df['atr_diff'] > 0)
                #         # & (df.feature1 < -0.2)
                #         # & (df.feature1 > 0)
                #         # & (df.feature1_diff > 0)
        
                #     , "stg2_long"] = 2 # without atr
                #     df.loc[
                #         (
                #             (df.stg23_short < 0)  # | (df.stg12_short < 0)
                #         )
                #         #& (df['atr_diff'] > 0)
                #         # & (df.feature1 < -0.2)
                #         # & (df.feature1 > 0)
                #         # & (df.feature1_diff > 0)
        
                #     , "stg2_short"] = -2 # without atr










                #     # df.loc[
                #     #     (
                #     #         (df.stg11_long > 0) | (df.stg12_long > 0)
                #     #     )

                #     # , "stg1_long"] = 2 # without atr
                #     # df.loc[
                #     #     (
                #     #         (df.stg11_short < 0) | (df.stg12_short < 0)
                #     #     )

                #     # , "stg1_short"] = -2 # without atr


                #     # df.loc[
                #     #     (
                #     #         (df.stg11_short < 0) | (df.stg12_short < 0)
                #     #     )

                #     # , "stg1_long"] = 2 # without atr
                #     # df.loc[
                #     #     (
                #     #         (df.stg11_long > 0) | (df.stg12_long > 0)
                #     #     )

                #     # , "stg1_short"] = -2 # without atr






                #     # df.loc[
                #     #     (
                #     #         (df.stg13_long > 0)
                #     #     )

                #     # , "stg1_long"] = 2 # without atr
                #     # df.loc[
                #     #     (
                #     #         (df.stg13_short < 0)
                #     #     )

                #     # , "stg1_short"] = -2 # without atr





                #     # df.loc[
                #     #     (
                #     #         (df.stg13_short < 0)
                #     #     )

                #     # , "stg1_long"] = 2 # without atr
                #     # df.loc[
                #     #     (
                #     #         (df.stg13_long > 0)
                #     #     )

                #     # , "stg1_short"] = -2 # without atr


































                #     # df.loc[
                #     #     # 추세 픽
                #     #     (
                #     #         ( # 진입전 추세 재확인 필요, atr14 증가하는지 확인 필요
                #     #             # (df['minima_peak_x_rsi'] > 0) #& (df['percentage_change2'] > 3) # & (df['macd_diff_200'] > df['macds_diff_200']) & (df['macd_diff_200'] > 0) & (df['macds_diff_200'] > 0) & (df['macd_diff_700'] > 0) & (df['macds_diff_700'] > 0) & (df['MACD_12_200_9'].shift(2) < df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] > 0) & (df['k_diff'] > 0) & (df['d_diff'] > 0)
                #     #             # (df['rsi_percentage_change'] > 5) & (df['atr_percentage_change'] > 4) & (df['macd_percentage_change'] > 50) & (df['obv_percentage_change'] > 100) #& (df['atr_percentage_change'] > 4)
                #     #             # & (df['obv_percentage_change'] > 150)
                #     #             # (df['obv_percentage_change'].abs() < 50)
                #     #             (((df['sma50_diff'] < 0) & (df['sma200_diff'] < 0)) | ((df['sma50_diff'] > 0) & (df['sma200_diff'] < 0)))
                #     #             & (df.SMA_50 < df.SMA_200)
                #     #             & (df.sma50_percentage_change > df.sma200_percentage_change)
                #     #             & (df.macd_percentage_change > df.macds_percentage_change)
                #     #             # & (df.RSI_14 < 75)
                #     #             # & (df.DX_14 < 70)
                #     #             # &~ ((df.adx_diff < 0) & (df.atr_diff < 0) & (df.atr_p < 0.5))

                #     #             # & (((df['sma50_diff'] > df['sma200_diff'])))
                #     #             # & (df.adx_diff > 0)

                #     #             # & (df.bbb_diff > 0)
                                
                #     #             # & (df.atr_diff > 0)
                #     #             # & ((df.MACD_12_26_9 > 0) & (df.MACDh_12_26_9 > 0) & (df.macd_diff > 0) & (df.macdh_diff > 0))
                #     #             & (df.obv_diff > 0)
                #     #             # & (df.atr_p < 0.7)


                #     #             # (df['minima_peak_x_macd'].shift(2) > 0) & (df['macd_diff_200'] > df['macds_diff_200']) & (df['macd_diff_200'].shift(2) > 0) & (df['macds_diff_200'].shift(2) > 0) & (df['macd_diff_700'].shift(2) > 0) & (df['macds_diff_700'].shift(2) > 0) & (df['MACD_12_200_9'].shift(2) < df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] > 0) & (df['k_diff'] > 0) & (df['d_diff'] > 0)
                #     #         )


                #     #         & 

                #     #         (
                #     #             # (
                #     #             #     ((df.dx_diff < 0) & (df.DMN_14 > df.DMP_14) & (df.intersect == 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.high_diff > 0)           & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff > 0)
                #     #             # )
                #     #             #     |
                #     #             # (
                #     #             #     ((df.dx_diff < 0) & (dmn_set_2 > dmp_set_2) & (df.intersect > 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.high_diff > 0)           & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff > 0)
                #     #             # )
                #     #             #     |
                #     #             (
                #     #                 (df.adx_diff > 0) & (df.DMP_14 > df.DMN_14) #& (df.atr_p_diff < 0)# & (df.high_diff > 0) & (df.rsi_diff > 0) # & (df.obv_diff > 0) & (df.j_diff > 0) & (df['atr_AMATe_LR_8_21_2'] > 0)
                #     #             )
                #     #         )
                #     #         # & (
                #     #         #     (df.vwma7_diff > 0)
                #     #         # )
                #     #     )
                #     #     # &~(
                #     #     #     # ((df.bbb_diff > 0) & (df.atr_diff > 0) & (df.adx_diff > 0) & ((df.MACD_12_26_9 > 0) & (df.MACDh_12_26_9 > 0) & (df.macd_diff > 0) & (df.macdh_diff > 0))) |
                #     #     #     # (df.bbb_diff > 0) | (df.atr_diff > 0) |
                #     #     #     # (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_macd.shift(1) < 0) | (df.maxima_peak_x_rsi.shift(1) < 0)
                #     #     #     (df.maxima_peak_x_close < 0) | (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_dmp < 0) | (df.minima_peak_x_dmn < 0)
                #     #     # )
                #     #     # &~(
                #     #     #     # ((df.bbb_diff > 0) & (df.atr_diff > 0) & (df.adx_diff > 0) & ((df.MACD_12_26_9 > 0) & (df.MACDh_12_26_9 > 0) & (df.macd_diff > 0) & (df.macdh_diff > 0))) |
                #     #     #     # (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_macd.shift(1) < 0) | (df.maxima_peak_x_rsi.shift(1) < 0)
                #     #     #     (df.adx_diff < 0) & (df.atr_diff < 0) #& (df.atr_p < 0.7)
                #     #     # )
                #     #     &~ (
                #     #         (
                #     #             (
                #     #                 ((df.DX_14 > 40) & (df.adx_diff < 0) & (df.DMP_14 > df.DMN_14) & (df.intersect == 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.low_diff < 0)          & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff < 0)
                #     #             )


                #     #         )
                #     #     )                
                #     # , "stg2_long"] = 2 # without atr

                #     # df.loc[
                #     #     # 추세 픽
                #     #     (
                #     #         ( # 진입전 추세 재확인 필요, atr14 증가하는지 확인 필요
                #     #             # (df['maxima_peak_x_rsi'] < 0) #& (df['percentage_change2'] < -3) # & (df['macd_diff_200'] < df['macds_diff_200']) & (df['macd_diff_200'] < 0) & (df['macds_diff_200'] < 0) & (df['macd_diff_700'] < 0) & (df['macds_diff_700'] < 0) & (df['MACD_12_200_9'].shift(2) > df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] < 0) & (df['k_diff'] < 0) & (df['d_diff'] < 0)
                #     #             # (df['rsi_percentage_change'] < -5) & (df['atr_percentage_change'] > 4) & (df['macd_percentage_change'] < -50) & (df['obv_percentage_change'] < -100) #& (df['atr_percentage_change'] > 4)
                #     #             # (df['obv_percentage_change'] < -150)
                #     #             # (df['obv_percentage_change'].abs() < 50)
                #     #             # & (((df['sma50_diff'] > 0) & (df['sma200_diff'] > 0)))
                #     #             (((df['sma50_diff'] > 0) & (df['sma200_diff'] > 0)) | ((df['sma50_diff'] < 0) & (df['sma200_diff'] > 0)))
                #     #             & (df.SMA_50 > df.SMA_200)
                #     #             & (df.sma50_percentage_change < df.sma200_percentage_change)
                #     #             & (df.macd_percentage_change < df.macds_percentage_change)
                #     #             # & (df.RSI_14 > 25)
                #     #             # & (df.DX_14 < 70)
                #     #             # &~ ((df.adx_diff < 0) & (df.atr_diff < 0) & (df.atr_p < 0.5))
                #     #             # &~ (df.atr_diff < 0)

                #     #             # & (((df['sma50_diff'] < df['sma200_diff'])))
                #     #             # &~ (df.adx_diff > 0)

                #     #             # & (df.bbb_diff > 0)
                                
                #     #             # & (df.atr_diff > 0)
                #     #             # & ((df.MACD_12_26_9 < 0) & (df.MACDh_12_26_9 < 0) & (df.macd_diff < 0) & (df.macdh_diff < 0))
                #     #             & (df.obv_diff < 0)
                #     #             # & (df.atr_p < 0.7)

                                
                #     #             # (df['maxima_peak_x_macd'].shift(2) < 0) & (df['macd_diff_200'] < df['macds_diff_200']) & (df['macd_diff_200'].shift(2) < 0) & (df['macds_diff_200'].shift(2) < 0) & (df['macd_diff_700'].shift(2) < 0) & (df['macds_diff_700'].shift(2) < 0) & (df['MACD_12_200_9'].shift(2) > df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] < 0) & (df['k_diff'] < 0) & (df['d_diff'] < 0)
                #     #         )


                #     #         & 

                #     #         (
                #     #             # (
                #     #             #     ((df.dx_diff < 0) & (df.DMP_14 > df.DMN_14) & (df.intersect == 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.low_diff < 0)          & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff < 0)
                #     #             # )
                #     #             #     |
                #     #             # (
                #     #             #     ((df.dx_diff < 0) & (dmp_set_2 > dmn_set_2) & (df.intersect > 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.low_diff < 0)          & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff < 0)
                #     #             # )
                #     #             #     |
                #     #             (
                #     #                 (df.dx_diff > 0) & (df.DMN_14 > df.DMP_14) #& (df.atr_p_diff < 0)#& (df.intersect == 0) & (df.low_diff < 0)  # & (df.obv_diff < 0) & (df.j_diff < 0) & (df['atr_AMATe_LR_8_21_2'] > 0)
                #     #             )
                #     #         )
                #     #         # & (
                #     #         #     (df.vwma7_diff < 0)
                #     #         # )
                #     #     )
                #     #     # &~(
                #     #     #     # ((df.bbb_diff > 0) & (df.atr_diff > 0) & (df.adx_diff > 0) & ((df.MACD_12_26_9 < 0) & (df.MACDh_12_26_9 < 0) & (df.macd_diff < 0) & (df.macdh_diff < 0))) |
                #     #     #     # (df.bbb_diff > 0) | (df.atr_diff > 0) |
                #     #     #     # (df.minima_peak_x_macd > 0) | (df.minima_peak_x_rsi > 0) | (df.minima_peak_x_macd.shift(1) > 0) | (df.minima_peak_x_rsi.shift(1) > 0)
                #     #     #     (df.minima_peak_x_close > 0) | (df.minima_peak_x_macd > 0) | (df.minima_peak_x_rsi > 0) | (df.minima_peak_x_dmp > 0) | (df.maxima_peak_x_dmn > 0)
                #     #     # )
                #     #     # &~(
                #     #     #     # ((df.bbb_diff > 0) & (df.atr_diff > 0) & (df.adx_diff > 0) & ((df.MACD_12_26_9 > 0) & (df.MACDh_12_26_9 > 0) & (df.macd_diff > 0) & (df.macdh_diff > 0))) |
                #     #     #     # (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_macd.shift(1) < 0) | (df.maxima_peak_x_rsi.shift(1) < 0)
                #     #     #     (df.adx_diff < 0) & (df.atr_diff < 0) #& (df.atr_p < 0.7)
                #     #     # )
                #     #     &~ (
                            
                #     #         (
                #     #             (
                #     #                 ((df.DX_14 > 40) & (df.adx_diff < 0) & (df.DMN_14 > df.DMP_14) & (df.intersect == 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.high_diff > 0)           & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff > 0)
                #     #             )


                #     #         )
                #     #     )
                #     # , "stg2_short"] = -2 # without atr





                #     # # # 5 -re1 #################&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&###############################################################################################################################
                #     # # # 5 -re1 #################&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&###############################################################################################################################
                #     # # # 5 -re1 #################&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&###############################################################################################################################

                #     # # # 5 -re2 #################&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&###############################################################################################################################
                #     # # # 5 -re2 #################&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&###############################################################################################################################
                #     # # # 5 -re2 #################&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&###############################################################################################################################










                #     df.loc[
                #         # 추세 픽
                #         (
                #             # ( # 진입전 추세 재확인 필요, atr14 증가하는지 확인 필요
                #             #     # (df['minima_peak_x_rsi'] > 0) #& (df['percentage_change2'] > 3) # & (df['macd_diff_200'] > df['macds_diff_200']) & (df['macd_diff_200'] > 0) & (df['macds_diff_200'] > 0) & (df['macd_diff_700'] > 0) & (df['macds_diff_700'] > 0) & (df['MACD_12_200_9'].shift(2) < df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] > 0) & (df['k_diff'] > 0) & (df['d_diff'] > 0)
                #             #     # (df['rsi_percentage_change'] > 5) & (df['atr_percentage_change'] > 4) & (df['macd_percentage_change'] > 50) & (df['obv_percentage_change'] > 100) #& (df['atr_percentage_change'] > 4)
                #             #     # & (df['obv_percentage_change'] > 150)
                #             #     # (df['obv_percentage_change'].abs() < 50)
                #             #     (((df['sma50_diff'] < 0) & (df['sma200_diff'] < 0)) | ((df['sma50_diff'] > 0) & (df['sma200_diff'] < 0)))
                #             #     & (df.SMA_50 < df.SMA_200)
                #             #     & (df.sma50_percentage_change > df.sma200_percentage_change)
                #             #     & (df.macd_percentage_change > df.macds_percentage_change)
                #             #     # & (df.RSI_14 < 75)
                #             #     # & (df.DX_14 < 70)
                #             #     # &~ ((df.adx_diff < 0) & (df.atr_diff < 0) & (df.atr_p < 0.5))

                #             #     # & (((df['sma50_diff'] > df['sma200_diff'])))
                #             #     # & (df.adx_diff > 0)

                #             #     # & (df.bbb_diff > 0)
                                
                #             #     # & (df.atr_diff > 0)
                #             #     # & ((df.MACD_12_26_9 > 0) & (df.MACDh_12_26_9 > 0) & (df.macd_diff > 0) & (df.macdh_diff > 0))
                #             #     & (df.obv_diff > 0)
                #             #     # & (df.atr_p < 0.7)


                #             #     # (df['minima_peak_x_macd'].shift(2) > 0) & (df['macd_diff_200'] > df['macds_diff_200']) & (df['macd_diff_200'].shift(2) > 0) & (df['macds_diff_200'].shift(2) > 0) & (df['macd_diff_700'].shift(2) > 0) & (df['macds_diff_700'].shift(2) > 0) & (df['MACD_12_200_9'].shift(2) < df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] > 0) & (df['k_diff'] > 0) & (df['d_diff'] > 0)
                #             # )


                #             # & 

                #             (

                #                 # (
                #                 #     (df.dx_diff < 0) & (dmn_set_2 > dmp_set_2) & (df.intersect > 0) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.high_diff > 0)           & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff > 0)
                #                 # )
                #                 #     |
                #                 (
                #                     # (df.DMP_14 > df.DMN_14) & (df.intersect > 0) & (df.ADX_14 > 25) # & (df.high_diff > 0) & (df.rsi_diff > 0) # & (df.obv_diff > 0) & (df.j_diff > 0) & (df['atr_AMATe_LR_8_21_2'] > 0)
                #                     # (df.MACD_50_75_35 > 0) & (df.close > df.SMA_200)
                #                     ((df.DMP_14 > df.DMN_14) & (df.intersect > 0))
                #                     & (df['close'] < df['close'].quantile(0.2))
                #                     # & (df.obv_diff > 0)
                #                     & ((df['close'].shift(1) > df['close']) & (df['close'].shift(-1) > df['close'])) # minima

                #                     & (abs(df['DX_200'] - df['ADX_200']) > 1)

                #                     # & (df['lowess'] > df['close'])
                #                     # & (df['lowess_diff'].shift(1) > 0) & (df['lowess_diff'] > 0)
                #                     # & (df['lowess'] > df['close'])
                #                     # & (df['lowess_diff'] > 0)
                #                     & (df['lowess_1'] > df['close'])
                #                     # & (df['lowess_1_diff'] > 0)
                #                     # & (df['ema200_diff'] > 0)
                #                     # & ((df['close'].shift(1) < df['close']) & (df['close'].shift(-1) < df['close'])) # maxima
                #                     # & (df.rsi_diff > 0)
                #                     # & (df.macdh_diff > 0)
                #                     # &~ (df.maxima_peak_x_close < 0)



                #                     # & 
                                    
                #                     # ((df['J_9_3'].shift(1) < df['K_9_3'].shift(1)) & (df['K_9_3'].shift(1) < df['D_9_3'].shift(1)))
                #                     # & ((df['J_9_3'] > df['K_9_3']) & (df['K_9_3'] > df['D_9_3']))
                #                 )
                #             )
                #             # & (
                #             #     (df.vwma7_diff > 0)
                #             # )
                #         )
                #         # &~(
                #         #     # ((df.bbb_diff > 0) & (df.atr_diff > 0) & (df.adx_diff > 0) & ((df.MACD_12_26_9 > 0) & (df.MACDh_12_26_9 > 0) & (df.macd_diff > 0) & (df.macdh_diff > 0))) |
                #         #     # (df.bbb_diff > 0) | (df.atr_diff > 0) |
                #         #     # (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_macd.shift(1) < 0) | (df.maxima_peak_x_rsi.shift(1) < 0)
                #         #     (df.maxima_peak_x_close < 0) | (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_dmp < 0) | (df.minima_peak_x_dmn < 0)
                #         # )
                #         # &~(
                #         #     # ((df.bbb_diff > 0) & (df.atr_diff > 0) & (df.adx_diff > 0) & ((df.MACD_12_26_9 > 0) & (df.MACDh_12_26_9 > 0) & (df.macd_diff > 0) & (df.macdh_diff > 0))) |
                #         #     # (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_macd.shift(1) < 0) | (df.maxima_peak_x_rsi.shift(1) < 0)
                #         #     (df.adx_diff < 0) & (df.atr_diff < 0) #& (df.atr_p < 0.7)
                #         # )
                #         # &~ (
                #         #     (
                #         #         (
                #         #             ((df.DX_14 > 40) & (df.adx_diff < 0) & (df.DMP_14 > df.DMN_14) & (df.intersect == 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.low_diff < 0)          & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff < 0)
                #         #         )


                #         #     )
                #         # )
                #     , "stg3_long"] = 2 # without atr

                #     df.loc[
                #         # 추세 픽
                #         (
                #             # ( # 진입전 추세 재확인 필요, atr14 증가하는지 확인 필요
                #             #     # (df['maxima_peak_x_rsi'] < 0) #& (df['percentage_change2'] < -3) # & (df['macd_diff_200'] < df['macds_diff_200']) & (df['macd_diff_200'] < 0) & (df['macds_diff_200'] < 0) & (df['macd_diff_700'] < 0) & (df['macds_diff_700'] < 0) & (df['MACD_12_200_9'].shift(2) > df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] < 0) & (df['k_diff'] < 0) & (df['d_diff'] < 0)
                #             #     # (df['rsi_percentage_change'] < -5) & (df['atr_percentage_change'] > 4) & (df['macd_percentage_change'] < -50) & (df['obv_percentage_change'] < -100) #& (df['atr_percentage_change'] > 4)
                #             #     # (df['obv_percentage_change'] < -150)
                #             #     # (df['obv_percentage_change'].abs() < 50)
                #             #     # & (((df['sma50_diff'] > 0) & (df['sma200_diff'] > 0)))
                #             #     (((df['sma50_diff'] > 0) & (df['sma200_diff'] > 0)) | ((df['sma50_diff'] < 0) & (df['sma200_diff'] > 0)))
                #             #     & (df.SMA_50 > df.SMA_200)
                #             #     & (df.sma50_percentage_change < df.sma200_percentage_change)
                #             #     & (df.macd_percentage_change < df.macds_percentage_change)
                #             #     # & (df.RSI_14 > 25)
                #             #     # & (df.DX_14 < 70)
                #             #     # &~ ((df.adx_diff < 0) & (df.atr_diff < 0) & (df.atr_p < 0.5))
                #             #     # &~ (df.atr_diff < 0)

                #             #     # & (((df['sma50_diff'] < df['sma200_diff'])))
                #             #     # &~ (df.adx_diff > 0)

                #             #     # & (df.bbb_diff > 0)
                                
                #             #     # & (df.atr_diff > 0)
                #             #     # & ((df.MACD_12_26_9 < 0) & (df.MACDh_12_26_9 < 0) & (df.macd_diff < 0) & (df.macdh_diff < 0))
                #             #     & (df.obv_diff < 0)
                #             #     # & (df.atr_p < 0.7)

                                
                #             #     # (df['maxima_peak_x_macd'].shift(2) < 0) & (df['macd_diff_200'] < df['macds_diff_200']) & (df['macd_diff_200'].shift(2) < 0) & (df['macds_diff_200'].shift(2) < 0) & (df['macd_diff_700'].shift(2) < 0) & (df['macds_diff_700'].shift(2) < 0) & (df['MACD_12_200_9'].shift(2) > df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] < 0) & (df['k_diff'] < 0) & (df['d_diff'] < 0)
                #             # )


                #             # & 

                #             (
                #                 # (
                #                 #     ((df.dx_diff < 0) & (dmn_set_2 < dmp_set_2) & (df.intersect > 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.low_diff < 0)          & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff < 0)
                #                 # )
                #                 #     |
                #                 (
                #                     # (df.DMP_14 < df.DMN_14) & (df.intersect > 0) & (df.ADX_14 > 25) #& (df.intersect == 0) & (df.low_diff < 0)  # & (df.obv_diff < 0) & (df.j_diff < 0) & (df['atr_AMATe_LR_8_21_2'] > 0)
                #                     # (df.MACD_50_75_35 < 0) & (df.close > df.SMA_200)
                #                     ((df.DMP_14 < df.DMN_14) & (df.intersect > 0))
                #                     & (df['close'] > df['close'].quantile(0.8))
                #                     # & (df.obv_diff < 0)
                #                     # & ((df['close'].shift(1) > df['close']) & (df['close'].shift(-1) > df['close'])) # minima
                #                     & ((df['close'].shift(1) < df['close']) & (df['close'].shift(-1) < df['close'])) # maxima

                #                     & (abs(df['DX_200'] - df['ADX_200']) > 1)

                #                     # & (df['lowess'] < df['close'])
                #                     # & (df['lowess_diff'].shift(1) < 0) & (df['lowess_diff'] < 0)
                #                     # & (df['lowess'] < df['close'])
                #                     # & (df['lowess_diff'] < 0)
                #                     & (df['lowess_1'] < df['close'])
                #                     # & (df['lowess_1_diff'] < 0)
                #                     # & (df['ema200_diff'] < 0)
                #                     # & (df.rsi_diff < 0)
                #                     # & (df.macdh_diff < 0)
                #                     # & 
                                    
                #                     # ((df['J_9_3'].shift(1) > df['K_9_3'].shift(1)) & (df['K_9_3'].shift(1) > df['D_9_3'].shift(1)))
                #                     # & ((df['J_9_3'] < df['K_9_3']) & (df['K_9_3'] < df['D_9_3']))                            
                #                 )
                #             )
                #             # & (
                #             #     (df.vwma7_diff < 0)
                #             # )
                #         )
                #         # &~(
                #         #     # ((df.bbb_diff > 0) & (df.atr_diff > 0) & (df.adx_diff > 0) & ((df.MACD_12_26_9 < 0) & (df.MACDh_12_26_9 < 0) & (df.macd_diff < 0) & (df.macdh_diff < 0))) |
                #         #     # (df.bbb_diff > 0) | (df.atr_diff > 0) |
                #         #     # (df.minima_peak_x_macd > 0) | (df.minima_peak_x_rsi > 0) | (df.minima_peak_x_macd.shift(1) > 0) | (df.minima_peak_x_rsi.shift(1) > 0)
                #         #     (df.minima_peak_x_close > 0) | (df.minima_peak_x_macd > 0) | (df.minima_peak_x_rsi > 0) | (df.minima_peak_x_dmp > 0) | (df.maxima_peak_x_dmn > 0)
                #         # )
                #         # &~(
                #         #     # ((df.bbb_diff > 0) & (df.atr_diff > 0) & (df.adx_diff > 0) & ((df.MACD_12_26_9 > 0) & (df.MACDh_12_26_9 > 0) & (df.macd_diff > 0) & (df.macdh_diff > 0))) |
                #         #     # (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_macd.shift(1) < 0) | (df.maxima_peak_x_rsi.shift(1) < 0)
                #         #     (df.adx_diff < 0) & (df.atr_diff < 0) #& (df.atr_p < 0.7)
                #         # )
                #         # &~ (
                            
                #         #     (
                #         #         (
                #         #             ((df.DX_14 > 40) & (df.adx_diff < 0) & (df.DMN_14 > df.DMP_14) & (df.intersect == 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.high_diff > 0)           & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff > 0)
                #         #         )


                #         #     )
                #         # )
                #     , "stg3_short"] = -2 # without atr



                #     # # # 5 -re2 #################&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&###############################################################################################################################
                #     # # # 5 -re2 #################&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&###############################################################################################################################
                #     # # # 5 -re2 #################&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&###############################################################################################################################


                #     # 6 #################&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&###############################################################################################################################
                #     # 6 #################&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&###############################################################################################################################
                #     # 6 #################&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&###############################################################################################################################







                #     df.loc[
                #         # 추세 픽
                #             (
                #                 ( # 진입전 추세 재확인 필요, atr14 증가하는지 확인 필요
                #                     (
                #                         # ((df['anomalies_close'] > 0) & ((df.maxima_peak_x_dx < 0) & (df.maxima_peak_x_atr14 < 0)) & (df.DX_14 > 40) & (df.DMP_14 < df.DMN_14))
                #                         # & ((df.minima_peak_x_close > 0) | (df.minima_peak_x_rsi > 0) | (df.minima_peak_x_obv > 0) | (df.minima_peak_x_macd > 0))


                #                         ((df['anomalies_close'] > 0) & (df['close'] < df['close'].quantile(0.5)) & (df.DX_14 > 40) & (df.DMP_14 < df.DMN_14) & (df.macdh_diff > 0))
                #                         & ((df.minima_peak_x_close > 0) | (df.minima_peak_x_rsi > 0) | (df.minima_peak_x_obv > 0) | (df.minima_peak_x_macd > 0))
                #                         & ((df.maxima_peak_x_dx < 0) | (df.maxima_peak_x_atr14 < 0))
                #                         # & (abs(df['DMP_14'] - df['DMN_14']) > 5)

                #                         # & (df['lowess'] > df['high'])
                #                         # & (df['lowess_diff'].shift(1) > 0) & (df['lowess_diff'] > 0)
                #                         & (df['lowess'] > df['close'])
                #                         # & (df['lowess_diff'] > 0)
                #                         & (df['lowess_1'] > df['close'])
                #                         # & (df['lowess_1_diff'] > 0)
                #                         & (df['lowess_MACD_12_26_9'] > df['close'])


                #                         # & ((df.dx_diff < 0))
                #                         # # & ((df.adx_diff > 0))
                #                         # & (df.DX_14 < df.ADX_14)
                                        
                #                         # & (df['ema200_diff'] > 0)
                #                         # & (df['EMA_200'] < df['EMA_50'])
                #                         # & (df['volume_change'].abs() > 200)
                                        

                #                         # & ((df['anomalies_close'] > 0) | (df['anomalies_RSI_14'] > 0))

                #                         # ((df['AnomalyDetection_close'] > 0) & (df.maxima_peak_x_dx < 0) & (df.DX_14 > 40) & (df.DMP_14 < df.DMN_14))
                #                         # & ((df.minima_peak_x_close > 0) | (df.minima_peak_x_rsi > 0) | (df.minima_peak_x_obv > 0) | (df.minima_peak_x_macd > 0)) 
                #                         # # & ((df['anomalies_close'] > 0) | (df['anomalies_RSI_14'] > 0))
                #                     )
                #                     #     |
                #                     # ( # 진입전 추세 재확인 필요, atr14 증가하는지 확인 필요
                #                     #     ((df['minima_peak_x_rsi'] > 0) | (df['minima_peak_x_macd'] > 0)) #& (df['percentage_change2'] > 3) # & (df['macd_diff_200'] > df['macds_diff_200']) & (df['macd_diff_200'] > 0) & (df['macds_diff_200'] > 0) & (df['macd_diff_700'] > 0) & (df['macds_diff_700'] > 0) & (df['MACD_12_200_9'].shift(2) < df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] > 0) & (df['k_diff'] > 0) & (df['d_diff'] > 0)
                #                     #     # (df['rsi_percentage_change'] > 5) & (df['atr_percentage_change'] > 4) & (df['macd_percentage_change'] > 50) & (df['obv_percentage_change'] > 100) #& (df['atr_percentage_change'] > 4)
                #                     #     # & (df['obv_percentage_change'] < -150)
                #                     #     # & (df['obv_percentage_change'] > 50)
                #                     #     # & (df['rsi_percentage_change'] > 5)
                #                     #     # & (df['macd_percentage_change'] > 5)
                #                     #     & (df['volume_change'].abs() > 1000)
                #                     #     # & (df['volume_change'].abs() > 200)
                #                     #     # & (df['volume_change'].abs() < 500)
                #                     #     & (df['close'] < df['SMA_50'] )
                #                     #     & (df['SMA_50'] < df['SMA_200'] )
                #                     #     # (df['obv_percentage_change'].abs() < 50)
                #                     #     # & (((df['sma50_diff'] > 0) & (df['sma200_diff'] > 0)))
                #                     #     & (((df['sma50_diff'] < 0) & (df['sma200_diff'] < 0)))

                                        
                #                     #     # (df['minima_peak_x_macd'].shift(2) > 0) & (df['macd_diff_200'] > df['macds_diff_200']) & (df['macd_diff_200'].shift(2) > 0) & (df['macds_diff_200'].shift(2) > 0) & (df['macd_diff_700'].shift(2) > 0) & (df['macds_diff_700'].shift(2) > 0) & (df['MACD_12_200_9'].shift(2) < df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] > 0) & (df['k_diff'] > 0) & (df['d_diff'] > 0)
                #                     # )
                                    












                #                     #     |
                #                     # (
                #                     #     ((df['anomaly_type_feature1_n'] > 0) & (df['anomaly_type_close_n'] > 0))
                #                     #         |
                #                     #     ((df['anomaly_type_feature2_n'] > 0) & (df['anomaly_type_close_n'] > 0))
                #                     # )
                #                 )
                #                 # & ( 
                #                 #     (df['SMA_50'] < df['SMA_200'])
                #                 #     & (df['sma50_diff'] < df['sma200_diff'])
                #                 #     & (df['sma50_diff'] < 0)
                #                 #     & (df['sma200_diff'] < 0)
                #                 #     & (df.RSI_14 < 75)

                #                 # )
                #                 # & (
                #                 #     (
                #                 #         ((df.dx_diff < 0) & (df.DMN_14 > df.DMP_14) & (df.intersect == 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.high_diff > 0)           & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff > 0)
                #                 #     )
                #                 #         |
                #                 #     (
                #                 #         ((df.dx_diff < 0) & (dmn_set_2 > dmp_set_2) & (df.intersect > 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.high_diff > 0)           & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff > 0)
                #                 #     )
                #                 #         |
                #                 #     (
                #                 #         (df.dx_diff > 0) & (df.DMP_14 > df.DMN_14) #& (df.atr_p_diff < 0)# & (df.high_diff > 0) & (df.rsi_diff > 0) # & (df.obv_diff > 0) & (df.j_diff > 0) & (df['atr_AMATe_LR_8_21_2'] > 0)
                #                 #     )
                #                 # )
                #                 # & (
                #                 #     (df.vwma7_diff > 0)
                #                 # )
                #             )
                #             # &~(
                #             #     # (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_macd.shift(1) < 0) | (df.maxima_peak_x_rsi.shift(1) < 0)
                #             #     (df.maxima_peak_x_close < 0) | (df.maxima_peak_x_macd < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_dmp < 0) | (df.minima_peak_x_dmn < 0)
                #             # )

                #         , "stg5_long"] = 2 # without atr

                #     df.loc[
                #         # 추세 픽
                #         (
                #             ( # 진입전 추세 재확인 필요, atr14 증가하는지 확인 필요
                #                 (
                #                     # ((df['anomalies_close'] > 0) & ((df.maxima_peak_x_dx < 0) & (df.maxima_peak_x_atr14 < 0)) & (df.DX_14 > 40) & (df.DMP_14 > df.DMN_14))
                #                     # & ((df.maxima_peak_x_close < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_obv < 0) | (df.maxima_peak_x_macd < 0))


                #                     ((df['anomalies_close'] > 0) & (df['close'] > df['close'].quantile(0.5)) & (df.DX_14 > 40) & (df.DMP_14 > df.DMN_14) & (df.macdh_diff < 0))
                #                     & ((df.maxima_peak_x_close < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_obv < 0) | (df.maxima_peak_x_macd < 0))
                #                     & ((df.maxima_peak_x_dx < 0) | (df.maxima_peak_x_atr14 < 0))
                #                     # & (abs(df['DMP_14'] - df['DMN_14']) > 5)

                #                     # & (df['lowess'] < df['low'])
                #                     # & (df['lowess_diff'].shift(1) < 0) & (df['lowess_diff'] < 0)
                #                     & (df['lowess'] < df['close'])
                #                     # & (df['lowess_diff'] < 0)
                #                     & (df['lowess_1'] < df['close'])
                #                     # & (df['lowess_1_diff'] < 0)
                #                     & (df['lowess_MACD_12_26_9'] < df['close'])

                #                     # & ((df.dx_diff < 0))
                #                     # # & ((df.adx_diff > 0))
                #                     # & (df.DX_14 < df.ADX_14)
                                    
                #                     # & (df['ema200_diff'] < 0)
                #                     # & (df['EMA_200'] > df['EMA_50'])
                #                     # & (df['volume_change'].abs() > 200)



                #                     # ((df['AnomalyDetection_close'] > 0) & (df.maxima_peak_x_dx < 0) & (df.DX_14 > 40) & (df.DMP_14 > df.DMN_14))
                #                     # & ((df.maxima_peak_x_close < 0) | (df.maxima_peak_x_rsi < 0) | (df.maxima_peak_x_obv < 0) | (df.maxima_peak_x_macd < 0))
                #                     # # & ((df['anomalies_close'] > 0) | (df['anomalies_RSI_14'] > 0))                            
                #                 )
                #                 #     |

                #                 # ( # 진입전 추세 재확인 필요, atr14 증가하는지 확인 필요
                #                 #     ((df['maxima_peak_x_rsi'] < 0) | (df['maxima_peak_x_macd'] < 0)) #& (df['percentage_change2'] < -3) # & (df['macd_diff_200'] < df['macds_diff_200']) & (df['macd_diff_200'] < 0) & (df['macds_diff_200'] < 0) & (df['macd_diff_700'] < 0) & (df['macds_diff_700'] < 0) & (df['MACD_12_200_9'].shift(2) > df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] < 0) & (df['k_diff'] < 0) & (df['d_diff'] < 0)
                #                 #     # (df['rsi_percentage_change'] < -5) & (df['atr_percentage_change'] > 4) & (df['macd_percentage_change'] < -50) & (df['obv_percentage_change'] < -100) #& (df['atr_percentage_change'] > 4)
                #                 #     # & (df['obv_percentage_change'] > 150)
                #                 #     # & (df['obv_percentage_change'] < -50)
                #                 #     # & (df['macd_percentage_change'] < -5)
                #                 #     # & (df['rsi_percentage_change'] < -5)
                #                 #     & (df['volume_change'].abs() > 1000)
                #                 #     # & (df['volume_change'].abs() > 200)
                #                 #     # & (df['volume_change'].abs() < 300)
                #                 #     & (df['close'] > df['SMA_50'] )
                #                 #     & (df['SMA_50'] > df['SMA_200'] )
                #                 #     # (df['obv_percentage_change'].abs() < 50)
                #                 #     & (((df['sma50_diff'] > 0) & (df['sma200_diff'] > 0)))
                #                 #     # & (((df['sma50_diff'] < 0) & (df['sma200_diff'] < 0)))


                #                 #     # (df['maxima_peak_x_macd'].shift(2) < 0) & (df['macd_diff_200'] < df['macds_diff_200']) & (df['macd_diff_200'].shift(2) < 0) & (df['macds_diff_200'].shift(2) < 0) & (df['macd_diff_700'].shift(2) < 0) & (df['macds_diff_700'].shift(2) < 0) & (df['MACD_12_200_9'].shift(2) > df['MACD_12_200_9']) #& (df['adx_200_diff'].shift(2) < 0) & (df['atr200_diff'].shift(2) > 0) # # #& (df.ATRr_1 > df.close*1/100) # & (df['adx_200_diff'] > 0) #& (df['atr_diff'] < 0) #& (df['j_diff'] < 0) & (df['k_diff'] < 0) & (df['d_diff'] < 0)
                #                 # )





















                #                 #     |
                #                 # (
                #                 #     ((df['anomaly_type_feature1_n'] < 0) & (df['anomaly_type_close_n'] < 0))
                #                 #         |
                #                 #     ((df['anomaly_type_feature2_n'] < 0) & (df['anomaly_type_close_n'] < 0))
                #                 # )
                #             )
                #             # & ( 
                #             #     (df['SMA_50'] > df['SMA_200'])
                #             #     & (df['sma50_diff'] > df['sma200_diff'])
                #             #     & (df['sma50_diff'] > 0)
                #             #     & (df['sma200_diff'] > 0)
                #             #     & (df.RSI_14 > 25)
                #             # )
                #             # & (
                #             #     (
                #             #         ((df.dx_diff < 0) & (df.DMP_14 > df.DMN_14) & (df.intersect == 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.low_diff < 0)          & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff < 0)
                #             #     )
                #             #         |
                #             #     (
                #             #         ((df.dx_diff < 0) & (dmp_set_2 > dmn_set_2) & (df.intersect > 0) ) #& (df.atr_p_diff < 0)# & (df['atr_AMATe_LR_8_21_2'] > 0) #  & (df.low_diff < 0)          & (df['atr_AMATe_SR_8_21_2'] > 0)  & (df.rsi_diff < 0)
                #             #     )
                #             #         |
                #             #     (
                #             #         (df.dx_diff > 0) & (df.DMN_14 > df.DMP_14) #& (df.atr_p_diff < 0)#& (df.intersect == 0) & (df.low_diff < 0)  # & (df.obv_diff < 0) & (df.j_diff < 0) & (df['atr_AMATe_LR_8_21_2'] > 0)
                #             #     )
                #             # )
                #             # & (
                #             #     (df.vwma7_diff < 0)
                #             # )
                #         )
                #         # &~(
                #         #     # (df.minima_peak_x_macd > 0) | (df.minima_peak_x_rsi > 0) | (df.minima_peak_x_macd.shift(1) > 0) | (df.minima_peak_x_rsi.shift(1) > 0)
                #         #     (df.minima_peak_x_close > 0) | (df.minima_peak_x_macd > 0) | (df.minima_peak_x_rsi > 0) | (df.minima_peak_x_dmp > 0) | (df.maxima_peak_x_dmn > 0)
                #         # )

                #     , "stg5_short"] = -2 # without atr




                #     df.loc[
                #         # 추세 픽
                #             (
                #                 ( # 진입전 추세 재확인 필요, atr14 증가하는지 확인 필요
                #                     (
                #                         # (((df['anomaly_type_MACD_12_26_9_n'] > 0))) # | ((df['anomaly_type_RSI_14_n'] > 0)))
                #                         # (((df['anomaly_type_MACD_50_75_35_n'] > 0))) # | ((df['anomaly_type_RSI_14_n'] > 0)))


                #                         (((df['anomaly_type_close_n'] > 0))) # | ((df['anomaly_type_RSI_14_n'] > 0)))                            
                #                         & (df.bbb_diff < 0)
                #                         & (df['close'] > df['BBL_21_2.0'])

                #                         & (df['lowess'] > df['close'])
                #                         # & (df['lowess_diff'] > 0)
                #                         & (df['lowess_1'] > df['close'])
                #                         # & (df['lowess_1_diff'] > 0)

                #                         # & (df['wma7_diff'] > 0)
                #                         # & (df['ema200_diff'] < 0)
                #                         # & (df['EMA_200'] < df['EMA_50'])
                #                         # & (df.MACD_12_26_9 < 0) & (df.MACDh_12_26_9 < 0)


                #                         & ((df.dx_200_diff < 0))
                #                         & (df.DX_200 < df.ADX_200)
                                        
                #                         # & ((df.dx_diff < 0))
                #                         # # & ((df.adx_diff > 0))
                #                         # & (df.DX_14 < df.ADX_14)
                #                         # # & (abs(df['DMP_14'] - df['DMN_14']) > 5)
                                        
                #                         # & (df.feature1 < -0.2)
                #                         # & (df.feature1_diff > 0)
                #                         # & (df['ema50_diff'] > 0)
                #                         # & (df['volume_change'].abs() > 400)
                #                     )
                #                 )
                #             )

                #         , "stg32_long"] = 2 # without atr

                #     df.loc[
                #         # 추세 픽
                #             (
                #                 ( # 진입전 추세 재확인 필요, atr14 증가하는지 확인 필요
                #                     (
                #                         # (((df['anomaly_type_MACD_12_26_9_n'] < 0))) #  | ((df['anomaly_type_RSI_14_n'] < 0)))
                #                         # (((df['anomaly_type_MACD_50_75_35_n'] < 0))) #  | ((df['anomaly_type_RSI_14_n'] < 0)))
                #                         (((df['anomaly_type_close_n'] < 0))) #  | ((df['anomaly_type_RSI_14_n'] < 0)))
                #                         & (df.bbb_diff < 0)
                #                         & (df['close'] < df['BBU_21_2.0'])
                #                         # & (df['lowess'] < df['low'])
                #                         # & (df['lowess_diff'].shift(1) < 0) & (df['lowess_diff'] < 0)

                #                         & (df['lowess'] < df['close'])
                #                         # & (df['lowess_diff'] < 0)
                #                         & (df['lowess_1'] < df['close'])
                #                         # & (df['lowess_1_diff'] < 0)

                #                         # & (df['wma7_diff'] < 0)
                #                         # & (df['ema200_diff'] > 0)
                #                         # & (df['EMA_200'] > df['EMA_50'])
                #                         # & (df.MACD_12_26_9 > 0) & (df.MACDh_12_26_9 > 0)

                #                         & ((df.dx_200_diff < 0))
                #                         & (df.DX_200 < df.ADX_200)

                #                         # & ((df.dx_diff < 0))
                #                         # # & ((df.adx_diff > 0))
                #                         # & (df.DX_14 < df.ADX_14)
                #                         # # & (abs(df['DMP_14'] - df['DMN_14']) > 5)
                                        
                #                         # & (df.feature1 < -0.2)
                #                         # & (df.feature1_diff > 0)
                #                         # & (df['ema50_diff'] < 0)
                #                         # & (df['volume_change'].abs() > 400)
                #                     )
                #                 )
                #             )

                #     , "stg32_short"] = -2 # without atr




                #     # df.loc[
                #     #     (
                #     #         (df.stg1_long > 0) | ((df.stg2_long.shift(1) > 0) & (df.obv_diff > 0))
                #     #     )

                #     # , "stg3_long"] = 2 # without atr
                #     # df.loc[
                #     #     (
                #     #         (df.stg1_short < 0) | ((df.stg2_short.shift(1) < 0) & (df.obv_diff < 0))
                #     #     )

                #     # , "stg3_short"] = -2 # without atr


                #     # df.loc[
                #     #     (
                #     #         (df.stg7_long > 0)
                #     #     )

                #     # , "stg3_long"] = 2 # without atr
                #     # df.loc[
                #     #     (
                #     #         (df.stg7_short < 0)
                #     #     )

                #     # , "stg3_short"] = -2 # without atr



                #     df.loc[
                #         (
                #             (df.stg5_long > 0) | (df.stg32_long > 0)
                #         )

                #     , "stg10_long"] = 2 # without atr
                #     df.loc[
                #         (
                #             (df.stg5_short < 0) | (df.stg32_short < 0)
                #         )

                #     , "stg10_short"] = -2 # without atr


























                #     # 6 #################&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&###############################################################################################################################
                #     # 6 #################&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&###############################################################################################################################
                #     # 6 #################&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&###############################################################################################################################








                #     df.loc[
                #         (
                #             (df.stg23_long > 0) # | (df.stg55_long > 0) | (df.stg23_long > 0) | (df.stg32_long > 0) | (df.stg5_long > 0) # | (df.stg3_long > 0)
                #         )
                #         #& (df['atr_diff'] > 0)
                #         # & (df.feature1 < -0.2)
                #         # & (df.feature1 > 0)
                #         # & (df.feature1_diff > 0)

                #     , "stgT_long"] = 2 # without atr
                #     df.loc[
                #         (
                #             (df.stg23_short < 0) # | (df.stg55_short < 0) | (df.stg23_short < 0) | (df.stg32_short < 0) | (df.stg5_short < 0) # | (df.stg3_short < 0)
                #         )
                #         #& (df['atr_diff'] > 0)
                #         # & (df.feature1 < -0.2)
                #         # & (df.feature1 > 0)
                #         # & (df.feature1_diff > 0)

                #     , "stgT_short"] = -2 # without atr



                #     # df.loc[
                #     #     (
                #     #         (df.stg11_long > 0) | (df.stg55_long > 0) | (df.stg23_long > 0) | (df.stg32_long > 0) | (df.stg5_long > 0) # | (df.stg3_long > 0)
                #     #     )
                #     #     #& (df['atr_diff'] > 0)
                #     #     # & (df.feature1 < -0.2)
                #     #     # & (df.feature1 > 0)
                #     #     # & (df.feature1_diff > 0)

                #     # , "stgT_long"] = 2 # without atr
                #     # df.loc[
                #     #     (
                #     #         (df.stg11_short < 0) | (df.stg55_short < 0) | (df.stg23_short < 0) | (df.stg32_short < 0) | (df.stg5_short < 0) # | (df.stg3_short < 0)
                #     #     )
                #     #     #& (df['atr_diff'] > 0)
                #     #     # & (df.feature1 < -0.2)
                #     #     # & (df.feature1 > 0)
                #     #     # & (df.feature1_diff > 0)

                #     # , "stgT_short"] = -2 # without atr




                globals()['df_'+interval] = df

    intervals_3 = ['5m', '15m']
    smallest_interval = min_interval_finder(intervals_3, point_frame)
    common_long, common_short = multiple_frame_stg(intervals_3)
    update_stg_values(common_long, common_short, smallest_interval)

    interval_not_in_df_intv_ = ['15m', '30m', '1h', '2h', '4h']
    for intv_ in intervals:
        if intv_ not in interval_not_in_df_intv_:
            df_intv_ = globals()[f'df_{intv_}']
            if intv_ == '1m':
                df_intv_.loc[
                    (df_intv_['stg1_long'] > 0) | (df_intv_['stg3_long'] > 0),
                    'stgT_long'
                ] = 2  # without atr

                df_intv_.loc[
                    (df_intv_['stg1_short'] < 0) | (df_intv_['stg3_short'] < 0),
                    'stgT_short'
                ] = -2  # without atr

            else:
                df_intv_.loc[
                    (df_intv_['stg1_long'] > 0) | (df_intv_['stg2_long'] > 0) | (df_intv_['stg3_long'] > 0) | (df_intv_['stg10_long'] > 0),
                    'stgT_long'
                ] = 2  # without atr

                df_intv_.loc[
                    (df_intv_['stg1_short'] < 0) | (df_intv_['stg2_short'] < 0) | (df_intv_['stg3_short'] < 0) | (df_intv_['stg10_short'] < 0),
                    'stgT_short'
                ] = -2  # without atr
    return

def time_to_seconds_converter_cal(atr_time):
    if atr_time:
        if atr_time[-1] in x_to_seconds_frame:
            exit_order_waiting_seconds = int(atr_time.split(atr_time[-1])[0])*x_to_seconds_frame[atr_time[-1]]
        else:
            exit_order_waiting_seconds = 60
    else:
        exit_order_waiting_seconds = 60
    return exit_order_waiting_seconds

def point_sum_cal(stg_type, hmm_list, hmm_list_name, point_sum):
    #if stg_type == 'stg1':
        #point_sum = 0
    for i in hmm_list:
        if i[-1] in point_frame:
            point_single = int(i.split(i[-1])[0])*point_frame[i[-1]]
        else:
            point_single = 0
        if 'plus' in hmm_list_name:
            point_sum += point_single
        else: # 'minus' in hmm_list_name:
            point_sum -= point_single
    return point_sum

def hmm_list_dict_maker(hmm_list): # 가장큰 리스트 절대값으로 비교
    point_sum = 0
    for i in hmm_list:
        if i[-1] in point_frame:
            point_single = int(i.split(i[-1])[0])*point_frame[i[-1]]
        else:
            point_single = 0
        point_sum += point_single
    return point_sum

def biggest_hmm_list_finder(biggest_hmm_list_dict):
    return max(biggest_hmm_list_dict.items(), key=operator.itemgetter(1))[0]

def sum_of_df_values_to_list(values_to_list):
    point_sum = 0
    for i in values_to_list:
        if i.split('_')[0][-1] in point_frame:
            point_single = int(i.split('_')[0].split(i.split('_')[0][-1])[0])*point_frame[i.split('_')[0][-1]]
        else:
            point_single = 0
        if 'long' in i.split('_')[1]:
            point_sum += point_single
        else: # 'minus' in hmm_list_name:
            point_sum -= point_single
    return point_sum

def peaker_frame_appender(interval, side_key, strategy_value):
    #print('&&&&&& appender &&&&&&&&  appender &&&&&&')
    #print(interval+'_'+side_key+'_'+strategy_value)
    #print(globals()['maxima_minima_current_peak_time_'+interval+'_'+side_key+'_'+strategy_value])

    new_peak = {'peaker_pk':interval+'_'+side_key+'_'+strategy_value, 'peaked_time':datetime.datetime.now().replace(microsecond=0), 'open_time':globals()['maxima_minima_current_peak_time_'+interval+'_'+side_key+'_'+strategy_value]}
    # globals()['peaker_frame'] = globals()['peaker_frame'].append(new_peak, ignore_index = True)
    globals()['peaker_frame'] = pd.concat([globals()['peaker_frame'], pd.DataFrame([new_peak])], ignore_index=True)
    
    #print('&&&&&& appender &&&&&&&&  appender &&&&&&')
    return

def peaker_frame_dropper(interval, side_key, strategy_value):
    #print('@@@@@@@@ dropper @@@@@@@@  dropper @@@@@@@@')
    #print(interval+'_'+side_key+'_'+strategy_value)
    #print('비교2:', globals()['maxima_minima_last_peak_time_'+interval+'_'+side_key+'_'+strategy_value])
    #print(globals()['peaker_frame'][(globals()['peaker_frame']['peaker_pk'] == interval+'_'+side_key+'_'+strategy_value) & (globals()['peaker_frame']['close_time'] == globals()['maxima_minima_last_peak_time_'+interval+'_'+side_key+'_'+strategy_value])])
    #print('i:', globals()['peaker_frame'][(globals()['peaker_frame']['peaker_pk']==interval+'_'+side_key+'_'+strategy_value) & (globals()['peaker_frame']['close_time']== globals()['maxima_minima_last_peak_time_'+interval+'_'+side_key+'_'+strategy_value])].index)

    i = globals()['peaker_frame'][(globals()['peaker_frame']['peaker_pk']==interval+'_'+side_key+'_'+strategy_value) & (globals()['peaker_frame']['open_time']== globals()['maxima_minima_last_peak_time_'+interval+'_'+side_key+'_'+strategy_value])].index
    globals()['peaker_frame'].drop(labels=i, axis=0, inplace=True)
    #print('@@@@@@@@ dropper @@@@@@@@  dropper @@@@@@@@')
    return

def continuous_sequence_finder(random_intervals):
    # Finds the largest number in the second list that forms a continuous sequence in the first list.
    full_intervals = globals()['valid_intervals'] # valid_intervals
    micro_interval_pick = ''
    max_length = 0

    for start_index in range(len(full_intervals)):
        for end_index in range(start_index + 1, len(full_intervals) + 1):
            full_intervals_sequence = full_intervals[start_index:end_index]
            if full_intervals_sequence == random_intervals[:len(full_intervals_sequence)]:
                full_intervals_sequence_length = len(full_intervals_sequence)
                if full_intervals_sequence_length > max_length:
                    max_length = full_intervals_sequence_length
                    micro_interval_pick = full_intervals_sequence[-1]  # Select the last element from the full_intervals_sequence
    return micro_interval_pick

def confirmer():
    for side_key, direction, strategy_value in itertools.product(side.keys(), directions, strategy): # 변수 최초 생성
        default_list_name = 'hmm_' + side_key + '_' + direction + '_' + strategy_value
        locals()[default_list_name] = []

    atr_look_up_point_single_list = []
    atr_dicts = {}
    stg_type_orgin = stg_type = '' # default, stg1, stg5
    peaker_side = ''
    peaker_option = ''
    success = 0
    point_sum =0
    divergence_name = ''

    total_length = 0
    max_priority = float('-inf')
    max_dicts = {}
    dicts = {}
    forward_lists_updated = False
    max_sum = 0
    max_list_name = None
    max_list_value = []
    scalping_indicator_confirmer = 0
    scalping_switch = ''
    scalping_direction_pick = None

    # stg1_long_exist = 0
    # stg1_short_exist = 0
    # stg1_and_stg10_at_the_sametime = 0
    # stg3_long_exist = 0
    # stg3_short_exist = 0
    # stg3_and_stg10_at_the_sametime = 0

    pick_time =  ''
    close_price_high = 0
    close_price_low = 0

    peak_calc(market_id, intervals) # current_peak_time 업데이트!

    ############################################# back_tester #######################################################################################################
    #globals()['interval_not_in_stg0'] = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '8h', '12h', '1d', '3d']
    #globals()['interval_not_in_stg1'] = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '8h', '12h', '1d', '3d']
    #globals()['interval_not_in_stg0'] = ['1m', '3m', '5m', '15m', '2h', '8h', '12h', '1d', '3d'] # > 즉, 만 ['30m', '1h', '4h', '6h'] 포함
    #globals()['interval_not_in_stg1'] = ['1m', '3m', '5m', '2h', '8h', '12h', '1d', '3d'] # > 즉, 만 ['15m', '30m', '1h', '4h', '6h'] 포함

    #globals()['interval_not_in_stg0'] = ['1m', '3m', '15m', '30m', '2h', '4h', '6h', '8h', '12h', '1d', '3d'] # => single timeframe, 5m, 1h
    #globals()['interval_not_in_stg1'] = ['1m', '3m', '15m', '30m', '2h', '4h', '6h', '8h', '12h', '1d', '3d'] # => single timeframe, 5m, 1h
    #globals()['interval_not_in_stg0'] = ['1m', '3m', '15m', '30m', '2h', '4h', '6h', '8h', '12h', '1d', '3d'] # => single timeframe, 5m, 1h
    #globals()['interval_not_in_stg1'] = ['1m', '3m', '15m', '30m', '2h', '4h', '6h', '8h', '12h', '1d', '3d'] # => single timeframe, 5m, 1h
    
    globals()['interval_not_in_stg1'] = ['1m', '3m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d'] # '1m', '5m', '15m'
    # globals()['interval_not_in_stg2'] = ['1m', '3m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d']
    # globals()['interval_not_in_stg3'] = ['3m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d'] # '1m', '5m', '15m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d'
    globals()['interval_not_in_stg2'] = ['1m', '3m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d'] # '1m', '5m', '15m'
    # globals()['interval_not_in_stg3'] =  ['1m', '3m', '5m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d']
    globals()['interval_not_in_stg3'] =  ['3m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d'] # '5m', '15m'
    globals()['interval_not_in_stg10'] = ['3m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d'] # '5m', '15m'
    globals()['interval_not_in_stg110'] = ['3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d'] # '1m'
    globals()['interval_not_in_stg_scalping_0'] = ['1m', '3m', '5m', '15m', '30m']
    ############################################# back_tester #######################################################################################################

    # if exchange_id in exchange_id_list_df_1h:
    #     if  globals()['df_1h']['atr_AMATe_LR_8_21_2'][-1]: # 변동성 있으면 민감하므로 '5m', '15m' interval 포함, 너무잦은 픽은 손절주의로 미포함, 장투 '4h', '6h' 포함
    #         ############################################# back_tester #######################################################################################################
    #         #globals()['interval_not_in_stg0'] = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '8h', '12h', '1d', '3d']
    #         #globals()['interval_not_in_stg1'] = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '8h', '12h', '1d', '3d']
    #         #globals()['interval_not_in_stg0'] = ['1m', '3m', '5m', '15m', '2h', '8h', '12h', '1d', '3d'] # > 즉, 만 ['30m', '1h', '4h', '6h'] 포함
    #         #globals()['interval_not_in_stg1'] = ['1m', '3m', '5m', '2h', '8h', '12h', '1d', '3d'] # > 즉, 만 ['15m', '30m', '1h', '4h', '6h'] 포함

    #         #globals()['interval_not_in_stg0'] = ['1m', '3m', '15m', '30m', '2h', '4h', '6h', '8h', '12h', '1d', '3d'] # => single timeframe, 5m, 1h
    #         #globals()['interval_not_in_stg1'] = ['1m', '3m', '15m', '30m', '2h', '4h', '6h', '8h', '12h', '1d', '3d'] # => single timeframe, 5m, 1h
    #         #globals()['interval_not_in_stg0'] = ['1m', '3m', '15m', '30m', '2h', '4h', '6h', '8h', '12h', '1d', '3d'] # => single timeframe, 5m, 1h
    #         #globals()['interval_not_in_stg1'] = ['1m', '3m', '15m', '30m', '2h', '4h', '6h', '8h', '12h', '1d', '3d'] # => single timeframe, 5m, 1h
            
    #         globals()['interval_not_in_stg1'] = ['1m', '3m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d']
    #         globals()['interval_not_in_stg2'] = ['1m', '3m', '5m', '15m', '30m', '3d']
    #         # globals()['interval_not_in_stg3'] = ['1m', '3m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d']
    #         globals()['interval_not_in_stg3'] = ['15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d']
    #         globals()['interval_not_in_stg10'] = ['15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d']
    #         globals()['interval_not_in_stg_scalping_0'] = ['1m', '3m', '5m', '15m', '30m']
    #         ############################################# back_tester #######################################################################################################
    #     else: # 변동성 없으면 
    #         #globals()['interval_not_in_stg0'] = ['1m', '3m', '8h', '12h', '1d', '3d'] # > 즉, 만 ['5m', '15m', '30m', '1h', '2h', '4h', '6h'] 포함
    #         #globals()['interval_not_in_stg1'] = ['1m', '3m', '8h', '12h', '1d', '3d'] # > 즉, 만 ['5m', '15m', '30m', '1h', '2h', '4h', '6h'] 포함

    #         #globals()['interval_not_in_stg0'] = ['3m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d'] # > 즉, 만 ['1m', '5m'] 포함
    #         #globals()['interval_not_in_stg1'] = ['3m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d'] # > 즉, 만 ['1m', '5m'] 포함
    #         globals()['interval_not_in_stg1'] = ['1m', '3m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d']
    #         globals()['interval_not_in_stg2'] = ['1m', '3m', '5m', '15m', '30m', '3d']
    #         # globals()['interval_not_in_stg3'] = ['1m', '3m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d']
    #         globals()['interval_not_in_stg3'] = ['15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d']
    #         globals()['interval_not_in_stg10'] = ['15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d']
    #         #globals()['interval_not_in_stg0'] = []
    #         #globals()['interval_not_in_stg1'] = []
    #         globals()['interval_not_in_stg_scalping_0'] = ['1m', '3m', '5m', '15m', '30m']
    # else:
    #     if  globals()['df_15m']['atr_AMATe_LR_8_21_2'][-1]: # 변동성 있으면 민감하므로 '1m', '5m', '15m' interval 포함, 너무잦은 픽은 손절주의로 미포함, 장투 '4h', '6h' 포함
    #         ############################################# back_tester #######################################################################################################
    #         #globals()['interval_not_in_stg0'] = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '8h', '12h', '1d', '3d']
    #         #globals()['interval_not_in_stg1'] = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '8h', '12h', '1d', '3d']
    #         #globals()['interval_not_in_stg0'] = ['3m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d'] # > 즉, 만 ['1m', '5m', '15m', '30m'] 포함
    #         #globals()['interval_not_in_stg1'] = ['1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d'] # > 즉, 만 ['1m', '3m', '5m', '15m', '30m'] 포함
    #         globals()['interval_not_in_stg1'] = ['1m', '3m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d']
    #         globals()['interval_not_in_stg2'] = ['1m', '3m', '5m', '15m', '30m', '3d']
    #         # globals()['interval_not_in_stg3'] = ['1m', '3m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d']
    #         globals()['interval_not_in_stg3'] = ['15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d']
    #         globals()['interval_not_in_stg10'] = ['15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d']
    #         globals()['interval_not_in_stg_scalping_0'] = ['1m', '3m', '5m', '15m', '30m']

    #         #globals()['interval_not_in_stg0'] = [] # => single timeframe, 5m, 15m, 30m, 1h
    #         #globals()['interval_not_in_stg1'] = [] # => single timeframe, 5m, 15m, 30m, 1h
    #         ############################################# back_tester #######################################################################################################
    #     else: # 변동성 없으면 
    #         #globals()['interval_not_in_stg0'] = ['3m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d'] # > 즉, 만 ['1m', '5m'] 포함
    #         #globals()['interval_not_in_stg1'] = ['3m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d'] # > 즉, 만 ['1m', '5m'] 포함
    #         globals()['interval_not_in_stg1'] = ['1m', '3m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d']
    #         globals()['interval_not_in_stg2'] = ['1m', '3m', '5m', '15m', '30m', '3d']
    #         # globals()['interval_not_in_stg3'] = ['1m', '3m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d']
    #         globals()['interval_not_in_stg3'] = ['15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d']
    #         globals()['interval_not_in_stg10'] = ['15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d']
    #         globals()['interval_not_in_stg_scalping_0'] = ['1m', '3m', '5m', '15m', '30m']
    #         #globals()['interval_not_in_stg0'] = ['1m', '3m'] # => single timeframe, 5m, 15m, 30m, 1h
    #         #globals()['interval_not_in_stg1'] = ['1m'] # => single timeframe, 5m, 15m, 30m, 1h
    #         #globals()['interval_not_in_stg0'] = []
    #         #globals()['interval_not_in_stg1'] = []













































    # if len(globals()['valid_intervals']) == 0:
    #     globals()['valid_intervals'] = intervals

    # stg pick 결정 용도
    # hmm_long_reverse_stg5 리스트 값 신규 업데이트
    for interval_tmp, side_key, strategy_value in itertools.product(globals()['valid_intervals'], side.keys(), strategy):
        if interval_tmp not in globals()['interval_not_in_' + strategy_value]:
            column_name = strategy_value + '_' + side_key
            # Dataframe 에서 가장 마지막 peak 추출하여 current_peak, last_peak변수에 저장
            variable_name_current_peak = 'maxima_minima_current_peak_time_' + interval_tmp + '_' + side_key + '_' + strategy_value
            variable_name_last_peak = 'maxima_minima_last_peak_time_' + interval_tmp + '_' + side_key + '_' + strategy_value
            # print(interval_tmp)
            # print(f'변경전 : {side_key}_L: {globals()[variable_name_last_peak]}')   
            # print(f'변경전 : {side_key}_C: {globals()[variable_name_current_peak]} \n')   

            if column_name in globals()['df_' + interval_tmp].columns:
                df_tmp = globals()['df_' + interval_tmp]
                threshold = globals()['counter_light_weight_' + side_key]
                mask = (df_tmp[column_name] < threshold) if side_key == 'short' else (df_tmp[column_name] > threshold)
                globals()[variable_name_current_peak] = df_tmp.peak_time[mask][-1] if mask.any() else default_time # Dataframe 에서 가장 마지막 peak 추출하여 current_peak 지속 업데이트, 만약 일시적으로 peak 있다 사라진 경우 다시 초기화 값 입력
                #print(interval_tmp, side_key, strategy_value, globals()[variable_name_last_peak], globals()[variable_name_current_peak])
                ################################################################################
                ################################################################################
                ################################################################################
                # 원복 필요
                # print(globals()['peaker_frame'])
                if globals()[variable_name_current_peak] != globals()[variable_name_last_peak]:
                # if (globals()[variable_name_current_peak] != globals()[variable_name_last_peak]) or (globals()[variable_name_current_peak] == globals()[variable_name_last_peak]):
                    # print('\n 마지막거 아닌거?', datetime.datetime.now().replace(microsecond=0))
                    # print(interval_tmp, side_key, strategy_value, globals()[variable_name_last_peak], globals()[variable_name_current_peak])
                    # hmm_ 에 넣어서 strategy 계산하기 전에, 현재 픽인지, 과거 픽이 지금 뜬건지 확인 후, 과거 픽이라면 disregard 할 것.
                    # 현재 픽인지 과거 픽인지 확인.
                    ########### stg10 은 findpeaks 실시간 픽 사용 시, stg3 은 find_peaks 실시간 픽 사용 아닐시
                    # if (strategy_value == 'stg10' and globals()[variable_name_current_peak] == df_tmp['open_time2'].iloc[-1]) or \
                    # (strategy_value == 'stg3' and globals()[variable_name_current_peak] in df_tmp['open_time2'].iloc[-2:]):
                    ########### stg3, stg10 둘다 findpeaks 실시간 픽 사용 시
                    # if (globals()[variable_name_current_peak] in df_tmp['open_time2'].iloc[-1:]):

                    # print('[prev]:-----------------------------')
                    # print(globals()['peaker_frame'])
                    if globals()[variable_name_current_peak] > globals()[variable_name_last_peak]:  # forward
                        if (
                            ((strategy_value in ['stg110']) and (globals()[variable_name_current_peak] == df_tmp['open_time2'].iloc[-1])) or
                            # ((strategy_value in ['stg1']) and (globals()[variable_name_current_peak] in df_tmp['open_time2'].iloc[-1:])) or
                            # ((strategy_value in ['stg1']) and (interval_tmp in ['1m']) and (globals()[variable_name_current_peak] in df_tmp['open_time2'].iloc[-3:])) or
                            # ((strategy_value in ['stg3', 'stg9']) and (globals()[variable_name_current_peak] in df_tmp['open_time2'].iloc[-4:])) or
                            # ((strategy_value in ['stg2']) and (globals()[variable_name_current_peak] in df_tmp['open_time2'].iloc[-6:])) or
                            ((strategy_value in ['stg1', 'stg2', 'stg3', 'stg9', 'stg10', 'stg100']) and (globals()[variable_name_current_peak] in df_tmp['open_time2'].iloc[-2:]))
                        ):
                            locals()['hmm_' + side_key + '_forward_' + strategy_value].append(interval_tmp)
                            # print('[prev]:&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&---------')
                            # print('[prev]:&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&---------')
                            # print('[prev]:&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&---------')
                            # print('[prev]:&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&---------')
                            # print('[prev]:&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&---------')
                            # print('[prev]:&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&---------')
                        
                        peaker_frame_appender(interval_tmp, side_key, strategy_value)
                        # if strategy_value == 'stg1':
                        #     if side_key == 'long':
                        #         stg1_long_exist = 1
                        #     elif side_key == 'short':
                        #         stg1_short_exist = 1
                        # if strategy_value == 'stg3':
                        #     if side_key == 'long':
                        #         stg3_long_exist = 1
                        #     elif side_key == 'short':
                        #         stg3_short_exist = 1
                    
                    else: # reverse
                        locals()['hmm_' + side_key + '_reverse_' + strategy_value].append(interval_tmp)
                        peaker_frame_dropper(interval_tmp, side_key, strategy_value)
                    # print('\n\n')
                    # print('[next]:-----------------------------')
                    # print(globals()['peaker_frame'])
                    



                    globals()[variable_name_last_peak] = globals()[variable_name_current_peak]
                





    # value 가 있는 리스트 존재 확인
    for side_key, direction, strategy_value in itertools.product(side.keys(), directions, strategy):
        list_name = 'hmm_' + side_key + '_' + direction + '_' + strategy_value
        list_value = locals().get(list_name, [])
        if list_value:  # Only consider lists that have values
            total_length += 1
    
    if total_length > 0: # 만약 value 가 있는 리스트 존재시 아래시작
        # pick frame filter 시작!
        # 값 있는 forward list 선택, forward list 없을경우 reverse list 선택, priority: forward > reverse

        for direction in directions:
            if forward_lists_updated and direction == 'reverse':
                break  # Skip further checks for 'reverse' lists
            for side_key in side.keys():
                for strategy_value in strategy:
                    list_name = 'hmm_' + side_key + '_' + direction + '_' + strategy_value
                    list_value = locals().get(list_name, [])
                    if list_value:  # Only consider lists that have values
                        dicts.update({list_name: list_value})
                        if strategy_value in ['stg3', 'stg10'] and len(list_value) > 1:  # Additional condition: list_value > 2
                            dicts.update({list_name.split('stg')[0] + 'stg' + str(int(list_name.split('stg')[-1])**2): list_value})                        
                        # if strategy_value == 'stg2' and len(list_value) > 1:  # Additional condition: list_value > 2
                        #     dicts.update({list_name.split('stg')[0] + 'stg' + str(int(list_name.split('stg')[-1])**2): list_value})
                        # elif strategy_value == 'stg3' and list_name and len(list_value) > 1:  # Additional condition: list_value > 2
                        #     dicts.update({list_name.split('stg')[0] + 'stg' + str(int(list_name.split('stg')[-1])**2): list_value})
                        if direction == 'forward':
                            forward_lists_updated = True

        # stg 선택, priority: stg0 < stg1 < stg5
        for side_key in side.keys():
            for dct in dicts:
                priority_str = dct.split('stg')[-1]
                priority = int(priority_str)
                if priority > max_priority:
                    max_priority = priority
                    max_dicts = {dct: dicts[dct]}
                elif priority == max_priority:
                    max_dicts.update({dct: dicts[dct]})

        # long, short 선택, priority: long == short
        for list_name, list_value in max_dicts.items():
            sum_intervals = sum([int(interval[:-1]) * point_frame.get(interval[-1], 0) for interval in list_value])
            #print(list_name, list_value, sum_intervals)
            if sum_intervals > max_sum:
                max_list_name = list_name
                max_list_value = list_value
                max_sum = sum_intervals
        #print(max_list_name, max_list_value, max_sum)
        # output: ('hmm_long_reverse_stg5', ['15m', '3m'], 18)


        # return 값 결정
        # if peaker_option == 'forward':
        # success = 1
        
        stg_type = stg_type_ = stg_type_orgin = max_list_name.split('_')[3]
        # if stg_type in ['stg1', 'stg2', 'stg3']:
        # if stg_type in ['stg1']:
        #     success = 0 # stg1, stg2 은 macd 지표만사용하며 단타용도임
        #     scalping_indicator_confirmer = 1
        point_sum = max_sum
        peaker_side = max_list_name.split('_')[1]
        peaker_option = max_list_name.split('_')[2]
        for i in max_list_value: # atr_look_up_list
            if i[-1] in point_frame:
                point_single = int(i.split(i[-1])[0])*point_frame[i[-1]]
                atr_look_up_point_single_list.append(point_single)
                atr_dicts[i] = point_single
        atr_max_value = list(atr_dicts.values())
        atr_max_key= list(atr_dicts.keys())

        if stg_type_ == 'stg9':
            stg_type_orgin = 'stg3'
        elif stg_type_ == 'stg100':
            stg_type_orgin = 'stg10'

        if peaker_option == 'reverse':
            globals()['atr_pick'] = ''
            globals()['atr_given'] = 0
            globals()['adx_given'] = 0
            success = 0
        else:
            globals()['atr_pick'] = atr_max_key[atr_max_value.index(max(atr_max_value))]
            globals()['atr_given'] = float(exchange.price_to_precision(market_id, globals()['df_' + globals()['atr_pick']]['ATRr_14'][-1]))
            globals()['adx_given'] = float(globals()['df_' + globals()['atr_pick']]['ADX_14'][-1])



            # 실시간 peack
            # if stg_type in ['stg3', 'stg9', 'stg10', 'stg100'] and (globals()["maxima_minima_last_peak_time_" + atr_max_key[atr_max_value.index(max(atr_max_value))] + "_" + peaker_side + "_" + stg_type_] == globals()["df_" + atr_max_key[atr_max_value.index(max(atr_max_value))]]["open_time2"].iloc[-3]):
            # if stg_type in ['stg1'] and (globals()['atr_pick'] in ['1m']) and (globals()["maxima_minima_last_peak_time_" + atr_max_key[atr_max_value.index(max(atr_max_value))] + "_" + peaker_side + "_" + stg_type_orgin] in globals()["df_" + atr_max_key[atr_max_value.index(max(atr_max_value))]]["open_time2"].iloc[-3:]):
            #     success = 1
          

            # elif stg_type in ['stg1'] and (globals()["maxima_minima_last_peak_time_" + atr_max_key[atr_max_value.index(max(atr_max_value))] + "_" + peaker_side + "_" + stg_type_orgin] in globals()["df_" + atr_max_key[atr_max_value.index(max(atr_max_value))]]["open_time2"].iloc[-1:]):
            #     success = 1
            # elif stg_type in ['stg2'] and (globals()["maxima_minima_last_peak_time_" + atr_max_key[atr_max_value.index(max(atr_max_value))] + "_" + peaker_side + "_" + stg_type_] == globals()["df_" + atr_max_key[atr_max_value.index(max(atr_max_value))]]["open_time2"].iloc[-1]):
            if stg_type in ['stg110'] and (globals()["maxima_minima_last_peak_time_" + atr_max_key[atr_max_value.index(max(atr_max_value))] + "_" + peaker_side + "_" + stg_type_orgin] == globals()["df_" + atr_max_key[atr_max_value.index(max(atr_max_value))]]["open_time2"].iloc[-1]):
                success = 1
            # elif stg_type in ['stg3', 'stg9'] and (globals()["maxima_minima_last_peak_time_" + atr_max_key[atr_max_value.index(max(atr_max_value))] + "_" + peaker_side + "_" + stg_type_orgin] in globals()["df_" + atr_max_key[atr_max_value.index(max(atr_max_value))]]["open_time2"].iloc[-4:]):
            #     success = 1
            # elif stg_type in ['stg2'] and (globals()["maxima_minima_last_peak_time_" + atr_max_key[atr_max_value.index(max(atr_max_value))] + "_" + peaker_side + "_" + stg_type_orgin] in globals()["df_" + atr_max_key[atr_max_value.index(max(atr_max_value))]]["open_time2"].iloc[-6:]):
            #     success = 1
            elif stg_type in ['stg1', 'stg2', 'stg3', 'stg9', 'stg10', 'stg100'] and (globals()["maxima_minima_last_peak_time_" + atr_max_key[atr_max_value.index(max(atr_max_value))] + "_" + peaker_side + "_" + stg_type_orgin] in globals()["df_" + atr_max_key[atr_max_value.index(max(atr_max_value))]]["open_time2"].iloc[-2:]):
                success = 1
            # if stg_type in ['stg3', 'stg9', 'stg10', 'stg100'] and (globals()["maxima_minima_last_peak_time_" + atr_max_key[atr_max_value.index(max(atr_max_value))] + "_" + peaker_side + "_" + stg_type_] == globals()["df_" + atr_max_key[atr_max_value.index(max(atr_max_value))]]["open_time2"].iloc[-3]):
            else:
                success = 0


        print('\n')
        print(f'[success : {success}]')
        print(f'[{stg_type}, {peaker_side} {peaker_option}, {atr_max_key[atr_max_value.index(max(atr_max_value))]}]')
        print(f'[{peaker_side}_last pick 시간: {globals()["maxima_minima_last_peak_time_" + atr_max_key[atr_max_value.index(max(atr_max_value))] + "_" + peaker_side + "_" + stg_type_orgin]}, 현재시간: {globals()["df_" + atr_max_key[atr_max_value.index(max(atr_max_value))]]["open_time2"].iloc[-1]}]')



    # if scalping_indicator_confirmer:
    #     formatted_datetime = datetime.datetime.now().replace(microsecond=0)
    #     print('\n', formatted_datetime)
    #     print(f"scalping_indicator_confirmer: [{scalping_indicator_confirmer}], success: [{success}], [{peaker_side}], [{peaker_option}] [{stg_type}], picked_list: [{max_list_name}], picked_interval: {max_list_value}, point_sum: [{point_sum}점], atr_pick: [{str(globals()['atr_pick'])}], atr_given: [{str(globals()['atr_given'])}]")

    if success:
        if stg_type in ['stg1', 'stg2', 'stg3', 'stg9', 'stg10', 'stg100', 'stg110'] and peaker_option == 'forward':
            divergence_name = ''
            # print('-------------------------------')
            formatted_datetime = datetime.datetime.now().replace(microsecond=0)
            c_p = ticker_calc(market_id)[1]
            # print('current_time: ', formatted_datetime)
            # print(f"success: [{success}], [{peaker_side}], [{peaker_option}] [{stg_type}], picked_list: [{max_list_name}], picked_interval: {max_list_value}, point_sum: [{point_sum}점], atr_pick: [{str(globals()['atr_pick'])}], atr_given: [{str(globals()['atr_given'])}]")
            if globals()['atr_pick'] != '' and peaker_option == 'forward':
                if stg_type == 'stg9':
                    stg_type_orgin = 'stg3'
                elif stg_type == 'stg100':
                    stg_type_orgin = 'stg10'
                else:
                    stg_type_orgin = stg_type
                pick_time = globals()['df_' + globals()['atr_pick']].loc[(globals()['df_' + globals()['atr_pick']][stg_type_orgin + '_' + peaker_side] != 0), 'open_time2'].iloc[-1]
                close_price_high = globals()['df_' + globals()['atr_pick']].loc[(globals()['df_' + globals()['atr_pick']][stg_type_orgin + '_' + peaker_side] != 0), 'high'].iloc[-1]   
                close_price_low = globals()['df_' + globals()['atr_pick']].loc[(globals()['df_' + globals()['atr_pick']][stg_type_orgin + '_' + peaker_side] != 0), 'low'].iloc[-1]
                is_divergence_point_2 = globals()['df_' + globals()['atr_pick']].loc[(globals()['df_' + globals()['atr_pick']][stg_type_orgin + '_' + peaker_side] != 0), 'divergence_point_2'].iloc[-1]
                is_divergence_name = globals()['df_' + globals()['atr_pick']].loc[(globals()['df_' + globals()['atr_pick']][stg_type_orgin + '_' + peaker_side] != 0), 'divergence_name'].iloc[-1]
                if is_divergence_point_2 != 0:
                    divergence_name = is_divergence_name                
                # print('pick_time: ', pick_time, ', current_time: ', (formatted_datetime), ', [calc_time: ', formatted_datetime - pick_time, ']')
                # if peaker_side == 'long':
                #     print('close_price_low: ', close_price_low, ', current_price: ', c_p, ', [calc_price: ', float(c_p) - float(close_price_low), ', perct: ', ((float(c_p) - float(close_price_low)) * 100) / float(c_p), ']')
                # elif peaker_side == 'short':
                #     print('close_price_high: ', close_price_high, ', current_price: ', c_p, ', [calc_price: ', float(close_price_high) - float(c_p), ', perct: ', ((float(close_price_high) - float(c_p)) * 100) / float(c_p), ']')

            # for side_key, direction, strategy_value in itertools.product(side.keys(), directions, strategy):
            #     default_list_name = 'hmm_' + side_key + '_' + direction + '_' + strategy_value
            #     print(default_list_name, locals()[default_list_name])
            # print(dicts)
            # print(globals()['peaker_frame'])


        # 다이버사용 원할시 아래구문 주석 해제
        # divergence_name = ''
        # if is_divergence_point_2 != 0:
        #     divergence_name = is_divergence_name
        # if stg_type in ['stg3', 'stg9'] and peaker_option == 'forward' and :
        #     divergence_name = globals()['df_' + globals()['atr_pick']].loc[(globals()['df_' + globals()['atr_pick']].divergence_point_2 != 0), 'divergence_name'].iloc[-1]
        


            # if (peaker_side == 'long' and stg1_long_exist == 1) or (peaker_side == 'short' and stg1_short_exist == 1):
            #     stg1_and_stg10_at_the_sametime = 1            
            # if (peaker_side == 'long' and stg3_long_exist == 1) or (peaker_side == 'short' and stg3_short_exist == 1):
            #     stg3_and_stg10_at_the_sametime = 1

        #     min_close_time = globals()['df_' + globals()['atr_pick']].loc[(globals()['df_' + globals()['atr_pick']].divergence_point_2 > 0), 'close_time'].iloc[-1]
        #     max_close_time = globals()['df_' + globals()['atr_pick']].loc[(globals()['df_' + globals()['atr_pick']].divergence_point_2 < 0), 'close_time'].iloc[-1]

        #     print('----------------')
        #     print(stg_type, side_key, peaker_option)
        #     print('divergence_name: ', divergence_name)
        #     print('timeframe: ', globals()['atr_pick'])
        #     diver_side = str(divergence_name.split('_')[-1])
        #     if diver_side == 'uptrend':
        #         print('min_close_time: ', min_close_time)
        #     else:
        #         print('max_close_time: ', max_close_time)
        #     print('current time: ', formatted_datetime)




    # 단타용 추세 결정 용도
    # 전체 trend 검토
    trend_directions = []
    long_trend_intervals = []
    short_trend_intervals = []
    # print('------------------')
    # print('scalping_list 미들: ', end='')
    for interval_tmp, side_key, strategy_scalping_value in itertools.product(globals()['valid_intervals'], side.keys(), strategy_scalping): # ['stg_scalping_0']
        if interval_tmp not in globals()['interval_not_in_' + strategy_scalping_value]:
            column_name = strategy_scalping_value + '_' + side_key
            variable_name_current_peak = 'maxima_minima_current_peak_time_' + interval_tmp + '_' + side_key + '_' + strategy_scalping_value
            variable_name_last_peak = 'maxima_minima_last_peak_time_' + interval_tmp + '_' + side_key + '_' + strategy_scalping_value
            if column_name in globals()['df_' + interval_tmp].columns:
                df_tmp = globals()['df_' + interval_tmp]
                threshold = globals()['counter_light_weight_' + side_key]
                mask = (df_tmp[column_name][-1] < threshold) if side_key == 'short' else (df_tmp[column_name][-1] > threshold)
                # print(mask)
                if mask:
                    # print(interval_tmp + '_' + side_key + ', ', end='')
                    trend_directions.append(side_key)
                    if side_key == 'long':
                        long_trend_intervals.append(interval_tmp)
                    elif side_key == 'short':
                        short_trend_intervals.append(interval_tmp)

    # macro trend 산출. Direction determination logic 큰그림 단순 비교, 롱이많은지 숏이많은지 비교, trend_macro_state
    direction_weights = {"long": 1, "short": -1}
    cumulative_sum = sum(direction_weights.get(direction, 0) for direction in trend_directions)
    if cumulative_sum > 0:
        globals()['trend_macro_state'] = 'long'
    elif cumulative_sum < 0:
        globals()['trend_macro_state'] = 'short'
    else:
        globals()['trend_macro_state'] = 'neutral'

    # micro trend(long, shot 각각) 산출. 롱추세 숏추세 interval 뽑아서 연속되는 인터발의 가장센거 (마지막거 string으로 추출 '3m', '1h')
    globals()['long_trend_micro_interval_pick'] = continuous_sequence_finder(long_trend_intervals)
    globals()['short_trend_micro_interval_pick'] = continuous_sequence_finder(short_trend_intervals)

    # micro trend(long, shot) 비교, 우세한 micro trend 찾기
    # globals()['long_trend_micro_interval_pick'] = '3m'
    # globals()['short_trend_micro_interval_pick'] = '1h'
    if globals()['long_trend_micro_interval_pick'] == '' and globals()['short_trend_micro_interval_pick'] != '':
        trend_micro_state = 'short'
    elif globals()['short_trend_micro_interval_pick'] == '' and globals()['long_trend_micro_interval_pick'] != '':
        trend_micro_state = 'long'
    else:
        if globals()['long_trend_micro_interval_pick'] == '' or globals()['short_trend_micro_interval_pick'] == '':
            trend_micro_state = 'neutral'
        else:
            long_interval_minutes = int(globals()['long_trend_micro_interval_pick'][:-1]) * point_frame.get(globals()['long_trend_micro_interval_pick'][-1])
            short_interval_minutes = int(globals()['short_trend_micro_interval_pick'][:-1]) * point_frame.get(globals()['short_trend_micro_interval_pick'][-1])

            if long_interval_minutes > short_interval_minutes:
                trend_micro_state = 'long'
            elif long_interval_minutes < short_interval_minutes:
                trend_micro_state = 'short'
            else:
                trend_micro_state = 'neutral'

     # macro 와 micro 추세가 일치할때 scalping_direction_pick 결정
    if globals()['trend_macro_state'] == trend_micro_state:
        scalping_direction_pick = trend_micro_state

    # print('')
    # print('long_trend_micro_interval_pick: [', globals()['long_trend_micro_interval_pick'], '], long_trend_intervals', long_trend_intervals)
    # print('short_trend_micro_interval_pick: [', globals()['short_trend_micro_interval_pick'], '], short_trend_intervals', short_trend_intervals)
    # print('trend_macro_state: [', trend_macro_state, '], trend_micro_state: [', trend_micro_state, ']')


    # scalping_indicator_confirmer peak 있고 scalping_direction_pick 과 같은방향일때 scalping start!
    if scalping_indicator_confirmer == 1 and peaker_option == 'forward' and peaker_side == scalping_direction_pick:
        scalping_switch = 'on'

    if globals()['long_trend_micro_interval_pick'] != '':
        globals()['long_trend_atr_given'] = float(exchange.price_to_precision(market_id, globals()['df_' + globals()['long_trend_micro_interval_pick']]['ATRr_14'][-1]))
    if globals()['short_trend_micro_interval_pick'] != '':
        globals()['short_trend_atr_given'] = float(exchange.price_to_precision(market_id, globals()['df_' + globals()['short_trend_micro_interval_pick']]['ATRr_14'][-1]))

    




    # if scalping_switch == 'on':
    # print('--------------------------------------')
    # print('scalping_indicator_confirmer: ', scalping_indicator_confirmer)
    # print('stg_type: ', stg_type, 'peaker_side: ',  peaker_side, 'peaker_option: ', peaker_option)
    # print('trend_macro_state: ', trend_macro_state)
    # print('trend_micro_state: ', trend_micro_state)
    # print('long_trend_intervals: ', long_trend_intervals)
    # print('long_trend_micro_interval_pick: ', globals()['long_trend_micro_interval_pick'])
    # print('short_trend_intervals: ', short_trend_intervals)
    # print('short_trend_micro_interval_pick: ', globals()['short_trend_micro_interval_pick'])
    # print('scalping_switch: ', scalping_switch)


    return stg_type, pick_time, close_price_high, close_price_low, point_sum, success, peaker_side, peaker_option, scalping_direction_pick, scalping_switch, scalping_indicator_confirmer, divergence_name


def volatility_checker():
    # Initialize counters
    high_volatility_count = 0
    low_volatility_count = 0
    high_list = []
    low_list = []

    # Iterate over the valid intervals
    for interval in globals()['valid_intervals']:
        if globals()['df_' + interval]['atr_AMATe_LR_8_21_2'][-1] == 1:
            high_volatility_count += 1
            high_list.append(interval)
        elif globals()['df_' + interval]['atr_AMATe_SR_8_21_2'][-1] == 1:
            low_volatility_count += 1
            low_list.append(interval)

    # Determine the overall volatility state
    if high_volatility_count > low_volatility_count:
        globals()['volatility_macro_state'] = 1
    else:
        globals()['volatility_macro_state'] = 0
    high_list_len = len(high_list)
    low_list_len = len(low_list)

    # Finds the largest number in the second list that forms a continuous sequence in the first list.
    globals()['volatility_micro_interval_pick'] = continuous_sequence_finder(high_list)
    if globals()['volatility_micro_interval_pick'] != '':
        globals()['volatility_atr_given'] = float(exchange.price_to_precision(market_id, globals()['df_' + globals()['volatility_micro_interval_pick']]['ATRr_14'][-1]))
    
    return high_list, low_list, high_list_len, low_list_len

def big_boss_point_cal(market_id, interval_):
    """
    가장 최근에 생긴 pick의 interval_ 이 atr_pick 보다 큰지, 작은지 확인
    """
    point_of_atr_pick_ = 0
    point_big_boss_interval_ = 0
    is_big_boss_interval_grater_then_atr_pick_ = False
    if (globals()['atr_pick'] != '') and (globals()['atr_pick'][-1] in point_frame):
        point_of_atr_pick_ = int(globals()['atr_pick'].split(globals()['atr_pick'][-1])[0])*point_frame[globals()['atr_pick'][-1]]
    if (interval_[-1] in point_frame):
        point_big_boss_interval_ = int(interval_.split(interval_[-1])[0])*point_frame[interval_[-1]]
    if point_big_boss_interval_ > point_of_atr_pick_:
        is_big_boss_interval_grater_then_atr_pick_ = True
    return is_big_boss_interval_grater_then_atr_pick_

# def check_condition(side_, current_d_):
#     """
#     가장 최근에 생긴 pick의 방향이 전체 추세와 일치하는지 확인
#     """
#     # condition_ = (current_d_['atr_diff'] > 0) # & (current_d_['dx_diff'] > 0)
#     if (current_d_['atr_diff'] > 0):
#         big_boss_trend_checker = True
#     elif (current_d_['atr_diff'] < 0):
#         big_boss_trend_checker = False
#     else:
#         big_boss_trend_checker = 0
#     # print(big_boss_trend_checker)
#     return big_boss_trend_checker


# def check_condition(side_, current_d_):
#     """
#     가장 최근에 생긴 pick의 방향이 전체 추세와 일치하는지 확인
#     """

#     big_boss_trend_checker = ''
#     condition_1 = ((current_d_['rsi_diff'] > 0) & (current_d_['macd_diff'] > 0)) & (current_d_['atr_diff'] > 0) & (current_d_['dx_diff'] > 0) & (current_d_['dmp_diff'] > 0) & (current_d_['RSI_14'] < 70)
#     condition_2 = ((current_d_['rsi_diff'] < 0) & (current_d_['macd_diff'] < 0)) & (current_d_['atr_diff'] > 0) & (current_d_['dx_diff'] > 0) & (current_d_['dmn_diff'] > 0) & (current_d_['RSI_14'] > 30)
#     if condition_1:
#         big_boss_trend_checker = 'long'
#     elif condition_2:
#         big_boss_trend_checker = 'short'
#     return big_boss_trend_checker

def check_condition(side, current_d_):
    """
    가장 최근에 생긴 pick의 방향이 전체 추세와 일치하는지 확인
    """

    big_boss_trend_checker = ''
    condition_1 = ((current_d_['rsi_diff'] > 0) & (current_d_['macd_diff'] > 0)) & (current_d_['atr_diff'] > 0) & (current_d_['dx_diff'] > 0) & (current_d_['dmp_diff'] > 0) & (current_d_['RSI_14'] < 70)
    condition_2 = ((current_d_['rsi_diff'] < 0) & (current_d_['macd_diff'] < 0)) & (current_d_['atr_diff'] > 0) & (current_d_['dx_diff'] > 0) & (current_d_['dmn_diff'] > 0) & (current_d_['RSI_14'] > 30)

    if side == 'long':
        # if condition_1:
        big_boss_trend_checker = 'long'
    elif side == 'short':
        # if condition_2:
        big_boss_trend_checker = 'short'
    return big_boss_trend_checker

# def big_boss_trend(market_id, intervals):
#     """
#     가장 최근에 생긴 pick의  interval_, 방향, 확인
#     """
#     big_boss_trend_checker = False
#     interval_not_in_big_boss_trend = ['1m', '3m', '5m', '15m', '30m']
#     latest_non_zero = interval_ = side_= last_time_ = None  # Initialize latest_non_zero
#     for interval in intervals:
#         if interval not in interval_not_in_big_boss_trend:
#             df_interval = globals()['df_' + interval]
#             current_d_ = df_interval.iloc[-1]
#             non_zero_rows = df_interval[(df_interval['stg2_short'] != 0) | (df_interval['stg2_long'] != 0)]
#             if not non_zero_rows.empty:
#                 last_non_zero = non_zero_rows.iloc[-1]
#                 last_time = last_non_zero['close_time']
#                 last_values = last_non_zero[['stg2_short', 'stg2_long']]
#                 if latest_non_zero is None or last_time > latest_non_zero[0]:
#                     latest_non_zero = (last_time, last_values, interval)
#                     interval_ = interval
#                     side_ = "long" if last_values['stg2_long'] != 0 else "short"
#                     last_time_ = last_time
#                     big_boss_trend_checker = check_condition(side_, current_d_)
#     return interval_, side_, last_time_, big_boss_trend_checker

# def big_boss_trend_re(market_id, intervals):
#     """
#     가장 최근에 생긴 pick의  interval_, 방향, 확인
#     """
#     # big_boss_trend_checker = False
#     interval_not_in_big_boss_trend = ['1m', '3m', '5m', '15m', '30m', '1h', '3d']
#     interval_ = side_ = last_time_ = big_boss_trend_checker_ = ''
#     # latest_non_zero = interval_ = side_= last_time_ = None  # Initialize latest_non_zero
#     for interval in intervals:
#         if interval not in interval_not_in_big_boss_trend:
#             df_interval = globals()['df_' + interval]
#             current_d_ = df_interval.iloc[-1]
#             non_zero_rows = df_interval[(df_interval['stg2_short'] != 0) | (df_interval['stg2_long'] != 0)]
#             if not non_zero_rows.empty:
#                 last_non_zero = non_zero_rows.iloc[-1]
#                 last_time = last_non_zero['close_time']
#                 last_values = last_non_zero[['stg2_short', 'stg2_long']]
#                 # symbol_ticker_last = ticker_calc(market_id)[1]
#                 if last_values['stg2_long'] != 0: # long 일 경우
#                     # peak_close_price_ = last_non_zero['low']
#                     # if (latest_non_zero is None or last_time > latest_non_zero[0]) and (peak_close_price_ <= symbol_ticker_last):
#                     # latest_non_zero = (last_time, last_values, interval)
#                     # interval_ = interval
#                     side_ = "long"
#                     # last_time_ = last_time
#                     # big_boss_trend_checker = check_condition(side_, current_d_)
#                 else: # short 일 경우
#                     # peak_close_price_ = last_non_zero['high']
#                     # if (latest_non_zero is None or last_time > latest_non_zero[0]) and (peak_close_price_ >= symbol_ticker_last):
#                     # latest_non_zero = (last_time, last_values, interval)
#                     # interval_ = interval
#                     side_ = "short"
#                     # last_time_ = last_time
#                     # big_boss_trend_checker = check_condition(side_, current_d_)
#                 big_boss_trend_checker = check_condition(side, current_d_)
#                 if big_boss_trend_checker:
#                     # print('\n-----------')
#                     # print(interval)
#                     interval_ = interval
#                     # side_ = side
#                     last_time_ = last_time
#                     big_boss_trend_checker_ = big_boss_trend_checker
#     #                 print(interval_, side_, last_time_, big_boss_trend_checker_)
#     # print('**********')
#     # print(interval_, side_, last_time_, big_boss_trend_checker)
#     return interval_, side_, last_time_, big_boss_trend_checker_

def big_boss_trend_re(market_id, intervals):
    """
    가장 최근에 생긴 pick의  interval_, 방향, 확인
    """
    interval_not_in_big_boss_trend = ['30m', '1h', '2h', '4h']
    # interval_ = side_ = last_time_ = big_boss_trend_checker_ = ''
    # latest_non_zero = interval_ = side_= last_time_ = None  # Initialize latest_non_zero
    latest_non_zero = interval_ = side_= side= last_time_ = None  # Initialize latest_non_zero
    big_boss_trend_checker_ = ''
    for interval in intervals:
        if interval not in interval_not_in_big_boss_trend:
            big_boss_trend_checker = ''
            df_interval = globals()['df_' + interval]
            current_d_ = df_interval.iloc[-1]
            # non_zero_rows = df_interval[(df_interval['stgU_short'] != 0) | (df_interval['stgU_long'] != 0)]
            non_zero_rows = df_interval[
                ((df_interval['stgU_short'] != 0) & pd.notna(df_interval['stgU_short'])) |
                ((df_interval['stgU_long'] != 0) & pd.notna(df_interval['stgU_long']))
            ]
            if not non_zero_rows.empty:
                last_non_zero = non_zero_rows.iloc[-1]
                last_time = last_non_zero['close_time']
                last_values = last_non_zero[['stgU_short', 'stgU_long']]
                symbol_ticker_last_ = ticker_calc(market_id)[1]
                # if last_values['stgU_long'] > 0: # long 일 경우
                if pd.notna(last_values['stgU_long']) and (last_values['stgU_long'] > 0):
                    peak_close_price_ = last_non_zero['low']
                    if (latest_non_zero is None or last_time > latest_non_zero[0]): # and (peak_close_price_ <= symbol_ticker_last_):
                        latest_non_zero = (last_time, last_values, interval)
                        interval_ = interval
                        side = "long"
                        last_time_ = last_time
                        # big_boss_trend_checker = check_condition(side, current_d_)
                        big_boss_trend_checker = "long"
                # elif last_values['stgU_short'] < 0: # short 일 경우
                elif pd.notna(last_values['stgU_short']) and (last_values['stgU_short'] < 0):
                    peak_close_price_ = last_non_zero['high']
                    if (latest_non_zero is None or last_time > latest_non_zero[0]): # and (peak_close_price_ >= symbol_ticker_last_):
                        latest_non_zero = (last_time, last_values, interval)
                        interval_ = interval
                        side = "short"
                        last_time_ = last_time
                        # big_boss_trend_checker = check_condition(side, current_d_)
                        big_boss_trend_checker = "short"
                # big_boss_trend_checker = check_condition(side, current_d_)
                if big_boss_trend_checker:
                    interval_ = interval
                    side_ = side
                    last_time_ = last_time
                    big_boss_trend_checker_ = big_boss_trend_checker
    return interval_, side_, last_time_, big_boss_trend_checker_





def big_boss_trend_re_2(market_id, intervals):
    """
    가장 최근에 생긴 pick의  interval_, 방향, 확인
    """
    interval_not_in_big_boss_trend = ['30m', '1h', '2h', '4h']
    # interval_ = side_ = last_time_ = big_boss_trend_checker_ = ''
    # latest_non_zero = interval_ = side_= last_time_ = None  # Initialize latest_non_zero
    latest_non_zero = interval_ = side_= side= last_time_ = None  # Initialize latest_non_zero
    big_boss_trend_checker_ = ''
    for interval in intervals:
        if interval not in interval_not_in_big_boss_trend:
            big_boss_trend_checker = ''
            df_interval = globals()['df_' + interval]
            recent_5_rows = df_interval.tail(3)
            current_d_ = df_interval.iloc[-1]
            # non_zero_rows = df_interval[(df_interval['stgT_short'] != 0) | (df_interval['stgT_long'] != 0)]
            non_zero_rows = recent_5_rows[
                ((recent_5_rows['stgT_short'] != 0) & pd.notna(recent_5_rows['stgT_short'])) |
                ((recent_5_rows['stgT_long'] != 0) & pd.notna(recent_5_rows['stgT_long']))
            ]
            print('\n\n\n\n22222222####%%%%%%%%%%%%%%%%%%%%####################**********')
            print('2', interval)
            print(non_zero_rows[['stgT_short', 'stgT_long']].tail(3))
            if not non_zero_rows.empty:
                last_non_zero = non_zero_rows.iloc[-1]
                last_time = last_non_zero['close_time']
                last_values = last_non_zero[['stgT_short', 'stgT_long']]
                print('last_values-2:', last_values)
                # print(type(last_values))
                symbol_ticker_last_ = ticker_calc(market_id)[1]

                # # 현재 시간으로부터 30분 이내의 데이터만 고려
                # time_threshold = pd.Timestamp.now() - pd.Timedelta(minutes=30)
                # if last_time >= time_threshold:
                #     # 최신 데이터로 업데이트

                # if last_values['stgT_long'] > 0: # long 일 경우
                if pd.notna(last_values['stgT_long']) and (last_values['stgT_long'] > 0):
                    print('들어왔음_long-2')
                    peak_close_price_ = last_non_zero['low']
                    if (latest_non_zero is None or last_time > latest_non_zero[0]): # and (peak_close_price_ <= symbol_ticker_last_):
                        latest_non_zero = (last_time, last_values, interval)
                        interval_ = interval
                        side = "long"
                        last_time_ = last_time
                        # big_boss_trend_checker = check_condition(side, current_d_)
                        big_boss_trend_checker = "long"
                        print(interval_, side, last_time_)
                # elif last_values['stgT_short'] < 0: # short 일 경우
                elif pd.notna(last_values['stgT_short']) and (last_values['stgT_short'] < 0):
                    print('들어왔음_short-2')
                    peak_close_price_ = last_non_zero['high']
                    if (latest_non_zero is None or last_time > latest_non_zero[0]): # and (peak_close_price_ >= symbol_ticker_last_):
                        latest_non_zero = (last_time, last_values, interval)
                        interval_ = interval
                        side = "short"
                        last_time_ = last_time
                        # big_boss_trend_checker = check_condition(side, current_d_)
                        big_boss_trend_checker = "short"
                        print(interval_, side, last_time_)
                # big_boss_trend_checker = check_condition(side, current_d_)
                if big_boss_trend_checker:
                    # print('\n-----------')
                    # print(interval)
                    interval_ = interval
                    side_ = side
                    last_time_ = last_time
                    big_boss_trend_checker_ = big_boss_trend_checker
                    print("결정-2: ", interval_, side, last_time_)
                    # print(interval_, side_, last_time_, big_boss_trend_checker_)
    print('**********222222222222')
    print(interval_, side_, last_time_, big_boss_trend_checker_)
    return interval_, side_, last_time_, big_boss_trend_checker_


def big_boss_trend_re_3(market_id, intervals):
    """
    가장 최근에 생긴 pick의 interval_, 방향, 확인
    """
    interval_not_in_big_boss_trend = ['1m', '30m']
    interval_ = side_ = last_time_ = None  # Initialize latest_non_zero
    big_boss_trend_checker_ = False

    for interval in intervals:
        if interval not in interval_not_in_big_boss_trend:
            df_interval = globals()['df_' + interval]

            if (
                (0.4 < df_interval['combined_diff'].iloc[-1] < 0.9)
                and (df_interval['combined_diff_diff'].iloc[-1] > 0)
            ):
                last_time_ = df_interval['close_time'].iloc[-1]  # 최근 시간을 가져오는 부분 수정
                interval_ = interval
                big_boss_trend_checker_ = True  # 트렌드가 성립한 경우

    return interval_, side_, last_time_, big_boss_trend_checker_






# def stop_loss_calc(stg_type, success, peaker_side, symbol_ticker_last, close_price_low, close_price_high, pick_max, pick_min):
#     # 역지, 
#     # max(픽봉의 꼬리, 현재가의 꼬리, ) + 수수료만큼, 
#     # exit price가 4퍼 이상될 시 포지 진입x

#     if (stg_type in ['stg3', 'stg9', 'stg10', 'stg100']) and (success == 1) and (globals()['atr_pick'] != ''):
#         pric_ = globals()['atr_pick']
#         l_p_l = globals()['df_' + pric_].low[-1] # 현재봉의 가장 낮은가
#         l_p_h = globals()['df_' + pric_].high[-1] # 현재봉의 가장 높은가
#     else:
#         interval = '5m'
#         l_p_l = globals()['df_' + interval].low[-1] # 현재봉의 가장 낮은가
#         l_p_h = globals()['df_' + interval].high[-1] # 현재봉의 가장 높은가
#         df_interval = globals()['df_' + interval]
#         if peaker_side == 'long': 
#             non_zero_rows = df_interval[(df_interval['stg10_long'] != 0)]
#             if not non_zero_rows.empty:
#                 last_non_zero = non_zero_rows.iloc[-1]
#                 close_price_low = last_non_zero['low']
#             else:
#                 close_price_low = symbol_ticker_last

#         else:
#             non_zero_rows = df_interval[(df_interval['stg10_short'] != 0)]
#             if not non_zero_rows.empty:
#                 last_non_zero = non_zero_rows.iloc[-1]
#                 close_price_high = last_non_zero['high']
#             else:
#                 close_price_high = symbol_ticker_last

#     if peaker_side == 'long': 
#         current_p_ = symbol_ticker_last
#         stop_loss = (min(close_price_low, l_p_l) * 1.0008)
#         stop_loss_ = float(exchange.price_to_precision(market_id, stop_loss)) # current_position 이 short 일때 사용
#         percentage_difference_stop_loss_ = ((stop_loss_ - current_p_) / current_p_) * 100
#         target_p_ = pick_max
#         percentage_difference_target_p_ = ((target_p_ - current_p_) / current_p_) * 100
#     else: 
#         current_p_ = symbol_ticker_last
#         stop_loss = (max(close_price_high, l_p_h) * 1.0008)
#         stop_loss_ = float(exchange.price_to_precision(market_id, stop_loss)) # current_position 이 short 일때 사용
#         percentage_difference_stop_loss_ = ((current_p_ - stop_loss_) / current_p_) * 100
#         target_p_ = pick_min
#         percentage_difference_target_p_ = ((current_p_ - target_p_) / current_p_) * 100

#     # print(stg_type, success, peaker_side, symbol_ticker_last, close_price_low, close_price_high, pick_max, pick_min)
#     # print(stop_loss_, percentage_difference_stop_loss_, target_p_, percentage_difference_target_p_)    
#     return stop_loss_, percentage_difference_stop_loss_, target_p_, percentage_difference_target_p_




# def stop_loss_calc(stg_type, success, peaker_side, position_entry_price, symbol_ticker_last, close_price_low, close_price_high, pick_max, pick_min, stop_loss_range):
#     # 역지, 
#     # max(픽봉의 꼬리, 현재가의 꼬리, ) + 수수료만큼, 
#     # exit price가 4퍼 이상될 시 포지 진입x

#     # if (globals()['atr_pick'] != ''):
#     #     interval = globals()['atr_pick']
#     # else:
#     #     if stop_loss_range == "tight":
#     #         interval = '15m'
#     #     else:
#     #         interval = '15m'
#     # if stop_loss_range == "tight":
#     #     l_p_l = globals()['df_' + interval]['low'].tail(2).min() # 15m기준 마지막 3개 중 가장 낮은가
#     #     l_p_h = globals()['df_' + interval]['high'].tail(2).max() # 15m기준 마지막 3개 중 가장 높은가
#     # else:
#     #     l_p_l = globals()['df_' + interval]['low'].tail(5).min() # 15m기준 마지막 3개 중 가장 낮은가
#     #     l_p_h = globals()['df_' + interval]['high'].tail(5).max() # 15m기준 마지막 3개 중 가장 높은가

#     # # interval = '15m'
#     # l_p_l = globals()['df_' + interval]['low'].tail(2).min() # 15m기준 마지막 3개 중 가장 낮은가
#     # l_p_h = globals()['df_' + interval]['high'].tail(2).max() # 15m기준 마지막 3개 중 가장 높은가

#     if stop_loss_range == "tight":
#         stop_loss_range_constant = 0.003
#         # if (globals()['atr_pick'] != ''):
#         #     interval = globals()['atr_pick']
#         # else:        
#         #     interval = '5m'
#         interval = '5m'
#         l_p_l = globals()['df_' + interval]['low'].tail(3).min() # 15m기준 마지막 3개 중 가장 낮은가
#         l_p_h = globals()['df_' + interval]['high'].tail(3).max() # 15m기준 마지막 3개 중 가장 높은가            
#     else:
#         stop_loss_range_constant = 0.003
#         interval = '5m'
#         l_p_l = globals()['df_' + interval]['low'].tail(3).min() # 15m기준 마지막 3개 중 가장 낮은가
#         l_p_h = globals()['df_' + interval]['high'].tail(3).max() # 15m기준 마지막 3개 중 가장 높은가

#     if peaker_side == 'long':
#         current_p_ = symbol_ticker_last
#         stop_loss = (l_p_l * (1 - stop_loss_range_constant)) # 0.0015 ==  0.15%, 0.15% * 레버리지 배율

#         if position_entry_price:
#             break_even = (position_entry_price * (1 + stop_loss_range_constant))
#             stop_loss = min(break_even, stop_loss)
        
#         stop_loss_ = float(exchange.price_to_precision(market_id, stop_loss)) # current_position 이 short 일때 사용
#         percentage_difference_stop_loss_ = ((stop_loss_ - current_p_) / current_p_) * 100
#         target_p_ = pick_max
#         percentage_difference_target_p_ = ((target_p_ - current_p_) / current_p_) * 100

#     else:
#         current_p_ = symbol_ticker_last
#         stop_loss = (l_p_h * (1 + stop_loss_range_constant))

#         if position_entry_price:
#             break_even = (position_entry_price * (1 - stop_loss_range_constant))
#             stop_loss = max(break_even, stop_loss)

#         stop_loss_ = float(exchange.price_to_precision(market_id, stop_loss)) # current_position 이 short 일때 사용
#         percentage_difference_stop_loss_ = ((current_p_ - stop_loss_) / current_p_) * 100
#         target_p_ = pick_min
#         percentage_difference_target_p_ = ((current_p_ - target_p_) / current_p_) * 100

#     # print(stg_type, success, peaker_side, symbol_ticker_last, close_price_low, close_price_high, pick_max, pick_min)
#     # print(stop_loss_, percentage_difference_stop_loss_, target_p_, percentage_difference_target_p_)    
#     return stop_loss_, percentage_difference_stop_loss_, target_p_, percentage_difference_target_p_



def l_p_l_and_l_p_h_calc(symbol_ticker_last):
    # if (globals()['atr_pick'] != ''):
    #     interval_ = globals()['atr_pick']
    # else:
    #     interval_ = '15m'

    if 'df_5m' in globals():
        interval_ = '5m'
    else:
        interval_ = '1m'

    l_p_l = globals()['df_' + interval_]['low'].tail(2).min() # 15m기준 마지막 3개 중 가장 낮은가
    l_p_h = globals()['df_' + interval_]['high'].tail(2).max() # 15m기준 마지막 3개 중 가장 높은가
    last_peaked_price = symbol_ticker_last
    return l_p_l, l_p_h, last_peaked_price



def stop_loss_calc(stg_type, success, peaker_side, position_entry_price, symbol_ticker_last, close_price_low, close_price_high, pick_max, pick_min, stop_loss_range):

    break_even_1 = 0
    break_even_2 = 0
    stop_loss_1 = 0
    stop_loss_2 = 0
    take_profit_1 = 0
    take_profit_2 = 0
    take_profit_3 = 0
    break_even_zero = 0
    max_profit_60percent = 0

    if position_entry_price and position_entry_price != 0:

        if (globals()['atr_pick'] != ''):
            interval_ = globals()['atr_pick']
        else:
            # interval_ = '5m'

            if 'df_5m' in globals():
                interval_ = '5m'
            else:
                interval_ = '1m'

        # interval_ = '5m'
        stop_loss_range_constant = (globals()['df_' + interval_]['atr_p'].iloc[-1])/100

        # stop_loss_range_constant = 0.003
        break_even_range_constant = stop_loss_range_constant * 2


        entry_time = datetime.datetime.fromtimestamp(position_entry_time)

        if (position_entry_time != 0) and (df_1m['open_time2'] >= entry_time).any():
            df_after_entry = df_1m[df_1m['open_time2'] >= entry_time][['high', 'low']]
            all_time_high = df_after_entry['high'].max()
            all_time_low = df_after_entry['low'].min()            
        else:
            df_after_entry = 0
            all_time_high = 0
            all_time_low = 0



        if peaker_side == 'long':
            current_p_ = symbol_ticker_last

            # break_even = max((last_peaked_price * (1 + break_even_range_constant)), (position_entry_price * (1 + break_even_range_constant)))

            # if l_p_l and l_p_l != 0:
            #     stop_loss = (l_p_l * (1 - stop_loss_range_constant)) # 0.0015 ==  0.15%, 0.15% * 레버리지 배율
            # else:
            #     stop_loss = (position_entry_price * (1 - stop_loss_range_constant))

            # if l_p_h and l_p_h != 0:
            #     take_profit = (l_p_h * (1 + stop_loss_range_constant)) # 0.0015 ==  0.15%, 0.15% * 레버리지 배율
            # else:
            #     take_profit = (position_entry_price * (1 + stop_loss_range_constant))


            # break_even_zero = (position_entry_price * (1 + 0.0012))

            break_even_1 = last_peaked_price * (1 + break_even_range_constant)
            break_even_2 = position_entry_price * (1 + break_even_range_constant)
            # stop_loss_1 = l_p_l * (1 - stop_loss_range_constant)
            stop_loss_2 = position_entry_price * (1 - stop_loss_range_constant)
            # take_profit_1 = l_p_h * (1 + stop_loss_range_constant)
            # take_profit_2 = l_p_l * (1 + stop_loss_range_constant)
            take_profit_3 = position_entry_price * (1 + stop_loss_range_constant)
            break_even_zero = position_entry_price * (1 + 0.0012)

            if (all_time_high > break_even_1) and (break_even_1 > position_entry_price):
                max_profit_60percent = all_time_high - (all_time_high - position_entry_price) * 0.3 # long
                # max_profit_60percent = all_time_low + (position_entry_price - all_time_low) * 0.4 # short
            else:
                max_profit_60percent = 0

            # 리스트 정렬
            values = [break_even_1, break_even_2, stop_loss_1, stop_loss_2, take_profit_1, take_profit_2, take_profit_3, break_even_zero, max_profit_60percent, stric_exit_min_price]
            filtered_values = [value for value in values if value != 0] # 0 제외
            sorted_values = sorted(filtered_values, reverse=True)

            # 초기 stop_loss 값 설정 (기본값은 최소값)
            stop_loss = sorted_values[-1]

            # 조건문을 통해 stop_loss 결정
            for value in sorted_values:
                if current_p_ > value:
                    stop_loss = value
                    break
            
            print(f' interval_:{interval_}, current_p_:{current_p_}, stric_exit_min_price:{stric_exit_min_price}, stop_loss_range_constant:{stop_loss_range_constant}, break_even_range_constant:{break_even_range_constant}, break_even:{break_even_2}, l_p_l:{l_p_l}, stop_loss:{stop_loss}')

            stop_loss_ = float(exchange.price_to_precision(market_id, stop_loss)) # current_position 이 short 일때 사용
            percentage_difference_stop_loss_ = ((stop_loss_ - current_p_) / current_p_) * 100
            target_p_ = pick_max
            percentage_difference_target_p_ = ((target_p_ - current_p_) / current_p_) * 100

        elif peaker_side == 'short':
            current_p_ = symbol_ticker_last

            # break_even = min((last_peaked_price * (1 - break_even_range_constant)), (position_entry_price * (1 - break_even_range_constant)))

            # if l_p_h and l_p_h != 0:
            #     stop_loss = (l_p_h * (1 + stop_loss_range_constant))
            # else:
            #     stop_loss = (position_entry_price * (1 + stop_loss_range_constant))

            # if l_p_l and l_p_l != 0:
            #     take_profit = (l_p_l * (1 - stop_loss_range_constant)) # 0.0015 ==  0.15%, 0.15% * 레버리지 배율
            # else:
            #     take_profit = (position_entry_price * (1 - stop_loss_range_constant))

            # break_even_zero = (position_entry_price * (1 - 0.0012))

            break_even_1 = last_peaked_price * (1 - break_even_range_constant)
            break_even_2 = position_entry_price * (1 - break_even_range_constant)
            # stop_loss_1 = l_p_h * (1 + stop_loss_range_constant)
            stop_loss_2 = position_entry_price * (1 + stop_loss_range_constant)
            # take_profit_1 = l_p_l * (1 - stop_loss_range_constant)
            # take_profit_2 = l_p_h * (1 - stop_loss_range_constant)
            take_profit_3 = position_entry_price * (1 - stop_loss_range_constant)
            break_even_zero = position_entry_price * (1 - 0.0012)

            if (all_time_low < break_even_1) and (break_even_1 < position_entry_price):
                # max_profit_60percent = all_time_high - (all_time_high - position_entry_price) * 0.4 # long
                max_profit_60percent = all_time_low + (position_entry_price - all_time_low) * 0.3 # short
            else:
                max_profit_60percent = 0

            # 리스트 정렬
            values = [break_even_1, break_even_2, stop_loss_1, stop_loss_2, take_profit_1, take_profit_2, take_profit_3, break_even_zero, max_profit_60percent, stric_exit_min_price]
            filtered_values = [value for value in values if value != 0] # 0 제외
            sorted_values = sorted(filtered_values, reverse=False)

            # 초기 stop_loss 값 설정 (기본값은 최소값)
            stop_loss = sorted_values[-1]

            # 조건문을 통해 stop_loss 결정
            for value in sorted_values:
                if current_p_ < value:
                    stop_loss = value
                    break

            print(f' interval_:{interval_}, current_p_:{current_p_}, stric_exit_min_price:{stric_exit_min_price}, stop_loss_range_constant:{stop_loss_range_constant}, break_even_range_constant:{break_even_range_constant}, break_even:{break_even_2}, l_p_l:{l_p_l}, stop_loss:{stop_loss}')
            stop_loss_ = float(exchange.price_to_precision(market_id, stop_loss)) # current_position 이 short 일때 사용
            percentage_difference_stop_loss_ = ((current_p_ - stop_loss_) / current_p_) * 100
            target_p_ = pick_min
            percentage_difference_target_p_ = ((current_p_ - target_p_) / current_p_) * 100

            print('---------------------------')
            print(f'break_even: {break_even_2}')
            print('---------------------------')
            # print(stg_type, success, peaker_side, symbol_ticker_last, close_price_low, close_price_high, pick_max, pick_min)
            # print(stop_loss_, percentage_difference_stop_loss_, target_p_, percentage_difference_target_p_)
    else:
        stop_loss_ = percentage_difference_stop_loss_ = target_p_ = percentage_difference_target_p_ = 0

    return stop_loss_, percentage_difference_stop_loss_, target_p_, percentage_difference_target_p_




























def target_p_re(cumulate_lv, peaker_side, s_l_p, position_entry_price, position_size, symbol_ticker_last):
    if peaker_side == 'long':
        t_p = 2*(position_entry_price - s_l_p) + position_entry_price
    elif peaker_side == 'short':
        t_p = position_entry_price - 2*(s_l_p - position_entry_price)

    if position_size == 0:
        pick_min = max([i for i in cumulate_lv if symbol_ticker_last>i], default=symbol_ticker_last) # current_position 이 short 일때 사용
        pick_max = min([i for i in cumulate_lv if symbol_ticker_last<i], default=symbol_ticker_last) # current_position 이 long 일때 사용
    else:
        pick_min = max([i for i in cumulate_lv if t_p>=i], default=t_p) # current_position 이 short 일때 사용
        pick_max = min([i for i in cumulate_lv if t_p<=i], default=t_p) # current_position 이 long 일때 사용
    pick_min = float(exchange.price_to_precision(market_id, pick_min)) # current_position 이 short 일때 사용
    pick_max = float(exchange.price_to_precision(market_id, pick_max)) # current_position 이 long 일때 사용
    return(pick_max, pick_min)






def max_atr_tr_interval_confirm():
    # tr_intervals = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d']
    # tr_intervals = ['1m', '5m', '15m']
    tr_intervals = ['5m']
    trend_result = trade_type = target_range = stop_loss_range = ""
    # tr_intervals = intervals
    current_atr_p = atr_p_store = atr_p_store_interval = atr_p_store_trend_result = atr_p_store_timestamp = None
    latest_interval = None
    latest_timestamp = None
    timestamp = None
    tr_interval = None
    
    re_intvs = []
    current_time = dt.datetime.now()
    long_count_ = 0
    short_count_ = 0
    condition_met_for_all_intervals = True
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    for interval in tr_intervals:
        df_name  = 'df_' + interval
        if df_name  in globals():
            re_intvs.append(interval)
            df_interval = globals()['df_' + interval]


            # Get the last and second-to-last elements of the 'DX_14' column
            last_value = df_interval['DX_14'].iloc[-1]
            second_last_value = df_interval['DX_14'].iloc[-2]

            # Calculate the absolute difference
            absolute_difference = abs(last_value - second_last_value)

            # Check if the absolute difference is greater than 20% of the second-to-last value
            threshold = 6 * second_last_value


            condition = (
                # (
                #     (absolute_difference > threshold) &
                #     # (df_interval['adx_diff'].iloc[-1] > 0) &
                #     (df_interval['dx_diff'].iloc[-1] > 0) &
                #     (df_interval['DX_14'].iloc[-2] < 5) &
                #     (df_interval['DX_14'].iloc[-1] > 30)
                #     # (df_interval['DX_14'].iloc[-1] > df_interval['DMN_14'].iloc[-1]) &
                #     # (df_interval['DX_14'].iloc[-1] > df_interval['DMP_14'].iloc[-1]) &
                #     # (df_interval['DX_14'].iloc[-1] > df_interval['ADX_14'].iloc[-1])
                # ) |
                # (
                #     # (df_interval['adx_diff'].iloc[-1] > 0) &
                #     (df_interval['dx_diff'].iloc[-1] < 0) &
                #     (df_interval['DX_14'].iloc[-2] > 55) &
                #     (df_interval['DX_14'].iloc[-1] < 20)
                #     # (df_interval['DX_14'].iloc[-1] < df_interval['DMN_14'].iloc[-1]) &
                #     # (df_interval['DX_14'].iloc[-1] < df_interval['DMP_14'].iloc[-1]) &
                #     # (df_interval['DX_14'].iloc[-1] < df_interval['ADX_14'].iloc[-1])
                # )







                (
                    (
                        (
                            # (df_interval.minima_peak_x_atr14 > 0) & (df_interval.minima_peak_x_dx > 0)
                            (df_interval['atr_diff'].iloc[-1] > 0) & (df_interval['dx_diff'].iloc[-1] > 0) & (df_interval['DMP_14'].iloc[-1] > df_interval['DMN_14'].iloc[-1]) & (df_interval.rsi_diff.iloc[-1] > 0) & (df_interval.macd_diff.iloc[-1] > 0) & (df_interval['atr_AMATe_LR_8_21_2'].iloc[-1] > 0)
                            | (df_interval['atr_diff'].iloc[-1] > 0) & (df_interval['dx_diff'].iloc[-1] < 0) & (df_interval['DMN_14'].iloc[-2] > df_interval['DMP_14'].iloc[-2]) & (df_interval.rsi_diff.iloc[-1] > 0) & (df_interval.macd_diff.iloc[-1] > 0) & (df_interval['atr_AMATe_LR_8_21_2'].iloc[-1] > 0)
                            | (df_interval['atr_diff'].iloc[-1] < 0) & (df_interval['dx_diff'].iloc[-1] > 0) & (df_interval['DMP_14'].iloc[-1] > df_interval['DMN_14'].iloc[-1]) & (df_interval.rsi_diff.iloc[-1] > 0) & (df_interval.macd_diff.iloc[-1] > 0) & (df_interval['atr_AMATe_SR_8_21_2'].iloc[-1] > 0)
                            | (df_interval['atr_diff'].iloc[-1] < 0) & (df_interval['dx_diff'].iloc[-1] < 0) & (df_interval['DMN_14'].iloc[-2] > df_interval['DMP_14'].iloc[-2]) & (df_interval.rsi_diff.iloc[-1] > 0) & (df_interval.macd_diff.iloc[-1] > 0) & (df_interval['atr_AMATe_SR_8_21_2'].iloc[-1] > 0)
                        )
                        # & (df_interval.ATRr_14 > df_interval.close*0.7/100)
                    )
                    &~(
                        (df_interval.maxima_peak_x_close.iloc[-1] < 0) | (df_interval.maxima_peak_x_MACD_12_26_9.iloc[-1] < 0) | (df_interval.maxima_peak_x_MACD_50_75_35.iloc[-1] < 0) | (df_interval.maxima_peak_x_rsi.iloc[-1] < 0) | (df_interval.maxima_peak_x_dmp.iloc[-1] < 0) | (df_interval.minima_peak_x_dmn.iloc[-1] < 0)
                    )
                )
                |
                (
                    (
                        (
                            # (df_interval.minima_peak_x_atr14 > 0) & (df_interval.minima_peak_x_dx > 0)
                            (df_interval['atr_diff'].iloc[-1] > 0) & (df_interval['dx_diff'].iloc[-1] > 0) & (df_interval['DMN_14'].iloc[-1] > df_interval['DMP_14'].iloc[-1]) & (df_interval.rsi_diff.iloc[-1] < 0)  & (df_interval.macd_diff.iloc[-1] < 0) & (df_interval['atr_AMATe_LR_8_21_2'].iloc[-1] > 0)
                            | (df_interval['atr_diff'].iloc[-1] > 0) & (df_interval['dx_diff'].iloc[-1] < 0) & (df_interval['DMP_14'].iloc[-2] > df_interval['DMN_14'].iloc[-2]) & (df_interval.rsi_diff.iloc[-1] < 0) & (df_interval.macd_diff.iloc[-1] < 0) & (df_interval['atr_AMATe_LR_8_21_2'].iloc[-1] > 0)
                            | (df_interval['atr_diff'].iloc[-1] < 0) & (df_interval['dx_diff'].iloc[-1] > 0) & (df_interval['DMN_14'].iloc[-1] > df_interval['DMP_14'].iloc[-1]) & (df_interval.rsi_diff.iloc[-1] < 0) & (df_interval.macd_diff.iloc[-1] < 0) & (df_interval['atr_AMATe_SR_8_21_2'].iloc[-1] > 0)
                            | (df_interval['atr_diff'].iloc[-1] < 0) & (df_interval['dx_diff'].iloc[-1] < 0) & (df_interval['DMP_14'].iloc[-2] > df_interval['DMN_14'].iloc[-2]) & (df_interval.rsi_diff.iloc[-1] < 0) & (df_interval.macd_diff.iloc[-1] < 0) & (df_interval['atr_AMATe_SR_8_21_2'].iloc[-1] > 0)
                        )
                        # & (df_interval.ATRr_14 > df_interval.close*0.7/100)
                    )
                    &~(
                        (df_interval.minima_peak_x_close.iloc[-1] > 0) | (df_interval.minima_peak_x_MACD_12_26_9.iloc[-1] > 0) | (df_interval.minima_peak_x_MACD_50_75_35.iloc[-1] > 0) | (df_interval.minima_peak_x_rsi.iloc[-1] > 0) | (df_interval.minima_peak_x_dmp.iloc[-1] > 0) | (df_interval.maxima_peak_x_dmn.iloc[-1] > 0)
                    )
                )








            )


            # ##################################################################################
            # absolute_difference = abs(df_interval['dx_diff'])

            # # Check if the absolute difference is greater than 20% of the second-to-last value
            # # threshold = 1.2 * (df_interval['DX_14'])
            # threshold = 6 * df_interval['DX_14'].shift(1)
            # # print(absolute_difference)
            # condition = (absolute_difference > threshold)\
            # & ((df_interval['atr_diff']) > 0) \
            # & ((df_interval['adx_diff']) > 0) \
            # & ((df_interval['dx_diff']) > 0) \
            # & ((df_interval['DX_14'].shift(1)) < 5) \
            # & ((df_interval['DX_14']) > 30) \
            # & ((df_interval['DX_14']) > (df_interval['DMN_14'])) \
            # & ((df_interval['DX_14']) > (df_interval['DMP_14'])) \
            # & ((df_interval['DX_14']) > (df_interval['ADX_14']))
            # ##################################################################################















            # condition = (df_interval['atr_diff'].iloc[-1] > 0) & (df_interval['adx_diff'].iloc[-1] > 0)
            # condition = (df_interval['atr_diff'].iloc[-1] > 0)

            # condition = (df_interval['ATRr_14'].iloc[-1] > (df_interval['close'].iloc[-1]*1/100)) \
            #             & ((df_interval['ADX_14'].iloc[-1]) > 25) \
            #             & ((df_interval['atr_diff'].iloc[-1]) > 0) \
            #             & ((df_interval['adx_diff'].iloc[-1]) > 0)

            # condition = ((df_interval['DX_14'].iloc[-1]) > 25) \
            #             & ((df_interval['atr_diff'].iloc[-1]) > 0) \
            #             & ((df_interval['adx_diff'].iloc[-1]) > 0)

            # condition = ((df_interval['atr_diff'].iloc[-1]) > 0) # & ((df_interval['adx_diff'].iloc[-1]) > 0)
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    #         condition = True
    #         if condition:
    #             if (df_interval['atr_p_diff'].iloc[-1] > 0):
    #                 current_atr_p = (df_interval['atr_p_diff'].iloc[-1]) / (df_interval['atr_p_diff'].iloc[-2])

    #             if (current_atr_p is not None) and ((atr_p_store is None) or (current_atr_p > atr_p_store)):
    #                 atr_p_store = current_atr_p
    #                 atr_p_store_interval = interval
    #                 # atr_p_store_trend_result = trend_result
    #                 atr_p_store_timestamp = globals()['df_' + atr_p_store_interval]['open_time2'].iloc[-1]

    # # trend_result = atr_p_store_trend_result
    # tr_interval = atr_p_store_interval
    # print(f"current | {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ")
    # if atr_p_store is not None:
    #     atr_p_store = round(atr_p_store, 3)
    # print(f'[{tr_interval}] [{atr_p_store}] *** [{atr_p_store_timestamp} total_count: 동점]')
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
            condition = True
            if condition:
                if (
                    ((interval == '1m') & ((df_interval['ATRr_14'].iloc[-1]) > ((df_interval['close'].iloc[-1])*0.7/100)))
                    |
                    ((interval == '5m') & ((df_interval['ATRr_14'].iloc[-1]) > ((df_interval['close'].iloc[-1])*0.9/100)))
                    |
                    ((interval == '15m') & ((df_interval['ATRr_14'].iloc[-1]) > ((df_interval['close'].iloc[-1])*1.1/100)))
                ):# & (df_interval['atr_diff'].iloc[-1] > 0) & (df_interval['adx_diff'].iloc[-1] > 0) & (df_interval['ADX_14'].iloc[-1] > 25) :
                    current_atr_p = (df_interval['atr_diff'].iloc[-1])
                    timestamp = df_interval['open_time2'].iloc[-1]
                if (timestamp is not None) and ((latest_timestamp is None) or (timestamp > latest_timestamp)):
                    atr_p_store = current_atr_p
                    latest_timestamp = timestamp
                    latest_interval = interval
    tr_interval = latest_interval
    print(f"current | {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ")
    if atr_p_store is not None:
        atr_p_store = round(atr_p_store, 3)
    print(f'[{tr_interval}] [{atr_p_store}] *** [{latest_timestamp} total_count: 동점]')
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    # if (long_count_ > 0) and (long_count_ > short_count_) :
    #     trend_result = 'long'
    #     tr_interval = latest_interval
    #     print(f"current | {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ")
    #     print(f'[{trend_result}] [{tr_interval}] *** [{latest_timestamp} long_count: {long_count_}]')
        

    # elif (short_count_ > 0) and (short_count_ > long_count_) : # if (short_count_ > 0):
    #     trend_result = 'short'
    #     tr_interval = latest_interval
    #     print(f"current | {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ")
    #     print(f'[{trend_result}] [{tr_interval}] *** [{latest_timestamp} short_count: {short_count_}]')
        

    # elif (long_count_ > 0) or (short_count_ > 0): # 동점일경우 atr_p_diff 가 큰 interval의 방향으로 결정
    #     trend_result = atr_p_store_trend_result
    #     tr_interval = atr_p_store_interval
    #     print(f"current | {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ")
    #     print(f'[{trend_result}] [{tr_interval}] *** [{atr_p_store_timestamp} total_count: 동점]')

    # else:
    #     trend_result = ''        
    #     print(f"current | {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ")
    # print('-----------')
    # print(latest_timestamp, latest_interval)
    # print('-----------\n')
    # if not latest_interval:
    #     latest_interval = re_intvs[-1]
    return tr_interval











def tr_interval_confirm():
    tr_interval = max_atr_tr_interval_confirm()
    tr_intervals = [tr_interval]
    # tr_intervals = ['1m', '3m', '5m', '15m', '30m', '1h']
    # tr_intervals = ['1m', '5m', '15m']

    # tr_intervals = ['5m', '1h']
    trend_result = trade_type = target_range = stop_loss_range = ""
    # tr_intervals = intervals
    current_atr_p = atr_p_store = atr_p_store_interval = atr_p_store_trend_result = atr_p_store_timestamp = None
    latest_interval = None
    latest_timestamp = None
    timestamp = None
    tr_interval = None
    
    re_intvs = []
    current_time = dt.datetime.now()
    long_count_ = 0
    short_count_ = 0
    condition_met_for_all_intervals = True
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    for interval in tr_intervals:
        if interval:
            df_name  = 'df_' + interval
            if df_name  in globals():
                re_intvs.append(interval)
                df_interval = globals()['df_' + interval]


                # Get the last and second-to-last elements of the 'DX_14' column
                last_value = df_interval['DX_14'].iloc[-1]
                second_last_value = df_interval['DX_14'].iloc[-2]

                # Calculate the absolute difference
                absolute_difference = abs(last_value - second_last_value)

                # Check if the absolute difference is greater than 20% of the second-to-last value
                threshold = 6 * second_last_value


                condition = (
                    # (
                    #     (absolute_difference > threshold) &
                    #     # (df_interval['adx_diff'].iloc[-1] > 0) &
                    #     (df_interval['dx_diff'].iloc[-1] > 0) &
                    #     (df_interval['DX_14'].iloc[-2] < 5) &
                    #     (df_interval['DX_14'].iloc[-1] > 30)
                    #     # (df_interval['DX_14'].iloc[-1] > df_interval['DMN_14'].iloc[-1]) &
                    #     # (df_interval['DX_14'].iloc[-1] > df_interval['DMP_14'].iloc[-1]) &
                    #     # (df_interval['DX_14'].iloc[-1] > df_interval['ADX_14'].iloc[-1])
                    # ) |
                    # (
                    #     # (df_interval['adx_diff'].iloc[-1] > 0) &
                    #     (df_interval['dx_diff'].iloc[-1] < 0) &
                    #     (df_interval['DX_14'].iloc[-2] > 55) &
                    #     (df_interval['DX_14'].iloc[-1] < 20)
                    #     # (df_interval['DX_14'].iloc[-1] < df_interval['DMN_14'].iloc[-1]) &
                    #     # (df_interval['DX_14'].iloc[-1] < df_interval['DMP_14'].iloc[-1]) &
                    #     # (df_interval['DX_14'].iloc[-1] < df_interval['ADX_14'].iloc[-1])
                    # )







                    (
                        (
                            (
                                # (df_interval.minima_peak_x_atr14 > 0) & (df_interval.minima_peak_x_dx > 0)
                                (df_interval['atr_diff'].iloc[-1] > 0) & (df_interval['dx_diff'].iloc[-1] > 0) & (df_interval['DMP_14'].iloc[-1] > df_interval['DMN_14'].iloc[-1]) & (df_interval.rsi_diff.iloc[-1] > 0) & (df_interval.macd_diff.iloc[-1] > 0) & (df_interval['atr_AMATe_LR_8_21_2'].iloc[-1] > 0)
                                | (df_interval['atr_diff'].iloc[-1] > 0) & (df_interval['dx_diff'].iloc[-1] < 0) & (df_interval['DMN_14'].iloc[-2] > df_interval['DMP_14'].iloc[-2]) & (df_interval.rsi_diff.iloc[-1] > 0) & (df_interval.macd_diff.iloc[-1] > 0) & (df_interval['atr_AMATe_LR_8_21_2'].iloc[-1] > 0)
                                | (df_interval['atr_diff'].iloc[-1] < 0) & (df_interval['dx_diff'].iloc[-1] > 0) & (df_interval['DMP_14'].iloc[-1] > df_interval['DMN_14'].iloc[-1]) & (df_interval.rsi_diff.iloc[-1] > 0) & (df_interval.macd_diff.iloc[-1] > 0) & (df_interval['atr_AMATe_SR_8_21_2'].iloc[-1] > 0)
                                | (df_interval['atr_diff'].iloc[-1] < 0) & (df_interval['dx_diff'].iloc[-1] < 0) & (df_interval['DMN_14'].iloc[-2] > df_interval['DMP_14'].iloc[-2]) & (df_interval.rsi_diff.iloc[-1] > 0) & (df_interval.macd_diff.iloc[-1] > 0) & (df_interval['atr_AMATe_SR_8_21_2'].iloc[-1] > 0)
                            )
                            # & (df_interval.ATRr_14 > df_interval.close*0.7/100)
                        )
                        &~(
                            (df_interval.maxima_peak_x_close.iloc[-1] < 0) | (df_interval.maxima_peak_x_MACD_12_26_9.iloc[-1] < 0) | (df_interval.maxima_peak_x_rsi.iloc[-1] < 0) | (df_interval.maxima_peak_x_dmp.iloc[-1] < 0) | (df_interval.minima_peak_x_dmn.iloc[-1] < 0)
                        )
                    )
                    |
                    (
                        (
                            (
                                # (df_interval.minima_peak_x_atr14 > 0) & (df_interval.minima_peak_x_dx > 0)
                                (df_interval['atr_diff'].iloc[-1] > 0) & (df_interval['dx_diff'].iloc[-1] > 0) & (df_interval['DMN_14'].iloc[-1] > df_interval['DMP_14'].iloc[-1]) & (df_interval.rsi_diff.iloc[-1] < 0)  & (df_interval.macd_diff.iloc[-1] < 0) & (df_interval['atr_AMATe_LR_8_21_2'].iloc[-1] > 0)
                                | (df_interval['atr_diff'].iloc[-1] > 0) & (df_interval['dx_diff'].iloc[-1] < 0) & (df_interval['DMP_14'].iloc[-2] > df_interval['DMN_14'].iloc[-2]) & (df_interval.rsi_diff.iloc[-1] < 0) & (df_interval.macd_diff.iloc[-1] < 0) & (df_interval['atr_AMATe_LR_8_21_2'].iloc[-1] > 0)
                                | (df_interval['atr_diff'].iloc[-1] < 0) & (df_interval['dx_diff'].iloc[-1] > 0) & (df_interval['DMN_14'].iloc[-1] > df_interval['DMP_14'].iloc[-1]) & (df_interval.rsi_diff.iloc[-1] < 0) & (df_interval.macd_diff.iloc[-1] < 0) & (df_interval['atr_AMATe_SR_8_21_2'].iloc[-1] > 0)
                                | (df_interval['atr_diff'].iloc[-1] < 0) & (df_interval['dx_diff'].iloc[-1] < 0) & (df_interval['DMP_14'].iloc[-2] > df_interval['DMN_14'].iloc[-2]) & (df_interval.rsi_diff.iloc[-1] < 0) & (df_interval.macd_diff.iloc[-1] < 0) & (df_interval['atr_AMATe_SR_8_21_2'].iloc[-1] > 0)
                            )
                            # & (df_interval.ATRr_14 > df_interval.close*0.7/100)
                        )
                        &~(
                            (df_interval.minima_peak_x_close.iloc[-1] > 0) | (df_interval.minima_peak_x_MACD_12_26_9.iloc[-1] > 0) | (df_interval.minima_peak_x_rsi.iloc[-1] > 0) | (df_interval.minima_peak_x_dmp.iloc[-1] > 0) | (df_interval.maxima_peak_x_dmn.iloc[-1] > 0)
                        )
                    )








                )


                # ##################################################################################
                # absolute_difference = abs(df_interval['dx_diff'])

                # # Check if the absolute difference is greater than 20% of the second-to-last value
                # # threshold = 1.2 * (df_interval['DX_14'])
                # threshold = 6 * df_interval['DX_14'].shift(1)
                # # print(absolute_difference)
                # condition = (absolute_difference > threshold)\
                # & ((df_interval['atr_diff']) > 0) \
                # & ((df_interval['adx_diff']) > 0) \
                # & ((df_interval['dx_diff']) > 0) \
                # & ((df_interval['DX_14'].shift(1)) < 5) \
                # & ((df_interval['DX_14']) > 30) \
                # & ((df_interval['DX_14']) > (df_interval['DMN_14'])) \
                # & ((df_interval['DX_14']) > (df_interval['DMP_14'])) \
                # & ((df_interval['DX_14']) > (df_interval['ADX_14']))
                # ##################################################################################















                # condition = (df_interval['atr_diff'].iloc[-1] > 0) & (df_interval['adx_diff'].iloc[-1] > 0)
                # condition = (df_interval['atr_diff'].iloc[-1] > 0)

                # condition = (df_interval['ATRr_14'].iloc[-1] > (df_interval['close'].iloc[-1]*1/100)) \
                #             & ((df_interval['ADX_14'].iloc[-1]) > 25) \
                #             & ((df_interval['atr_diff'].iloc[-1]) > 0) \
                #             & ((df_interval['adx_diff'].iloc[-1]) > 0)

                # condition = ((df_interval['DX_14'].iloc[-1]) > 25) \
                #             & ((df_interval['atr_diff'].iloc[-1]) > 0) \
                #             & ((df_interval['adx_diff'].iloc[-1]) > 0)

                # condition = ((df_interval['atr_diff'].iloc[-1]) > 0) # & ((df_interval['adx_diff'].iloc[-1]) > 0)
                
                condition = True
                if condition:
                    trend_result, trade_type, stop_loss_range, target_range = tr_confirm(interval)
                    print(f"[{interval}], [{trend_result}], [{round(df_interval['atr_p_diff'].iloc[-1], 3)}]")
                    if trend_result == 'long':
                        long_count_ += 1
                        timestamp = df_interval['open_time2'].iloc[-1]
                        current_atr_p = df_interval['atr_p_diff'].iloc[-1]
                    elif trend_result == 'short':
                        short_count_ += 1
                        timestamp = df_interval['open_time2'].iloc[-1]
                        current_atr_p = df_interval['atr_p_diff'].iloc[-1]
                    else:
                        condition_met_for_all_intervals = False                    
                    
                    if (timestamp is not None) and ((latest_timestamp is None) or (timestamp > latest_timestamp)):
                        latest_timestamp = timestamp
                        latest_interval = interval
                    
                    if (current_atr_p is not None) and ((atr_p_store is None) or (current_atr_p > atr_p_store)):
                        atr_p_store = current_atr_p
                        atr_p_store_interval = interval
                        atr_p_store_trend_result = trend_result
                        atr_p_store_timestamp = globals()['df_' + atr_p_store_interval]['open_time2'].iloc[-1]


    if (long_count_ > 0) and (long_count_ > short_count_) :
        trend_result = 'long'
        # trend_result = 'short'
        tr_interval = latest_interval
        print(f"current | {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ")
        print(f'[{trend_result}] [{tr_interval}] *** [{latest_timestamp} long_count: {long_count_}]')
        

    elif (short_count_ > 0) and (short_count_ > long_count_) : # if (short_count_ > 0):
        trend_result = 'short'
        # trend_result = 'long'
        tr_interval = latest_interval
        print(f"current | {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ")
        print(f'[{trend_result}] [{tr_interval}] *** [{latest_timestamp} short_count: {short_count_}]')
        

    elif (long_count_ > 0) or (short_count_ > 0): # 동점일경우 atr_p_diff 가 큰 interval의 방향으로 결정
        trend_result = atr_p_store_trend_result
        tr_interval = atr_p_store_interval
        print(f"current | {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ")
        print(f'[{trend_result}] [{tr_interval}] *** [{atr_p_store_timestamp} total_count: 2동점]')

    else:
        trend_result = ''        
        print(f"current | {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ")
    # print('-----------')
    # print(latest_timestamp, latest_interval)
    # print('-----------\n')
    # if not latest_interval:
    #     latest_interval = re_intvs[-1]
    return trend_result, tr_interval, trade_type, stop_loss_range, target_range, long_count_, short_count_


















# def tr_confirm(interval):
#     # interval = '15m'
#     if interval:

#         atr_p_set_1 = globals()['df_' + interval]['atr_p'].iloc[-1]
#         atr_set_1 = globals()['df_' + interval]['ATRr_14'].iloc[-1]
#         adx14_set_1 = globals()['df_' + interval]['ADX_14'].iloc[-1]
#         dx14_set_1 = globals()['df_' + interval]['DX_14'].iloc[-1]
#         dmn_set_1 = globals()['df_' + interval]['DMN_14'].iloc[-1]
#         dmp_set_1 = globals()['df_' + interval]['DMP_14'].iloc[-1]

#         atr_p_set_2 = globals()['df_' + interval]['atr_p'].iloc[-2]
#         atr_set_2 = globals()['df_' + interval]['ATRr_14'].iloc[-2]        
#         adx14_set_2 = globals()['df_' + interval]['ADX_14'].iloc[-2]
#         dx14_set_2 = globals()['df_' + interval]['DX_14'].iloc[-2]
#         dmn_set_2 = globals()['df_' + interval]['DMN_14'].iloc[-2]
#         dmp_set_2 = globals()['df_' + interval]['DMP_14'].iloc[-2]

#         atr_p_diff = globals()['df_' + interval]['atr_p_diff'].iloc[-1]
#         atr_diff = globals()['df_' + interval]['atr_diff'].iloc[-1]
#         adx_diff = globals()['df_' + interval]['adx_diff'].iloc[-1]
#         dx_diff = globals()['df_' + interval]['dx_diff'].iloc[-1]
#         rsi_diff = globals()['df_' + interval]['rsi_diff'].iloc[-1]
#         j_diff = globals()['df_' + interval]['j_diff'].iloc[-1]
#         atr14_set = globals()['df_' + interval]['ATRr_14'].iloc[-1]
#         close_set = globals()['df_' + interval]['close'].iloc[-1]

#         trend_result = trade_type = target_range = stop_loss_range = intersect = ""
#         if (dmn_set_2 > dmp_set_2) & (dmn_set_1 > dmp_set_1):
#             intersect = 0
#         elif (dmn_set_2 > dmp_set_2) & (dmn_set_1 < dmp_set_1):
#             intersect = 1
#         elif (dmn_set_2 < dmp_set_2) & (dmn_set_1 > dmp_set_1):
#             intersect = 1
#         elif (dmn_set_2 < dmp_set_2) & (dmn_set_1 < dmp_set_1):
#             intersect = 0
#         # print('\n----------')
#         # print(f'atr_diff: {atr_diff}, dx_diff: {dx_diff}, dmn_set: {dmn_set}, dmp_set: {dmp_set}')
#         # print(f'\
#         #     [atr_p_set_1]: {round(atr_p_set_1, 3)}, [atr_p_diff]: {round(atr_p_diff, 3)}\n\
#         #     [atr_set_1]: {round(atr_set_1, 3)}, [atr_diff]: {round(atr_diff, 3)}\n\
#         #     [adx14_set_1]: {round(adx14_set_1, 3)}, [adx_diff]: {round(adx_diff, 3)}\n\
#         #     [dx14_set_1]: {round(dx14_set_1, 3)}, [dx_diff]: {round(dx_diff, 3)}\n\
#         #     [dmp_set_1]: {round(dmp_set_1, 3)}, [dmn_set_1]: {round(dmn_set_1, 3)}\
#         # ')
#         # trending_trade 일때는 진입직전 추세확인 필요
#         # counter_trade 일때는 진입직전 추세확인 불필요
#         if (dx_diff > 0) and (atr_p_diff > 0) : # and (adx14_set_1 > 20):#  and (adx_diff > 0) : # atr 오르고 adx 오를때 추세방향으로 진입
#             target_range = "wide"
#             stop_loss_range = "tight"
#             if intersect: # 올리면서 intersect 일경우, 마지막 상태로 방향 결정
#                 if (dmp_set_1 > dmn_set_1):
#                     trend_result = "long"
#                     trade_type = "trending_trade"
#                     globals()['aa'] = '1'
#                 elif (dmn_set_1 > dmp_set_1):
#                     trend_result = "short"
#                     trade_type = "trending_trade"
#                     globals()['aa'] = '2'
#             else:
#                 if (dmp_set_1 > dmn_set_1):
#                     trend_result = "long"
#                     trade_type = "trending_trade"
#                     globals()['aa'] = '3'
#                 elif (dmn_set_1 > dmp_set_1):
#                     trend_result = "short"
#                     trade_type = "trending_trade"
#                     globals()['aa'] = '4'
#         # elif (dx_diff < 0) and (atr_p_diff > 0): # 내리면서 intersect 일경우, 마지막의 전상태로 방향  결정
#         elif (dx_diff < 0) and (atr_p_diff > 0) : # and (adx14_set_1 > 20):# and (adx_diff > 0) : # atr 오르고 adx 오를때 추세방향으로 진입

#             target_range = "wide"
#             stop_loss_range = "tight"
#             if intersect:
#                 if (dmn_set_2 > dmp_set_2):
#                     trend_result = "long"
#                     trade_type = "trending_trade"
#                     globals()['aa'] = '5'
#                 elif (dmp_set_2 > dmn_set_2):
#                     trend_result = "short"
#                     trade_type = "trending_trade"
#                     globals()['aa'] = '6'
#             else:
#                 if (dmn_set_1 > dmp_set_1):
#                     trend_result = "long"
#                     trade_type = "trending_trade"
#                     globals()['aa'] = '7'
#                 elif (dmp_set_1 > dmn_set_1):
#                     trend_result = "short"
#                     trade_type = "trending_trade"
#                     globals()['aa'] = '8'
#         else: # dx_diff == 0:
#             """
#             adx = 0 양방 카운트 단타, 역지를 넓히고 수익은 짧게 자주먹는다.
#             atr 는 계속오르므로 변동이 크고, adx는 고정되므로 곧 꺽일수있고 로컬 max pick 이 나올수 있음.
#             """
#             trend_result = ""
#             trade_type = "counter_trade"
#             target_range = "wide"
#             stop_loss_range = "tight"
#     else:
#         trend_result = ""
#         trade_type = "counter_trade"
#         target_range = "wide"
#         stop_loss_range = "tight"

#     # if ((interval == '1m') and(dx_diff > 0)) and ((dx14_set_1 < 25) or (dx14_set_1 < adx14_set_1) or (dx14_set_1 < dmn_set_1) or (dx14_set_1 < dmp_set_1)):        
#     #     if trend_result == "short":
#     #         trend_result = "long"
#     #     elif trend_result == "long":
#     #         trend_result = "short"
#     # print(interval, trend_result, (dt.datetime.now().replace(microsecond=0).isoformat()).split('T')[-1])        
#     return trend_result, trade_type, stop_loss_range, target_range

def tr_confirm(interval):
    # interval = '15m'
    if interval:

        atr_p_set_1 = globals()['df_' + interval]['atr_p'].iloc[-1]
        atr_set_1 = globals()['df_' + interval]['ATRr_14'].iloc[-1]

        adx14_set_1 = globals()['df_' + interval]['ADX_14'].iloc[-1]
        dx14_set_1 = globals()['df_' + interval]['DX_14'].iloc[-1]
        dmn_set_1 = globals()['df_' + interval]['DMN_14'].iloc[-1]
        dmp_set_1 = globals()['df_' + interval]['DMP_14'].iloc[-1]

        adx5_set_1 = globals()['df_' + interval]['ADX_5'].iloc[-1]
        dx5_set_1 = globals()['df_' + interval]['DX_5'].iloc[-1]
        dmn5_set_1 = globals()['df_' + interval]['DMN_5'].iloc[-1]
        dmp5_set_1 = globals()['df_' + interval]['DMP_5'].iloc[-1]

        adx200_set_1 = globals()['df_' + interval]['ADX_200'].iloc[-1]
        dx200_set_1 = globals()['df_' + interval]['DX_200'].iloc[-1]
        dmn200_set_1 = globals()['df_' + interval]['DMN_200'].iloc[-1]
        dmp200_set_1 = globals()['df_' + interval]['DMP_200'].iloc[-1]

        macd_set_1 = globals()['df_' + interval]['MACD_12_26_9'].iloc[-1]
        macdh_set_1 = globals()['df_' + interval]['MACDh_12_26_9'].iloc[-1]
        macd_200_set_1 = globals()['df_' + interval]['MACD_50_200_9'].iloc[-1]
        macdh_200_set_1 = globals()['df_' + interval]['MACDh_50_200_9'].iloc[-1]

        macd_set_2 = globals()['df_' + interval]['MACD_12_26_9'].iloc[-2]
        macdh_set_2 = globals()['df_' + interval]['MACDh_12_26_9'].iloc[-2]
        macd_200_set_2 = globals()['df_' + interval]['MACD_50_200_9'].iloc[-2]
        macdh_200_set_2 = globals()['df_' + interval]['MACDh_50_200_9'].iloc[-2]


        j_set_1 = globals()['df_' + interval]['J_9_3'].iloc[-1]
        k_set_1 = globals()['df_' + interval]['K_9_3'].iloc[-1]
        d_set_1 = globals()['df_' + interval]['D_9_3'].iloc[-1]

        rsi_6_set_1 = globals()['df_' + interval]['RSI_6'].iloc[-1]
        rsi_12_set_1 = globals()['df_' + interval]['RSI_12'].iloc[-1]
        rsi_24_set_1 = globals()['df_' + interval]['RSI_24'].iloc[-1]

        atr_p_set_2 = globals()['df_' + interval]['atr_p'].iloc[-2]
        atr_set_2 = globals()['df_' + interval]['ATRr_14'].iloc[-2]

        adx14_set_2 = globals()['df_' + interval]['ADX_14'].iloc[-2]
        dx14_set_2 = globals()['df_' + interval]['DX_14'].iloc[-2]
        dmn_set_2 = globals()['df_' + interval]['DMN_14'].iloc[-2]
        dmp_set_2 = globals()['df_' + interval]['DMP_14'].iloc[-2]

        adx5_set_2 = globals()['df_' + interval]['ADX_5'].iloc[-2]
        dx5_set_2 = globals()['df_' + interval]['DX_5'].iloc[-2]
        dmn5_set_2 = globals()['df_' + interval]['DMN_5'].iloc[-2]
        dmp5_set_2 = globals()['df_' + interval]['DMP_5'].iloc[-2]

        adx200_set_2 = globals()['df_' + interval]['ADX_200'].iloc[-2]
        dx200_set_2 = globals()['df_' + interval]['DX_200'].iloc[-2]
        dmn200_set_2 = globals()['df_' + interval]['DMN_200'].iloc[-2]
        dmp200_set_2 = globals()['df_' + interval]['DMP_200'].iloc[-2]

        j_set_2 = globals()['df_' + interval]['J_9_3'].iloc[-2]
        k_set_2 = globals()['df_' + interval]['K_9_3'].iloc[-2]
        d_set_2 = globals()['df_' + interval]['D_9_3'].iloc[-2]

        rsi_6_set_2 = globals()['df_' + interval]['RSI_6'].iloc[-2]
        rsi_12_set_2 = globals()['df_' + interval]['RSI_12'].iloc[-2]
        rsi_24_set_2 = globals()['df_' + interval]['RSI_24'].iloc[-2]

        atr_p_diff = globals()['df_' + interval]['atr_p_diff'].iloc[-1]
        atr_diff = globals()['df_' + interval]['atr_diff'].iloc[-1]

        atr5_p_diff = globals()['df_' + interval]['atr5_p_diff'].iloc[-1]
        atr5_diff = globals()['df_' + interval]['atr5_diff'].iloc[-1]

        atr200_p_diff = globals()['df_' + interval]['atr200_p_diff'].iloc[-1]
        atr200_diff = globals()['df_' + interval]['atr200_diff'].iloc[-1]

        adx_diff = globals()['df_' + interval]['adx_diff'].iloc[-1]
        dx_diff = globals()['df_' + interval]['dx_diff'].iloc[-1]

        adx_5_diff = globals()['df_' + interval]['adx_5_diff'].iloc[-1]
        dx_5_diff = globals()['df_' + interval]['dx_5_diff'].iloc[-1]
        dmp_5_diff = globals()['df_' + interval]['dmp_5_diff'].iloc[-1]
        dmn_5_diff = globals()['df_' + interval]['dmn_5_diff'].iloc[-1]

        adx_200_diff = globals()['df_' + interval]['adx_200_diff'].iloc[-1]
        dx_200_diff = globals()['df_' + interval]['dx_200_diff'].iloc[-1]

        rsi_diff = globals()['df_' + interval]['rsi_diff'].iloc[-1]

        j_diff = globals()['df_' + interval]['j_diff'].iloc[-1]
        k_diff = globals()['df_' + interval]['k_diff'].iloc[-1]
        d_diff = globals()['df_' + interval]['d_diff'].iloc[-1]
        
        atr14_set = globals()['df_' + interval]['ATRr_14'].iloc[-1]
        close_set = globals()['df_' + interval]['close'].iloc[-1]
        macd_diff = globals()['df_' + interval]['macd_diff'].iloc[-1]
        macdh_diff = globals()['df_' + interval]['macdh_diff'].iloc[-1]
        macd_diff_200 = globals()['df_' + interval]['macd_diff_200'].iloc[-1]
        macdh_diff_200 = globals()['df_' + interval]['macdh_diff_200'].iloc[-1]


        high_diff = globals()['df_' + interval]['high_diff'].iloc[-1]
        low_diff = globals()['df_' + interval]['low_diff'].iloc[-1]
        close_diff = globals()['df_' + interval]['close_diff'].iloc[-1]

        dmn_diff = globals()['df_' + interval]['dmn_diff'].iloc[-1]
        dmp_diff = globals()['df_' + interval]['dmp_diff'].iloc[-1]

        dmn_200_diff = globals()['df_' + interval]['dmn_200_diff'].iloc[-1]
        dmp_200_diff = globals()['df_' + interval]['dmp_200_diff'].iloc[-1]

        rsi_AMATe_LR_8_21_2 = globals()['df_' + interval]['rsi_AMATe_LR_8_21_2'].iloc[-1]
        kdj_AMATe_LR_8_21_2 = globals()['df_' + interval]['kdj_AMATe_LR_8_21_2'].iloc[-1]
        high_AMATe_LR_8_21_2 = globals()['df_' + interval]['high_AMATe_LR_8_21_2'].iloc[-1]

        rsi_AMATe_SR_8_21_2 = globals()['df_' + interval]['rsi_AMATe_SR_8_21_2'].iloc[-1]
        kdj_AMATe_SR_8_21_2 = globals()['df_' + interval]['kdj_AMATe_SR_8_21_2'].iloc[-1]
        low_AMATe_SR_8_21_2 = globals()['df_' + interval]['low_AMATe_SR_8_21_2'].iloc[-1]

        obv_diff = globals()['df_' + interval]['obv_diff'].iloc[-1]

        rsi_6_diff = globals()['df_' + interval]['rsi_6_diff'].iloc[-1]
        rsi_12_diff = globals()['df_' + interval]['rsi_12_diff'].iloc[-1]
        rsi_24_diff = globals()['df_' + interval]['rsi_24_diff'].iloc[-1]

        bbp21_set_1 = globals()['df_' + interval]['BBP_21_2.0'].iloc[-1]
        bbb_diff = globals()['df_' + interval]['bbb_diff'].iloc[-1]

        maxima_peak_x_close_set_1 = globals()['df_' + interval]['maxima_peak_x_close'].iloc[-1]
        maxima_peak_x_macd_set_1 = globals()['df_' + interval]['maxima_peak_x_MACD_12_26_9'].iloc[-1]
        maxima_peak_x_rsi_set_1 = globals()['df_' + interval]['maxima_peak_x_rsi'].iloc[-1]
        maxima_peak_x_dmp_set_1 = globals()['df_' + interval]['maxima_peak_x_dmp'].iloc[-1]
        minima_peak_x_dmn_set_1 = globals()['df_' + interval]['minima_peak_x_dmn'].iloc[-1]

        minima_peak_x_close_set_1 = globals()['df_' + interval]['minima_peak_x_close'].iloc[-1]
        minima_peak_x_macd_set_1 = globals()['df_' + interval]['minima_peak_x_MACD_12_26_9'].iloc[-1]
        minima_peak_x_rsi_set_1 = globals()['df_' + interval]['minima_peak_x_rsi'].iloc[-1]
        minima_peak_x_dmp_set_1 = globals()['df_' + interval]['minima_peak_x_dmp'].iloc[-1]
        maxima_peak_x_dmn_set_1 = globals()['df_' + interval]['maxima_peak_x_dmn'].iloc[-1]

        trend_result = trade_type = target_range = stop_loss_range = intersect = ""
        if (dmn_set_2 > dmp_set_2) & (dmn_set_1 > dmp_set_1):
            intersect = 0
        elif (dmn_set_2 > dmp_set_2) & (dmn_set_1 < dmp_set_1):
            intersect = 1
        elif (dmn_set_2 < dmp_set_2) & (dmn_set_1 > dmp_set_1):
            intersect = 1
        elif (dmn_set_2 < dmp_set_2) & (dmn_set_1 < dmp_set_1):
            intersect = 0
        # print('\n----------')
        # print(f'atr_diff: {atr_diff}, dx_diff: {dx_diff}, dmn_set: {dmn_set}, dmp_set: {dmp_set}')
        # print(f'\
        #     [atr_p_set_1]: {round(atr_p_set_1, 3)}, [atr_p_diff]: {round(atr_p_diff, 3)}\n\
        #     [atr_set_1]: {round(atr_set_1, 3)}, [atr_diff]: {round(atr_diff, 3)}\n\
        #     [adx14_set_1]: {round(adx14_set_1, 3)}, [adx_diff]: {round(adx_diff, 3)}\n\
        #     [dx14_set_1]: {round(dx14_set_1, 3)}, [dx_diff]: {round(dx_diff, 3)}\n\
        #     [dmp_set_1]: {round(dmp_set_1, 3)}, [dmn_set_1]: {round(dmn_set_1, 3)}\
        # ')
        # trending_trade 일때는 진입직전 추세확인 필요
        # counter_trade 일때는 진입직전 추세확인 불필요

        # if (dx_diff < 0) and (dmn_set_1 > dmp_set_1) and (df.intersect == 0): # and (atr_p_diff > 0) : # and (adx14_set_1 > 20):#  and (adx_diff > 0) : # atr 오르고 adx 오를때 추세방향으로 진입
        if (
                # (
                #     # (atr_diff > 0) & (obv_diff > 0) & (adx_diff > 0) & (j_diff > 0)& (k_diff > 0)& (d_diff > 0) & (j_set_2 < d_set_2) & (j_set_1 > d_set_1)
                #     (obv_diff > 0) & (j_diff > 0)& (k_diff > 0)& (d_diff > 0) & (j_set_2 < d_set_2) & (j_set_1 > d_set_1)

                # )
                #     &
                # (
                #     # (atr_diff > 0) & (df.obv_diff > 0) & (df.rsi_6_diff > 0)& (df.rsi_12_diff > 0)& (df.rsi_24_diff > 0) & (rsi_6_set_2 < rsi_24_set_2) & (rsi_6_set_1 > rsi_24_set_1)
                #     (obv_diff > 0) & (rsi_6_diff > 0)& (rsi_12_diff > 0)& (rsi_24_diff > 0) & (rsi_6_set_2 < rsi_24_set_2) & (rsi_6_set_1 > rsi_24_set_1)

                # )



                # (
                #     # (atr_diff > 0) & (df.obv_diff > 0) & (df.rsi_6_diff > 0)& (df.rsi_12_diff > 0)& (df.rsi_24_diff > 0) & (rsi_6_set_2 < rsi_24_set_2) & (rsi_6_set_1 > rsi_24_set_1)
                #     (atr_diff > 0) & (macdh_set_1 > 0) & (macdh_diff > 0)

                # )

                ########################## 됨2 ##########################
                #########################################################
                #########################################################

                # (
                #     # (macdh_set_1 > 0) & (macdh_diff > 0) & (j_diff > 0) & (atr_p_diff > 0)
                #     (macdh_200_set_2 < 0) and (macdh_200_set_1 > 0) and (macdh_diff_200 > 0) and (rsi_diff > 0) # and (atr200_p_diff > 0) # & (adx_200_diff > 0)# & (atr_p_diff > 0)
                #     # (atr_diff > 0) & (macd_diff_200 > 0) & (macdh_200_set_1 > 0) & (macdh_diff_200 > 0)
                # )

                ########################## 됨2 ##########################
                #########################################################
                #########################################################
                # (
                    
                #     (macd_200_set_1 > 0) & (macdh_200_set_1 > 0) & (macdh_diff_200 > 0) & (macd_diff_200 > 0) # & (df.MACDh_7_200_9 > 0) & (df['MACDh_7_200_9'].shift(1) < 0) 
                #     & (j_diff > 0)
                #     & (obv_diff > 0)
                #     & (rsi_diff > 0)
                # )
                #     &~
                # (


                #     (
                #         (adx_diff < 0) & (atr_diff < 0) 
                #     )
                #         |
                #     (
                #         (atr_diff > 0) 
                #         & (j_diff > 0)& (k_diff > 0)& (d_diff > 0)
                #         & (rsi_6_diff > 0)& (rsi_12_diff > 0)& (rsi_24_diff > 0)
                #     )     
                # )
                #########################################################
                #########################################################
                #########################################################

                # (
                #     (atr5_p_diff > 0) and (adx_5_diff > 0) # and (adx_5_diff > 0)#  and (adx_200_diff > 0) and (rsi_diff > 0)# and (atr_p_diff > 0) and (j_diff > 0)
                # )
                #     and
                # (
                #     (
                #         (dx_5_diff < 0) and (dmn5_set_1 > dmp5_set_1) and (intersect == 0) and (dmp_5_diff > 0) # and (adx_5_diff > 0)#  and (adx_200_diff > 0) and (rsi_diff > 0)# and (atr_p_diff > 0) and (j_diff > 0)
                #     )
                #         |
                #     (
                #         (dx_5_diff < 0) and (dmn5_set_2 > dmp5_set_2) and (intersect > 0) and (dmp_5_diff > 0) # and (adx_5_diff > 0)#  and (adx_200_diff > 0) and (rsi_diff > 0)# and (atr_p_diff > 0) and (j_diff > 0)
                #     )
                #         |
                #     (
                #         (dx_5_diff > 0) and (dmp5_set_1 > dmn5_set_1) and (dmp_5_diff > 0) # and (adx_5_diff > 0)#  and (rsi_diff > 0)# and (atr_p_diff > 0) and (j_diff > 0)
                #     )
                # )

                #########################################################
                #########################################################
                #########################################################

                (
                    (
                        (
                            (dx_diff < 0) and (dmn_set_1 > dmp_set_1) and (intersect == 0) and (atr_diff > 0)# and (macdh_diff > 0) # and (atr_p_diff > 0)
                        )
                            |
                        (
                            (dx_diff < 0) and (dmn_set_2 > dmp_set_2) and (intersect > 0) and (atr_diff > 0)# and (macdh_diff > 0) # and (atr_p_diff > 0)
                        )
                            |
                        (
                            (dx_diff > 0) and (dmp_set_1 > dmn_set_1) and (atr_diff > 0)# and (macdh_diff > 0) # and (atr_p_diff > 0)
                        )
                    )
                        and
                    (
                        (bbp21_set_1 > 0.5)
                    )
                        and
                    (
                        (bbb_diff > 0)
                    )
                )
                &~(
                    (maxima_peak_x_close_set_1 < 0) | (maxima_peak_x_macd_set_1 < 0) | (maxima_peak_x_rsi_set_1 < 0) | (maxima_peak_x_dmp_set_1 < 0) | (minima_peak_x_dmn_set_1 < 0)
                )
            ):
            target_range = "wide"
            stop_loss_range = "tight"
            trend_result = "long"
            trade_type = "trending_trade"
            globals()['aa'] = '1'
        elif (


                # (
                #     # (atr_diff > 0) & (obv_diff < 0) & (adx_diff > 0) & (j_diff < 0)& (k_diff < 0)& (d_diff < 0) & (j_set_2 > d_set_2) & (j_set_1 < d_set_1)
                #     (obv_diff < 0) & (j_diff < 0)& (k_diff < 0)& (d_diff < 0) & (j_set_2 > d_set_2) & (j_set_1 < d_set_1)
                # )
                #     &
                # (
                #     # (atr_diff > 0) & (df.obv_diff < 0) & (df.rsi_6_diff < 0)& (df.rsi_12_diff < 0)& (df.rsi_24_diff < 0) & (rsi_6_set_2 > rsi_24_set_2) & (rsi_6_set_1 < rsi_24_set_1)
                #     (obv_diff < 0) & (rsi_6_diff < 0)& (rsi_12_diff < 0)& (rsi_24_diff < 0) & (rsi_6_set_2 > rsi_24_set_2) & (rsi_6_set_1 < rsi_24_set_1)

                # )


                # (
                #     # (atr_diff > 0) & (df.obv_diff > 0) & (df.rsi_6_diff > 0)& (df.rsi_12_diff > 0)& (df.rsi_24_diff > 0) & (rsi_6_set_2 < rsi_24_set_2) & (rsi_6_set_1 > rsi_24_set_1)
                #     (atr_diff > 0) & (macdh_set_1 < 0) & (macdh_diff < 0)

                # )
                ########################## 됨2 ##########################
                #########################################################
                #########################################################
                # (
                #     # (macdh_set_1 < 0) & (macdh_diff < 0) & (j_diff < 0) & (atr_p_diff > 0)
                #     (macdh_200_set_2 > 0) and (macdh_200_set_1 < 0) and (macdh_diff_200 < 0) and (rsi_diff < 0) # and (atr200_p_diff > 0) # & (adx_200_diff > 0)# & (atr_p_diff > 0)
                #     # (atr_diff > 0) & (macd_diff_200 < 0) & (macdh_200_set_1 < 0) & (macdh_diff_200 < 0)
                # )
                ########################## 됨2 ##########################
                #########################################################
                #########################################################
                # (
                #     (macd_200_set_1 < 0) & (macdh_200_set_1 < 0) & (macdh_diff_200 < 0) & (macd_diff_200 < 0) # & (df.MACDh_7_200_9 > 0) & (df['MACDh_7_200_9'].shift(1) < 0) 
                #     & (j_diff < 0)
                #     & (obv_diff < 0)
                #     & (rsi_diff < 0)
                # )
                #     &~
                # (
                #     (
                #         (adx_diff < 0) & (atr_diff < 0) 
                #     )
                #         |
                #     (
                #         (atr_diff > 0) 
                #         & (j_diff < 0)& (k_diff < 0)& (d_diff < 0)
                #         & (rsi_6_diff < 0)& (rsi_12_diff < 0)& (rsi_24_diff < 0)
                #     )
                # )
                #########################################################
                #########################################################
                #########################################################
                # (
                #     (atr5_p_diff > 0) and (adx_5_diff > 0) # and (adx_5_diff > 0)#  and (adx_200_diff > 0) and (rsi_diff > 0)# and (atr_p_diff > 0) and (j_diff > 0)
                # )
                #     and
                # (
                #     (
                #         (dx_5_diff < 0) and (dmp5_set_1 > dmn5_set_1) and (intersect == 0) and (dmn_5_diff > 0) # and (adx_5_diff > 0)# and (adx_200_diff > 0) and (rsi_diff < 0)# and (atr_p_diff > 0) and (j_diff < 0)
                #     )
                #         |
                #     (
                #         (dx_5_diff < 0) and (dmp5_set_2 > dmn5_set_2) and (intersect > 0) and (dmn_5_diff > 0) # and (adx_5_diff > 0)# and (adx_200_diff > 0) and (rsi_diff < 0)# and (atr_p_diff > 0) and (j_diff < 0)
                #     )
                #         |
                #     (
                #         (dx_5_diff > 0) and (dmn5_set_1 > dmp5_set_1) and (dmn_5_diff > 0) # and (adx_5_diff > 0)# and (adx_200_diff > 0) and (rsi_diff < 0)# and (atr_p_diff > 0) and (j_diff < 0)
                #     )
                # )
                #########################################################
                #########################################################
                #########################################################

                (
                    (
                        (
                            (dx_diff < 0) and (dmp_set_1 > dmn_set_1) and (intersect == 0) and (atr_diff > 0)# and (macdh_diff < 0) # and (atr_p_diff > 0)
                        )
                            |
                        (
                            (dx_diff < 0) and (dmp_set_2 > dmn_set_2) and (intersect > 0) and (atr_diff > 0)# and (macdh_diff < 0) # and (atr_p_diff > 0)
                        )
                            |
                        (
                            (dx_diff > 0) and (dmn_set_1 > dmp_set_1) and (atr_diff > 0)# and (macdh_diff < 0) # and (atr_p_diff > 0)
                        )
                    )
                        and
                    (
                        (bbp21_set_1 < 0.5)
                    )
                        and
                    (
                        (bbb_diff > 0)
                    )
                )
                &~(
                    (minima_peak_x_close_set_1 < 0) | (minima_peak_x_macd_set_1 < 0) | (minima_peak_x_rsi_set_1 < 0) | (minima_peak_x_dmp_set_1 < 0) | (maxima_peak_x_dmn_set_1 < 0)
                )
            ):
            target_range = "wide"
            stop_loss_range = "tight"
            trend_result = "short"
            trade_type = "trending_trade"
            globals()['aa'] = '2'

        else:
            target_range = "wide"
            stop_loss_range = "tight"
            trend_result = ""
            trade_type = "trending_trade"
            globals()['aa'] = '3'

    else:
        trend_result = ""
        trade_type = "counter_trade"
        target_range = "wide"
        stop_loss_range = "tight"
        globals()['aa'] = '5'
    
    return trend_result, trade_type, stop_loss_range, target_range
############################################ back_tester #######################################################################################################
def back_tester_1(bal, position_size, position_entry_price, scale_order_position_amount, price):
    if (position_size*scale_order_position_amount >= 0):
        position_entry_price = ((position_size * position_entry_price) + (scale_order_position_amount * price)) / (position_size + scale_order_position_amount)
        position_size += scale_order_position_amount
        if scale_order_position_amount > 0:
            fee = price * scale_order_position_amount * 0.0008
            fee = round(fee, 3)
            pnl = 0
            bal = bal - fee + pnl
            bal = round(bal, 3)
        else:
            fee = price * scale_order_position_amount * -1 * 0.0008
            fee = round(fee, 3)
            pnl = 0
            bal = bal - fee + pnl
            bal = round(bal, 3)
    else: #(position_size*scale_order_position_amount < 0)
        if position_size*position_size >= scale_order_position_amount*scale_order_position_amount: # position_size > scale_order_position_amount
            if scale_order_position_amount > 0:
                fee = price * scale_order_position_amount * 0.0008
                fee = round(fee, 3)
                pnl = (position_entry_price - price)*scale_order_position_amount
                pnl = round(pnl, 3)
                bal = bal - fee + pnl
                bal = round(bal, 3)
            else:
                fee = price * scale_order_position_amount * -1 * 0.0008
                fee = round(fee, 3)
                pnl = (price - position_entry_price)*scale_order_position_amount * -1
                pnl = round(pnl, 3)
                bal = bal - fee + pnl
                bal = round(bal, 3)
            position_entry_price = position_entry_price
            position_size += scale_order_position_amount
        else : # position_size < scale_order_position_amount
            if scale_order_position_amount > 0:
                fee = price * scale_order_position_amount * 0.0008
                fee = round(fee, 3)
                if position_size > 0:
                    pnl = (price - position_entry_price)*position_size
                    pnl = round(pnl, 3)
                else: # position_size < 0:
                    pnl = (price - position_entry_price)*position_size
                    pnl = round(pnl, 3)
                bal = bal - fee + pnl
                bal = round(bal, 3)
            else:
                fee = price * scale_order_position_amount * -1 * 0.0008
                fee = round(fee, 3)
                if position_size > 0:

                    pnl = (price - position_entry_price)*position_size
                    pnl = round(pnl, 3)
                else: # position_size < 0:
                    pnl = (price - position_entry_price)*position_size * -1
                    pnl = round(pnl, 3)
                bal = bal - fee + pnl
                bal = round(bal, 3)
            position_entry_price = price
            position_size += scale_order_position_amount
    return bal, fee, pnl, position_size, position_entry_price

def balance_checker(bal, position_size, position_entry_price, price):
    # if position_size > 0:
    #     fee = price * position_size * 0.0004
    # else: # position_size < 0:
    #     fee = price * position_size * 0.0004 * -1
    fee = 0
    pnl = (price - position_entry_price) * position_size
    bal = round(bal - fee + pnl, 3)
    return bal


def back_tester_open_position_tracer(amount, position_entry_price):
    global position_size_
    global position_entry_price_

    position_size_ += amount
    
    if position_size_ == 0:
        position_entry_price_ = 0
    else:
        position_entry_price_ = position_entry_price

    return position_size_, position_entry_price_

def balance_tracer(amount, position_size, position_entry_price, current_price, bal, action):
    fee = 0
    if action:
        fee = current_price * abs(amount) * 0.08 /100
        pnl = (current_price - position_entry_price) * position_size
        bal = round(bal - fee + pnl, 3)
    else:
        pnl = (current_price - position_entry_price) * position_size
        bal = round(bal - fee + pnl, 3)
    return bal, pnl






def calculate_predicted_change(market_id):
    # 1. 데이터 가져오기
    df = klines(market_id=market_id, interval='1d', limit=2)

    # 2. 전날 데이터
    low_yesterday = df['low'][-2]
    high_yesterday = df['high'][-2]
    Volume_yesterday = df['volume'][-2]
    Change_yesterday = ((high_yesterday - low_yesterday) / low_yesterday) * 100  # 전날 변동성 (%)
    C_yesterday = Change_yesterday / Volume_yesterday  # 전날 변동 상수

    # 3. 오늘 데이터 (현재까지)
    low_today = df['low'][-1]
    high_today = df['high'][-1]
    Volume_today_so_far = df['volume'][-1]
    Change_today_so_far = ((high_today - low_today) / low_today) * 100  # 현재까지의 변동성 (%)
    C_today = Change_today_so_far / Volume_today_so_far  # 현재까지의 변동 상수

    ##########################################################################################
    # # 4. 오늘 최종 거래량 (예측 또는 마감 시점 데이터)
    # # 현재 시간을 UTC 기준으로 설정
    # current_time_utc = datetime.datetime.utcnow()  # UTC 기준 현재 시간

    # # CME 정규 거래 시작 시간 (UTC)
    # cme_open_time_utc = current_time_utc.replace(hour=14, minute=30, second=0, microsecond=0)

    # # 만약 시간이 이미 시작 시간 이전이라면, 남은 시간을 24시간 기준으로 설정
    # if current_time_utc < cme_open_time_utc:
    #     remaining_time_utc = (cme_open_time_utc - current_time_utc).total_seconds() / 3600  # 남은 시간 (단위: 시간)
    # else:
    #     remaining_time_utc = 24 - (current_time_utc - cme_open_time_utc).total_seconds() / 3600  # 남은 시간 (24시간 기준)

    ##########################################################################################
    # 5. 오늘 최종 거래량 (예측 또는 마감 시점 데이터)
    # 현재 시간을 UTC 기준으로 설정
    current_time_utc = datetime.datetime.utcnow()  # UTC 기준 현재 시간
    midnight_utc = current_time_utc.replace(hour=0, minute=0, second=0, microsecond=0) + datetime.timedelta(days=1)  # 다음날 UTC 자정 시간
    remaining_time_utc = (midnight_utc - current_time_utc).total_seconds() / 3600  # 남은 시간 (단위: 시간)
    ##########################################################################################

    # 거래량 예측: 현재까지 거래량 * 남은 시간 비율로 조정
    # Volume_today_final = Volume_today_so_far * (1 + remaining_time_utc / 24)  # 남은 시간을 반영한 거래량 예측
    Volume_today_final = (Volume_today_so_far * 24) / (24 - remaining_time_utc)

    ##########################################################################################
    # # 6. 오늘 최종 변동성 예측
    Predicted_Change_today = C_today * Volume_today_final



    # Volume_prediction_1 = Change_yesterday/C_today
    # Predicted_Change_today = Volume_prediction_1 * C_today
    ##########################################################################################

    return Predicted_Change_today

# def calculate_max_leverage(predicted_change):
#     # 변동성이 1보다 작을 경우, 최대 레버리지를 100으로 고정
#     if predicted_change < 1:
#         return 100
#     # 그렇지 않으면 기존 방식으로 계산
#     return 100 / predicted_change

def calculate_max_leverage(predicted_change, market_max_leverage, max_leverage=100, min_leverage=1):
    """
    변동성과 마켓에서 허용하는 최대 레버리지를 고려하여 최적 레버리지를 계산합니다.
    
    Args:
        predicted_change (float): 예상 변동성 (0 ~ 100%)
        market_max_leverage (int): 마켓에서 허용하는 최대 레버리지
        max_leverage (int): 계산된 최대 레버리지 (기본값 100)
        min_leverage (int): 최소 레버리지 (기본값 1)
    
    Returns:
        float: 최적 레버리지 값
    """
    # 변동성 제한
    if predicted_change < 1:
        predicted_change = 1  # 최소 변동성 1%로 고정
    elif predicted_change > 100:
        predicted_change = 100  # 최대 변동성 100%로 고정

    # 안전 마진 설정
    safety_margin = 1.5  # 예상 변동성 대비 여유 배수

    # 예측 변동성에 안전 마진 적용
    adjusted_change = predicted_change * safety_margin

    # 계산된 레버리지
    leverage = max_leverage / (1 + adjusted_change)

    # 최소, 최대 레버리지 제한
    leverage = max(min_leverage, min(leverage, max_leverage))

    # 마켓에서 허용하는 최대 레버리지와 비교하여 최종 조정
    if leverage > market_max_leverage:
        leverage = market_max_leverage

    return leverage

# 예시로 예측된 변동성을 5%로 설정
















###################################################################################################################################################
# loop 시작
# initial 변수 선언 초기값 세팅... maxima_minima_last_peak_time_3d_long = pd.Timestamp(2019, 12, 22, 13, 30, 59)
###################################################################################################################################################
#print('1:', (dt.datetime.now().replace(microsecond=0).isoformat()).split('T')[-1])















peak_calc(market_id, intervals) # df_1m, df_3m, df_5m, df_15m, df_30m, df_1h, df_2h, df_4h, df_6h, df_8h, df_12h, df_1d, df_3d = peak_calc(market_id, intervals) # run and create df!
for interval_tmp, side_key, strategy_value, current_last_value in itertools.product(globals()['valid_intervals'], side.keys(), strategy, current_last.values()):
    column_name = strategy_value + '_' + side_key
    variable_name = 'maxima_minima_' + current_last_value + '_peak_time_' + interval_tmp + '_' + side_key + '_' + strategy_value
    globals()[variable_name] = default_time

    if column_name in globals()['df_' + interval_tmp].columns:
        df_tmp = globals()['df_' + interval_tmp]
        threshold = globals()['counter_light_weight_' + side_key]

        mask = (df_tmp[column_name] < threshold) if side_key == 'short' else (df_tmp[column_name] > threshold)
        if mask.any():
            globals()[variable_name] = df_tmp.peak_time[mask][-1]
    if current_last_value == 'current':
        peaker_frame_appender(interval_tmp, side_key, strategy_value)

#print('default 변수 선언 완료')
initial_start_time = datetime.datetime.now().replace(microsecond=0)

message = f"Game Starts Now. Trader: {trader_name}"
send_to_telegram(symbol, message)
exit_status = '0. initiate engin'
# check_pointer(trader_name, exchange_id, market_id, balance_currency, loop, max_waiting_in_second, exit_status, message)
#print('4-', (dt.datetime.now().replace(microsecond=0).isoformat()).split('T')[-1])
############################################# back_tester #######################################################################################################
# bal_1 = bal_2 = 100
# pnl_ = 0
# position_size_ = 0
# position_entry_price_ = 0
# position_side_ = ''
############################################# back_tester #######################################################################################################

###########################################################################
# while True:
#     if position_side == 'long': # 포지션 롱 일때
#         if (scaled_level_n < 0.8) or (stric_exit_min_price > symbol_ticker_last) : # 물타기 아직 없거나, 손실중이라면 atr 역지
#             - atr 역지
#         else: # 물타기 했거나, 이익중이라면 stric_exit_min_price 과 chandelier_exit_long 중 큰값으로 선택해서 익절 역지
#             if (symbol_ticker_last > chandelier_exit_long) and (symbol_ticker_last > stric_exit_min_price): # 현재가가  수익권 이면서, chandelier보다도 클때
#                 - max(stric_exit_min_price, chandelier_exit_long)  익절 역지
#             elif (symbol_ticker_last < chandelier_exit_short): # 현재가가  수익권 이지만, chandelier보다는 작을때
#                 - min(stric_exit_min_price, chandelier_exit_long)  익절 역지
#         - exit_order를 재산출하여 새롭게 생성

#     elif position_side == 'short': # 포지션 롱 일때
#         if (scaled_level_n < 0.8) or (stric_exit_min_price < symbol_ticker_last) : # 물타기 아직 없거나, 손실중이라면 atr 역지
#             - atr 역지
#         else: # 물타기 했거나, 이익중이라면 stric_exit_min_price 과 chandelier_exit_short 중 작은값으로 선택해서 익절 역지
#             if (symbol_ticker_last < chandelier_exit_short) and (symbol_ticker_last < stric_exit_min_price): # 현재가가  수익권 이면서, chandelier보다도 작을때
#                 - min(stric_exit_min_price, chandelier_exit_short)  익절 역지
#             elif (symbol_ticker_last > chandelier_exit_short): # 현재가가  수익권 이지만, chandelier보다는 클때
#                 - max(stric_exit_min_price, chandelier_exit_short)  익절 역지
#         - exit_order를 재산출하여 새롭게 생성
#     ###########################################################################
#     if success:
#         ###########################################################################
#         if (peaker_side =='long') and (peaker_option == 'forward'):
#             if (position_size != 0): # 포지션존재
#                 if position_side == 'long': # 포지션 롱 일때
#                     if (stric_exit_min_price > symbol_ticker_last): # 1. 손실중 & 포지션이랑 peaker 방향같을때
#                         - 물타기
#                         if 현재 position_value 가 25배 넘을경우:
#                             - atr 역지
#                     else: # (stric_exit_min_price < symbol_ticker_last): # 2. 수익중 & 포지션이랑 peaker 방향같을때,  => 불타기
#                         - 불타기
#                         if (symbol_ticker_last > chandelier_exit_long) and (symbol_ticker_last > stric_exit_min_price): # 현재가가  수익권 이면서, chandelier보다도 클때
#                             - max(stric_exit_min_price, chandelier_exit_long)  익절 역지
#                         elif (symbol_ticker_last < chandelier_exit_short): # 현재가가  수익권 이지만, chandelier보다는 작을때
#                             - min(stric_exit_min_price, chandelier_exit_long)  익절 역지
#                     - exit_order를 재산출하여 새롭게 생성

#                 elif position_side == 'short': # 포지션 숏 일때
#                     if (stg_type == 'stg5') or (stg_type == 'stg1'):
#                         -포지션 시장가 종료
#                     else:
#                         if (scaled_level_n < 0.8) or (stric_exit_min_price < symbol_ticker_last) : # 물타기 아직 없거나, 손실중이라면 atr 역지
#                             - atr 역지
#                         else: # 물타기 했거나, 이익중이라면 stric_exit_min_price 과 chandelier_exit_short 중 작은값으로 선택해서 익절 역지
#                             if (symbol_ticker_last < chandelier_exit_short) and (symbol_ticker_last < stric_exit_min_price): # 현재가가  수익권 이면서, chandelier보다도 작을때
#                                 - min(stric_exit_min_price, chandelier_exit_short)  익절 역지
#                             elif (symbol_ticker_last > chandelier_exit_short): # 현재가가  수익권 이지만, chandelier보다는 클때
#                                 - max(stric_exit_min_price, chandelier_exit_short)  익절 역지
#                         - exit_order를 재산출하여 새롭게 생성

#             else: # 포지션 없을때
#                 - 'long' 진입
#         ##########################################################################
#         # elif (peaker_side =='long') and (peaker_option == 'reverse'):
#         #     if position_side == 'long':
#         #         익절
#         ##########################################################################
#         elif (peaker_side =='short') and (peaker_option == 'forward'):
#             if (position_size != 0): # 포지션존재
#                 if position_side == 'long': # 포지션 롱 일때
#                     if (stg_type == 'stg5') or (stg_type == 'stg1'):
#                         -포지션 시장가 종료
#                     else:
#                         if (scaled_level_n < 0.8) or (stric_exit_min_price > symbol_ticker_last) : # 물타기 아직 없거나, 손실중이라면 atr 역지
#                             - atr 역지
#                         else: # 물타기 했거나, 이익중이라면 stric_exit_min_price 과 chandelier_exit_long 중 큰값으로 선택해서 익절 역지
#                             if (symbol_ticker_last > chandelier_exit_long) and (symbol_ticker_last > stric_exit_min_price): # 현재가가  수익권 이면서, chandelier보다도 클때
#                                 - max(stric_exit_min_price, chandelier_exit_long)  익절 역지
#                             elif (symbol_ticker_last < chandelier_exit_short): # 현재가가  수익권 이지만, chandelier보다는 작을때
#                                 - min(stric_exit_min_price, chandelier_exit_long)  익절 역지
#                         - exit_order를 재산출하여 새롭게 생성

#                 elif position_side == 'short': # 포지션 롱 일때
#                     if (stric_exit_min_price < symbol_ticker_last): # 1. 손실중 & 포지션이랑 peaker 방향같을때
#                         - 물타기
#                         if 현재 position_value 가 25배 넘을경우:
#                             - atr 역지

#                     else: # (stric_exit_min_price < symbol_ticker_last): # 2. 수익중 & 포지션이랑 peaker 방향같을때,  => 불타기
#                         - 불타기
#                         if (symbol_ticker_last < chandelier_exit_short) and (symbol_ticker_last < stric_exit_min_price): # 현재가가  수익권 이면서, chandelier보다도 작을때
#                             - min(stric_exit_min_price, chandelier_exit_short)  익절 역지
#                         elif (symbol_ticker_last > chandelier_exit_short): # 현재가가  수익권 이지만, chandelier보다는 클때
#                             - max(stric_exit_min_price, chandelier_exit_short)  익절 역지
#                     - exit_order를 재산출하여 새롭게 생성

#             else: # 포지션 없을때
#                 - 'short' 진입
#         #########################################################################
#         # elif (peaker_side =='short') and (peaker_option == 'reverse'):
#         #     if position_side == 'short':
#         #         익절
#         #########################################################################


reverse_confirmer = 0
stg_type_fixed = ''
stop_loss_ = 0
max_attempts = 10
retry_delay = 30
attempts = 0
error_message = ['too much', 'too many', 'internal error', 'no longer available', 'overloaded']
instance = 5
# max_leverage = 50/instance
ah_ = 1
clear_cnt = 0
p_clear_cnt = 0
######################################################################################################################################################
if exchange_id == 'binanceusdm':

    try:
        # 마진 모드 변경
        margin_mode = 'CROSS'  # 'ISOLATED' 또는 'CROSS'
        response = exchange.set_margin_mode(symbol=symbol, marginMode=margin_mode)
    except Exception as e:
        print("오류 발생:", str(e))

    # Set initial parameters
    # symbol = "RENUSDT"
    initial_leverage = 50  # Starting leverage
    min_leverage = 1        # Minimum leverage to attempt
    c_l = initial_leverage
    t_s = 15

    # Try setting leverage
    while c_l >= min_leverage:
        try:
            rp = exchange.fapiprivate_post_leverage({
                'symbol': symbol,
                'leverage': c_l
            })
            print(f"Leverage set successfully to {c_l}: {rp}")
            break  # Exit loop if successful
        except Exception as e:
            print(f"Failed to set leverage to {c_l}: {str(e)}")
            if c_l > min_leverage:
                print(f"Retrying after {t_s} seconds...")
                time.sleep(t_s)  # Wait for t_s seconds before retrying
            c_l -= 1  # Decrease leverage and try again

    # If no leverage value worked
    if c_l < min_leverage:
        print("Unable to set leverage within the allowed range.")
elif exchange_id == 'bybit':

    try:
        # 마진 모드 변경
        margin_mode = 'cross'  # 'ISOLATED' 또는 'CROSS'
        response = exchange.set_margin_mode(symbol=l_s, marginMode=margin_mode)
    except Exception as e:
        print("오류 발생:", str(e))

    # Set initial parameters
    # symbol = "RENUSDT"
    initial_leverage = 50  # Starting leverage
    min_leverage = 1        # Minimum leverage to attempt
    c_l = initial_leverage
    t_s = 15

    # Try setting leverage
    while c_l >= min_leverage:
        try:
            rp = exchange.set_leverage(c_l, l_s)
            print(f"Leverage set successfully to {c_l}: {rp}")
            break  # Exit loop if successful
        except Exception as e:
            print(f"Failed to set leverage to {c_l}: {str(e)}")
            if c_l > min_leverage:
                print(f"Retrying after {t_s} seconds...")
                time.sleep(t_s)  # Wait for t_s seconds before retrying
            c_l -= 1  # Decrease leverage and try again

    # If no leverage value worked
    if c_l < min_leverage:
        c_l = min_leverage
        print("Unable to set leverage within the allowed range.")
market_max_leverage = c_l  # c_l이 이미 마켓에서 허용하는 최대 레버리지
######################################################################################################################################################
predicted_change = calculate_predicted_change(market_id)
max_leverage = calculate_max_leverage(predicted_change, market_max_leverage)*3
lev_limit = calculate_max_leverage(predicted_change, market_max_leverage)*3
r = 1.6
stopPrice_const = 2
atr_const = 0.7 # 70%
atr_const2 = .87 # 100%
taker_fee = 0.12
total_played = 0
total_wins = 0
total_losses = 0
win_rate = 0
loss_rate = 0

timesp, symbol_ticker_last, last_24hr_volatility = ticker_calc(market_id)
raw_min_qty = max(float(min_cost_in_usdt)/float(symbol_ticker_last), float(min_order_qty))
round_up_min_qty = round_up(raw_min_qty, market_amount_precision)
min_order_amount = float(exchange.amount_to_precision(market_id, round_up_min_qty))
wallet_balance = wallet_balance_fix = balance_calc(exchange_id, balance_currency)
position_side, position_size, position_value, position_entry_price, position_entry_time, liquidation_price, unrealised_pnl, roe_pcnt = position_calc(exchange_id, market_id)
scale_order_position_amount_calc_val = scale_order_position_amount_calc(min_order_amount, wallet_balance, max_leverage, position_size, r, scale_order_max_limit)
initial_order_amount = scale_order_position_amount_calc_val[1]
scale_order_position_amount = scale_order_position_amount_calc_val[2]
scaled_level_n = scale_order_position_amount_calc_val[3]
bul_scale_order_position_amount = scale_order_position_amount_calc_val[4]
stric_exit_min_price = stric_exit_price_calc_min(position_side, position_entry_price)
stop_loss_pnl = stop_loss_price_and_pnl_calc(wallet_balance, position_size, position_side, position_entry_price, stop_lose_limit_percent_range)[1]
new_position_size = initial_order_amount

stric_exit_min_price = 0
l_p_l = 0
l_p_h = 0
last_peaked_price = 0
######################################################################################################################################################
while True:
    try:
        exchange.load_time_difference()
        ah_ += 1
        positive_pnl = 0
        formatted_datetime = datetime.datetime.now().replace(microsecond=0)
        # print('\n' + str(formatted_datetime))
        #print('==============')
        #print('5-', (dt.datetime.now().replace(microsecond=0).isoformat()).split('T')[-1])
        start = time.time()
        stg_type = ''
        stg_type, pick_time, close_price_high, close_price_low, point_sum, success, peaker_side, peaker_option, scalping_direction_pick, scalping_switch, scalping_indicator_confirmer, divergence_name = confirmer()       
        # peaker_side = check_and_reverse(peaker_side, stg_type, peaker_option, reverse_confirmer)
        # tr_interval = tr_interval_confirm()
        tr_interval = globals()['atr_pick']
        trend_result, trade_type, stop_loss_range, target_range = tr_confirm(tr_interval)
        # for interval in intervals:
        #     trend_result_1, trade_type_1, stop_loss_range_1, target_range_1 = tr_confirm(interval)
        #     print(f"{tr_interval} [{interval} {trend_result_1}] {(dt.datetime.now().replace(microsecond=0).isoformat()).split('T')[-1]}")
        # print(f'stg_type_org:{stg_type},and success: {success}')
        # if (stg_type == "stg1") and (trend_result in ["long", "short"]): # and (ah_%2 == 0):
        # if (trend_result in ["long", "short"]): # and (ah_%2 == 0):
        #     success = 1
        #     peaker_side = trend_result
        #     peaker_option = 'forward'
        # elif (stg_type == "stg1"):
        #     success = 0
        # print(f'나머지:{ah_%2},and success: {success}, trend_result:{trend_result}\n')
        
        
        
        formatted_time = dt.datetime.now().replace(microsecond=0).isoformat().replace('T', ' ')
        # print(f"ID:{ah_}, [stg_type:{stg_type}], [success:{success}], [tr_interval:{tr_interval}], [trend_result:{trend_result}], [aa:{globals()['aa']}]")
        symbol_ticker_last = ticker_calc(market_id)[1]
        print(f'[{formatted_time}]')
        print(f'[bal:{round(wallet_balance, 2)}, pnl:${round(unrealised_pnl, 2)}({roe_pcnt}), position_size_:{position_size}({position_side} ${position_value})]')
        print(f'[stg_type_fixed: {stg_type_fixed}, reverse_confirmer: {reverse_confirmer}]')
        print(f'[진입가:{position_entry_price}, 현재가:{symbol_ticker_last}, 역지가:{stop_loss_}]')
        print(f'[총 횟수:{total_played}, 익절:{total_wins}({win_rate}), 손절:{total_losses}({loss_rate}), new_position_size:{new_position_size}(${round((new_position_size*symbol_ticker_last), 2)})]')
        interval_, side_, last_time_, big_boss_trend_checker = big_boss_trend_re_3(symbol, globals()['valid_intervals'])
        interval_2, side_2, last_time_2, big_boss_trend_checker_2 = big_boss_trend_re_2(symbol, globals()['valid_intervals'])
        print('\n')
        high_list, low_list, high_list_len, low_list_len = volatility_checker() # 실행 해야 globals()['volatility_macro_state'])변수가 업데이트 됨.
        # if success:
        #     print('\n' + str(formatted_datetime))
        #     print('volatility_macro_state: ', globals()['volatility_macro_state'])
        #     print('volatility_micro_interval_pick: ', globals()['volatility_micro_interval_pick'])
        #     print('volatility_atr_given: ', globals()['volatility_atr_given'])
        #     print('-----------------------')
        # if (globals()['atr_pick'] == '1m') or (globals()['atr_pick'] == ''):
        #     exit_order_waiting_seconds = int(cpu['info']['exit_order_waiting_seconds'])
        # else:
        #     exit_order_waiting_seconds = (time_to_seconds_converter_cal(globals()['atr_pick'])/5)
        
        # if scalping_direction_pick == 'long':
        #     print('스켈핑 결정은?: ' + str(scalping_switch) + '스켈핑 방향?: ' + str(scalping_direction_pick) + '스켈핑 interval?: ' + globals()['long_trend_micro_interval_pick'] + ', volatility_micro_interval_pick: ' + globals()['volatility_micro_interval_pick'] + ', volatility_macro_state: ' + str(globals()['volatility_macro_state']) + ', atr_pick: ' + str(globals()['atr_pick']))
        # elif scalping_direction_pick == 'short':
        #     print('스켈핑 결정은?: ' + str(scalping_switch) + '스켈핑 방향?: ' + str(scalping_direction_pick) + '스켈핑 interval?: ' + globals()['short_trend_micro_interval_pick'] + ', volatility_micro_interval_pick: ' + globals()['volatility_micro_interval_pick'] + ', volatility_macro_state: ' + str(globals()['volatility_macro_state']) + ', atr_pick: ' + str(globals()['atr_pick']))
        # else:
        #     print('스켈핑 결정은?: ' + str(scalping_switch) + '스켈핑 방향?: ' + str(scalping_direction_pick) + ', volatility_micro_interval_pick: ' + globals()['volatility_micro_interval_pick'] + ', volatility_macro_state: ' + str(globals()['volatility_macro_state']) + ', atr_pick: ' + str(globals()['atr_pick']))

        #print(f'high_list_len: [{high_list_len}], high_list: {high_list}')
        #print(f'low_list_len: [{low_list_len}], low_list: {low_list}')
        # if globals()['volatility_macro_state'] == 1:

        ######################################################################################################################################################
        # max_leverage = 50/3/instance
        # lev_limit = 20/3/instance
        # r = 1.1
        # scale_order_max_limit = 0
        # stopPrice_const = 2
        # atr_const = 0.7 # 70%
        # atr_const2 = .87 # 100%
        # if target_range == "tight":
        #     taker_fee = 0.08
        # else:
        #     taker_fee = 0.08
        ######################################################################################################################################################
        # if big_boss_trend_checker:
        #     max_leverage = int(cpu['info']['max_leverage'])/3
        #     lev_limit = int(cpu['info']['limit_leverage'])/3
        #     r = 1.1
        #     scale_order_max_limit = 0
        #     stopPrice_const = 2
        # else:
        #     max_leverage = int(cpu['info']['max_leverage'])
        #     lev_limit = int(cpu['info']['limit_leverage'])
        #     r = float(cpu['info']['r'])
        #     scale_order_max_limit = int(cpu['info']['scale_order_max_limit'])
        #     stopPrice_const = 2
        ######################################################################################################################################################

        # print('max_leverage: ' + str(max_leverage) + ', lev_limit: ' + str(lev_limit))
        #print('stg_type:', stg_type, 'peaker_side:', peaker_side, 'peaker_option:', peaker_option, 'success:', success)
        ######################################################################################################################################################
        #print('-')
        timesp, symbol_ticker_last, last_24hr_volatility = ticker_calc(market_id)
        raw_min_qty = max(float(min_cost_in_usdt)/float(symbol_ticker_last), float(min_order_qty))
        round_up_min_qty = round_up(raw_min_qty, market_amount_precision)
        min_order_amount = float(exchange.amount_to_precision(market_id, round_up_min_qty))
        position_side, position_size, position_value, position_entry_price, position_entry_time, liquidation_price, unrealised_pnl, roe_pcnt = position_calc(exchange_id, market_id)
        open_order_counter, open_order_side, open_order_size, open_order_price, open_order_type, stop_market_counter = open_order_calc(exchange_id, market_id) # exit_order check
        exit_order_position_amount = exit_order_position_amount_calc(position_size)
        predicted_change = calculate_predicted_change(market_id)
        max_leverage = calculate_max_leverage(predicted_change, market_max_leverage)*3
        lev_limit = calculate_max_leverage(predicted_change, market_max_leverage)*3
        wallet_balance = balance_calc(exchange_id, balance_currency)
        scale_order_position_amount_calc_val = scale_order_position_amount_calc(min_order_amount, wallet_balance, max_leverage, position_size, r, scale_order_max_limit)
        initial_order_amount = scale_order_position_amount_calc_val[1]
        scale_order_position_amount = scale_order_position_amount_calc_val[2]
        scaled_level_n = scale_order_position_amount_calc_val[3]
        bul_scale_order_position_amount = scale_order_position_amount_calc_val[4]
        stric_exit_min_price = stric_exit_price_calc_min(position_side, position_entry_price)
        stop_loss_pnl = stop_loss_price_and_pnl_calc(wallet_balance, position_size, position_side, position_entry_price, stop_lose_limit_percent_range)[1]
        cumulate_lv = cumulate_lv_calc(market_id, intervals)
        symbol_ticker_last = ticker_calc(market_id)[1]
        pick_max, pick_min, previous_price_frequency, greedy_percentage, picked_interval = pick_calc(cumulate_lv, position_side, position_size, position_entry_price, symbol_ticker_last, scalping_direction_pick)
        l_p_l, l_p_h, last_peaked_price = l_p_l_and_l_p_h_calc(symbol_ticker_last)
        stop_loss_, percentage_difference_stop_loss_, target_p_, percentage_difference_target_p_ = stop_loss_calc(stg_type, success, position_side, position_entry_price, symbol_ticker_last, close_price_low, close_price_high, pick_max, pick_min, stop_loss_range)

        if trade_type == "counter_trade": # 
            # intv_ = globals()['atr_pick']
            # part = globals()['df_' + intv_]['ATRr_1'].tail(5).max() # picked interval 기준 마지막 3개 중 가장 높은 변동가
            # percentage_ = (part / symbol_ticker_last) * 100
            # percentage_re_ = percentage_ * (-1) * 3 # 가장 높은 변동가의 2배
            # percentage_difference_target_p_constant = percentage_ * 3 #  가장 높은 변동가의 3배
            # percentage_difference_stop_loss_constant = (max((stop_lose_limit_percent_range*(-1)), percentage_re_)) / max_leverage # 역지가 까지 너무 멀다면 포지 진입 포기
            percentage_difference_target_p_constant = 100 # 잡은 수량의 2.5%: # 타겟가 까지 너무 멀다면 포지 진입 포기
            percentage_difference_stop_loss_constant = -100
            # percentage_difference_stop_loss_constant = stop_lose_limit_percent_range / max_leverage * (-1) # 지가 까지 너무 멀다면 포지 진입 포기
        else: # trade_type == "trending_trade" 일때는 역지가 넓어도 무시
            percentage_difference_target_p_constant = 100 # 잡은 수량의 2.5%: # 타겟가 까지 너무 멀다면 포지 진입 포기
            percentage_difference_stop_loss_constant = -100
            # percentage_difference_stop_loss_constant = stop_lose_limit_percent_range / max_leverage * (-1) # 지가 까지 너무 멀다면 포지 진입 포기
        
        end = time.time()
        total_time = end - start
        elapsed_times.append(round(total_time, 2))
        timeline_peaked_ = globals()['atr_pick']
        # time.sleep(10)
        # all_columns_string = ', '.join(df_1m.columns)
        # print(all_columns_string)
        # print(globals()['peaker_frame'])
        ######################################################################################################################################################
        #print('6-', (dt.datetime.now().replace(microsecond=0).isoformat()).split('T')[-1])
        ######################################################################################################################################################
        ######################################################################################################################################################
        ######################################################################################################################################################
        ######################################################################################################################################################
        ######################################################################################################################################################














        ######################################################################################################################################################
        #####################################################################################################################################################
        #####################################################################################################################################################
        #####################################################################################################################################################
        #####################################################################################################################################################
        #####################################################################################################################################################


        # if not success: # 1 ~ 2.
        if (not success) and (not big_boss_trend_checker_2):
            if position_size: # 포지션존재 1.
                if position_side == 'long':
                    exit_side = 'sell'
                    limit_type_only_case_price_pick_param = float(1 - 0.0005)
                    if stric_exit_min_price < symbol_ticker_last:
                        positive_pnl = 1 # 수익중

                elif position_side == 'short':
                    exit_side = 'buy'
                    limit_type_only_case_price_pick_param = float(1 + 0.0005)
                    if stric_exit_min_price > symbol_ticker_last:
                        positive_pnl = 1 # 수익중


                # 임계치 확인 손절여부?
                if (
                    (
                        (scaled_level_n > (scale_order_max_limit + 0.6)) and 
                        (df_1h['feature1'].iloc[-1] > 0) and 
                        (df_1h['feature1_diff'].iloc[-1] > 0)
                    ) or 
                    (
                        (scaled_level_n > (scale_order_max_limit - 0.6)) and 
                        (positive_pnl == 1)
                    )
                ): # option2: 물타기 횟수 임계치 초과, 변동성 클때 시장가 손절, 수익중일 경우 무조건 익절 1-1.
                    max_waiting_in_second = open_position_stop_market_waiting_seconds
                    if open_order_counter > 0:
                        exchange.cancel_all_orders(market_id) # open_order 모두 취소
                    if exchange_id == 'huobi':
                        limit_type_only_case_price_pick = float(exchange.price_to_precision(market_id, float(symbol_ticker_last)*(limit_type_only_case_price_pick_param)))
                        exchange.create_order(symbol=market_id, type='limit', side=exit_side, amount=exit_order_position_amount, price=limit_type_only_case_price_pick, params=params)
                    else:
                        exchange.create_order(symbol=market_id, type='market', side=exit_side, amount=exit_order_position_amount, params=params) # 손절
                        stg_type_fixed = ''
                    exit_status = '1-1. 물타기/불타기 횟수 임계치 초과, (변동성 클때 시장가 손절, 수익중일 경우 무조건 익절)'
                    message = f"Trader: {trader_name}, 물타기/불타기 임계치 {scaled_level_n} 번의 시도에 이르렀습니다. open oder를 취소하고 포지션을 시장가손절하였습니다. 총 손절액은 ${unrealised_pnl}입니다."
                    # check_pointer(trader_name, exchange_id, market_id, balance_currency, loop, max_waiting_in_second, exit_status, message)
                    print(message)
                    send_to_telegram(trader_name, symbol, message)
                    # break

                if (unrealised_pnl < stop_loss_pnl): # option1: 손해액 임계치 초과, 시장가 손절, 1-2.
                    if open_order_calc(exchange_id, market_id)[0] > 0:
                        exchange.cancel_all_orders(market_id) # open_order 모두 취소
                    if exchange_id == 'huobi':
                        limit_type_only_case_price_pick = float(exchange.price_to_precision(market_id, float(symbol_ticker_last)*(limit_type_only_case_price_pick_param)))
                        exchange.create_order(symbol=market_id, type='limit', side=exit_side, amount=exit_order_position_amount, price=limit_type_only_case_price_pick, params=params)
                    else:
                        exchange.create_order(symbol=market_id, type='market', side=exit_side, amount=exit_order_position_amount, params=params) # 손절
                        stg_type_fixed = ''
                    exit_status = '1-2. 손해액 임계치 초과, 시장가 손절'
                    message = f'Trader: {trader_name}, 거래의 손해액이 임계치(총 balance 의 -{stop_lose_limit_percent_range}%, ${stop_loss_pnl}) USD 을 넘었으므로, open oder를 취소하고 포지션을 시장가손절하였습니다. 총 손절액은 ${unrealised_pnl}입니다.'
                    # check_pointer(trader_name, exchange_id, market_id, balance_currency, loop, max_waiting_in_second, exit_status, message)
                    print(message)
                    send_to_telegram(trader_name, symbol, message)
                    # break

                if open_order_counter != 0: # open_order 있으면 entry_order 인지 exit_order 인지 구분, 1-3.
                    if ((position_side == 'long') and (stric_exit_min_price > open_order_price)) or ((position_side == 'short') and (stric_exit_min_price < open_order_price)): # entry_order 일경우 취소
                        exchange.cancel_all_orders(market_id)
                        exit_status = '1-3. entry_order 취소'
                        message = '현재의 open_order는 entry_order 이므로 취소되었습니다.'
                        # check_pointer(trader_name, exchange_id, market_id, balance_currency, loop, max_waiting_in_second, exit_status, message)
                        print(message)

                if (open_order_counter == 0) or (((time.time() - exit_order_timestamp) > exit_order_waiting_seconds)): # 시간경과: exit_order 재산출, 1-4.
                    message = '시간경과: exit에 실패하였으므로 모든 exit_order 를 취소하겠습니다.'
                    if (open_order_counter == 0):
                        message = '현재 exit_order가 존재하지 않습니다.'
                    exit_status = '1-4. 시간경과: exit에 실패'
                    max_waiting_in_second = ''
                    # check_pointer(trader_name, exchange_id, market_id, balance_currency, loop, max_waiting_in_second, exit_status, message)
                    print(message)
                    ############################################################################################################
                    #if open_order_calc(exchange_id, market_id)[0] > 0:
                    #    exchange.cancel_all_orders(market_id) # open_order 모두 취소
                    exchange.cancel_all_orders(market_id) # open_order 모두 취소, exit_order 재산출
                    symbol_ticker_last = ticker_calc(market_id)[1]
                    cumulate_lv = cumulate_lv_calc(market_id, intervals)
                    pick_max, pick_min, previous_price_frequency, greedy_percentage, picked_interval = pick_calc(cumulate_lv, position_side, position_size, position_entry_price, symbol_ticker_last, scalping_direction_pick)
                    # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@2')
                    # print('cumulate_lv: ', cumulate_lv)
                    # print('symbol_ticker_last: ', symbol_ticker_last, ', position_side: ', position_side, ', position_entry_price: ', position_entry_price, ', long_pick_max: ', pick_max, ', short_pick_min: ', pick_min)
                    if position_side == 'long': # 포지션 롱 일때
                        ############################################################################################################
                        if exit_order_position_amount != 0:
                            exchange.create_order(symbol=market_id, type='limit', side='sell', amount=exit_order_position_amount, price=pick_max, params = {'reduceOnly': True}) # exit_order 재생성
                            exit_order_timestamp = time.time()
                            max_waiting_in_second = exit_order_waiting_seconds
                            exit_status = '1-4-1. exit_order 재생성'
                            message = f'exit_order를 재생성 하였습니다. {picked_interval} interval 기준으로 전 가격의 frequency는 {previous_price_frequency}, greedy_percentage는 {greedy_percentage} 입니다.'
                            # check_pointer(trader_name, exchange_id, market_id, balance_currency, loop, max_waiting_in_second, exit_status, message)
                            print(message)
                            ############################################################################################################
                            symbol_ticker_last = ticker_calc(market_id)[1]
                            if (((position_value) > (wallet_balance * lev_limit)) or ((unrealised_pnl / wallet_balance) * 100 >= 3)): # if 현재 position_value 가 25배 넘을경우 atr 역지
                                # stopPrice = float(exchange.price_to_precision(market_id, (position_entry_price - ((pick_max - position_entry_price)/stopPrice_const))))
                                l_p_l, l_p_h, last_peaked_price = l_p_l_and_l_p_h_calc(symbol_ticker_last)
                                stop_loss_, percentage_difference_stop_loss_, target_p_, percentage_difference_target_p_ = stop_loss_calc(stg_type, success, position_side, position_entry_price, symbol_ticker_last, close_price_low, close_price_high, pick_max, pick_min, stop_loss_range)
                                if stop_loss_ < symbol_ticker_last: # 현재 포지가 long 일때
                                    exchange.create_order(symbol=market_id, type='market', side='sell', amount=exit_order_position_amount, params={'stopPrice': stop_loss_, 'triggerDirection': 'below'}) # stop market 재생성
                                    exit_status = '1-4-2. 역지'
                                    message = f'시간경과로 역지 재생성 하였습니다.'
                                    message += ' [2:1]: ' + str((pick_max - position_entry_price)/stopPrice_const) + ', 손절가: ' + str(position_entry_price - ((pick_max - position_entry_price)/stopPrice_const))
                                    # check_pointer(trader_name, exchange_id, market_id, balance_currency, loop, max_waiting_in_second, exit_status, message)
                                    print(message)
                        else :
                            max_waiting_in_second = ''
                            exit_status = '1-4-6. '
                            message = f'현재 포지션이 즉시 사라진것같습니다. exit_order가 생성되지 않았습니다.'
                            # check_pointer(trader_name, exchange_id, market_id, balance_currency, loop, max_waiting_in_second, exit_status, message)
                            print(message)
                    elif position_side == 'short':
                        if exit_order_position_amount != 0:
                            exchange.create_order(symbol=market_id, type='limit', side='buy', amount=exit_order_position_amount, price=pick_min, params = {'reduceOnly': True}) # exit_order 재생성
                            exit_order_timestamp = time.time()
                            max_waiting_in_second = exit_order_waiting_seconds
                            exit_status = '1-4-7. exit_order 재생성'
                            message = f'exit_order를 재생성 하였습니다. {picked_interval} interval 기준으로 전 가격의 frequency는 {previous_price_frequency}, greedy_percentage는 {greedy_percentage} 입니다.'
                            # check_pointer(trader_name, exchange_id, market_id, balance_currency, loop, max_waiting_in_second, exit_status, message)
                            print(message)
                            ############################################################################################################
                            symbol_ticker_last = ticker_calc(market_id)[1]
                            if ((((-1 * position_value)) > (wallet_balance * lev_limit)) or ((unrealised_pnl / wallet_balance) * 100 >= 3)): # if 현재 position_value 가 25배 넘을경우 atr 역지
                                # stopPrice = float(exchange.price_to_precision(market_id, (position_entry_price + ((position_entry_price - pick_min)/stopPrice_const))))
                                l_p_l, l_p_h, last_peaked_price = l_p_l_and_l_p_h_calc(symbol_ticker_last)
                                stop_loss_, percentage_difference_stop_loss_, target_p_, percentage_difference_target_p_ = stop_loss_calc(stg_type, success, position_side, position_entry_price, symbol_ticker_last, close_price_low, close_price_high, pick_max, pick_min, stop_loss_range)
                                if stop_loss_ > symbol_ticker_last: # 현재 포지가 short 일때
                                    exchange.create_order(symbol=market_id, type='market', side='buy', amount=exit_order_position_amount, params={'stopPrice': stop_loss_, 'triggerDirection': 'above'}) # stop market 재생성
                                    exit_status = '1-4-8. 역지'
                                    message = f'시간경과로 역지 재생성 하였습니다.'
                                    message += '[2:1]: ' + str((position_entry_price - pick_min)/stopPrice_const) + ', 손절가: ' + str(position_entry_price + ((position_entry_price - pick_min)/stopPrice_const))
                                    # check_pointer(trader_name, exchange_id, market_id, balance_currency, loop, max_waiting_in_second, exit_status, message)
                                    print(message)
                        else :
                            max_waiting_in_second = ''
                            exit_status = '1-4-12. '
                            message = f'현재 포지션이 즉시 사라진것같습니다. exit_order가 생성되지 않았습니다.'
                            # check_pointer(trader_name, exchange_id, market_id, balance_currency, loop, max_waiting_in_second, exit_status, message)
                            print(message)
                #print('7-', (dt.datetime.now().replace(microsecond=0).isoformat()).split('T')[-1])
                # time.sleep(3)
            else: # 포지션 없는동안 2.
                if (stop_market_counter): # 포지 종료 후 나머지 stop_market_order 존재 또는 open_order가 1개만 존재할 시 취소 2-1
                    #print(f'포지 종료 후 나머지 open_order 존재 하여 취소하였습니다.')
                    exchange.cancel_all_orders(market_id) # open_order 모두 취소, 포지션 변경 준비
                    max_waiting_in_second = ''
                    exit_status = '2-1.'
                    message = f"포지 종료 후 나머지 stop_market_order 존재 하여 취소하였습니다."
                    # check_pointer(trader_name, exchange_id, market_id, balance_currency, loop, max_waiting_in_second, exit_status, message)
                    print(message)
                    #print('initial bid-', (dt.datetime.now().replace(microsecond=0).isoformat()).split('T')[-1])
                    # open_order_counter, open_order_side, open_order_size, open_order_price, open_order_type, stop_market_counter = open_order_calc(exchange_id, market_id) # 2. entry_order 확인
                    # entry order check, entry order 없거나, entry waiting 시간 경과 시 재산출
                
                elif (open_order_counter > 0) and ((entry_order_timestamp != 0) and (time.time() - entry_order_timestamp > entry_order_waiting_seconds)): # entry_order 재산출 2-2
                    exchange.cancel_all_orders(market_id) # open_order 모두 취소
                    message = '시간경과: entry에 실패하였으므로 모든 entry_order 를 취소하겠습니다.'
                    print(message)
                elif (open_order_counter > 0):
                    message = 'entry order 대기중입니다.'
                    print(message)
                else:
                    message = '현재 entry_order가 존재하지 않습니다.'
                    exit_status = '2-2. '
                    max_waiting_in_second = ''
                    # check_pointer(trader_name, exchange_id, market_id, balance_currency, loop, max_waiting_in_second, exit_status, message)
                    print(message)
                    ############################################################################################################
        else: #if success # 3 ~ 4.
            #print('9-', (dt.datetime.now().replace(microsecond=0).isoformat()).split('T')[-1])
            #print(f'!!!!!!!!! do it!!!success, point_sum is {point_sum} !!!!!!!!!')
            # is_big_boss_interval_grater_then_atr_pick_ = big_boss_point_cal(market_id, interval_)            
            loop_counter += 1
            max_waiting_in_second = ''
            exit_status = '3 ~ 10. peak deliver'
            # send_to_telegram(trader_name, symbol, message)
            # check_pointer(trader_name, exchange_id, market_id, balance_currency, loop, max_waiting_in_second, exit_status, message)
            print(message)
            #print('10-', (dt.datetime.now().replace(microsecond=0).isoformat()).split('T')[-1])
            # if (peaker_side =='long') and (scalping_direction_pick !='short') and (peaker_option == 'forward'): # if peacker == 'long_forward' # 3 ~ 6.
            # if (
            #     ((peaker_side =='long') and (peaker_option == 'forward') and (scalping_direction_pick !='long') and (divergence_name == 'regular_divergence_uptrend')) or \
            #     ((peaker_side =='long') and (peaker_option == 'forward') and (scalping_direction_pick !='short') and (divergence_name == 'hidden_divergence_uptrend')) \
            # ): # if peacker == 'long_forward' # 3 ~ 6.

            # if ((is_big_boss_interval_grater_then_atr_pick_ == False) and (peaker_side =='long') and (peaker_option == 'forward')) or ((is_big_boss_interval_grater_then_atr_pick_ == True) and (side_ =='long') and (big_boss_trend_checker == True) and (peaker_side =='long') and (peaker_option == 'forward')):# and (globals()['volatility_macro_state'] != 1 ): # if peacker == 'long_forward' # 3 ~ 6.
            # if ((stg_type in ['stg3', 'stg9']) and  (side_ =='long') and (big_boss_trend_checker == True) and (peaker_side =='long') and (peaker_option == 'forward')):# and (globals()['volatility_macro_state'] != 1 ): # if peacker == 'long_forward' # 3 ~ 6.
            # if ((stg_type in ['stg3', 'stg9']) and  (side_ =='long') and (peaker_side =='long') and (peaker_option == 'forward')):# and (globals()['volatility_macro_state'] != 1 ): # if peacker == 'long_forward' # 3 ~ 6.


            if stg_type in ['stg1', 'stg2', 'stg3', 'stg9', 'stg10', 'stg100', 'stg110']:
                message = f"-peak is delivered, diff: [{globals()['df_1m']['combined_diff_filtered_diff'].iloc[-1]}, {globals()['df_5m']['combined_diff_filtered_diff'].iloc[-1]}, {globals()['df_15m']['combined_diff_filtered_diff'].iloc[-1]}, {globals()['df_1m']['second_combined_diff_filtered'].iloc[-1]}, {globals()['df_5m']['second_combined_diff_filtered'].iloc[-1]}, {globals()['df_15m']['second_combined_diff_filtered'].iloc[-1]}], stg_type: [{stg_type},{peaker_side},{peaker_option},{point_sum}], divergence_name: [{divergence_name}], -big_boss_추세:[{side_}, {interval_}, {last_time_}], -현재_추세와_big_boss_추세_일치:[{big_boss_trend_checker}]]"
                message += f"-latest elapsed time: [{elapsed_times[-2]}], Initial Order Amount: [{initial_order_amount}]"
                send_to_telegram(trader_name, symbol, message)
            elif big_boss_trend_checker_2:
                stg_type = 'stgZ'
                peaker_side = big_boss_trend_checker_2
                peaker_option = 'forward'
                globals()['atr_pick'] = interval_2
                message = f"-peak is delivered, stg_type: [{stg_type},{peaker_side},{peaker_option},{interval_2}], divergence_name: [{divergence_name}], -big_boss_추세:[{side_}, {interval_}, {last_time_}], -현재_추세와_big_boss_추세_일치:[{big_boss_trend_checker}]]"
                message += f"-latest elapsed time: [{elapsed_times[-2]}], Initial Order Amount: [{initial_order_amount}]"
                # send_to_telegram(trader_name, symbol, message)

            if (stg_type in ['stg1', 'stg2', 'stg3', 'stg9', 'stg10', 'stg100', 'stg110', 'stgZ']): # and ((df_1h['ATRr_14'].iloc[-1]) > (df_1h['close'].iloc[-1] * 1.5/100)):
                if (

                        (    (stg_type in ['stg110'])
                            and (peaker_side == 'long')
                            and (peaker_option == 'forward')
                        )
                        or
                        (    (stg_type in ['stg10', 'stg100'])
                            # and (df_15m.feature1.iloc[-1] > 0)
                            and (peaker_side == 'long')
                            and (peaker_option == 'forward')
                            and (globals()['df_1m']['second_combined_diff_filtered'].iloc[-1] < 0.7)
                            and (globals()['df_5m']['second_combined_diff_filtered'].iloc[-1] < 0.7)
                            and ((globals()['df_15m']['second_combined_diff_filtered'].iloc[-1]) < 0.7)
                            # and (globals()['df_1m']['second_combined_diff_filtered_diff'].iloc[-1] > 0)
                            # and not ((df_1h['feature1'].iloc[-1] > 0) and (df_1h['feature1_diff'].iloc[-1] > 0))
                        )
                        or
                        (
                            (stg_type in ['stg3', 'stg9'])
                            # and (df_15m.feature1.iloc[-1] > 0)
                            and (peaker_side == 'long')
                            and (peaker_option == 'forward')
                            # and not ((df_1h['feature1'].iloc[-1] > 0) and (df_1h['feature1_diff'].iloc[-1] > 0))
                            # and ((df_1m['stg3_long'].iloc[-1] > 0) or (df_5m['stg3_long'].iloc[-1] > 0) or (df_15m['stg3_long'].iloc[-1] > 0))
                            # and ((df_4h['stg3_long'].iloc[-1] > 0) or (df_6h['stg3_long'].iloc[-1] > 0) or (df_8h['stg3_long'].iloc[-1] > 0))
                            # and (trade_type == "counter_trade")
                            # and (trend_result == 'long')
                            # and (trend_result != "short")
                            # and (df_4h['adx_diff'].iloc[-1] < 0)
                            # and (df_4h['ADX_14'].iloc[-1] < 25)
                            # and (df_1m['RSI_14'].iloc[-1] < 65)
                            # and (df_5m['RSI_14'].iloc[-1] < 80)
                            # and ((globals()['df_' + globals()['atr_pick']]['dmp_diff'].iloc[-1] > 0) or (globals()['df_' + globals()['atr_pick']]['dmn_diff'].iloc[-1] < 0))
                            # and (globals()['df_' + globals()['atr_pick']]['j_diff'].iloc[-1] > 0)
                            # and (globals()['df_' + globals()['atr_pick']]['rsi_diff'].iloc[-1] > 0)
                        )
                        or
                        (
                            (stg_type in ['stg2'])
                            # and (position_size == 0)
                            # and (df_15m.feature1.iloc[-1] > 0)
                            and (peaker_side == 'long')
                            and (peaker_option == 'forward')
                            # and (globals()['df_15m']['second_combined_diff_filtered'].iloc[-1] < 0.25)
                            # and (globals()['df_' + globals()['atr_pick']]['high'].max() > globals()['df_' + globals()['atr_pick']]['high'].iloc[-1])
                            # and (globals()['df_' + globals()['atr_pick']]['low'].min() < globals()['df_' + globals()['atr_pick']]['low'].iloc[-1])
                            # and not ((df_1h['feature1'].iloc[-1] > 0) and (df_1h['feature1_diff'].iloc[-1] > 0))
                            # and (trade_type == "trending_trade")
                            # and (trend_result == 'long')
                            # and (trend_result != "short")
                            # and (df_4h['adx_diff'].iloc[-1] > 0)
                            # and (df_4h['ADX_14'].iloc[-1] > 25)
                            # and ((df_4h['ATRr_1'].iloc[-1]) > (df_4h['close'].iloc[-1] * 0.8/100))
                            # and (globals()['df_' + globals()['atr_pick']]['adx_diff'].iloc[-1] > 0)
                            # and (df_1m['RSI_14'].iloc[-1] < 65)
                            # and (df_5m['RSI_14'].iloc[-1] < 80)
                            # and (globals()['df_' + globals()['atr_pick']]['j_diff'].iloc[-1] > 0)
                            # and (globals()['df_' + globals()['atr_pick']]['rsi_diff'].iloc[-1] > 0)
                        )
                        or
                        (
                            (stg_type in ['stg1'])
                            and (peaker_side == 'long')
                            and (peaker_option == 'forward')
                            and (globals()['df_1m']['second_combined_diff_filtered'].iloc[-1] < 0.7)
                            and (globals()['df_5m']['second_combined_diff_filtered'].iloc[-1] < 0.7)
                            and ((globals()['df_15m']['second_combined_diff_filtered'].iloc[-1]) < 0.7)
                            # and (globals()['df_15m']['second_combined_diff_filtered'].iloc[-1] < 0.3)
                            # and (globals()['df_4h']['second_combined_diff_filtered'].iloc[-1] < 0.3)



                            # and 
                            # (
                            #     (
                            #         (stg_type_fixed in ['stg1']) 
                            #         and (position_size != 0)
                            #         and (position_side == 'short')
                            #         and (stric_exit_min_price > symbol_ticker_last)
                            #     )
                            #     or
                            #     (
                            #         (stg_type_fixed in ['stg1']) 
                            #         and (position_size != 0)
                            #         and (position_side == 'long')
                            #     )
                            #     or 
                            #     (
                            #         position_size == 0
                            #     )
                            # )





                            # and ((df_1h['feature1'].iloc[-1] > 0) and (df_1h['feature1_diff'].iloc[-1] > 0))
                            # and not (df_15m.feature1.iloc[-1] < -0.2)
                            # and (df_15m.feature1.iloc[-1] > 0)
                            # and (position_size == 0)
                            # and (globals()['df_' + globals()['atr_pick']]['high'].max() > globals()['df_' + globals()['atr_pick']]['high'].iloc[-1])
                            # and (globals()['df_' + globals()['atr_pick']]['low'].min() < globals()['df_' + globals()['atr_pick']]['low'].iloc[-1])
                            
                            # and (globals()['df_' + globals()['atr_pick']]['lowess_1'].iloc[-1] > globals()['df_' + globals()['atr_pick']]['close'].iloc[-1])
                            # and (globals()['df_' + globals()['atr_pick']]['lowess_1_diff'].iloc[-1] > 0)

                            # and (trade_type == "trending_trade")
                            # and (trend_result == 'long')
                            # and (trend_result != "short")
                            # and (df_4h['adx_diff'].iloc[-1] > 0)
                            # and (df_15m['adx_diff'].iloc[-1] > 0)
                            # and (df_4h['ADX_14'].iloc[-1] > 25)
                            # and ((df_4h['ATRr_1'].iloc[-1]) > (df_4h['close'].iloc[-1] * 0.8/100))
                            # and (globals()['df_' + globals()['atr_pick']]['adx_diff'].iloc[-1] > 0)
                            # and (df_1m['RSI_14'].iloc[-1] < 65)
                            # and (df_5m['RSI_14'].iloc[-1] < 80)
                            # and (globals()['df_' + globals()['atr_pick']]['j_diff'].iloc[-1] > 0)
                            # and (globals()['df_' + globals()['atr_pick']]['rsi_diff'].iloc[-1] > 0)
                        )
                        or
                        (
                            (stg_type in ['stgZ'])
                            # and ((stg_type_fixed in ['stg1']) or (position_size == 0))
                            # and ((df_1h['feature1'].iloc[-1] > 0) and (df_1h['feature1_diff'].iloc[-1] > 0))
                            # and not (df_15m.feature1.iloc[-1] < -0.2)
                            # and (df_15m.feature1.iloc[-1] > 0)
                            # and (position_size == 0)
                            and (peaker_side == 'long')
                            and (peaker_option == 'forward')
                            and (globals()['df_1m']['second_combined_diff_filtered'].iloc[-1] < 0.7)
                            and (globals()['df_5m']['second_combined_diff_filtered'].iloc[-1] < 0.7)
                            and ((globals()['df_15m']['second_combined_diff_filtered'].iloc[-1]) < 0.7)
                            and (df_1m['RSI_14'].iloc[-1] < 70)
                            and (df_5m['RSI_14'].iloc[-1] < 70)
                            and (df_1m['macd_diff_35'].iloc[-3] > 0)
                            and (df_1m['macdh_diff_35'].iloc[-3] > 0)
                            # and (globals()['df_' + globals()['atr_pick']]['high'].max() > globals()['df_' + globals()['atr_pick']]['high'].iloc[-1])
                            # and (globals()['df_' + globals()['atr_pick']]['low'].min() < globals()['df_' + globals()['atr_pick']]['low'].iloc[-1])
                            
                            # and (globals()['df_' + globals()['atr_pick']]['lowess_1'].iloc[-1] > globals()['df_' + globals()['atr_pick']]['close'].iloc[-1])
                            # and (globals()['df_' + globals()['atr_pick']]['lowess_1_diff'].iloc[-1] > 0)

                            # and (trade_type == "trending_trade")
                            # and (trend_result == 'long')
                            # and (trend_result != "short")
                            # and (df_4h['adx_diff'].iloc[-1] > 0)
                            # and (df_15m['adx_diff'].iloc[-1] > 0)
                            # and (df_4h['ADX_14'].iloc[-1] > 25)
                            # and ((df_4h['ATRr_1'].iloc[-1]) > (df_4h['close'].iloc[-1] * 0.8/100))
                            # and (globals()['df_' + globals()['atr_pick']]['adx_diff'].iloc[-1] > 0)
                            # and (df_1m['RSI_14'].iloc[-1] < 65)
                            # and (df_5m['RSI_14'].iloc[-1] < 80)
                            # and (globals()['df_' + globals()['atr_pick']]['j_diff'].iloc[-1] > 0)
                            # and (globals()['df_' + globals()['atr_pick']]['rsi_diff'].iloc[-1] > 0)
                        )
                ):
                    if (position_size != 0): # 포지션존재 # 3 ~ 5.
                        loop = 'long-forward-' + str(loop_counter)
                        message = f'현재 포지션이 존재합니다. stric_exit_min_price={stric_exit_min_price}'
                        exit_status = '3 ~ 5. peak deliver, 포지 점검'
                        # check_pointer(trader_name, exchange_id, market_id, balance_currency, loop, max_waiting_in_second, exit_status, message)
                        print(message)
                        symbol_ticker_last = ticker_calc(market_id)[1]
                        if (position_side == 'long'): #and (globals()['trend_macro_state'] == 'long'):
                            if (position_entry_price > symbol_ticker_last): # 1. 손실중 & 포지션이랑 peaker 방향같을때,  => 물타기 3.
                                scale_order_after_seconds = scale_order_timestamp + dt.timedelta(seconds = scale_order_waiting_seconds) # 물타기 주기
                                current_timestamp = dt.datetime.now().replace(microsecond=0)
                                if current_timestamp < scale_order_after_seconds: # 마지막 물타기로부터 일정기간 경과했는지?
                                    message = f'※ {globals()["atr_pick"]}, 물타기 기간이 너무 이릅니다. {scale_order_timestamp.astimezone(pytz.timezone("Asia/Seoul"))}에 {scaled_level_n}번째 물타기 진입에 성공하였으며 {scale_order_timestamp + dt.timedelta(seconds = scale_order_waiting_seconds) - dt.datetime.now().replace(microsecond=0)} 초 후에 {round((scaled_level_n + 1), 2)}번째 물타기 시도가 가능합니다.'
                                    exit_status = '3-1. ※물타기 기간 점검'
                                    # check_pointer(trader_name, exchange_id, market_id, balance_currency, loop, max_waiting_in_second, exit_status, message)
                                    print(message)
                                    send_to_telegram(trader_name, symbol, message)
                                elif (position_value + (scale_order_position_amount*symbol_ticker_last)) > (wallet_balance * lev_limit): # if 현재 position_value 가 25배 넘을경우, 물타기 pass!
                                    if inverse_exchange:
                                        message = f'※ {globals()["atr_pick"]}, 현재 들고 있는 position_value의 값이 약 ${position_value}이며, 물탈경우 wallet_balance ${wallet_balance} 대비 Leverage Limit({lev_limit}) 배를 초과하게 됩니다. 물타기를 pass 하겠습니다.'
                                    else:
                                        message = f'※ {globals()["atr_pick"]}, 현재 들고 있는 position_value의 값이 약 ${position_value}이며, 물탈경우 wallet_balance ${wallet_balance} 대비 Leverage Limit({lev_limit}) 배를 초과하게 됩니다. 물타기를 pass 하겠습니다.'
                                    exit_status = '3-2. ※물타기 lev_limit 점검'
                                    # check_pointer(trader_name, exchange_id, market_id, balance_currency, loop, max_waiting_in_second, exit_status, message)
                                    print(message)
                                    send_to_telegram(trader_name, symbol, message)
                                # elif (position_entry_price*(1 - (previous_price_frequency_val/100)) < symbol_ticker_last):
                                #     message = f'진입가(${round(position_entry_price, 2)}) - 현재가(${round(symbol_ticker_last, 2)}) 의 변동성(${round((symbol_ticker_last - position_entry_price), 2)}, {(symbol_ticker_last/position_entry_price - 1)*100})%이 지난 15분봉의 변동성({round(previous_price_frequency_val, 2)}%) 보다 작습니다. ※ 물린가격이 임계치 이하이므로 물타기를 pass 하겠습니다.'
                                #     check_pointer(trader_name, exchange_id, market_id, balance_currency, loop, max_waiting_in_second, exit_status, message)
                                #     print(message)
                                # elif (globals()['atr_peak']) and (globals()['atr_given']!= 0) and ((position_entry_price - (atr_given * 1)) < symbol_ticker_last): # atr 변동보다 더 움직였을때만 물타고, 안움직였을때는 물타기 pass!
                                #     message = f'진입가(${position_entry_price}) - 현재가(${symbol_ticker_last}) 의 변동성(${symbol_ticker_last - position_entry_price}, {(symbol_ticker_last/position_entry_price - 1)*100})%이 atr(* 1) 변동성(${atr_given * 1}) 보다 작습니다. ※ 물린가격이 임계치 이하이므로 물타기를 pass 하겠습니다.'
                                #     # check_pointer(trader_name, exchange_id, market_id, balance_currency, loop, max_waiting_in_second, exit_status, message)
                                #     print(message)
                                # elif position_entry_price * (1 - (greedy_percentage_calc(globals()['atr_peak'])[1]) / 100) < symbol_ticker_last: # 현재가가 진입가 대비 greedy_percentage보다 더 움직였을때,
                                elif (position_entry_price * (1 - (greedy_percentage_calc(globals()['atr_pick'])[1]) / 100) < symbol_ticker_last): # 현재가가 진입가 대비 greedy_percentage보다 더 움직였을때,
                                    message = f' {globals()["atr_pick"]}, 진입가(${position_entry_price}) - 현재가(${symbol_ticker_last}) 의 변동성(${position_entry_price - symbol_ticker_last}, {(1 - symbol_ticker_last/position_entry_price)*100}%)이 greedy_percentage 변동성('
                                    message += str(greedy_percentage_calc(globals()['atr_pick'])[1]) + '%)보다 작습니다. ※ 물린가격이 임계치 이하이므로 물타기를 pass 하겠습니다.'
                                    exit_status = '3-3. ※물타기 가격변동 점검'
                                    send_to_telegram(trader_name, symbol, message)
                                    # check_pointer(trader_name, exchange_id, market_id, balance_currency, loop, max_waiting_in_second, exit_status, message)
                                    #print(message)
                                # ############################################################################################################
                                # elif (scalping_direction_pick in ['long', 'short']) and (position_entry_price * (1 - (greedy_percentage_calc(globals()[scalping_direction_pick + '_trend_micro_interval_pick'])[1]) / 100) < symbol_ticker_last): # 현재가가 진입가 대비 greedy_percentage보다 더 움직였을때,
                                #     message = f'진입가(${position_entry_price}) - 현재가(${symbol_ticker_last}) 의 변동성(${symbol_ticker_last - position_entry_price}, {(symbol_ticker_last/position_entry_price - 1)*100}%)이 greedy_percentage 변동성('
                                #     message += str(greedy_percentage_calc(globals()[scalping_direction_pick + '_trend_micro_interval_pick'])[2]) + ', ' + str(greedy_percentage_calc(globals()[scalping_direction_pick + '_trend_micro_interval_pick'])[1]) + '%)보다 작습니다. ※ 물린가격이 임계치 이하이므로 물타기를 pass 하겠습니다.'
                                #     exit_status = '3-3. ※물타기 가격변동 점검'
                                #     # check_pointer(trader_name, exchange_id, market_id, balance_currency, loop, max_waiting_in_second, exit_status, message)
                                #     #print(message)
                                # elif (scalping_direction_pick in ['neutral']) and (position_entry_price * (1 - (greedy_percentage_calc(globals()['long_trend_micro_interval_pick'])[1]) / 100) < symbol_ticker_last): # 현재가가 진입가 대비 greedy_percentage보다 더 움직였을때,
                                #     message = f'진입가(${position_entry_price}) - 현재가(${symbol_ticker_last}) 의 변동성(${symbol_ticker_last - position_entry_price}, {(symbol_ticker_last/position_entry_price - 1)*100}%)이 greedy_percentage 변동성('
                                #     message += str(greedy_percentage_calc(globals()['long_trend_micro_interval_pick'])[2]) + ', ' + str(greedy_percentage_calc(globals()['long_trend_micro_interval_pick'])[1]) + '%)보다 작습니다. ※ 물린가격이 임계치 이하이므로 물타기를 pass 하겠습니다.'
                                #     exit_status = '3-3. ※물타기 가격변동 점검'
                                #     # check_pointer(trader_name, exchange_id, market_id, balance_currency, loop, max_waiting_in_second, exit_status, message)
                                #     #print(message)
                                # ############################################################################################################

                                else: #물타기로부터 기간 만족
                                    exchange.cancel_all_orders(market_id) # open_order 모두 취소, 포지션 변경 준비
                                    message = "현재의 포지션은 손실중입니다. 물타기로부터 기간 만족하므로 물타기 시도를 시작하겠습니다."
                                    exit_status = '3-4. ※물타기 시작'
                                    # check_pointer(trader_name, exchange_id, market_id, balance_currency, loop, max_waiting_in_second, exit_status, message)
                                    print(message)
                                    if exchange_id == 'huobi':
                                        limit_type_only_case_price_pick = float(exchange.price_to_precision(market_id, float(symbol_ticker_last)*(1 + 0.0005)))
                                        exchange.create_order(symbol=market_id, type='limit', side='buy', amount=scale_order_position_amount, price=limit_type_only_case_price_pick, params=params)
                                    else:
                                        exchange.create_order(symbol=market_id, type='market', side='buy', amount=scale_order_position_amount, params=params) # 물타기
                                        stg_type_fixed = stg_type
                                    exit_status = '3-5. 물타기 완료'
                                    scale_order_timestamp = dt.datetime.now().replace(microsecond=0) # 물탄시점 저장
                                    #print('-')
                                    position_side, position_size, position_value, position_entry_price, position_entry_time, liquidation_price, unrealised_pnl, roe_pcnt = position_calc(exchange_id, market_id)
                                    wallet_balance = balance_calc(exchange_id, balance_currency)
                                    scaled_level_n = scale_order_position_amount_calc(min_order_amount, wallet_balance, max_leverage, position_size, r, scale_order_max_limit)[3]
                                    message = f'stg_type: {stg_type}, {globals()["atr_pick"]}, {scale_order_timestamp.astimezone(pytz.timezone("Asia/Seoul"))}에 {scaled_level_n}번째 물타기 진입에 성공하였습니다.'
                                    send_to_telegram(trader_name, symbol, message)
                                    # check_pointer(trader_name, exchange_id, market_id, balance_currency, loop, max_waiting_in_second, exit_status, message)
                                    print(message)
                                    ############################################################################################################
                                    cumulate_lv = cumulate_lv_calc(market_id, intervals)
                                    symbol_ticker_last = ticker_calc(market_id)[1]
                                    pick_max, pick_min, previous_price_frequency, greedy_percentage, picked_interval = pick_calc(cumulate_lv, position_side, position_size, position_entry_price, symbol_ticker_last, scalping_direction_pick)
                                    exit_order_position_amount = exit_order_position_amount_calc(position_size)
                                    if exit_order_position_amount != 0:
                                        exchange.create_order(symbol=market_id, type='limit', side='sell', amount=exit_order_position_amount, price=pick_max, params = {'reduceOnly': True}) # exit_order 재생성
                                        exit_order_timestamp = time.time()
                                        max_waiting_in_second = exit_order_waiting_seconds
                                        exit_status = '3-6. exit_order 재산출'
                                        message = f'exit_order를 재산출하여 새롭게 생성 하였습니다. {picked_interval} interval 기준으로 전 가격의 frequency는 {previous_price_frequency}, greedy_percentage는 {greedy_percentage} 입니다.'
                                        ############################################################################################################
                                        symbol_ticker_last = ticker_calc(market_id)[1]
                                        if (((position_value) > (wallet_balance * lev_limit)) or ((unrealised_pnl / wallet_balance) * 100 >= 3)): # if 현재 position_value 가 25배 넘을경우, atr 역지 생성
                                            # stopPrice = float(exchange.price_to_precision(market_id, (position_entry_price - ((pick_max - position_entry_price)/stopPrice_const))))
                                            l_p_l, l_p_h, last_peaked_price = l_p_l_and_l_p_h_calc(symbol_ticker_last)
                                            stop_loss_, percentage_difference_stop_loss_, target_p_, percentage_difference_target_p_ = stop_loss_calc(stg_type, success, position_side, position_entry_price, symbol_ticker_last, close_price_low, close_price_high, pick_max, pick_min, stop_loss_range)
                                            if stop_loss_ < symbol_ticker_last: # 현재 포지가 long 일때
                                                exchange.create_order(symbol=market_id, type='market', side='sell', amount=exit_order_position_amount, params={'stopPrice': stop_loss_, 'triggerDirection': 'below'}) # stop market 재생성
                                                exit_status = '3-7. exit_order 재산출하여 + 역지'
                                                message += f' 물타기 후 역지 재생성 하였습니다.'
                                                message += ' [2:1]: ' + str((pick_max - position_entry_price)/2) + ', 손절가: ' + str(position_entry_price - ((pick_max - position_entry_price)/stopPrice_const))
                                        ############################################################################################################
                                        # if (globals()['long_trend_atr_given'] != 0) and (exchange_id == 'binanceusdm') and ((position_value + (scale_order_position_amount*symbol_ticker_last)) > (wallet_balance * lev_limit)): # if 현재 position_value 가 25배 넘을경우, atr 역지 생성
                                        #     stopPrice = float(exchange.price_to_precision(market_id, (position_entry_price - (globals()['long_trend_atr_given'] * 1.0))))
                                        #     if stopPrice < symbol_ticker_last: # 현재 포지가 long 일때
                                        #         exchange.create_order(symbol=market_id, type='stop_market', side='sell', amount=exit_order_position_amount, params={'stopPrice': stopPrice}) # stop market 재생성
                                        #         exit_status = '3-7. exit_order 재산출하여 + long_trend_atr_given 역지'
                                        #         message += f' 손실중이므로 long_trend_atr_given 역지 재생성 하였습니다.'
                                        #         message += ' long_trend_atr_given * 1.0: ' + str(globals()['long_trend_atr_given'] * 1.0) + ', 손절가: ' + str(position_entry_price  - (globals()['long_trend_atr_given'] * 1.0))
                                        #         #print(message)
                                        ############################################################################################################
                                    else :
                                        max_waiting_in_second = ''
                                        exit_status = '3-8. '
                                        message = f'현재 포지션이 즉시 사라진것같습니다. exit_order가 생성되지 않았습니다.'
                                    # check_pointer(trader_name, exchange_id, market_id, balance_currency, loop, max_waiting_in_second, exit_status, message)
                                    print(message)
                                    ###########################################################################################################


                        elif (position_side == 'short'): # and (position_entry_price > symbol_ticker_last): # 수익중 반대진입.
                            # if ((stg_type in ['stg10', 'stg100']) or ((stg_type in ['stg3', 'stg9']) and (scalping_direction_pick =='long'))):
                            # if not ((stg_type == 'stg10') and (globals()['atr_pick'] == '1m')):
                            # if globals()['trend_macro_state'] == 'long':
                            # if scalping_direction_pick == 'long' : # long trending scalping start
                            # if globals()['trend_macro_state'] == 'long':
                            exchange.cancel_all_orders(market_id) # open_order 모두 취소, 포지션 변경 준비
                            loop = f'long-forward-new-{loop_counter}'
                            message = '현재의 포지션은 이익중입니다. 종료후 반대진입 시도를 시작하겠습니다.'
                            exit_status = '6-1. 반대진입 시작'
                            # check_pointer(trader_name, exchange_id, market_id, balance_currency, loop, max_waiting_in_second, exit_status, message)
                            print(message)
                            symbol_ticker_last = ticker_calc(market_id)[1]
                            exit_order_position_amount_re = (exit_order_position_amount + initial_order_amount)

                            if exchange_id == 'huobi':
                                limit_type_only_case_price_pick = float(exchange.price_to_precision(market_id, float(symbol_ticker_last)*(1 + 0.0005)))
                                exchange.create_order(symbol=market_id, type='limit', side='buy', amount=exit_order_position_amount_re, price=limit_type_only_case_price_pick, params=params)
                            else:
                                #exchange.create_order(symbol=market_id, type='limit', side='buy', amount=exit_order_position_amount_re, price=pick_min, params=params)
                                exchange.create_order(symbol=market_id, type='market', side='buy', amount=exit_order_position_amount_re, params=params) # 시장가 진입
                                stg_type_fixed = stg_type
                                # limit_type_only_case_price_pick = float(exchange.price_to_precision(market_id, float(symbol_ticker_last)*(1 - 0.0005)))
                                # exchange.create_order(symbol=market_id, type='limit', side='buy', amount=exit_order_position_amount_re, price=limit_type_only_case_price_pick, params=params)

                            scale_order_timestamp = dt.datetime.now().replace(microsecond=0) # 진입시점 저장
                            message = f'stg_type: {stg_type}, {globals()["atr_pick"]}, entry_order lower_band 생성하였습니다.'
                            exit_status = '6-2. 포지션 진입'
                            send_to_telegram(trader_name, symbol, message)
                            # check_pointer(trader_name, exchange_id, market_id, balance_currency, loop, max_waiting_in_second, exit_status, message)
                            print(message)
                            ############################################################################################################
                            #print('-')
                            position_side, position_size, position_value, position_entry_price, position_entry_time, liquidation_price, unrealised_pnl, roe_pcnt = position_calc(exchange_id, market_id)
                            cumulate_lv = cumulate_lv_calc(market_id, intervals)
                            symbol_ticker_last = ticker_calc(market_id)[1]
                            pick_max, pick_min, previous_price_frequency, greedy_percentage, picked_interval = pick_calc(cumulate_lv, position_side, position_size, position_entry_price, symbol_ticker_last, scalping_direction_pick)
                            exit_order_position_amount = exit_order_position_amount_calc(position_size)
                            if exit_order_position_amount != 0:
                                exchange.create_order(symbol=market_id, type='limit', side='sell', amount=exit_order_position_amount, price=pick_max, params={'reduceOnly': True}) # exit_order 재생성
                                exit_order_timestamp = time.time()
                                max_waiting_in_second = exit_order_waiting_seconds
                                exit_status = '6-3. exit_order 재산출'
                                message = f'exit_order를 재산출하여 새롭게 생성 하였습니다. {picked_interval} interval 기준으로 전 가격의 frequency는 {previous_price_frequency}, greedy_percentage는 {greedy_percentage} 입니다.'
                                ############################################################################################################
                                symbol_ticker_last = ticker_calc(market_id)[1]
                                if (((position_value) > (wallet_balance * lev_limit)) or ((unrealised_pnl / wallet_balance) * 100 >= 3)): # if 현재 position_value 가 25배 넘을경우, atr 역지 생성
                                    # stopPrice = float(exchange.price_to_precision(market_id, (position_entry_price - ((pick_max - position_entry_price)/stopPrice_const))))
                                    l_p_l, l_p_h, last_peaked_price = l_p_l_and_l_p_h_calc(symbol_ticker_last)
                                    stop_loss_, percentage_difference_stop_loss_, target_p_, percentage_difference_target_p_ = stop_loss_calc(stg_type, success, position_side, position_entry_price, symbol_ticker_last, close_price_low, close_price_high, pick_max, pick_min, stop_loss_range)
                                    if stop_loss_ < symbol_ticker_last: # 현재 포지가 long 일때
                                        exchange.create_order(symbol=market_id, type='market', side='sell', amount=exit_order_position_amount, params={'stopPrice': stop_loss_, 'triggerDirection': 'below'}) # stop market 재생성
                                        exit_status = '3-7. exit_order 재산출하여 + 역지'
                                        message += f'역지 생성 하였습니다.'
                                        message += ' [2:1]: ' + str((pick_max - position_entry_price)/stopPrice_const) + ', 손절가: ' + str(position_entry_price - ((pick_max - position_entry_price)/stopPrice_const))
                            else :
                                max_waiting_in_second = ''
                                exit_status = '6-4. '
                                message = f'현재 포지션이 즉시 사라진것같습니다. exit_order가 생성되지 않았습니다.'

                            # check_pointer(trader_name, exchange_id, market_id, balance_currency, loop, max_waiting_in_second, exit_status, message)
                            print(message)
                            ############################################################################################################
                            
        
                
                    
                    
                    else: # 포지션 없을때 6.
                        # if ((stg_type in ['stg10', 'stg100']) or ((stg_type in ['stg3', 'stg9']) and (scalping_direction_pick =='long'))):
                        # if not ((stg_type == 'stg10') and (globals()['atr_pick'] == '1m')):
                        # if globals()['trend_macro_state'] == 'long':
                        # if scalping_direction_pick == 'long' : # long trending scalping start
                        # if globals()['trend_macro_state'] == 'long':
                        exchange.cancel_all_orders(market_id) # open_order 모두 취소, 포지션 변경 준비
                        loop = f'long-forward-new-{loop_counter}'
                        message = '현재 포지션이 존재하지 않습니다.'
                        exit_status = '6-1. 포지션 점검'
                        # check_pointer(trader_name, exchange_id, market_id, balance_currency, loop, max_waiting_in_second, exit_status, message)
                        print(message)
                        symbol_ticker_last = ticker_calc(market_id)[1]
                        if exchange_id == 'huobi':
                            limit_type_only_case_price_pick = float(exchange.price_to_precision(market_id, float(symbol_ticker_last)*(1 + 0.0005)))
                            exchange.create_order(symbol=market_id, type='limit', side='buy', amount=initial_order_amount, price=limit_type_only_case_price_pick, params=params)
                        else:
                            #exchange.create_order(symbol=market_id, type='limit', side='buy', amount=initial_order_amount, price=pick_min, params=params)
                            exchange.create_order(symbol=market_id, type='market', side='buy', amount=initial_order_amount, params=params) # 시장가 진입
                            stg_type_fixed = stg_type
                            # limit_type_only_case_price_pick = float(exchange.price_to_precision(market_id, float(symbol_ticker_last)*(1 - 0.0005)))
                            # exchange.create_order(symbol=market_id, type='limit', side='buy', amount=initial_order_amount, price=limit_type_only_case_price_pick, params=params)

                        scale_order_timestamp = dt.datetime.now().replace(microsecond=0) # 진입시점 저장
                        entry_order_timestamp = time.time()
                        message = f'stg_type: {stg_type}, {globals()["atr_pick"]}, entry_order lower_band 생성하였습니다.'
                        exit_status = '6-2. 포지션 진입'
                        send_to_telegram(trader_name, symbol, message)
                        # check_pointer(trader_name, exchange_id, market_id, balance_currency, loop, max_waiting_in_second, exit_status, message)
                        print(message)
                        ############################################################################################################
                        #print('-')
                        position_side, position_size, position_value, position_entry_price, position_entry_time, liquidation_price, unrealised_pnl, roe_pcnt = position_calc(exchange_id, market_id)
                        cumulate_lv = cumulate_lv_calc(market_id, intervals)
                        symbol_ticker_last = ticker_calc(market_id)[1]
                        pick_max, pick_min, previous_price_frequency, greedy_percentage, picked_interval = pick_calc(cumulate_lv, position_side, position_size, position_entry_price, symbol_ticker_last, scalping_direction_pick)
                        exit_order_position_amount = exit_order_position_amount_calc(position_size)
                        if exit_order_position_amount != 0:
                            exchange.create_order(symbol=market_id, type='limit', side='sell', amount=exit_order_position_amount, price=pick_max, params={'reduceOnly': True}) # exit_order 재생성
                            exit_order_timestamp = time.time()
                            max_waiting_in_second = exit_order_waiting_seconds
                            exit_status = '6-3. exit_order 재산출'
                            message = f'exit_order를 재산출하여 새롭게 생성 하였습니다. {picked_interval} interval 기준으로 전 가격의 frequency는 {previous_price_frequency}, greedy_percentage는 {greedy_percentage} 입니다.'
                            ############################################################################################################
                            symbol_ticker_last = ticker_calc(market_id)[1]
                            if (((position_value) > (wallet_balance * lev_limit)) or ((unrealised_pnl / wallet_balance) * 100 >= 3)): # if 현재 position_value 가 25배 넘을경우, atr 역지 생성
                                # stopPrice = float(exchange.price_to_precision(market_id, (position_entry_price - ((pick_max - position_entry_price)/stopPrice_const))))
                                l_p_l, l_p_h, last_peaked_price = l_p_l_and_l_p_h_calc(symbol_ticker_last)
                                stop_loss_, percentage_difference_stop_loss_, target_p_, percentage_difference_target_p_ = stop_loss_calc(stg_type, success, position_side, position_entry_price, symbol_ticker_last, close_price_low, close_price_high, pick_max, pick_min, stop_loss_range)
                                if stop_loss_ < symbol_ticker_last: # 현재 포지가 long 일때
                                    exchange.create_order(symbol=market_id, type='market', side='sell', amount=exit_order_position_amount, params={'stopPrice': stop_loss_, 'triggerDirection': 'below'}) # stop market 재생성
                                    exit_status = '3-7. exit_order 재산출하여 + 역지'
                                    message += f'역지 생성 하였습니다.'
                                    message += ' [2:1]: ' + str((pick_max - position_entry_price)/stopPrice_const) + ', 손절가: ' + str(position_entry_price - ((pick_max - position_entry_price)/stopPrice_const))
                        else :
                            max_waiting_in_second = ''
                            exit_status = '6-4. '
                            message = f'현재 포지션이 즉시 사라진것같습니다. exit_order가 생성되지 않았습니다.'

                        # check_pointer(trader_name, exchange_id, market_id, balance_currency, loop, max_waiting_in_second, exit_status, message)
                        print(message)
                        ############################################################################################################
                    #print('11-', (dt.datetime.now().replace(microsecond=0).isoformat()).split('T')[-1])
                # elif (peaker_side =='long') and (peaker_option == 'reverse') and (point_sum < 0): # if peacker == 'long_reverse'
                #     if (position_size != 0): # 포지션존재
                #         loop = 'long-reverse-' + str(loop_counter)
                #         message = f'현재 포지션이 존재합니다. stric_exit_min_price={stric_exit_min_price}'
                #         check_pointer(trader_name, exchange_id, market_id, balance_currency, loop, max_waiting_in_second, exit_status, message)
                #         #print(message)
                #         if position_side == 'long': # 포지션 롱 일때
                #             if (stric_exit_min_price < symbol_ticker_last): # 3. 수익중 & 포지션이랑 peaker 방향다를때,  => 익절
                #                 exchange.cancel_all_orders(market_id) # open_order 모두 취소

                #                 if exchange_id == 'huobi':
                #                     symbol_ticker_last = ticker_calc(market_id)[1]
                #                     limit_type_only_case_price_pick = float(exchange.price_to_precision(market_id, float(symbol_ticker_last)*(1 - 0.0005)))
                #                     exchange.create_order(symbol=market_id, type='limit', side='sell', amount=exit_order_position_amount, price=limit_type_only_case_price_pick, params=params)
                #                 else:
                #                     exchange.create_order(symbol=market_id, type='market', side='sell', amount=exit_order_position_amount, params=params) # 4. 익절
                #                 exit_status = '포지션과 peaker 방향 다름, 시장가 익절'
                #                 message = f'포지션과 peaker 방향이 다르므로 포지션을 시장가 익절하였습니다.'
                #                 check_pointer(trader_name, exchange_id, market_id, balance_currency, loop, max_waiting_in_second, exit_status, message)
                #                 #print(message)







                elif (
                        (
                            (stg_type in ['stg110'])
                            and (peaker_side == 'short')
                            and (peaker_option == 'forward')
                        )
                        or
                        (
                            (stg_type in ['stg10', 'stg100'])
                            # and (df_15m.feature1.iloc[-1] > 0)
                            and (peaker_side == 'short')
                            and (peaker_option == 'forward')
                            and (globals()['df_1m']['second_combined_diff_filtered'].iloc[-1] > -0.7)
                            and (globals()['df_5m']['second_combined_diff_filtered'].iloc[-1] > -0.7)
                            and ((globals()['df_15m']['second_combined_diff_filtered'].iloc[-1]) > -0.7)
                            # and (globals()['df_1m']['second_combined_diff_filtered_diff'].iloc[-1] < 0)
                            # and not ((df_1h['feature1'].iloc[-1] > 0) and (df_1h['feature1_diff'].iloc[-1] > 0))
                        )
                        or
                        (
                            (stg_type in ['stg3', 'stg9'])
                            # and (df_15m.feature1.iloc[-1] > 0)
                            and (peaker_side == 'short')
                            and (peaker_option == 'forward')
                            # and not ((df_1h['feature1'].iloc[-1] > 0) and (df_1h['feature1_diff'].iloc[-1] > 0))
                            # and ((df_5m['stg3_short'].iloc[-1] < 0) or (df_15m['stg3_short'].iloc[-1] < 0))
                            # and ((df_4h['stg3_short'].iloc[-1] < 0) or (df_6h['stg3_short'].iloc[-1] < 0) or (df_8h['stg3_short'].iloc[-1] < 0))
                            # and (trade_type == "counter_trade")
                            # and (trend_result == 'short')
                            # and (trend_result != "long")
                            # and (df_4h['adx_diff'].iloc[-1] < 0)
                            # and (df_4h['ADX_14'].iloc[-1] < 25)
                            # and (df_1m['RSI_14'].iloc[-1] < 65)
                            # and (df_5m['RSI_14'].iloc[-1] > 20)
                            # and ((globals()['df_' + globals()['atr_pick']]['dmp_diff'].iloc[-1] > 0) or (globals()['df_' + globals()['atr_pick']]['dmn_diff'].iloc[-1] < 0))
                            # and (globals()['df_' + globals()['atr_pick']]['j_diff'].iloc[-1] < 0)
                            # and (globals()['df_' + globals()['atr_pick']]['rsi_diff'].iloc[-1] < 0)
                        )
                        or
                        (
                            (stg_type in ['stg2'])
                            # and (df_15m.feature1.iloc[-1] > 0)
                            # and (position_size == 0)
                            and (peaker_side == 'short')
                            and (peaker_option == 'forward')
                            # and (globals()['df_15m']['second_combined_diff_filtered'].iloc[-1] > -0.25)
                            # and (globals()['df_' + globals()['atr_pick']]['low'].min() < globals()['df_' + globals()['atr_pick']]['low'].iloc[-1])
                            # and (globals()['df_' + globals()['atr_pick']]['high'].max() > globals()['df_' + globals()['atr_pick']]['high'].iloc[-1])
                            # and not ((df_1h['feature1'].iloc[-1] > 0) and (df_1h['feature1_diff'].iloc[-1] > 0))
                            # and (trade_type == "trending_trade")
                            # and (intersect == 1)
                            # and (trend_result == 'short')
                            # and (trend_result != "long")            
                            # and (df_4h['adx_diff'].iloc[-1] > 0)
                            # and (df_4h['ADX_14'].iloc[-1] > 25)
                            # and ((df_4h['ATRr_1'].iloc[-1]) > (df_4h['close'].iloc[-1] * 0.8/100))
                            # and (globals()['df_' + globals()['atr_pick']]['adx_diff'].iloc[-1] > 0)
                            # and (df_1m['RSI_14'].iloc[-1] < 65)
                            # and (df_5m['RSI_14'].iloc[-1] > 20)
                            # and (globals()['df_' + globals()['atr_pick']]['j_diff'].iloc[-1] < 0)
                            # and (globals()['df_' + globals()['atr_pick']]['rsi_diff'].iloc[-1] < 0)
                        )
                        or
                        (
                            (stg_type in ['stg1'])
                            and (peaker_side == 'short')
                            and (peaker_option == 'forward')
                            and (globals()['df_1m']['second_combined_diff_filtered'].iloc[-1] > -0.7)
                            and (globals()['df_5m']['second_combined_diff_filtered'].iloc[-1] > -0.7)
                            and ((globals()['df_15m']['second_combined_diff_filtered'].iloc[-1]) > -0.7)
                            # and (globals()['df_15m']['second_combined_diff_filtered'].iloc[-1] > -0.3)
                            # and (globals()['df_4h']['second_combined_diff_filtered'].iloc[-1] > -0.3)








                            # and 
                            # (
                            #     (
                            #         (stg_type_fixed in ['stg1']) 
                            #         and (position_size != 0)
                            #         and (position_side == 'long')
                            #         and (stric_exit_min_price < symbol_ticker_last)
                            #     )
                            #     or
                            #     (
                            #         (stg_type_fixed in ['stg1']) 
                            #         and (position_size != 0)
                            #         and (position_side == 'short')
                            #     )
                            #     or 
                            #     (
                            #         position_size == 0
                            #     )
                            # )











                            # and ((stg_type_fixed in ['stg1']) or (position_size == 0))
                            # and ((df_1h['feature1'].iloc[-1] > 0) and (df_1h['feature1_diff'].iloc[-1] > 0))
                            # and not (df_15m.feature1.iloc[-1] < -0.2)
                            # and (df_15m.feature1.iloc[-1] > 0)
                            # and (position_size == 0)
                            # and (globals()['df_' + globals()['atr_pick']]['low'].min() < globals()['df_' + globals()['atr_pick']]['low'].iloc[-1])
                            # and (globals()['df_' + globals()['atr_pick']]['high'].max() > globals()['df_' + globals()['atr_pick']]['high'].iloc[-1])

                            # and (globals()['df_' + globals()['atr_pick']]['lowess_1'].iloc[-1] < globals()['df_' + globals()['atr_pick']]['close'].iloc[-1])
                            # and (globals()['df_' + globals()['atr_pick']]['lowess_1_diff'].iloc[-1] < 0)

                            # and (trade_type == "trending_trade")
                            # and (intersect == 1)
                            # and (trend_result == 'short')
                            # and (trend_result != "long")            
                            # and (df_15m['adx_diff'].iloc[-1] > 0)
                            # and (df_4h['ADX_14'].iloc[-1] > 25)
                            # and ((df_4h['ATRr_1'].iloc[-1]) > (df_4h['close'].iloc[-1] * 0.8/100))
                            # and (globals()['df_' + globals()['atr_pick']]['adx_diff'].iloc[-1] > 0)
                            # and (df_1m['RSI_14'].iloc[-1] < 65)
                            # and (df_5m['RSI_14'].iloc[-1] > 20)
                            # and (globals()['df_' + globals()['atr_pick']]['j_diff'].iloc[-1] < 0)
                            # and (globals()['df_' + globals()['atr_pick']]['rsi_diff'].iloc[-1] < 0)
                        )
                        or
                        (
                            (stg_type in ['stgZ'])
                            # and ((stg_type_fixed in ['stg1']) or (position_size == 0))
                            # and ((df_1h['feature1'].iloc[-1] > 0) and (df_1h['feature1_diff'].iloc[-1] > 0))
                            # and not (df_15m.feature1.iloc[-1] < -0.2)
                            # and (df_15m.feature1.iloc[-1] > 0)
                            # and (position_size == 0)
                            and (peaker_side == 'short')
                            and (peaker_option == 'forward')
                            and (globals()['df_1m']['second_combined_diff_filtered'].iloc[-1] > -0.7)
                            and (globals()['df_5m']['second_combined_diff_filtered'].iloc[-1] > -0.7)
                            and ((globals()['df_15m']['second_combined_diff_filtered'].iloc[-1]) > -0.7)
                            and (df_1m['RSI_14'].iloc[-1] > 30)
                            and (df_5m['RSI_14'].iloc[-1] > 30)
                            and (df_1m['macd_diff_35'].iloc[-3] < 0)
                            and (df_1m['macdh_diff_35'].iloc[-3] < 0)
                            # and (globals()['df_' + globals()['atr_pick']]['low'].min() < globals()['df_' + globals()['atr_pick']]['low'].iloc[-1])
                            # and (globals()['df_' + globals()['atr_pick']]['high'].max() > globals()['df_' + globals()['atr_pick']]['high'].iloc[-1])

                            # and (globals()['df_' + globals()['atr_pick']]['lowess_1'].iloc[-1] < globals()['df_' + globals()['atr_pick']]['close'].iloc[-1])
                            # and (globals()['df_' + globals()['atr_pick']]['lowess_1_diff'].iloc[-1] < 0)

                            # and (trade_type == "trending_trade")
                            # and (intersect == 1)
                            # and (trend_result == 'short')
                            # and (trend_result != "long")            
                            # and (df_15m['adx_diff'].iloc[-1] > 0)
                            # and (df_4h['ADX_14'].iloc[-1] > 25)
                            # and ((df_4h['ATRr_1'].iloc[-1]) > (df_4h['close'].iloc[-1] * 0.8/100))
                            # and (globals()['df_' + globals()['atr_pick']]['adx_diff'].iloc[-1] > 0)
                            # and (df_1m['RSI_14'].iloc[-1] < 65)
                            # and (df_5m['RSI_14'].iloc[-1] > 20)
                            # and (globals()['df_' + globals()['atr_pick']]['j_diff'].iloc[-1] < 0)
                            # and (globals()['df_' + globals()['atr_pick']]['rsi_diff'].iloc[-1] < 0)
                        )
                ):
                    if (position_size != 0): # 포지션존재 7 ~ 9.
                        loop = 'short-forward-' + str(loop_counter)
                        message = f'현재 포지션이 존재합니다. stric_exit_min_price={stric_exit_min_price}'
                        exit_status = '7 ~ 9. peak deliver, 포지 점검'
                        # check_pointer(trader_name, exchange_id, market_id, balance_currency, loop, max_waiting_in_second, exit_status, message)
                        print(message)
                        symbol_ticker_last = ticker_calc(market_id)[1]
                        if (position_side == 'short'): # and (globals()['trend_macro_state'] == 'short'):
                            if (position_entry_price < symbol_ticker_last): # 1. 손실중 & 포지션이랑 peaker 방향같을때,  => 물타기 8.
                                scale_order_after_seconds = scale_order_timestamp + dt.timedelta(seconds = scale_order_waiting_seconds) # 물타기 주기
                                current_timestamp = dt.datetime.now().replace(microsecond=0)
                                if current_timestamp < scale_order_after_seconds: # 마지막 물타기로부터 일정기간 경과했는지?
                                    message = f'※ {globals()["atr_pick"]}, 물타기 기간이 너무 이릅니다. {scale_order_timestamp.astimezone(pytz.timezone("Asia/Seoul"))}에 {scaled_level_n}번째 물타기 진입에 성공하였으며 {scale_order_timestamp + dt.timedelta(seconds = scale_order_waiting_seconds) - dt.datetime.now().replace(microsecond=0)} 초 후에 {round((scaled_level_n + 1), 2)}번째 물타기 시도가 가능합니다.'
                                    exit_status = '8-1. ※물타기 기간 점검'
                                    # check_pointer(trader_name, exchange_id, market_id, balance_currency, loop, max_waiting_in_second, exit_status, message)
                                    print(message)
                                    send_to_telegram(trader_name, symbol, message)
                                elif ((-1 * position_value) + (scale_order_position_amount*symbol_ticker_last)) > (wallet_balance * lev_limit): # if 현재 position_value 가 25배 넘을경우, ※물타기 pass!
                                    if inverse_exchange:
                                        message = f'※ {globals()["atr_pick"]}, 현재 들고 있는 position_value의 값이 약 ${position_value}이며, 물탈경우 wallet_balance ${wallet_balance} 대비 Leverage Limit({lev_limit}) 배를 초과하게 됩니다. 물타기를 pass 하겠습니다.'
                                    else:
                                        message = f'※ {globals()["atr_pick"]}, 현재 들고 있는 position_value의 값이 약 ${position_value}이며, 물탈경우 wallet_balance ${wallet_balance} 대비 Leverage Limit({lev_limit}) 배를 초과하게 됩니다. 물타기를 pass 하겠습니다.'
                                    exit_status = '8-2. ※물타기 lev_limit 점검'
                                    # check_pointer(trader_name, exchange_id, market_id, balance_currency, loop, max_waiting_in_second, exit_status, message)
                                    print(message)
                                    send_to_telegram(trader_name, symbol, message)
                                # elif (position_entry_price*(1 + (previous_price_frequency_val/100)) > symbol_ticker_last):
                                #     message = f'현재가(${round(symbol_ticker_last, 2)}) - 진입가(${round(position_entry_price, 2)}) 의 변동성(${round((position_entry_price - symbol_ticker_last), 2)}, {(position_entry_price/symbol_ticker_last - 1)*100}%)이 지난 15분봉의 변동성({round(previous_price_frequency_val, 2)}%) 보다 작습니다. �� 물린가격이 임계치 이하이므로 물타기를 pass 하겠습니다.'
                                #     check_pointer(trader_name, exchange_id, market_id, balance_currency, loop, max_waiting_in_second, exit_status, message)
                                #     #print(message)
                                # elif (globals()['atr_peak']) and (globals()['atr_given']!= 0) and  ((position_entry_price + (atr_given * 1)) > symbol_ticker_last): # atr 변동보다 더 움직였을때만 물타고, 안움직였을때는 물타기 pass!
                                #     message = f'현재가(${symbol_ticker_last}) - 진입가(${position_entry_price}) 의 변동성(${position_entry_price - symbol_ticker_last}, {(position_entry_price/symbol_ticker_last - 1)*100}%)이 atr(* 1) 변동성(${atr_given * 1}) 보다 작습니다. ※ 물린가격이 임계치 이하이므로 물타기를 pass 하겠습니다.'
                                #     # check_pointer(trader_name, exchange_id, market_id, balance_currency, loop, max_waiting_in_second, exit_status, message)
                                #     print(message)
                                elif (position_entry_price * (1 + (greedy_percentage_calc(globals()['atr_pick'])[1]) / 100) > symbol_ticker_last): # 현재가가 진입가 대비 greedy_percentage보다 더 움직였을때,
                                    message = f' {globals()["atr_pick"]}, 현재가(${symbol_ticker_last}) - 진입가(${position_entry_price}) 의 변동성(${symbol_ticker_last - position_entry_price}, {(symbol_ticker_last/position_entry_price - 1)*100}%)이 greedy_percentage 변동성('
                                    message += str(greedy_percentage_calc(globals()['atr_pick'])[1]) + '%)보다 작습니다. ※ 물린가격이 임계치 이하이므로 물타기를 pass 하겠습니다.'
                                    exit_status = '8-3. ※물타기 가격 변동 점검'
                                    send_to_telegram(trader_name, symbol, message)
                                # ############################################################################################################
                                # elif (scalping_direction_pick in ['long', 'short']) and (position_entry_price * (1 + (greedy_percentage_calc(globals()[scalping_direction_pick + '_trend_micro_interval_pick'])[1]) / 100) > symbol_ticker_last): # 현재가가 진입가 대비 greedy_percentage보다 더 움직였을때,
                                #     message = f'진입가(${position_entry_price}) - 현재가(${symbol_ticker_last}) 의 변동성(${symbol_ticker_last - position_entry_price}, {(symbol_ticker_last/position_entry_price - 1)*100}%)이 greedy_percentage 변동성('
                                #     message += str(greedy_percentage_calc(globals()[scalping_direction_pick + '_trend_micro_interval_pick'])[2]) + ', ' + str(greedy_percentage_calc(globals()[scalping_direction_pick + '_trend_micro_interval_pick'])[1]) + '%)보다 작습니다. ※ 물린가격이 임계치 이하이므로 물타기를 pass 하겠습니다.'
                                #     exit_status = '8-3. ※물타기 가격 변동 점검'
                                #     # check_pointer(trader_name, exchange_id, market_id, balance_currency, loop, max_waiting_in_second, exit_status, message)
                                #     #print(message)
                                # elif (scalping_direction_pick in ['neutral']) and (position_entry_price * (1 + (greedy_percentage_calc(globals()['short_trend_micro_interval_pick'])[1]) / 100) > symbol_ticker_last): # 현재가가 진입가 대비 greedy_percentage보다 더 움직였을때,
                                #     message = f'진입가(${position_entry_price}) - 현재가(${symbol_ticker_last}) 의 변동성(${symbol_ticker_last - position_entry_price}, {(symbol_ticker_last/position_entry_price - 1)*100}%)이 greedy_percentage 변동성('
                                #     message += str(greedy_percentage_calc(globals()['short_trend_micro_interval_pick'])[2]) + ', ' + str(greedy_percentage_calc(globals()['short_trend_micro_interval_pick'])[1]) + '%)보다 작습니다. ※ 물린가격이 임계치 이하이므로 물타기를 pass 하겠습니다.'
                                #     exit_status = '8-3. ※물타기 가격 변동 점검'
                                #     # check_pointer(trader_name, exchange_id, market_id, balance_currency, loop, max_waiting_in_second, exit_status, message)
                                #     #print(message)
                                # ############################################################################################################
                                else: #물타기로부터 기간 만족
                                    exchange.cancel_all_orders(market_id) # open_order 모두 취소, 포지션 변경 준비
                                    message = f"stg_type: {stg_type}, 현재의 포지션은 손실중입니다. 물타기로부터 기간 만족하므로 물타기 시도를 시작하겠습니다."
                                    exit_status = '8-4. ※물타기 시작'
                                    # check_pointer(trader_name, exchange_id, market_id, balance_currency, loop, max_waiting_in_second, exit_status, message)
                                    print(message)
                                    if exchange_id == 'huobi':
                                        limit_type_only_case_price_pick = float(exchange.price_to_precision(market_id, float(symbol_ticker_last)*(1 - 0.0005)))
                                        exchange.create_order(symbol=market_id, type='limit', side='sell', amount=scale_order_position_amount, price=limit_type_only_case_price_pick, params=params)
                                    else:
                                        exchange.create_order(symbol=market_id, type='market', side='sell', amount=scale_order_position_amount, params=params) # 물타기
                                        stg_type_fixed = stg_type
                                    exit_status = '8-5. 물타기 완료'
                                    scale_order_timestamp = dt.datetime.now().replace(microsecond=0) # 물탄시점 저장
                                    #print('-')
                                    position_side, position_size, position_value, position_entry_price, position_entry_time, liquidation_price, unrealised_pnl, roe_pcnt = position_calc(exchange_id, market_id)
                                    wallet_balance = balance_calc(exchange_id, balance_currency)
                                    scaled_level_n = scale_order_position_amount_calc(min_order_amount, wallet_balance, max_leverage, position_size, r, scale_order_max_limit)[3]
                                    message = f'stg_type: {stg_type}, {globals()["atr_pick"]}, {scale_order_timestamp.astimezone(pytz.timezone("Asia/Seoul"))}에 {scaled_level_n}번째 물타기 진입에 성공하였습니다.'
                                    send_to_telegram(trader_name, symbol, message)
                                    # check_pointer(trader_name, exchange_id, market_id, balance_currency, loop, max_waiting_in_second, exit_status, message)
                                    print(message)
                                    ############################################################################################################
                                    cumulate_lv = cumulate_lv_calc(market_id, intervals)
                                    exit_order_position_amount = exit_order_position_amount_calc(position_size)
                                    symbol_ticker_last = ticker_calc(market_id)[1]
                                    pick_max, pick_min, previous_price_frequency, greedy_percentage, picked_interval = pick_calc(cumulate_lv, position_side, position_size, position_entry_price, symbol_ticker_last, scalping_direction_pick)
                                    if exit_order_position_amount != 0:
                                        exchange.create_order(symbol=market_id, type='limit', side='buy', amount=exit_order_position_amount, price=pick_min, params={'reduceOnly': True}) # exit_order 재생성
                                        exit_order_timestamp = time.time()
                                        max_waiting_in_second = exit_order_waiting_seconds
                                        exit_status = '8-6. exit_order 재산출'
                                        message = f'exit_order를 재산출하여 새롭게 생성 하였습니다. {picked_interval} interval 기준으로 전 가격의 frequency는 {previous_price_frequency}, greedy_percentage는 {greedy_percentage} 입니다.'
                                        ############################################################################################################
                                        symbol_ticker_last = ticker_calc(market_id)[1]
                                        if (((-1 * position_value) > (wallet_balance * lev_limit)) or ((unrealised_pnl / wallet_balance) * 100 >= 3)): # if 현재 position_value 가 25배 넘을경우, atr 역지 생성
                                            # stopPrice = float(exchange.price_to_precision(market_id, (position_entry_price + ((position_entry_price - pick_min)/stopPrice_const))))
                                            l_p_l, l_p_h, last_peaked_price = l_p_l_and_l_p_h_calc(symbol_ticker_last)
                                            stop_loss_, percentage_difference_stop_loss_, target_p_, percentage_difference_target_p_ = stop_loss_calc(stg_type, success, position_side, position_entry_price, symbol_ticker_last, close_price_low, close_price_high, pick_max, pick_min, stop_loss_range)
                                            if stop_loss_ > symbol_ticker_last: # 현재 포지가 short 일때
                                                exchange.create_order(symbol=market_id, type='market', side='buy', amount=exit_order_position_amount, params={'stopPrice': stop_loss_, 'triggerDirection': 'above'}) # stop market 재생성
                                                exit_status = '8-7. exit_order 재산출하여 + 역지'
                                                message += f' 물타기 후 역지 재생성 하였습니다.'
                                                message += '[2:1]: ' + str((position_entry_price - pick_min)/2) + ', 손절가: ' + str(position_entry_price + ((position_entry_price - pick_min)/stopPrice_const))
                                        ############################################################################################################
                                        # symbol_ticker_last = ticker_calc(market_id)[1]
                                        # if (globals()['short_trend_atr_given'] != 0) and (exchange_id == 'binanceusdm') and ((-1 * position_value) + (scale_order_position_amount*symbol_ticker_last) > (wallet_balance * lev_limit)): # if 현재 position_value 가 25배 넘을경우, atr 역지 생성
                                        #     stopPrice = float(exchange.price_to_precision(market_id, (position_entry_price + (globals()['short_trend_atr_given'] * 1.0))))
                                        #     if stopPrice > symbol_ticker_last: # 현재 포지가 short 일때
                                        #         exchange.create_order(symbol=market_id, type='stop_market', side='buy', amount=exit_order_position_amount, params={'stopPrice': stopPrice}) # stop market 재생성
                                        #         exit_status = '8-7. exit_order 재산출하여 + short_trend_atr_given 역지'
                                        #         message += f' 손실중이므로 short_trend_atr_given 역지 재생성 하였습니다.'
                                        #         message += ' short_trend_atr_given * 1.0: ' + str(globals()['short_trend_atr_given'] * 1.0) + ', 손절가: ' + str(position_entry_price  - (globals()['short_trend_atr_given'] * 1.0))
                                        #         #print(message)
                                        ############################################################################################################
                                    else:
                                        max_waiting_in_second = ''
                                        exit_status = '8-8. '
                                        message = f'현재 포지션이 즉시 사라진것같습니다. exit_order가 생성되지 않았습니다.'
                                    # check_pointer(trader_name, exchange_id, market_id, balance_currency, loop, max_waiting_in_second, exit_status, message)
                                    print(message)
                                    ############################################################################################################
                    
                    
                        elif (position_side == 'long'): # and (position_entry_price < symbol_ticker_last): # 수익중
                            # if ((stg_type in ['stg10', 'stg100']) or ((stg_type in ['stg3', 'stg9']) and (scalping_direction_pick =='short'))):
                            #if not ((stg_type == 'stg10') and (globals()['atr_pick'] == '1m')):
                            # if globals()['trend_macro_state'] == 'short':
                            # if scalping_direction_pick == 'short': # short trending scalping start
                            # if globals()['trend_macro_state'] == 'short':
                            exchange.cancel_all_orders(market_id) # open_order 모두 취소, 포지션 변경 준비
                            loop = f'short-forward-new-{loop_counter}'
                            message = '현재의 포지션은 이익중입니다. 종료후 반대진입 시도를 시작하겠습니다.'
                            exit_status = '10-1. 반대진입 시작'
                            # check_pointer(trader_name, exchange_id, market_id, balance_currency, loop, max_waiting_in_second, exit_status, message)
                            print(message)
                            symbol_ticker_last = ticker_calc(market_id)[1]
                            exit_order_position_amount_re = (exit_order_position_amount + initial_order_amount)

                            if exchange_id == 'huobi':
                                limit_type_only_case_price_pick = float(exchange.price_to_precision(market_id, float(symbol_ticker_last)*(1 - 0.0005)))
                                exchange.create_order(symbol=market_id, type='limit', side='sell', amount=exit_order_position_amount_re, price=limit_type_only_case_price_pick, params=params)
                            else:
                                #exchange.create_order(symbol=market_id, type='limit', side='sell', amount=exit_order_position_amount_re, price=pick_max, params=params)
                                exchange.create_order(symbol=market_id, type='market', side='sell', amount=exit_order_position_amount_re, params=params) # 시장가 진입
                                stg_type_fixed = stg_type
                                # limit_type_only_case_price_pick = float(exchange.price_to_precision(market_id, float(symbol_ticker_last)*(1 + 0.0005)))
                                # exchange.create_order(symbol=market_id, type='limit', side='sell', amount=exit_order_position_amount_re, price=limit_type_only_case_price_pick, params=params)

                            scale_order_timestamp = dt.datetime.now().replace(microsecond=0) # 진입시점 저장
                            message = f'stg_type: {stg_type}, {globals()["atr_pick"]}, entry_order upper_band 생성하였습니다.'
                            exit_status = '10-2. 포지션 진입'
                            send_to_telegram(trader_name, symbol, message)
                            # check_pointer(trader_name, exchange_id, market_id, balance_currency, loop, max_waiting_in_second, exit_status, message)
                            print(message)
                            ############################################################################################################
                            #print('-')
                            position_side, position_size, position_value, position_entry_price, position_entry_time, liquidation_price, unrealised_pnl, roe_pcnt = position_calc(exchange_id, market_id)
                            cumulate_lv = cumulate_lv_calc(market_id, intervals)
                            symbol_ticker_last = ticker_calc(market_id)[1]
                            pick_max, pick_min, previous_price_frequency, greedy_percentage, picked_interval = pick_calc(cumulate_lv, position_side, position_size, position_entry_price, symbol_ticker_last, scalping_direction_pick)
                            exit_order_position_amount = exit_order_position_amount_calc(position_size)
                            if exit_order_position_amount != 0:
                                exchange.create_order(symbol=market_id, type='limit', side='buy', amount=exit_order_position_amount, price=pick_min, params={'reduceOnly': True}) # exit_order 재생성
                                exit_order_timestamp = time.time()
                                max_waiting_in_second = exit_order_waiting_seconds
                                exit_status = '10-3. exit_order 재산출'
                                message = f'exit_order를 재산출하여 새롭게 생성 하였습니다. {picked_interval} interval 기준으로 전 가격의 frequency는 {previous_price_frequency}, greedy_percentage는 {greedy_percentage} 입니다.'
                                ############################################################################################################
                                symbol_ticker_last = ticker_calc(market_id)[1]
                                if (((-1 * position_value) > (wallet_balance * lev_limit)) or ((unrealised_pnl / wallet_balance) * 100 >= 3)): # if 현재 position_value 가 25배 넘을경우, atr 역지 생성
                                    # stopPrice = float(exchange.price_to_precision(market_id, (position_entry_price + ((position_entry_price - pick_min)/stopPrice_const))))
                                    l_p_l, l_p_h, last_peaked_price = l_p_l_and_l_p_h_calc(symbol_ticker_last)
                                    stop_loss_, percentage_difference_stop_loss_, target_p_, percentage_difference_target_p_ = stop_loss_calc(stg_type, success, position_side, position_entry_price, symbol_ticker_last, close_price_low, close_price_high, pick_max, pick_min, stop_loss_range)
                                    if stop_loss_ > symbol_ticker_last: # 현재 포지가 short 일때
                                        exchange.create_order(symbol=market_id, type='market', side='buy', amount=exit_order_position_amount, params={'stopPrice': stop_loss_, 'triggerDirection': 'above'}) # stop market 재생성
                                        exit_status = '8-7. exit_order 재산출하여 + 역지'
                                        message += f' 역지 생성 하였습니다.'
                                        message += '[2:1]: ' + str((position_entry_price - pick_min)/stopPrice_const) + ', 손절가: ' + str(position_entry_price + ((position_entry_price - pick_min)/stopPrice_const))
                                ############################################################################################################
                            else :
                                max_waiting_in_second = ''
                                exit_status = '10-4. '
                                message = f'현재 포지션이 즉시 사라진것같습니다. exit_order가 생성되지 않았습니다.'
                            # check_pointer(trader_name, exchange_id, market_id, balance_currency, loop, max_waiting_in_second, exit_status, message)
                            print(message)
                            ############################################################################################################
                        #print('12-', (dt.datetime.now().replace(microsecond=0).isoformat()).split('T')[-1])


                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    else: # 포지션 없을때 10.
                        # if ((stg_type in ['stg10', 'stg100']) or ((stg_type in ['stg3', 'stg9']) and (scalping_direction_pick =='short'))):
                        #if not ((stg_type == 'stg10') and (globals()['atr_pick'] == '1m')):
                        # if globals()['trend_macro_state'] == 'short':
                        # if scalping_direction_pick == 'short': # short trending scalping start
                        # if globals()['trend_macro_state'] == 'short':
                        exchange.cancel_all_orders(market_id) # open_order 모두 취소, 포지션 변경 준비
                        loop = f'short-forward-new-{loop_counter}'
                        message = '현재 포지션이 존재하지 않습니다.'
                        exit_status = '10-1. 포지션 점검'
                        # check_pointer(trader_name, exchange_id, market_id, balance_currency, loop, max_waiting_in_second, exit_status, message)
                        print(message)
                        symbol_ticker_last = ticker_calc(market_id)[1]
                        if exchange_id == 'huobi':
                            limit_type_only_case_price_pick = float(exchange.price_to_precision(market_id, float(symbol_ticker_last)*(1 - 0.0005)))
                            exchange.create_order(symbol=market_id, type='limit', side='sell', amount=initial_order_amount, price=limit_type_only_case_price_pick, params=params)
                        else:
                            #exchange.create_order(symbol=market_id, type='limit', side='sell', amount=initial_order_amount, price=pick_max, params=params)
                            exchange.create_order(symbol=market_id, type='market', side='sell', amount=initial_order_amount, params=params) # 시장가 진입
                            stg_type_fixed = stg_type
                            # limit_type_only_case_price_pick = float(exchange.price_to_precision(market_id, float(symbol_ticker_last)*(1 + 0.0005)))
                            # exchange.create_order(symbol=market_id, type='limit', side='sell', amount=initial_order_amount, price=limit_type_only_case_price_pick, params=params)
                        scale_order_timestamp = dt.datetime.now().replace(microsecond=0) # 진입시점 저장
                        entry_order_timestamp = time.time()
                        message = f'stg_type: {stg_type}, {globals()["atr_pick"]}, entry_order upper_band 생성하였습니다.'
                        exit_status = '10-2. 포지션 진입'
                        send_to_telegram(trader_name, symbol, message)
                        # check_pointer(trader_name, exchange_id, market_id, balance_currency, loop, max_waiting_in_second, exit_status, message)
                        print(message)
                        ############################################################################################################
                        #print('-')
                        position_side, position_size, position_value, position_entry_price, position_entry_time, liquidation_price, unrealised_pnl, roe_pcnt = position_calc(exchange_id, market_id)
                        cumulate_lv = cumulate_lv_calc(market_id, intervals)
                        symbol_ticker_last = ticker_calc(market_id)[1]
                        pick_max, pick_min, previous_price_frequency, greedy_percentage, picked_interval = pick_calc(cumulate_lv, position_side, position_size, position_entry_price, symbol_ticker_last, scalping_direction_pick)
                        exit_order_position_amount = exit_order_position_amount_calc(position_size)
                        if exit_order_position_amount != 0:
                            exchange.create_order(symbol=market_id, type='limit', side='buy', amount=exit_order_position_amount, price=pick_min, params={'reduceOnly': True}) # exit_order 재생성
                            exit_order_timestamp = time.time()
                            max_waiting_in_second = exit_order_waiting_seconds
                            exit_status = '10-3. exit_order 재산출'
                            message = f'exit_order를 재산출하여 새롭게 생성 하였습니다. {picked_interval} interval 기준으로 전 가격의 frequency는 {previous_price_frequency}, greedy_percentage는 {greedy_percentage} 입니다.'
                            ############################################################################################################
                            symbol_ticker_last = ticker_calc(market_id)[1]
                            if (((-1 * position_value) > (wallet_balance * lev_limit)) or ((unrealised_pnl / wallet_balance) * 100 >= 3)): # if 현재 position_value 가 25배 넘을경우, atr 역지 생성
                                # stopPrice = float(exchange.price_to_precision(market_id, (position_entry_price + ((position_entry_price - pick_min)/stopPrice_const))))
                                l_p_l, l_p_h, last_peaked_price = l_p_l_and_l_p_h_calc(symbol_ticker_last)
                                stop_loss_, percentage_difference_stop_loss_, target_p_, percentage_difference_target_p_ = stop_loss_calc(stg_type, success, position_side, position_entry_price, symbol_ticker_last, close_price_low, close_price_high, pick_max, pick_min, stop_loss_range)
                                if stop_loss_ > symbol_ticker_last: # 현재 포지가 short 일때
                                    exchange.create_order(symbol=market_id, type='market', side='buy', amount=exit_order_position_amount, params={'stopPrice': stop_loss_, 'triggerDirection': 'above'}) # stop market 재생성
                                    exit_status = '8-7. exit_order 재산출하여 + 역지'
                                    message += f' 역지 생성 하였습니다.'
                                    message += '[2:1]: ' + str((position_entry_price - pick_min)/stopPrice_const) + ', 손절가: ' + str(position_entry_price + ((position_entry_price - pick_min)/stopPrice_const))
                            ############################################################################################################
                        else :
                            max_waiting_in_second = ''
                            exit_status = '10-4. '
                            message = f'현재 포지션이 즉시 사라진것같습니다. exit_order가 생성되지 않았습니다.'
                        # check_pointer(trader_name, exchange_id, market_id, balance_currency, loop, max_waiting_in_second, exit_status, message)
                        print(message)
                        ############################################################################################################
                    #print('12-', (dt.datetime.now().replace(microsecond=0).isoformat()).split('T')[-1])



            # elif (peaker_side =='short') and (peaker_option == 'reverse') and (point_sum > 0): # if peacker == 'short_reverse'
            #     if (position_size != 0): # 포지션존재
            #         loop = 'short-reverse-' + str(loop_counter)
            #         message = f'현재 포지션이 존재합니다. stric_exit_min_price={stric_exit_min_price}'
            #         check_pointer(trader_name, exchange_id, market_id, balance_currency, loop, max_waiting_in_second, exit_status, message)
            #         #print(message)
            #         if position_side == 'short': # 포지션 숏 일때
            #             if (stric_exit_min_price > symbol_ticker_last): # 3. 수익중 & 포지션이랑 peaker 방향다를때,  => 익절
            #                 exchange.cancel_all_orders(market_id) # open_order 모두 취소
            #                 exit_order_position_amount = exit_order_position_amount_calc(position_size, min_order_amount)

            #                 if exchange_id == 'huobi':
            #                     symbol_ticker_last = ticker_calc(market_id)[1]
            #                     limit_type_only_case_price_pick = float(exchange.price_to_precision(market_id, float(symbol_ticker_last)*(1 + 0.0005)))
            #                     exchange.create_order(symbol=market_id, type='limit', side='buy', amount=exit_order_position_amount, price=limit_type_only_case_price_pick, params=params)
            #                 else:
            #                     exchange.create_order(symbol=market_id, type='market', side='buy', amount=exit_order_position_amount, params=params) # 4. 익절
            #                 exit_status = '포지션과 peaker 방향 다름, 시장가 익절'
            #                 message = f'포지션과 peaker 방향이 다르므로 포지션을 시장가 익절하였습니다.'
            #                 check_pointer(trader_name, exchange_id, market_id, balance_currency, loop, max_waiting_in_second, exit_status, message)
            #                 #print(message)
            ####################################################################################################################
            ####################################################################################################################
            ####################################################################################################################
            #print('loop' + str(loop_counter) + 'end!')
            #print('13-', (dt.datetime.now().replace(microsecond=0).isoformat()).split('T')[-1])






        
        
        
        
        #####################################################################################################################################################
        #####################################################################################################################################################
        #####################################################################################################################################################
        #####################################################################################################################################################






















            
            
            
            
            
            
            
            
            
            
            time.sleep(5)
            #####################################################################################################################################################
            #####################################################################################################################################################
            #####################################################################################################################################################
            #####################################################################################################################################################
            #####################################################################################################################################################

    except (Exception, KeyboardInterrupt) as e:
        # formatted_datetime = datetime.datetime.now().replace(microsecond=0)
        # Get the traceback information
        traceback_info = traceback.format_exc()
        
        # Include the traceback information in the error message
        error_message = f"Error in {trader_name} ({symbol}), attempts: {attempts},\n{traceback_info}"
        
        if attempts < max_attempts:
            if isinstance(e, KeyboardInterrupt):
                send_to_telegram(f"Trader: {trader_name}, {symbol}, attempts: {str(attempts)},\nerror:", "KeyboardInterrupt: Script execution was interrupted.")
                raise RuntimeError("KeyboardInterrupt: Script execution was interrupted.") from e
            else:
                send_to_telegram(f"Trader: {trader_name}, {symbol}, attempts: {str(attempts)},\nerror:", error_message)
                attempts += 1
                if any(substring in str(e).lower() for substring in error_message):
                    time.sleep(retry_delay)
                else:
                    time.sleep(20)
        else:
            if isinstance(e, KeyboardInterrupt):
                send_to_telegram(f"Trader: {trader_name}, {symbol}, attempts: Max number of retries met,\nerror:", "KeyboardInterrupt: Script execution was interrupted.")
            else:
                send_to_telegram(f'Trader: {trader_name}, {symbol}, attempts: Max number of retries met,\nerror:', error_message)
            raise RuntimeError("Max number of retries met.") from e