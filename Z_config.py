# Backtest - Z_config.py 
lookback_hours_parameter = 24*7*4*3 #Wie viele Stunden zurück soll es Backtesten? | Parameter die gut funktioniert haben in 'lookback_hours_parameter' sind gleichzusetzen mit 'fetch' in Live-Bot-Config da diese symbole per Fifo danach aussoertiert werden.


use_custom_backtest_datetime = False
backtest_datetime = "2025-03-22 17:35:00" #Von hier 'lookback_hours_parameter' Std. nach hinten
taker_fee_parameter = 0.00045 #Binance Taker fee standard. Soll so bleiben | nicht vorhanden in Live-Bot-Config
#
# Trade an manchen Tagen. - und dynamisch pro jede candle auch die funktion einfügen, dass es sequentiell alle 10min nach top 200 symbolen welche auf top 100 reduziert wurden welche dann eben in einer bestimmten zeit eine % spanne haben mussten. und am besten soll es auch day tracking machen können.  in dieser zeit schauen soll.
#######################
### Trade Management ##
#######################

start_balance_parameter = 25 #USDT per symbol/trade (pro Seite {buy/sell}). Jeder Trade wird zurückgesetzt | Live-Bot-Config: 'geld'
max_daily_trades_parameter = 100 # wird immer um 00:00 zurückgesetzt | Live-Bot-Config: 'max_daily_trades_parameter'
max_daily_loss_parameter = 5 #% max Verlust wird auch um 00:00 zurückgesetzt | Live-Bot-Config: 'max_daily_loss_parameter'
risk_per_trade = 1  #1 = 100% von 'start_balance_parameter' wird pro trade soviel usdt genutzt | nicht vorhanden in Live-Bot-Config
#
# # BASIERT AUF BASIS VON Einstiegspreis 5min candle anfangspreis. Also ungenau... (noch.)
# Binance Stop loss & TP benutzen?
use_standard_sl_tp = True #Live-Bot-Config: 'use_standard_sl_tp'
stop_loss_parameter_binance = 0.011 #Falls ja, hier STOP LOSS angeben - Binance Order | Live-Bot-Config: 'stop_loss_parameter_binance'
take_profit_parameter_binance = 0.01 #Falls ja, hier TAKE PROFIT angeben - Binance Order | Live-Bot-Config: 'take_profit_parameter_binance'
slippage = 0.003 # 0.001 = 0.1% ... Verfehle den Preis aufgrund von market order aber falschen preis bekommen = slippage %. also zu ungusten von mir.
#
### Note! - Wenn man ohne Stop_loss & Take profit backtesten will, sondern nur mit signalen, muss man stop loss und take profit VON BINANCE sehr hoch einstellen. also in diesem Fall MUSS: use_standard_sl_tp = True
#
#
## Workers - Threads #
max_worker_threads = 12 #
# Anzahl der gleichzeitig zu verarbeitenden Symbole
max_concurrent_symbols = 1  # Symbole gleichzeitig laufen lasse
#
### TRAILING Parameter:      (Wenn 'use_standard_sl_tp' = True werden diese nicht beachtet)
# # BASIERT AUF BASIS VON Einstiegspreis 5min candle anfangspreis. Also ungenau... (noch.)
# --- Trailing Stop Parameter (Optimierte Werte eingesetzt, wo vorhanden) ---
# use_standard_sl_tp = False # Optimiert -> Trailing/Multi-TP wird verwendet

activation_threshold = 0.7                   # Optimierter Wert (war param_trailing_activation_threshold)
trailing_distance = 5                      # Optimierter Wert (war param_trailing_distance)
adjustment_step = 0.2                        # Wert aus deinem Beispiel (nicht optimiert)
#
# Take Profit Parameter (Optimierte Werte eingesetzt, wo vorhanden)
take_profit_levels = [2.5, 4.0, 6]         # Optimierter Wert (war param_take_profit_levels)
take_profit_size_percentages = [20, 35, 45]  # Optimierter Wert (war param_take_profit_size_percentages)
third_level_trailing_distance = 1.5          # Wenn trailing_distance nicht hittet
enable_breakeven = True                     # Optimierter Wert
enable_trailing_take_profit = True          # Optimierter Wert (Kein spezielles Trailing für Stufe 3)

##################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################

# Master-Schalter für Filterung für Beobachtung in bestimmter Zeit oder mit bestimmter % - Änderung
filtering_active = True                      # Wert aus deinem Beispiel

# Observierungsstrategie: (Optimierte Werte eingesetzt)
beobachten_active = False                     # Wert aus deinem Beispiel (Annahme, da Parameter optimiert wurden)
seite = 'both'# z.B.: " seite = 'plus', 'minus' " - Konzentriert sich für Beobachtungszeiträume nur auf eine seite der % Veränderung
min_price_change_pct_min = 5.0               # Optimierter Wert
min_price_change_pct_max = 100.0              # Optimierter Wert (ACHTUNG: Deutlich niedriger als dein Bsp.!)
price_change_lookback_minutes = 60*12           # Optimierter Wert (ACHTUNG: Deutlich kürzer als dein Bsp.!)
symbol_observation_hours = 6                 # Optimierter Wert (ACHTUNG: Kürzer als dein Bsp.!)
close_position = False                        # Optimierter Wert

# Zeitfilter Einstellungen (Werte aus deinem Beispiel)
time_filter_active = False
trading_days = [0, 1, 2, 3, 4, 5]
trading_start_hour = 5
trading_start_minute = 0
trading_end_hour = 17
trading_end_minute = 00
trading_timezone = 'Europe/Berlin'

## Auf eine Seite begrenzen: (Werte aus deinem Beispiel)
allow_short = True
allow_long = True
#Vortag muss diese änderung gehabt haben (Werte aus deinem Beispiel)
filter_previous_day = False
previous_day_direction = 'plus'

############################################
###  Ab hier beginnen Tradingstrategien  ###
############################################

#EMA Bestimmungen gelten für Trendbestimmung an dem Interval
require_timeframe_alignment = False          # Optimierter Wert
### AB HIER HAT JEDER PARAMETER IN LIVE & BACKTEST CODE DIE EXAKT GLEICHEN NAMEN
interval = "5m"                             # Wert aus deinem Beispiel
ema_fast_parameter = 7                      # Optimierter Wert
ema_slow_parameter = 60                      # Optimierter Wert
ema_baseline_parameter = 50                  # Optimierter Wert
back_trend_periode = 3                       # Wert aus deinem Beispiel
#
interval_int_2 = "15m"                       # Wert aus deinem Beispiel (Irrelevant, da require_tf_alignment=False)
interval_int_3 = "15m"                       # Wert aus deinem Beispiel (Irrelevant, da require_tf_alignment=False)
min_trend_strength_parameter = 0.4           # Optimierter Wert
min_trend_duration_parameter = 3             # Optimierter Wert (als Integer)

# RSI Parameters (Optimierte Werte eingesetzt)
rsi_period = 11                              # Optimierter Wert
rsi_buy = 80                                 # Optimierter Wert
rsi_sell = 30                                # Optimierter Wert
rsi_exit_overbought = 80                     # Optimierter Wert
rsi_exit_oversold = 25                       # Optimierter Wert

# Volume Parameters (Werte aus deinem Beispiel, da nicht im optimierten Ergebnis)
volume_sma = 30
volume_multiplier = 0.6
volume_entry_multiplier = 0.5
min_volume = 0.01

# Momentum Parameters (Optimierte Werte eingesetzt)
use_momentum_check = True                    # Optimierter Wert
momentum_lookback = 10                       # Optimierter Wert
momentum = 0.015                             # Optimierter Wert

# Bollinger Bands Parameters (Optimierte Werte eingesetzt)
use_bb = True                               # Optimierter Wert
bb_period = 21                               # Wert aus deinem Beispiel (Irrelevant, da use_bb=False)
bb_deviation = 2.5                           # Wert aus deinem Beispiel (Irrelevant, da use_bb=False)

# MACD Parameters (Optimierte Werte eingesetzt)
use_macd = True                             # Optimierter Wert
macd_fast_period = 21                        # Wert aus deinem Beispiel (Irrelevant, da use_macd=False)
macd_slow_period = 36                        # Wert aus deinem Beispiel (Irrelevant, da use_macd=False)
macd_signal_period = 5                       # Wert aus deinem Beispiel (Irrelevant, da use_macd=False)

# ADX Parameters (Optimierte Werte eingesetzt)
use_adx = True                              # Optimierter Wert
adx_period = 7                               # Wert aus deinem Beispiel (Irrelevant, da use_adx=False)
adx_threshold = 8                            # Wert aus deinem Beispiel (Irrelevant, da use_adx=False)

# VWAP Parameters (Werte aus deinem Beispiel, da nicht optimiert)
use_vwap = True
use_advanced_vwap = True
advanced_vwap_periods = [20, 50, 100]      # Irrelevant, da use_advanced_vwap=False
advanced_vwap_std_threshold = 0.9          # Irrelevant, da use_advanced_vwap=False

# OBV Parameters (Werte aus deinem Beispiel, da nicht optimiert)
use_obv = True
obv_period = 11
debug_obv = False
debug_indicators = False

# Chaikin money flow Parameters (Werte aus deinem Beispiel, da nicht optimiert)
use_chaikin_money_flow = True
cmf_period = 1
cmf_threshold = 0.00

################################################################################################################################################################################################################################################
###########   EINSTELLUNGEN ZU ENDE   ##########################################################################################################################################################################################################
################################################################################################################################################################################################################################################






















#Ablage nicht verändern sondern einfach lassen.:
### Bot Configuration
ut_bot = False # Funktioniert wsl noch, aber wird sowieso nicht gebraucht
### Position Management (wurde hauptsächlich für UT Bot alert verwendet)
key_value_parameter = 1.0
atr_period_parameter = 14
#Zwischenfunktion für Stoploss / welcher genommen werden sollte
#Nicht beachten
buy_delay_1_candle_spaeter = False
if use_standard_sl_tp == True:
    stop_loss_parameter = stop_loss_parameter_binance
    take_profit_parameter = take_profit_parameter_binance
else:
    stop_loss_parameter = None
    take_profit_parameter = None
if interval:
    interval_int = interval
early_exit_check = True  # Wenn True: Beendet den Backtest, wenn aktuelle Systemzeit außerhalb des Handelszeitfensters liegt

your_csv_path_variable = "./strategy_results/full_data.csv"
