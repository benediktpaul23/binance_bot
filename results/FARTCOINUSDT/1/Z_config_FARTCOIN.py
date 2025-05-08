# Backtest - Z_config.py
lookback_hours_parameter = 24*7*4*12*2 #Wie viele Stunden zurück soll es Backtesten? | Parameter die gut funktioniert haben in 'lookback_hours_parameter' sind gleichzusetzen mit 'fetch' in Live-Bot-Config da diese symbole per Fifo danach aussoertiert werden.

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
use_standard_sl_tp = False #Live-Bot-Config: 'use_standard_sl_tp' <-- Updated from Trial 434
stop_loss_parameter_binance = 0.99 #Falls ja, hier STOP LOSS angeben - Binance Order | Live-Bot-Config: 'stop_loss_parameter_binance' <-- Updated from Trial 434
take_profit_parameter_binance = 1.0 #Falls ja, hier TAKE PROFIT angeben - Binance Order | Live-Bot-Config: 'take_profit_parameter_binance' <-- Updated from Trial 434
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
# use_standard_sl_tp = False # Optimiert -> Trailing/Multi-TP wird verwendet <-- Updated from Trial 434 (redundant comment, value set above)

activation_threshold = 0.12                   # Optimierter Wert (war param_trailing_activation_threshold) <-- Updated from Trial 434
trailing_distance = 1.56                      # Optimierter Wert (war param_trailing_distance) <-- Updated from Trial 434
adjustment_step = 0.98                        # Wert aus deinem Beispiel (nicht optimiert) <-- Updated from Trial 434
#
# Take Profit Parameter (Optimierte Werte eingesetzt, wo vorhanden)
take_profit_levels = [1.6, 3.6, 3.6]          # Optimierter Wert (war param_take_profit_levels) <-- Updated from Trial 434
take_profit_size_percentages = [33.33, 33.33, 33.33] # Optimierter Wert (war param_take_profit_size_percentages) <-- Updated from Trial 434
third_level_trailing_distance = 0.98          # Wenn trailing_distance nicht hittet <-- Updated from Trial 434
enable_breakeven = True                       # Optimierter Wert <-- Updated from Trial 434
enable_trailing_take_profit = True            # Optimierter Wert (Kein spezielles Trailing für Stufe 3) <-- Updated from Trial 434

##################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################

# Master-Schalter für Filterung für Beobachtung in bestimmter Zeit oder mit bestimmter % - Änderung
filtering_active = False                      # Wert aus deinem Beispiel <-- Updated from Trial 434

# Observierungsstrategie: (Optimierte Werte eingesetzt)
beobachten_active = False                     # Wert aus deinem Beispiel (Annahme, da Parameter optimiert wurden) <-- Updated from Trial 434
seite = 'long'                                # z.B.: " seite = 'plus', 'minus' " - Konzentriert sich für Beobachtungszeiträume nur auf eine seite der % Veränderung <-- Updated from Trial 434
min_price_change_pct_min = 0.99               # Optimierter Wert <-- Updated from Trial 434
min_price_change_pct_max = 31.47              # Optimierter Wert (ACHTUNG: Deutlich niedriger als dein Bsp.!) <-- Updated from Trial 434
price_change_lookback_minutes = 632           # Optimierter Wert (ACHTUNG: Deutlich kürzer als dein Bsp.!) <-- Updated from Trial 434
symbol_observation_hours = 3                  # Optimierter Wert (ACHTUNG: Kürzer als dein Bsp.!) <-- Updated from Trial 434
close_position = True                         # Optimierter Wert <-- Updated from Trial 434

# Zeitfilter Einstellungen (Werte aus deinem Beispiel)
time_filter_active = False
trading_days = [0, 1, 2, 3, 4, 5]
trading_start_hour = 5
trading_start_minute = 0
trading_end_hour = 17
trading_end_minute = 00
trading_timezone = 'Europe/Berlin'

## Auf eine Seite begrenzen: (Werte aus deinem Beispiel)
allow_short = True                            # <-- Updated from Trial 434
allow_long = True                             # <-- Updated from Trial 434
#Vortag muss diese änderung gehabt haben (Werte aus deinem Beispiel)
filter_previous_day = False                   # <-- Updated from Trial 434
previous_day_direction = 'bullish'            # <-- Updated from Trial 434

############################################
###  Ab hier beginnen Tradingstrategien  ###
############################################

#EMA Bestimmungen gelten für Trendbestimmung an dem Interval
require_timeframe_alignment = False          # Optimierter Wert -> Wert aus Trial 265 <-- Updated from Trial 434
### AB HIER HAT JEDER PARAMETER IN LIVE & BACKTEST CODE DIE EXAKT GLEICHEN NAMEN
interval = "5m"                             # Wert aus deinem Beispiel -> Wert aus Trial 265 (Annahme basierend auf interval_int_2/3) <-- Consistent with Trial 434 intervals
ema_fast_parameter = 7                       # Optimierter Wert -> Wert aus Trial 265 <-- Updated from Trial 434
ema_slow_parameter = 84                      # Optimierter Wert -> Wert aus Trial 265 <-- Updated from Trial 434
ema_baseline_parameter = 153                 # Optimierter Wert -> Wert aus Trial 265 <-- Updated from Trial 434
back_trend_periode = 3                       # Wert aus deinem Beispiel -> Wert aus Trial 265 <-- Updated from Trial 434
#
interval_int_2 = "5m"                        # Wert aus deinem Beispiel -> Wert aus Trial 265 <-- Updated from Trial 434
interval_int_3 = "5m"                        # Wert aus deinem Beispiel -> Wert aus Trial 265 <-- Updated from Trial 434
min_trend_strength_parameter = 4.61          # Optimierter Wert -> Wert aus Trial 265 <-- Updated from Trial 434
min_trend_duration_parameter = 10            # Optimierter Wert (als Integer) -> Wert aus Trial 265 <-- Updated from Trial 434

# RSI Parameters (Optimierte Werte eingesetzt)
rsi_period = 16                              # Optimierter Wert -> Wert aus Trial 265 <-- Updated from Trial 434
rsi_buy = 55                                 # Optimierter Wert -> Wert aus Trial 265 <-- Updated from Trial 434
rsi_sell = 29                                # Optimierter Wert -> Wert aus Trial 265 <-- Updated from Trial 434
rsi_exit_overbought = 84                     # Optimierter Wert -> Wert aus Trial 265 <-- Updated from Trial 434
rsi_exit_oversold = 20                       # Optimierter Wert -> Wert aus Trial 265 <-- Updated from Trial 434

# Volume Parameters (Werte aus deinem Beispiel, da nicht im optimierten Ergebnis)
volume_sma = 10                              # Wert aus Trial 265 <-- Updated from Trial 434
volume_multiplier = 1.99                     # Wert aus Trial 265 <-- Updated from Trial 434
volume_entry_multiplier = 0.74               # Wert aus Trial 265 <-- Updated from Trial 434
min_volume = 0.3564                          # Wert aus Trial 265 <-- Updated from Trial 434

# Momentum Parameters (Optimierte Werte eingesetzt)
use_momentum_check = False                   # Optimierter Wert -> Wert aus Trial 265 <-- Updated from Trial 434
momentum_lookback = 36                       # Optimierter Wert -> Wert aus Trial 265 <-- Updated from Trial 434
momentum = 0.6438                            # Optimierter Wert -> Wert aus Trial 265 <-- Updated from Trial 434

# Bollinger Bands Parameters (Optimierte Werte eingesetzt)
use_bb = True                                # Optimierter Wert -> Wert aus Trial 265 <-- Updated from Trial 434
bb_period = 11                               # Wert aus deinem Beispiel (Irrelevant, da use_bb=False) -> Wert aus Trial 265 (aber irrelevant) <-- Updated from Trial 434
bb_deviation = 1.73                          # Wert aus deinem Beispiel (Irrelevant, da use_bb=False) -> Wert aus Trial 265 (aber irrelevant) <-- Updated from Trial 434

# MACD Parameters (Optimierte Werte eingesetzt)
use_macd = False                             # Optimierter Wert -> Wert aus Trial 265 <-- Updated from Trial 434
macd_fast_period = 20                        # Wert aus deinem Beispiel (Irrelevant, da use_macd=False) -> Wert aus Trial 265 <-- Updated from Trial 434
macd_slow_period = 58                        # Wert aus deinem Beispiel (Irrelevant, da use_macd=False) -> Wert aus Trial 265 <-- Updated from Trial 434
macd_signal_period = 13                      # Wert aus deinem Beispiel (Irrelevant, da use_macd=False) -> Wert aus Trial 265 <-- Updated from Trial 434

# ADX Parameters (Optimierte Werte eingesetzt)
use_adx = True                               # Optimierter Wert -> Wert aus Trial 265 <-- Updated from Trial 434
adx_period = 35                              # Wert aus deinem Beispiel (Irrelevant, da use_adx=False) -> Wert aus Trial 265 <-- Updated from Trial 434
adx_threshold = 32.2                         # Wert aus deinem Beispiel (Irrelevant, da use_adx=False) -> Wert aus Trial 265 <-- Updated from Trial 434

# VWAP Parameters (Werte aus deinem Beispiel, da nicht optimiert)
use_vwap = False                             # Wert aus Trial 265 <-- Updated from Trial 434
use_advanced_vwap = False                    # Wert aus Trial 265 <-- Updated from Trial 434
advanced_vwap_periods = [38, 120, 296]       # Irrelevant, da use_advanced_vwap=False -> Wert aus Trial 265 <-- Updated from Trial 434
advanced_vwap_std_threshold = 0.89           # Irrelevant, da use_advanced_vwap=False -> Wert aus Trial 265 <-- Updated from Trial 434

# OBV Parameters (Werte aus deinem Beispiel, da nicht optimiert)
use_obv = False                              # Wert aus Trial 265 <-- Updated from Trial 434
obv_period = 11                              # Wert aus Trial 265 (aber irrelevant) <-- Updated from Trial 434

# Chaikin money flow Parameters (Werte aus deinem Beispiel, da nicht optimiert)
use_chaikin_money_flow = False               # Wert aus Trial 265 <-- Updated from Trial 434
cmf_period = 1                               # Wert aus Trial 265 (aber irrelevant) <-- Updated from Trial 434
cmf_threshold = 0.0                          # Wert aus Trial 265 (aber irrelevant) <-- Updated from Trial 434
################################################################################################################################################################################################################################################
###########   EINSTELLUNGEN ZU ENDE   ##########################################################################################################################################################################################################
################################################################################################################################################################################################################################################

#Ablage nicht verändern sondern einfach lassen.:
debug_obv = False
debug_indicators = False
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
    # Wenn use_standard_sl_tp False ist, wird Trailing Stop/TP Logik verwendet
    # Setze diese auf None oder einen Platzhalterwert, da sie nicht direkt für Binance Orders genutzt werden
    stop_loss_parameter = None # Oder z.B. 99.0 für interne Logik, falls nötig
    take_profit_parameter = None # Oder z.B. 99.0 für interne Logik, falls nötig
if interval:
    interval_int = interval
early_exit_check = True  # Wenn True: Beendet den Backtest, wenn aktuelle Systemzeit außerhalb des Handelszeitfensters liegt

your_csv_path_variable = "./strategy_results/full_data.csv"

if use_advanced_vwap == True:
    hoechste_zahl_advanced_vwap = max(advanced_vwap_periods)
else:
    # Stelle sicher, dass hoechste_zahl_advanced_vwap definiert ist, auch wenn use_advanced_vwap False ist
    hoechste_zahl_advanced_vwap = 0 # oder ein anderer sinnvoller Standardwert