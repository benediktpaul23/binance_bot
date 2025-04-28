#backtest_strategy.py. Version where it kind of did not work anymore. or lets say it showed me suddenly other solutions: "

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import logging
import Z_config
import utils.Backtest as Backtest
from collections import Counter
from utils.Backtest import append_to_csv
import math


# Globaler Tracker für offene Positionen
position_tracking = {}  # Format: {symbol: {'position_open': bool, 'observation_ended': bool}}


def verify_indicator_calculation(candle_data: pd.Series, symbol="UNKNOWN", label="Verification"):
    """
    Gibt Indikatorwerte für die übergebene Kerze (pd.Series) aus,
    wenn Z_config.debug_indicators = True ist. Speichert optional auch CSV.
    """
    # --- Prüfe Debug-Schalter ---
    if not getattr(Z_config, 'debug_indicators', False):
        return # Debugging ist deaktiviert

    # --- Validierung des Inputs ---
    if candle_data is None or not isinstance(candle_data, pd.Series) or candle_data.empty:
        # Verwende print hier, da logging evtl. noch nicht voll konfiguriert ist oder unterdrückt wird
        print(f"DEBUG ({symbol} - {label}): Keine gültigen Kerzendaten (Series) zum Verifizieren übergeben.")
        logging.warning(f"DEBUG ({symbol} - {label}): Keine gültigen Kerzendaten (Series) zum Verifizieren übergeben.")
        return

    # --- Hole Zeitstempel aus der Series ---
    timestamp = candle_data.name

    # --- Prüfe, ob Kerze vollständig ist (optional, aber informativ) ---
    is_complete = candle_data.get('is_complete', True) # Nimm an, sie ist vollständig, wenn Flag fehlt
    completeness_info = "" if is_complete else " [UNVOLLSTÄNDIG]"

    # --- Beginne die Ausgabe für DIESE Kerze ---
    # Nutze print, da der User explizit Debug-Prints sehen will
    print(f"\n--- DEBUG Indikatoren: {symbol} [{label}] @ {timestamp}{completeness_info} ---")
    max_key_len = 25 # Für Ausrichtung anpassen

    # --- Liste aller potenziell vorhandenen Indikator-Spalten ---
    # Passe diese Liste an ALLE Indikatoren an, die du berechnest und sehen willst
    indicator_columns = [
         # Basisdaten
         'open', 'high', 'low', 'close', 'volume', 'is_complete',
         # EMAs (aus calculate_trends_vectorized)
         'ema_fast', 'ema_slow', 'ema_baseline',
         # RSI (aus calculate_trends_vectorized)
         'rsi',
         # ATR (aus calculate_trends_vectorized)
         'atr',
         # Momentum (aus calculate_trends_vectorized)
         'momentum',
         # Volumen Indikatoren (aus calculate_trends_vectorized)
         'volume_sma', 'volume_multiplier', 'high_volume',
         # Trend Indikatoren (aus calculate_trends_vectorized)
         'trend', 'trend_strength', 'trend_duration',
         # Multi-Timeframe Trends (Namen anpassen, falls sie anders lauten!)
         f'trend_{Z_config.interval}', f'trend_{Z_config.interval_int_2}', f'trend_{Z_config.interval_int_3}',
         'all_trends_aligned', # Alignment Flag
         # Standard VWAP (aus calculate_trends_vectorized)
         'vwap', 'vwap_trend',
         # Bollinger Bands (aus calculate_trends_vectorized)
         'bb_upper', 'bb_middle', 'bb_lower', 'bb_signal',
         # MACD (aus calculate_trends_vectorized)
         'macd_line', 'macd_signal_line', 'macd_histogram', 'macd_signal', 'macd_crossover',
         # Erweiterte Indikatoren (aus advanced_indicators_file.py)
         'adx', 'adx_signal',
         'vwap_20', 'vwap_50', 'vwap_100', 'vwap_std', 'advanced_vwap_signal',
         'obv', 'obv_trend',
         'cmf', 'cmf_signal',
         # Strategie-Signale (können auch nützlich sein)
         'signal', 'trigger',
         # Kontext-Flags (falls vorhanden und relevant)
         'is_context', 'is_observation', 'is_post_observation'
    ]

    # Dictionary zur Definition der Formatierung für bestimmte Spalten
    # (None bedeutet Standard-String-Konvertierung, z.B. für Booleans)
    formatting_map = {
         'open': '.6f', 'high': '.6f', 'low': '.6f', 'close': '.6f',
         'volume': '_,.2f', 'volume_sma': '_,.2f', 'obv': '_,.2f',
         'ema_fast': '.6f', 'ema_slow': '.6f', 'ema_baseline': '.6f',
         'atr': '.6f', 'vwap': '.6f', 'bb_upper': '.6f', 'bb_middle': '.6f', 'bb_lower': '.6f',
         'macd_line': '.6f', 'macd_signal_line': '.6f', 'macd_histogram': '.6f', 'vwap_std': '.6f',
         'vwap_20': '.6f', 'vwap_50': '.6f', 'vwap_100': '.6f',
         'rsi': '.2f', 'adx': '.2f',
         'trend_strength': '.3f', 'volume_multiplier': '.3f',
         'momentum': '.4f', 'cmf': '.4f',
         'trend': 'd', 'trend_duration': 'd', 'vwap_trend': 'd', 'bb_signal': 'd',
         'macd_signal': 'd', 'macd_crossover': 'd', 'adx_signal': 'd',
         'advanced_vwap_signal': 'd', 'cmf_signal': 'd',
         'obv_trend': '.0f', # oft -1, 0, 1
         f'trend_{Z_config.interval}': 'd', f'trend_{Z_config.interval_int_2}': 'd', f'trend_{Z_config.interval_int_3}': 'd',
         'high_volume': None, 'all_trends_aligned': None, 'is_complete': None,
         'is_context': None, 'is_observation': None, 'is_post_observation': None,
         'signal': None, 'trigger': None # Strings
    }


    output_lines = []
    found_any = False
    for col in indicator_columns:
        if col in candle_data.index:
            found_any = True
            value = candle_data.get(col)
            fmt = formatting_map.get(col) # Hole Formatstring aus Map

            if pd.isna(value):
                formatted_value = "NaN"
            elif fmt:
                try:
                    if 'd' in fmt: # Integer
                        formatted_value = f"{int(value):{fmt.replace('d','')}}"
                    elif '_' in fmt: # Float/Int mit Tausendertrennzeichen
                         # Prüfe, ob es float oder int ist
                         if isinstance(value, (int, np.integer)):
                             formatted_value = f"{value:{fmt}}" # Direkt formatieren
                         else:
                             formatted_value = f"{float(value):{fmt}}".replace('_',' ') # Float-Konvertierung und Ersetzung
                    else: # Standard Float
                        formatted_value = f"{float(value):{fmt}}"
                except (ValueError, TypeError):
                    formatted_value = str(value) # Fallback
            else: # Kein Format angegeben (z.B. bool, string)
                formatted_value = str(value)

            output_lines.append(f"  {col:<{max_key_len}} : {formatted_value}")

    if not found_any:
        print(f"  Keine der gesuchten Indikator-Spalten in Kerze @ {timestamp} gefunden.")
    else:
        for line in output_lines:
            print(line) # Direkte Ausgabe auf Konsole

    # --- CSV Speicherung (SEHR RESSOURCENINTENSIV bei jeder Kerze!) ---
    save_csv_per_candle = False # <<< Standardmäßig AUS! Nur für gezieltes Debuggen aktivieren >>>
    if save_csv_per_candle:
         try:
             # Wähle Spalten für die CSV-Datei
             csv_columns = [
                  'ema_fast', 'ema_slow', 'ema_baseline', 'trend', 'trend_strength', 'trend_duration', 'rsi',
                  'volume_sma', 'volume_multiplier', 'high_volume', 'adx', 'adx_signal', 'vwap',
                  'vwap_trend', 'obv', 'obv_trend', 'cmf', 'cmf_signal', 'close' # Beispielspalten
             ]
             verification_data_dict = {'timestamp': str(timestamp), 'label': label, 'symbol': symbol}
             for col in csv_columns:
                   val = candle_data.get(col, np.nan)
                   # Standardwerte für NaN basierend auf vermutetem Typ (vereinfacht)
                   default_val = 0.0
                   if col in ['trend', 'trend_duration', 'adx_signal', 'vwap_trend', 'obv_trend', 'cmf_signal']: default_val = 0
                   elif col == 'high_volume': default_val = False
                   verification_data_dict[col] = val if pd.notna(val) else default_val

             df_to_save = pd.DataFrame([verification_data_dict])
             csv_file = 'indicator_verification_per_candle.csv' # Anderer Dateiname!
             df_to_save.to_csv(csv_file, mode='a', header=not os.path.exists(csv_file), index=False)
         except Exception as e:
             print(f"FEHLER beim Speichern der Kerzen-Verifikations-CSV für {symbol} @ {timestamp}: {e}")
             logging.error(f"FEHLER beim Speichern der Kerzen-Verifikations-CSV für {symbol} @ {timestamp}: {e}", exc_info=True)


from utils.symbol_filter import update_symbol_position_status


logger_apply_strat = logging.getLogger(__name__) # Eigener Logger

def apply_base_strategy(
    data: pd.DataFrame,
    symbol: str,
    # --- Parameter, die optimiert werden KÖNNEN (Defaults auf None) ---
    # Basis-Parameter
    min_trend_strength: float = None,
    rsi_buy: float = None,
    rsi_sell: float = None,
    rsi_exit_overbought: float = None,
    rsi_exit_oversold: float = None,
    min_trend_duration_parameter: int = None,
    # Indikator-Perioden etc. (Beispiele - passe sie an!)
    rsi_period: int = None,
    ema_fast_parameter: int = None,
    ema_slow_parameter: int = None,
    ema_baseline_parameter: int = None,
    volume_sma: int = None,                 # <<< Beispiel uncommentiert
    momentum_lookback: int = None,          # <<< Beispiel uncommentiert
    bb_period: int = None,                  # <<< Beispiel uncommentiert
    bb_deviation: float = None,             # <<< Beispiel uncommentiert
    macd_fast_period: int = None,           # <<< Beispiel uncommentiert
    macd_slow_period: int = None,           # <<< Beispiel uncommentiert
    macd_signal_period: int = None,         # <<< Beispiel uncommentiert
    adx_period: int = None,                 # <<< Beispiel uncommentiert
    cmf_period: int = None,                 # <<< Beispiel uncommentiert
    # Füge hier ALLE weiteren Parameter hinzu, die Optuna vorschlägt!

    # --- Feste Parameter (können weiterhin Defaults haben) ---
    allow_long: bool = None,
    allow_short: bool = None,
    use_ut_bot: bool = False
) -> pd.DataFrame:
    """
    MODIFIED: Generiert Signale basierend auf Indikatoren.
    Akzeptiert jetzt ALLE optimierbaren Parameter direkt als Keyword-Argumente.
    Verwendet ITERATIVE EMA/OBV-Werte (_iter Spalten).
    """
    global logger_apply_strat
    if logger_apply_strat is None:
        logger_apply_strat = logging.getLogger(__name__)
        if not logger_apply_strat.hasHandlers(): logging.basicConfig(level=logging.INFO)

    logger_apply_strat.info(f"Generating signals for {symbol} (using iterative EMAs/OBV).")

    # --- Hole Parameter: Priorisiere Argumente, sonst Fallback auf Z_config ---
    cfg_prefix = Z_config if hasattr(Z_config, '__dict__') else None

    # Lade ALLE Parameter aus der Signatur oder Z_config
    min_trend_strength_actual = min_trend_strength if min_trend_strength is not None else getattr(cfg_prefix, 'min_trend_strength_parameter', 1.0)
    rsi_buy_actual = rsi_buy if rsi_buy is not None else getattr(cfg_prefix, 'rsi_buy', 70)
    rsi_sell_actual = rsi_sell if rsi_sell is not None else getattr(cfg_prefix, 'rsi_sell', 30)
    rsi_exit_ob_actual = rsi_exit_overbought if rsi_exit_overbought is not None else getattr(cfg_prefix, 'rsi_exit_overbought', 80)
    rsi_exit_os_actual = rsi_exit_oversold if rsi_exit_oversold is not None else getattr(cfg_prefix, 'rsi_exit_oversold', 20)
    min_trend_duration_actual = min_trend_duration_parameter if min_trend_duration_parameter is not None else getattr(cfg_prefix, 'min_trend_duration_parameter', 3)

    # Periode-Parameter (werden extern für Indikatorberechnung genutzt)
    rsi_p_actual = rsi_period if rsi_period is not None else getattr(cfg_prefix, 'rsi_period', 8)
    ema_f_p_actual = ema_fast_parameter if ema_fast_parameter is not None else getattr(cfg_prefix, 'ema_fast_parameter', 7)
    ema_s_p_actual = ema_slow_parameter if ema_slow_parameter is not None else getattr(cfg_prefix, 'ema_slow_parameter', 60)
    ema_b_p_actual = ema_baseline_parameter if ema_baseline_parameter is not None else getattr(cfg_prefix, 'ema_baseline_parameter', 50)
    vol_sma_actual = volume_sma if volume_sma is not None else getattr(cfg_prefix, 'volume_sma', 30)
    mom_lookback_actual = momentum_lookback if momentum_lookback is not None else getattr(cfg_prefix, 'momentum_lookback', 15)
    bb_p_actual = bb_period if bb_period is not None else getattr(cfg_prefix, 'bb_period', 21)
    bb_dev_actual = bb_deviation if bb_deviation is not None else getattr(cfg_prefix, 'bb_deviation', 2.0)
    macd_f_actual = macd_fast_period if macd_fast_period is not None else getattr(cfg_prefix, 'macd_fast_period', 12)
    macd_s_actual = macd_slow_period if macd_slow_period is not None else getattr(cfg_prefix, 'macd_slow_period', 26)
    macd_sig_actual = macd_signal_period if macd_signal_period is not None else getattr(cfg_prefix, 'macd_signal_period', 9)
    adx_p_actual = adx_period if adx_period is not None else getattr(cfg_prefix, 'adx_period', 21)
    cmf_p_actual = cmf_period if cmf_period is not None else getattr(cfg_prefix, 'cmf_period', 20)
    # ... lade hier ALLE anderen Parameter mit dieser Logik ...

    allow_long_actual = allow_long if allow_long is not None else getattr(cfg_prefix, 'allow_long', True)
    allow_short_actual = allow_short if allow_short is not None else getattr(cfg_prefix, 'allow_short', True)

    # Lade Konfiguration für optionale Indikatoren direkt aus Z_config
    # (Diese werden normalerweise nicht optimiert, daher kein Argument-Check nötig)
    require_timeframe_alignment = getattr(cfg_prefix, 'require_timeframe_alignment', False)
    use_momentum_check = getattr(cfg_prefix, 'use_momentum_check', False)
    momentum_threshold = getattr(cfg_prefix, 'momentum_threshold', 0.001)
    use_bb = getattr(cfg_prefix, 'use_bb', False)
    use_macd = getattr(cfg_prefix, 'use_macd', False)
    use_adx = getattr(cfg_prefix, 'use_adx', False)
    use_vwap = getattr(cfg_prefix, 'use_vwap', False)
    use_advanced_vwap = getattr(cfg_prefix, 'use_advanced_vwap', False)
    use_obv = getattr(cfg_prefix, 'use_obv', False)
    use_cmf = getattr(cfg_prefix, 'use_chaikin_money_flow', False)

    # Logge die tatsächlich verwendeten Parameter
    logger_apply_strat.debug(f"[{symbol}] Using parameters: min_strength={min_trend_strength_actual:.2f}, "
                            f"rsi_levels=({rsi_sell_actual}/{rsi_buy_actual} exits:{rsi_exit_os_actual}/{rsi_exit_ob_actual}), "
                            f"min_duration={min_trend_duration_actual} | "
                            f"(Info: RSI Per: {rsi_p_actual}, EMA Pers: {ema_f_p_actual}/{ema_s_p_actual}/{ema_b_p_actual}, "
                            f"Vol SMA: {vol_sma_actual}, Mom LB: {mom_lookback_actual}, BB: {bb_p_actual}/{bb_dev_actual}, "
                            f"MACD: {macd_f_actual}/{macd_s_actual}/{macd_sig_actual}, ADX: {adx_p_actual}, CMF: {cmf_p_actual})")


    # --- Input DataFrame Validierung ---
    if data is None or data.empty:
        logger_apply_strat.error(f"Received empty or invalid data for {symbol}.")
        return pd.DataFrame(columns=['signal', 'trigger', 'symbol'])

    # --- Vorbereitung ---
    data = data.copy() # Mit Kopie arbeiten
    data['signal'] = "no_signal"
    data['trigger'] = None
    if 'symbol' not in data.columns: data['symbol'] = symbol
    if 'is_warmup' not in data.columns:
        logger_apply_strat.warning(f"[{symbol}] 'is_warmup' column missing. Assuming no warmup rows.")
        data['is_warmup'] = False
    data['is_warmup'] = data['is_warmup'].astype(bool)

    # --- Definiere benötigte Spalten ---
    required_cols = ['close', 'is_warmup',
                     'ema_fast_iter', 'ema_slow_iter', 'ema_baseline_iter',
                     'rsi', 'high_volume', 'trend_strength', 'trend_duration'] # Basis-Indikatoren

    # Füge Spalten hinzu, die von aktivierten optionalen Checks benötigt werden
    if use_obv: required_cols.append('obv_trend_iter')
    if use_momentum_check: required_cols.append('momentum')
    if use_bb: required_cols.append('bb_signal') # bb_signal wird extern berechnet
    if use_macd: required_cols.extend(['macd_signal', 'macd_crossover']) # Diese werden extern berechnet
    if use_adx: required_cols.append('adx_signal') # Extern berechnet
    if use_vwap: required_cols.append('vwap_trend') # Extern berechnet
    if use_advanced_vwap: required_cols.append('advanced_vwap_signal') # Extern berechnet
    if use_cmf: required_cols.append('cmf_signal') # Extern berechnet
    if require_timeframe_alignment: required_cols.append('all_trends_aligned')

    required_cols = sorted(list(set(required_cols)))

    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        logger_apply_strat.error(f"[{symbol}] Missing required columns: {missing_cols}")
        logger_apply_strat.error(f"Available columns: {data.columns.tolist()}")
        data['signal'] = 'no_signal'; data['trigger'] = None
        return data[['signal', 'trigger', 'symbol']]

    # --- Finde ersten gültigen Index ---
    non_warmup_data_indices = data.index[~data['is_warmup']]
    if non_warmup_data_indices.empty:
         logger_apply_strat.warning(f"[{symbol}] No non-warmup candles found.")
         return data

    check_cols_for_nan = [c for c in required_cols if c not in ['is_warmup', 'symbol']]
    # Temporäre Kopie nur für dropna
    temp_non_warmup_df = data.loc[non_warmup_data_indices, check_cols_for_nan].dropna()
    if temp_non_warmup_df.empty:
         logger_apply_strat.warning(f"[{symbol}] No rows with valid indicators found in non-warmup data.")
         return data
    first_valid_index = temp_non_warmup_df.index[0]
    del temp_non_warmup_df # Speicher freigeben

    logger_apply_strat.info(f"Generating signals loop for {symbol} starting from {first_valid_index}...")

    # --- Signal-Generierungs-Schleife ---
    valid_indices_to_process = data.loc[first_valid_index:].index

    for current_idx in valid_indices_to_process:
        if data.loc[current_idx, 'is_warmup']: continue

        current_candle = data.loc[current_idx]

        # --- Lese Indikatoren ---
        ema_f_val = current_candle.get('ema_fast_iter', np.nan)
        ema_s_val = current_candle.get('ema_slow_iter', np.nan)
        ema_b_val = current_candle.get('ema_baseline_iter', np.nan)
        close_val = current_candle.get('close', np.nan)
        obv_trend_iter_val = int(current_candle.get('obv_trend_iter', 0)) if use_obv else 0
        current_trend_strength = current_candle.get('trend_strength', 0.0)
        current_trend_duration = int(current_candle.get('trend_duration', 0))
        current_rsi = current_candle.get('rsi', 50.0)
        current_high_volume = current_candle.get('high_volume', False)
        current_momentum = current_candle.get('momentum', 0) if use_momentum_check else 0
        bb_signal = int(current_candle.get('bb_signal', 0)) if use_bb else 0
        macd_signal_val = int(current_candle.get('macd_signal', 0)) if use_macd else 0
        macd_crossover = int(current_candle.get('macd_crossover', 0)) if use_macd else 0
        adx_signal_on = bool(current_candle.get('adx_signal', 0)) if use_adx else True
        vwap_trend = int(current_candle.get('vwap_trend', 0)) if use_vwap else 0
        adv_vwap_sig = bool(current_candle.get('advanced_vwap_signal', 0)) if use_advanced_vwap else True
        cmf_signal = int(current_candle.get('cmf_signal', 0)) if use_cmf else 0
        all_trends_aligned = current_candle.get('all_trends_aligned', not require_timeframe_alignment)

        # --- Prüfe Strategie Conditions ---
        current_trend_iter = 0
        if not pd.isna([ema_f_val, ema_s_val, ema_b_val, close_val]).any():
            if ema_f_val > ema_s_val and close_val > ema_b_val: current_trend_iter = 1
            elif ema_f_val < ema_s_val and close_val < ema_b_val: current_trend_iter = -1

        # Verwende _actual Variablen für Schwellwerte etc.
        baseLongOk = (current_trend_iter == 1 and
                     current_trend_strength >= min_trend_strength_actual and
                     current_trend_duration >= min_trend_duration_actual and
                     current_rsi < rsi_buy_actual and
                     current_high_volume and
                     all_trends_aligned)

        baseShortOk = (current_trend_iter == -1 and
                      current_trend_strength >= min_trend_strength_actual and
                      current_trend_duration >= min_trend_duration_actual and
                      current_rsi > rsi_sell_actual and
                      current_high_volume and
                      all_trends_aligned)

        longCandCond = baseLongOk
        shortCandCond = baseShortOk

        # Optionale Indikatoren Checks
        if use_momentum_check:
            longCandCond &= (current_momentum >= momentum_threshold)
            shortCandCond &= (current_momentum <= -momentum_threshold)
        if use_bb:
             longCandCond &= (bb_signal >= 0)
             shortCandCond &= (bb_signal <= 0)
        if use_macd:
             longCandCond &= (macd_signal_val > 0 or macd_crossover > 0)
             shortCandCond &= (macd_signal_val < 0 or macd_crossover < 0)
        if use_adx:
             longCandCond &= adx_signal_on
             shortCandCond &= adx_signal_on
        if use_vwap:
             longCandCond &= (vwap_trend >= 0)
             shortCandCond &= (vwap_trend <= 0)
        if use_advanced_vwap:
             longCandCond &= adv_vwap_sig
             shortCandCond &= adv_vwap_sig # Logik prüfen
        if use_obv:
            longCandCond &= (obv_trend_iter_val >= 0)
            shortCandCond &= (obv_trend_iter_val <= 0)
        if use_cmf:
            longCandCond &= (cmf_signal >= 0)
            shortCandCond &= (cmf_signal <= 0)

        # --- Exit Conditions ---
        longStrategyExit = current_trend_iter == -1 or current_rsi > rsi_exit_ob_actual
        shortStrategyExit = current_trend_iter == 1 or current_rsi < rsi_exit_os_actual

        # --- Signal setzen ---
        final_signal = "no_signal"; final_trigger = None
        if allow_long_actual and longCandCond:
            final_signal = "buy"; final_trigger = "entry_long_signal"
        elif allow_short_actual and shortCandCond:
            final_signal = "sell"; final_trigger = "entry_short_signal"
        elif longStrategyExit:
            final_signal = "exit_long"; final_trigger = "strategy_exit"
        elif shortStrategyExit:
            final_signal = "exit_short"; final_trigger = "strategy_exit"

        data.loc[current_idx, 'signal'] = final_signal
        data.loc[current_idx, 'trigger'] = final_trigger

    # --- Ende der Schleife ---
    non_warmup_signals = data.loc[valid_indices_to_process, 'signal'].value_counts()
    logger_apply_strat.info(f"Signal generation finished for {symbol}. Signal counts (non-warmup, valid indicators):\n{non_warmup_signals}")

    return data


def validate_trade_timestamps(trade):
    """Stellt sicher, dass die Exit-Zeit nicht vor der Entry-Zeit liegt."""
    if 'exit_time' in trade and 'entry_time' in trade and trade['exit_time'] < trade['entry_time']:
        original_exit_time = trade['exit_time']
        original_entry_time = trade['entry_time']
        logging.warning(f"Detected exit time {original_exit_time} before entry time {original_entry_time} for {trade.get('symbol', 'N/A')} (Trigger: {trade.get('trigger', 'N/A')}) - Adjusting.")
        trade['exit_time'] = trade['entry_time'] + timedelta(minutes=1)
        logging.info(f"Adjusted exit time to {trade['exit_time']}")
    return trade


from utils.backtest_position_tracker import BacktestPositionTracker
from utils.backtest_position_tracker import _calculate_pnl

logger_get_multi = logging.getLogger(__name__) # Eigener Logger für diese Funktion


def get_multi_timeframe_data(symbol, end_time, initial_start_time, position_tracking=None):
    """
    MODIFIED VERSION 2: Fetches raw data ensuring enough history for BOTH
    indicator warm-up AND the configured lookback_hours_parameter.
    Marks warm-up candles. Alignment trends are still optional.
    Indicator calculation happens LATER. Correctly calculates fetch window and limit.

    Args:
        symbol (str): Trading symbol.
        end_time (datetime): Logical end time for the data needed (TZ-aware UTC).
        initial_start_time (datetime): Logical start time for the BACKTEST period (TZ-aware UTC).
                                       Wird jetzt hauptsächlich zur Orientierung/Logging verwendet.
        position_tracking (dict, optional): Passed state of open positions.

    Returns:
        pd.DataFrame or None: DataFrame with OHLCV, is_complete, is_warmup, symbol,
                              and potentially alignment trends, or None on failure.
    """
    log_prefix = f"get_multi (Backtest - {symbol})"
    logger_get_multi.debug(f"{log_prefix}: Starting function call.")
    try:
        # 1. Zeitstempel validieren
        if not all(isinstance(t, datetime) and t.tzinfo is not None for t in [end_time, initial_start_time]):
             logger_get_multi.error(f"{log_prefix}: Provided start/end times must be timezone-aware datetimes.")
             if isinstance(end_time, datetime) and end_time.tzinfo is None: end_time = end_time.replace(tzinfo=timezone.utc)
             if isinstance(initial_start_time, datetime) and initial_start_time.tzinfo is None: initial_start_time = initial_start_time.replace(tzinfo=timezone.utc)
             if not all(isinstance(t, datetime) and t.tzinfo is not None for t in [end_time, initial_start_time]):
                 logger_get_multi.error(f"{log_prefix}: Could not ensure timezone-aware datetimes. Returning None.")
                 return None
             logger_get_multi.warning(f"{log_prefix}: Corrected naive start/end times to UTC.")

        end_time_utc = end_time.astimezone(timezone.utc)
        initial_start_time_utc_ref = initial_start_time.astimezone(timezone.utc) # Referenzpunkt
        logger_get_multi.debug(f"{log_prefix}: Logical Backtest Time Range Requested: {initial_start_time_utc_ref} -> {end_time_utc}")

        # 2. Benötigte GESAMT-Historie berechnen
        required_warmup_candles = calculate_required_candles()
        interval_str = getattr(Z_config, 'interval', '5m')
        interval_minutes = Backtest.parse_interval_to_minutes(interval_str)
        if interval_minutes is None or interval_minutes <= 0:
             logger_get_multi.error(f"{log_prefix}: Invalid base interval '{interval_str}' parsed to {interval_minutes} minutes.")
             return None

        warmup_duration = timedelta(minutes=required_warmup_candles * interval_minutes)
        logger_get_multi.debug(f"{log_prefix}: Required indicator WARMUP duration: {warmup_duration} ({required_warmup_candles} candles * {interval_minutes} min)")

        # Hole lookback_hours_parameter aus Z_config
        lookback_hours = getattr(Z_config, 'lookback_hours_parameter', 24) # Default 24h

        # --- Sanity Check und Debugging für lookback_hours ---
        logger_get_multi.info(f"{log_prefix}: Read Z_config.lookback_hours_parameter = {lookback_hours} (Type: {type(lookback_hours)})")
        if not isinstance(lookback_hours, (int, float)) or lookback_hours <= 0:
            logger_get_multi.error(f"{log_prefix}: Invalid value read for lookback_hours_parameter: {lookback_hours}. Using fallback 24 hours.")
            lookback_hours = 24 # Fallback auf einen sinnvollen Wert
        # --- Ende Sanity Check ---

        backtest_duration = timedelta(hours=lookback_hours)
        logger_get_multi.debug(f"{log_prefix}: Requested BACKTEST duration: {backtest_duration} ({lookback_hours} hours)") # Wird jetzt korrekt geloggt

        total_fetch_duration = backtest_duration + warmup_duration
        logger_get_multi.debug(f"{log_prefix}: TOTAL fetch duration needed: {total_fetch_duration}")

        # 3. Fetch-Fenster bestimmen
        fetch_end = end_time_utc

        close_position_setting = getattr(Z_config, 'close_position', False)
        if not close_position_setting and position_tracking and symbol in position_tracking and position_tracking[symbol].get('position_open'):
            pos_data = position_tracking[symbol].get('position_data', {})
            entry_time_data = pos_data.get('entry_time')
            if entry_time_data and isinstance(entry_time_data, datetime):
                buffer_hours_ext = getattr(Z_config, 'fetch_extension_hours', 24)
                potential_extended_end = end_time_utc + timedelta(hours=buffer_hours_ext)
                if potential_extended_end > fetch_end:
                    logger_get_multi.info(f"{log_prefix}: Extending fetch end to {potential_extended_end} (close_position=False).")
                    fetch_end = potential_extended_end
            else:
                 logger_get_multi.warning(f"{log_prefix}: Open pos detected for {symbol}, but entry_time invalid/missing. Cannot extend fetch end accurately.")

        fetch_start_raw = fetch_end - total_fetch_duration
        logger_get_multi.debug(f"{log_prefix}: Raw fetch_start calculated based on total duration: {fetch_start_raw}")

        fetch_start = None
        if interval_minutes > 0:
            if fetch_start_raw.tzinfo is not None: fetch_start_naive = fetch_start_raw.replace(tzinfo=None)
            else: fetch_start_naive = fetch_start_raw
            minutes_since_midnight = fetch_start_naive.hour * 60 + fetch_start_naive.minute
            minutes_into_interval = minutes_since_midnight % interval_minutes
            fetch_start_aligned_naive = fetch_start_naive - timedelta(minutes=minutes_into_interval)
            fetch_start_aligned_naive = fetch_start_aligned_naive.replace(second=0, microsecond=0)
            fetch_start = fetch_start_aligned_naive.replace(tzinfo=timezone.utc)
            if fetch_start > fetch_start_raw:
                 logger_get_multi.warning(f"{log_prefix}: fetch_start alignment resulted in later time? {fetch_start} > {fetch_start_raw}. Reverting to raw.")
                 fetch_start = fetch_start_raw.replace(second=0, microsecond=0)
            logger_get_multi.debug(f"{log_prefix}: Aligned fetch_start (rounded down): {fetch_start}")
        else:
            logger_get_multi.warning(f"{log_prefix}: Invalid interval_minutes ({interval_minutes}), cannot align fetch_start.")
            fetch_start = fetch_start_raw.replace(second=0, microsecond=0)

        logger_get_multi.info(f"{log_prefix}: Final Fetch Window for API call: {fetch_start} -> {fetch_end}")

        # 4. Basisdaten holen
        # API Limit Berechnung basiert jetzt auf korrekter backtest_duration
        num_backtest_candles = math.ceil(backtest_duration.total_seconds() / (interval_minutes * 60)) if interval_minutes > 0 else 0
        total_candles_expected = required_warmup_candles + num_backtest_candles
        api_limit_parameter_for_fetch = total_candles_expected + getattr(Z_config, 'api_limit_buffer', 10) # Wie viele Kerzen WIR brauchen
        max_api_limit_per_call = 1499 # Was die API pro Call maximal gibt
        # WICHTIG: limit_force an fetch_data übergeben wir NICHT mehr direkt,
        # da fetch_data jetzt selbst batcht. Wir übergeben nur start und end.
        logger_get_multi.debug(f"{log_prefix}: Total candles needed (estimate): {total_candles_expected} (Warmup: {required_warmup_candles}, Backtest: {num_backtest_candles})")

        # Fetch-Aufruf (VERWENDET DIE NEUE fetch_data FUNKTION in Backtest.py)
        try:
            df_base = Backtest.fetch_data(
                symbol=symbol,
                interval=interval_str,
                end_time=fetch_end,           # Das Ende des Fensters
                start_time_force=fetch_start  # Der berechnete Start des Fensters
                # Kein limit_force mehr hier! fetch_data kümmert sich drum.
            )
        except AttributeError:
             logger_get_multi.critical(f"{log_prefix}: Backtest.fetch_data function not found or import failed!")
             return None
        except Exception as fetch_err:
             logger_get_multi.error(f"{log_prefix}: Error during Backtest.fetch_data call: {fetch_err}", exc_info=True)
             return None

        if df_base is None or df_base.empty:
            logger_get_multi.warning(f"{log_prefix}: No base data ({interval_str}) returned from fetch_data call.")
            return None

        # --- Mark Warm-up Candles ---
        first_backtest_candle_time = fetch_start + warmup_duration
        logger_get_multi.debug(f"{log_prefix}: First non-warmup candle expected around: {first_backtest_candle_time}")
        # Stelle sicher, dass die Spalte existiert
        if 'is_warmup' not in df_base.columns:
             df_base['is_warmup'] = False
        df_base.loc[df_base.index < first_backtest_candle_time, 'is_warmup'] = True
        warmup_count = df_base['is_warmup'].sum()
        non_warmup_count = len(df_base) - warmup_count
        logger_get_multi.info(f"{log_prefix}: Marked {warmup_count} candles as WARMUP (before {first_backtest_candle_time}). {non_warmup_count} candles remain for backtest period.")

        # --- Mark 'is_complete' --- (Bleibt wie bisher)
        if 'is_complete' not in df_base.columns:
            df_base['is_complete'] = True # Default

        current_utc_time_for_flag = datetime.now(timezone.utc)
        if not df_base.empty and fetch_end >= current_utc_time_for_flag - timedelta(minutes=interval_minutes):
             try:
                  last_idx = df_base.index[-1]
                  if last_idx >= fetch_end - timedelta(minutes=interval_minutes):
                       df_base.loc[last_idx, 'is_complete'] = False
                       logger_get_multi.info(f"Marked last fetched candle at {last_idx} as potentially incomplete (ends at or after {fetch_end - timedelta(minutes=interval_minutes)}).")
                  else:
                       logger_get_multi.debug(f"Last candle {last_idx} ends before {fetch_end - timedelta(minutes=interval_minutes)}, not marked incomplete.")
             except Exception as e_flag:
                  logger_get_multi.error(f"Error marking last candle incomplete: {e_flag}")

        # 5. HÖHERE TIMEFRAMES & ALIGNMENT (Optional)
        df_merged = df_base.copy()
        require_alignment = getattr(Z_config, 'require_timeframe_alignment', False)
        tf2_interval = getattr(Z_config, 'interval_int_2', None)
        tf3_interval = getattr(Z_config, 'interval_int_3', None)
        tf2_trend_col = f'trend_{tf2_interval}' if tf2_interval else None
        tf3_trend_col = f'trend_{tf3_interval}' if tf3_interval else None

        if tf2_trend_col: df_merged[tf2_trend_col] = 0
        if tf3_trend_col: df_merged[tf3_trend_col] = 0
        df_merged['all_trends_aligned'] = True

        if require_alignment:
             logger_get_multi.debug(f"{log_prefix}: Fetching/Calculating alignment trends...")
             # --- Fetching für TF2/TF3 (verwende den gleichen Start/End wie Basisdaten) ---
             # Die neue fetch_data Funktion holt automatisch genug Daten, auch für die längeren Intervalle
             # innerhalb des gleichen Zeitfensters fetch_start -> fetch_end.

             # TF2
             if tf2_interval:
                  logger_get_multi.debug(f"{log_prefix}: Fetching data for TF2 ({tf2_interval})...")
                  df_tf2_raw = Backtest.fetch_data(symbol=symbol, interval=tf2_interval, end_time=fetch_end, start_time_force=fetch_start)
                  if df_tf2_raw is not None and not df_tf2_raw.empty:
                      try:
                          df_tf2_indicators = indicators.calculate_indicators(df_tf2_raw, calculate_for_alignment_only=True)
                          if df_tf2_indicators is not None and 'trend' in df_tf2_indicators:
                               resampled_trend_tf2 = df_tf2_indicators['trend'].resample(interval_str).ffill()
                               df_merged = df_merged.join(resampled_trend_tf2.rename(tf2_trend_col), how='left')
                               df_merged[tf2_trend_col] = df_merged[tf2_trend_col].ffill().fillna(0).astype(int)
                               logger_get_multi.debug(f"{log_prefix}: TF2 trend merged.")
                          else: logger_get_multi.warning(f"{log_prefix}: Alignment trend calculation failed for {tf2_interval}.")
                      except AttributeError: logger_get_multi.error(f"{log_prefix}: indicators.calculate_indicators not found/imported.")
                      except Exception as indi_err: logger_get_multi.error(f"{log_prefix}: Error calculating TF2 indicators: {indi_err}")
                  else: logger_get_multi.warning(f"{log_prefix}: No data fetched for {tf2_interval}.")

             # TF3
             if tf3_interval:
                  logger_get_multi.debug(f"{log_prefix}: Fetching data for TF3 ({tf3_interval})...")
                  df_tf3_raw = Backtest.fetch_data(symbol=symbol, interval=tf3_interval, end_time=fetch_end, start_time_force=fetch_start)
                  if df_tf3_raw is not None and not df_tf3_raw.empty:
                       try:
                          df_tf3_indicators = indicators.calculate_indicators(df_tf3_raw, calculate_for_alignment_only=True)
                          if df_tf3_indicators is not None and 'trend' in df_tf3_indicators.columns:
                              resampled_trend_tf3 = df_tf3_indicators['trend'].resample(interval_str).ffill()
                              df_merged = df_merged.join(resampled_trend_tf3.rename(tf3_trend_col), how='left')
                              df_merged[tf3_trend_col] = df_merged[tf3_trend_col].ffill().fillna(0).astype(int)
                              logger_get_multi.debug(f"{log_prefix}: TF3 trend merged.")
                          else: logger_get_multi.warning(f"{log_prefix}: Alignment trend calculation failed for {tf3_interval}.")
                       except AttributeError: logger_get_multi.error(f"{log_prefix}: indicators.calculate_indicators not found/imported.")
                       except Exception as indi_err: logger_get_multi.error(f"{log_prefix}: Error calculating TF3 indicators: {indi_err}")
                  else: logger_get_multi.warning(f"{log_prefix}: No data fetched for {tf3_interval}.")

             # Alignment Flag Calculation
             if 'trend' not in df_merged.columns:
                 logger_get_multi.error(f"{log_prefix}: Base 'trend' column missing. Cannot calculate alignment.")
                 df_merged['all_trends_aligned'] = False
             elif tf2_trend_col and tf3_trend_col and all(c in df_merged for c in ['trend', tf2_trend_col, tf3_trend_col]):
                  try:
                      df_merged['trend'] = pd.to_numeric(df_merged['trend'], errors='coerce').fillna(0).astype(int)
                      df_merged[tf2_trend_col] = pd.to_numeric(df_merged[tf2_trend_col], errors='coerce').fillna(0).astype(int)
                      df_merged[tf3_trend_col] = pd.to_numeric(df_merged[tf3_trend_col], errors='coerce').fillna(0).astype(int)
                      trend_cols_align = ['trend', tf2_trend_col, tf3_trend_col]
                      all_equal = df_merged[trend_cols_align].apply(lambda x: x.nunique(dropna=False) == 1, axis=1)
                      trend_not_zero = df_merged['trend'] != 0
                      df_merged['all_trends_aligned'] = (all_equal & trend_not_zero)
                      logger_get_multi.debug(f"{log_prefix}: Alignment flag calculated.")
                  except Exception as align_calc_err:
                       logger_get_multi.error(f"{log_prefix}: Error calculating alignment flag: {align_calc_err}")
                       df_merged['all_trends_aligned'] = False
             elif tf2_trend_col and tf2_trend_col in df_merged and tf3_interval is None: # Nur TF1 und TF2 prüfen
                 try:
                     df_merged['trend'] = pd.to_numeric(df_merged['trend'], errors='coerce').fillna(0).astype(int)
                     df_merged[tf2_trend_col] = pd.to_numeric(df_merged[tf2_trend_col], errors='coerce').fillna(0).astype(int)
                     trend_cols_align = ['trend', tf2_trend_col]
                     all_equal = df_merged[trend_cols_align].apply(lambda x: x.nunique(dropna=False) == 1, axis=1)
                     trend_not_zero = df_merged['trend'] != 0
                     df_merged['all_trends_aligned'] = (all_equal & trend_not_zero)
                     logger_get_multi.debug(f"{log_prefix}: Alignment flag calculated (TF1 & TF2).")
                 except Exception as align_calc_err:
                      logger_get_multi.error(f"{log_prefix}: Error calculating TF1/TF2 alignment flag: {align_calc_err}")
                      df_merged['all_trends_aligned'] = False
             else: # TF2 oder TF3 nicht konfiguriert oder Spalte fehlt
                 logger_get_multi.warning(f"{log_prefix}: Cannot calculate full alignment flag (TF2/TF3 not fully available). Setting to False.")
                 df_merged['all_trends_aligned'] = False

        # 6. Rückgabe des DataFrames
        logger_get_multi.info(f"{log_prefix}: Successfully prepared RAW data including WARMUP ({warmup_count}) and backtest ({non_warmup_count}) candles. Final shape: {df_merged.shape}.")

        # Gib alle relevanten Spalten zurück (inkl. is_warmup)
        final_cols = ['open', 'high', 'low', 'close', 'volume', 'is_complete', 'is_warmup', 'symbol']
        if tf2_trend_col and tf2_trend_col in df_merged: final_cols.append(tf2_trend_col)
        if tf3_trend_col and tf3_trend_col in df_merged: final_cols.append(tf3_trend_col)
        if 'all_trends_aligned' in df_merged: final_cols.append('all_trends_aligned')

        # Wähle nur existierende Spalten aus df_merged aus
        cols_to_return = [col for col in final_cols if col in df_merged.columns]
        if df_merged.empty:
             logger_get_multi.warning(f"{log_prefix}: df_merged became empty before final column selection.")
             return None
        # Stelle sicher, dass die wichtigsten Spalten nicht nur NaNs enthalten (außer ggf. am Anfang)
        ohlcv_check = df_merged[['open', 'high', 'low', 'close', 'volume']].iloc[max(0, warmup_count):] # Prüfe nach Warmup
        if ohlcv_check.isnull().all().any():
            logger_get_multi.error(f"{log_prefix}: Critical NaNs detected in OHLCV data after warmup period. Returning None.")
            return None

        return df_merged[cols_to_return]

    except Exception as e:
        logger_get_multi.error(f"--- FATAL ERROR in get_multi_timeframe_data ({symbol}): {e} ---", exc_info=True)
        return None



# ------------------------------------------------------------------------------

# ==============================================================================
# Function: process_symbol
# Accepts position_tracking dictionary as argument
# ==============================================================================
from utils import Backtest
logger_process_symbol = logging.getLogger(__name__) # Eigener Logger

def process_symbol(symbol, df, symbol_data, start_time, position_tracking):
    """
    MODIFIED: Verarbeitet Signale und generiert Trades, ignoriert aber Warm-up-Kerzen
    für die eigentliche Trade-Logik (Entry/Exit).
    Nutzt das übergebene `position_tracking` Dictionary.

    Args:
        symbol (str): Das Handelssymbol.
        df (pd.DataFrame): DataFrame mit OHLCV-Daten und Indikator-/Signal-Spalten
                           (inkl. is_observation UND is_warmup Flags).
        symbol_data (dict): Zusätzliche Daten zum Symbol.
        start_time (datetime): Startzeit für den Kontext des Backtests in diesem Abschnitt.
        position_tracking (dict): Das Dictionary zur Verfolgung des globalen Positionsstatus.

    Returns:
        tuple: (list_of_trades, performance_dict)
    """
    log_prefix = f"process_symbol ({symbol})"
    logger_process_symbol.debug(f"{log_prefix}: Starting function call.")

    # --- Initial Check of Passed Dictionary ---
    if position_tracking is None or not isinstance(position_tracking, dict):
        logger_process_symbol.error(f"{log_prefix}: CRITICAL - Invalid 'position_tracking' dictionary passed.")
        return [], {'error': 'Invalid position_tracking dictionary received', 'symbol': symbol, 'total_trades': 0}

    try:
        # Initialize entry in position_tracking if symbol not present
        if symbol not in position_tracking:
            logger_process_symbol.warning(f"{log_prefix}: Initializing position_tracking entry for {symbol}.")
            position_tracking[symbol] = {'position_open': False, 'observation_ended': False, 'position_data': {}}

        # --- DataFrame Validation ---
        if df is None or df.empty:
            logger_process_symbol.error(f"{log_prefix}: No data provided.")
            return [], {'error': 'No data provided', 'symbol': symbol, 'total_trades': 0}

        # Ensure 'is_warmup' column exists
        if 'is_warmup' not in df.columns:
            logger_process_symbol.warning(f"{log_prefix}: 'is_warmup' column missing. Assuming no warmup rows.")
            df = df.copy()
            df['is_warmup'] = False

        required_cols_process = ['signal', 'open', 'high', 'low', 'close', 'is_warmup'] # is_warmup hinzugefügt
        if not all(col in df.columns for col in required_cols_process):
            missing = [col for col in required_cols_process if col not in df.columns]
            logger_process_symbol.error(f"{log_prefix}: DataFrame missing required columns for processing: {missing}.")
            return [], {'error': f"Missing columns: {missing}", 'symbol': symbol, 'total_trades': 0}

        # --- Setup ---
        tracker = BacktestPositionTracker() # Lokaler Tracker für diese Runde
        trades = [] # Lokale Liste für Trades dieser Runde
        fixed_start_balance = getattr(Z_config, 'start_balance_parameter', 25.0) # Default 25
        commission_rate = getattr(Z_config, 'taker_fee_parameter', 0.0004) # Default 0.04%
        daily_trades_count = {}
        daily_loss_tracker = {}
        previous_date = None
        close_position_on_timeout = getattr(Z_config, 'close_position', False)
        max_daily_trades = getattr(Z_config, 'max_daily_trades_parameter', 100)
        max_daily_loss_pct = getattr(Z_config, 'max_daily_loss_parameter', 100.0)
        close_positions_at_end_run = getattr(Z_config, 'close_positions_at_end', True)
        has_observation_markers = 'is_observation' in df.columns
        trades_dir = "./trades"
        os.makedirs(trades_dir, exist_ok=True)

        # --- Positionsrekonstruktion (aus globalem State) ---
        # This part remains the same, using the passed position_tracking dict
        if position_tracking[symbol].get('position_open') and position_tracking[symbol].get('observation_ended'):
             logger_process_symbol.info(f"{log_prefix}: Reconstructing open position state from passed dictionary.")
             position_data_global = position_tracking[symbol].get('position_data', {})
             if position_data_global and isinstance(position_data_global.get('entry_time'), datetime):
                  try:
                       entry_time_global = position_data_global['entry_time']
                       if entry_time_global.tzinfo is None: entry_time_global = entry_time_global.replace(tzinfo=timezone.utc)
                       else: entry_time_global = entry_time_global.astimezone(timezone.utc)

                       reconstructed_pos = tracker.open_position(
                           symbol=symbol, entry_price=position_data_global.get('entry_price'),
                           position_type=position_data_global.get('position_type'),
                           quantity=position_data_global.get('quantity'), entry_time=entry_time_global )

                       if reconstructed_pos and symbol in tracker.positions:
                           tracker.positions[symbol].update(position_data_global)
                           tracker.positions[symbol]['is_active'] = True
                           tracker.positions[symbol]['entry_time'] = entry_time_global
                           tracker.positions[symbol]['remaining_quantity'] = position_data_global.get('remaining_quantity', tracker.positions[symbol].get('quantity', 0)) # Safely get quantity
                           logger_process_symbol.info(f"{log_prefix}: Position reconstructed locally. Qty: {tracker.positions[symbol]['remaining_quantity']:.8f}, Entry: {entry_time_global}")
                       else:
                           logger_process_symbol.error(f"{log_prefix}: Failed local reconstruction. Resetting global state.")
                           position_tracking[symbol] = {'position_open': False, 'observation_ended': False, 'position_data': {}}
                  except Exception as e:
                       logger_process_symbol.error(f"{log_prefix}: Error during reconstruction: {e}", exc_info=True)
                       position_tracking[symbol] = {'position_open': False, 'observation_ended': False, 'position_data': {}}
             else:
                  logger_process_symbol.warning(f"{log_prefix}: Open flag set, but position_data invalid. Resetting global state.")
                  position_tracking[symbol] = {'position_open': False, 'observation_ended': False, 'position_data': {}}

        # --- Main Candle Loop ---
        logger_process_symbol.debug(f"{log_prefix}: Starting candle loop over {len(df)} rows (index 1 to {len(df) - 2})")
        for i in range(1, len(df) - 1): # Loop bis zur vorletzten Kerze
            try:
                current_candle = df.iloc[i]
                next_candle = df.iloc[i + 1]
                current_time = df.index[i]
                if not isinstance(current_time, pd.Timestamp): continue
                # Ensure timezone awareness (UTC)
                if current_time.tzinfo is None: current_time = current_time.tz_localize('UTC')
                elif str(current_time.tzinfo) != 'UTC': current_time = current_time.tz_convert('UTC')
            except Exception as e:
                logger_process_symbol.error(f"{log_prefix}: Error getting candle data at index {i}: {e}", exc_info=True)
                continue

            # --- Skip Warm-up Candles for Trade Logic ---
            is_warmup_candle = current_candle.get('is_warmup', False)
            if is_warmup_candle:
                # Use a less verbose level for frequent messages
                # logger_process_symbol.log(logging.DEBUG - 1, f"  Candle {i} @{current_time}: Skipping Warmup Candle")
                continue # Go to the next candle

            # --- Daily Limits & Date Tracking ---
            current_date = current_time.date()
            if previous_date is not None and current_date != previous_date:
                logger_process_symbol.debug(f"{log_prefix}: Date changed to {current_date}. Resetting daily limits.")
                daily_trades_count[current_date] = 0
                daily_loss_tracker[current_date] = 0.0
            previous_date = current_date
            if current_date not in daily_trades_count: daily_trades_count[current_date] = 0
            if current_date not in daily_loss_tracker: daily_loss_tracker[current_date] = 0.0

            # --- Observation Period Flags ---
            is_current_candle_in_observation = current_candle.get('is_observation', False) if has_observation_markers else True
            is_last_observation_candle = False
            if has_observation_markers and is_current_candle_in_observation and i + 1 < len(df):
                try:
                    # Check if the *next* candle is NOT in observation
                    if not df.iloc[i+1].get('is_observation', False):
                        is_last_observation_candle = True
                except IndexError: pass # Should not happen due to loop range

            # --- Active Position Handling (Only for Non-Warmup Candles) ---
            position = tracker.get_position(symbol)
            exit_events_recorded_this_candle = []

            if position and position.get("is_active"):
                logger_process_symbol.debug(f"  Candle {i} @{current_time}: Position Active - Checking exits (Non-Warmup Candle)...")

                # 1. Timeout Check (End of Observation)
                if has_observation_markers and close_position_on_timeout and is_last_observation_candle:
                    logger_process_symbol.info(f"{log_prefix}: Timeout Check triggered at end of observation @ {current_time}.")
                    close_price = current_candle.get('close')
                    if pd.notna(close_price):
                        timeout_exit = tracker._close_position(symbol, float(close_price), current_time, "observation_timeout")
                        if timeout_exit:
                             exit_events_recorded_this_candle.append(timeout_exit)
                             position = tracker.get_position(symbol) # Update local status
                             logger_process_symbol.info(f"   -> Position closed due to timeout.")
                        else: logger_process_symbol.error(f"{log_prefix}: Failed to execute timeout close for {symbol}")
                    else: logger_process_symbol.error(f"{log_prefix}: Cannot close on timeout: NaN close price for {symbol} at {current_time}")

                # 2. Main Exit Logic (SL/TP/Strategy) - if still active
                if position and position.get("is_active"):
                    if tracker.use_standard_sl_tp:
                        # Standard SL/TP Check
                        candle_low = current_candle.get('low'); candle_high = current_candle.get('high')
                        if pd.notna(candle_low) and pd.notna(candle_high):
                            # Check SL against low
                            sl_exit = tracker.update_position(symbol, float(candle_low), current_time)
                            if sl_exit:
                                logger_process_symbol.info(f"   Standard SL hit for {symbol} based on candle low {candle_low:.8f}")
                                exit_events_recorded_this_candle.append(sl_exit)
                                position = tracker.get_position(symbol)
                            # Check TP against high ONLY IF SL NOT hit
                            if position and position.get("is_active"):
                                tp_exit = tracker.update_position(symbol, float(candle_high), current_time)
                                if tp_exit:
                                    logger_process_symbol.info(f"   Standard TP hit for {symbol} based on candle high {candle_high:.8f}")
                                    exit_events_recorded_this_candle.append(tp_exit)
                                    position = tracker.get_position(symbol)
                        else: logger_process_symbol.error(f"{log_prefix}: Cannot check standard SL/TP for {symbol} at {current_time}: NaN low/high.")
                    else:
                        # Advanced SL/TP (Intra-Candle Simulation)
                        logger_process_symbol.debug(f"   Running intra-candle simulation for advanced exits @ {current_time}")
                        try:
                            simulated_exits, final_sim_state = tracker._simulate_intra_candle_advanced_exits(symbol, current_candle)
                            if simulated_exits:
                                logger_process_symbol.info(f"   Simulation found {len(simulated_exits)} exit events.")
                                exit_events_recorded_this_candle.extend(simulated_exits)
                            if final_sim_state and isinstance(final_sim_state, dict):
                                if final_sim_state != tracker.positions.get(symbol):
                                    logger_process_symbol.debug(f"    Updating local tracker state based on sim result. Active: {final_sim_state.get('is_active')}")
                                    tracker.positions[symbol] = final_sim_state
                                    position = final_sim_state # Update local variable
                                else:
                                     logger_process_symbol.debug(f"    Sim state matches current tracker state.")
                            else:
                                logger_process_symbol.error(f"{log_prefix}: Intra-candle simulation did not return a valid final state for {symbol}. Local state unchanged.")
                                position = tracker.get_position(symbol)
                        except Exception as sim_err:
                            logger_process_symbol.error(f"{log_prefix}: Error during intra-candle simulation for {symbol}: {sim_err}", exc_info=True)
                            position = tracker.get_position(symbol)

                # 3. Strategy Exit Check - if still active
                if position and position.get("is_active"):
                    sig_exit = current_candle.get('signal', 'no_signal')
                    close_price_sig = current_candle.get('close')
                    pos_type_local = position.get("position_type")
                    if pd.notna(close_price_sig) and \
                       ((sig_exit == "exit_long" and pos_type_local == "long") or \
                        (sig_exit == "exit_short" and pos_type_local == "short")):
                        logger_process_symbol.info(f"{log_prefix}: Strategy Exit Signal '{sig_exit}' found for {symbol} at {current_time}.")
                        strategy_exit_res = tracker._close_position(symbol, float(close_price_sig), current_time, "strategy_exit")
                        if strategy_exit_res:
                            exit_events_recorded_this_candle.append(strategy_exit_res)
                            position = tracker.get_position(symbol)
                            logger_process_symbol.info(f"   -> Position closed due to strategy signal.")
                        else:
                            logger_process_symbol.error(f"{log_prefix}: Failed to execute strategy exit for {symbol}")
                    elif pd.isna(close_price_sig) and sig_exit in ["exit_long", "exit_short"]:
                         logger_process_symbol.error(f"{log_prefix}: Cannot process strategy exit for {symbol} at {current_time}: NaN close price.")


                # --- Record Trades (if any exits occurred) ---
                if exit_events_recorded_this_candle:
                    logger_process_symbol.debug(f"  Candle {i}: Recording {len(exit_events_recorded_this_candle)} exit events...")
                    # Get initial data for PnL calc etc. from global state
                    initial_pos_data_for_rec = position_tracking[symbol].get('position_data', {})
                    # Fallback to local if global seems empty
                    if not initial_pos_data_for_rec or not initial_pos_data_for_rec.get('entry_price'):
                         local_pos_check = tracker.positions.get(symbol, {})
                         if local_pos_check.get('entry_price'):
                              initial_pos_data_for_rec = local_pos_check.copy()
                              logger_process_symbol.warning(f"{log_prefix}: Using LOCAL state for trade recording {symbol} (Global empty/invalid).")

                    if not initial_pos_data_for_rec or not isinstance(initial_pos_data_for_rec.get('entry_price'), (int, float)):
                         logger_process_symbol.error(f"Cannot record trades for {symbol} at {current_time}: Invalid initial position data for PnL. Skipping exits for this candle.")
                         continue # Skip to next candle

                    original_entry_price_rec = initial_pos_data_for_rec.get("entry_price")
                    original_entry_time_rec = initial_pos_data_for_rec.get("entry_time")
                    original_position_type_rec = initial_pos_data_for_rec.get("position_type")
                    original_position_id_rec = initial_pos_data_for_rec.get("position_id")
                    original_slippage_rec = initial_pos_data_for_rec.get('slippage', 0.0)
                    if original_entry_time_rec and original_entry_time_rec.tzinfo is None: original_entry_time_rec = original_entry_time_rec.replace(tzinfo=timezone.utc)

                    for idx, exit_event in enumerate(exit_events_recorded_this_candle):
                         if not isinstance(exit_event, dict): continue
                         exit_qty = exit_event.get("exit_quantity")
                         exit_price = exit_event.get("exit_price")
                         exit_time_rec = exit_event.get("exit_time")
                         exit_reason_rec = exit_event.get("exit_reason", "unknown")
                         is_full_exit = exit_event.get("full_exit", False)
                         level = exit_event.get("exit_level", 0)
                         rem_qty_after_exit = exit_event.get("remaining_quantity", 0)
                         be_active_at_exit = exit_event.get("breakeven_activated", False)
                         res_ts = exit_event.get("resolution_timestamp", pd.NaT)

                         if None in [original_entry_price_rec, exit_qty, exit_price, original_position_type_rec, original_position_id_rec] or pd.isna(exit_time_rec) or exit_qty <= 1e-9:
                              logger_process_symbol.warning(f"{log_prefix}: Skipping trade record event {idx+1} for {symbol} due to missing/invalid data (ExitQty: {exit_qty}, ExitPrice: {exit_price}).")
                              continue

                         try:
                             entry_p=float(original_entry_price_rec); exit_p=float(exit_price); exit_q=float(exit_qty)
                             pnl = _calculate_pnl(original_position_type_rec, entry_p, exit_p, exit_q, commission_rate)
                             comm_entry=entry_p*exit_q*commission_rate; comm_exit=exit_p*exit_q*commission_rate; total_comm=comm_entry+comm_exit
                             pnl_pct=(pnl/fixed_start_balance)*100 if fixed_start_balance!=0 else 0.0
                             if current_date in daily_loss_tracker: daily_loss_tracker[current_date] += pnl_pct
                             else: daily_loss_tracker[current_date] = pnl_pct
                             if not pd.notna(pnl): pnl = 0.0
                         except Exception as calc_err:
                             pnl,total_comm,pnl_pct=0.0,0.0,0.0
                             logger_process_symbol.error(f"{log_prefix}: Error calculating PnL/Comm for exit event {idx+1} ({symbol}): {calc_err}")

                         state_for_levels=initial_pos_data_for_rec
                         current_stop_rec = state_for_levels.get("current_stop", state_for_levels.get("initial_stop", 0))
                         tp_price_rec=0.0; tp_levels_rec=state_for_levels.get("take_profit_levels",[])
                         if tp_levels_rec:
                             tp_idx_to_use=level-1 if level>0 and exit_reason_rec and 'profit' in exit_reason_rec else state_for_levels.get("current_tp_level",0)
                             if 0<=tp_idx_to_use<len(tp_levels_rec): tp_price_rec=tp_levels_rec[tp_idx_to_use]
                             elif tp_levels_rec: tp_price_rec=tp_levels_rec[0]

                         duration_td=timedelta(0); duration_minutes=0.0; duration_hours=0.0
                         if original_entry_time_rec and exit_time_rec:
                             if exit_time_rec.tzinfo is None: exit_time_rec=exit_time_rec.replace(tzinfo=timezone.utc)
                             else: exit_time_rec=exit_time_rec.astimezone(timezone.utc)
                             if isinstance(original_entry_time_rec, datetime) and isinstance(exit_time_rec, datetime):
                                 duration_td=exit_time_rec-original_entry_time_rec
                                 duration_minutes=duration_td.total_seconds()/60
                                 duration_hours=duration_minutes/60
                             else:
                                 logger_process_symbol.warning(f"{log_prefix}: Invalid datetime objects for duration calculation.")

                         trade = {
                             'symbol': symbol, 'position_id': original_position_id_rec,
                             'entry_time': original_entry_time_rec, 'exit_time': exit_time_rec,
                             'signal': f"exit_{original_position_type_rec}", 'trigger': exit_reason_rec,
                             'entry_price': original_entry_price_rec, 'exit_price': exit_price,
                             'trend_strength': current_candle.get('trend_strength', 0),
                             'entry_balance': fixed_start_balance, 'exit_balance': fixed_start_balance + pnl,
                             'profit_loss': pnl, 'commission': total_comm,
                             'slippage': original_slippage_rec if idx == 0 else 0.0,
                             'stop_loss_price': current_stop_rec, 'take_profit_price': tp_price_rec,
                             'breakeven_activated': be_active_at_exit, 'is_partial_exit': not is_full_exit,
                             'exit_level': level, 'remaining_quantity': rem_qty_after_exit,
                             'resolution_timestamp': res_ts if pd.notna(res_ts) else None,
                             'return_pct': pnl_pct, 'duration': duration_minutes, 'duration_hours': duration_hours
                         }
                         trade = validate_trade_timestamps(trade)
                         trades.append(trade)

                         try:
                             append_to_csv([trade], "24h_signals.csv")
                         except NameError: logger_process_symbol.error(f"{log_prefix}: append_to_csv not found/imported.")
                         except Exception as csv_err: logger_process_symbol.error(f"{log_prefix}: Error appending trade to GLOBAL CSV for {symbol}: {csv_err}")

                         exit_type_log = f"{'FULL' if is_full_exit else 'PARTIAL'} EXIT ({exit_reason_rec})"
                         logger_process_symbol.info(f"{log_prefix}: TRADE RECORDED: {exit_type_log} ID: {original_position_id_rec} PnL: {trade['profit_loss']:.8f}")

                    # Update GLOBAL position_tracking AFTER processing ALL events for this candle
                    final_local_state = tracker.get_position(symbol)
                    if final_local_state and final_local_state.get('is_active'):
                        logger_process_symbol.info(f"{log_prefix}: Updating GLOBAL state after partial exit(s). Position remains ACTIVE.")
                        position_tracking[symbol]['position_open'] = True
                        position_tracking[symbol]['position_data'] = final_local_state.copy()
                        # Handle observation_ended flag
                        if has_observation_markers and is_last_observation_candle and not close_position_on_timeout:
                             position_tracking[symbol]['observation_ended'] = True
                             logger_process_symbol.info(f"    Marking observation_ended=True for {symbol}.")
                        else:
                             position_tracking[symbol]['observation_ended'] = position_tracking[symbol].get('observation_ended', False) and not is_last_observation_candle
                    else:
                        logger_process_symbol.info(f"{log_prefix}: Resetting GLOBAL state after full exit.")
                        position_tracking[symbol] = {'position_open': False, 'observation_ended': False, 'position_data': {}}
                        try: update_symbol_position_status(symbol, False)
                        except NameError: logger_process_symbol.debug("update_symbol_position_status not available.")
                        except Exception as e_usps: logger_process_symbol.error(f"Error update_symbol_position_status: {e_usps}")

                    logger_process_symbol.debug(f"  Global Dict State (After Exit Update): Open={position_tracking[symbol].get('position_open')}, ObsEnded={position_tracking[symbol].get('observation_ended')}")
                    continue # Skip Entry Logic for this candle after exits

            # --- Entry Logic (Only for Non-Warmup Candles) ---
            position = tracker.get_position(symbol) # Get current local status again
            signal = current_candle.get('signal', 'no_signal')
            is_current_candle_in_observation = current_candle.get('is_observation', False) if has_observation_markers else True

            can_open_new_base = (signal in ['buy', 'sell']) and (not position or not position.get("is_active", False))
            logger_process_symbol.debug(f"  Candle {i} @{current_time}: Entry Check (Non-Warmup): Signal='{signal}', LocalActive={position.get('is_active') if position else 'False'}, InObservation={is_current_candle_in_observation}, CanOpenBase={can_open_new_base}")

            if can_open_new_base and is_current_candle_in_observation:
                logger_process_symbol.debug(f"    Signal '{signal}' in observation window. Checking limits...")
                limit_trade = daily_trades_count.get(current_date, 0) >= max_daily_trades
                limit_loss = daily_loss_tracker.get(current_date, 0.0) <= -abs(max_daily_loss_pct)

                if limit_trade or limit_loss:
                    reason = "Trade-Limit" if limit_trade else "Verlust-Limit"
                    logger_process_symbol.debug(f"    Entry Blocked: Daily Limit hit ({reason}). Trades: {daily_trades_count.get(current_date, 0)}, Loss: {daily_loss_tracker.get(current_date, 0.0):.2f}%")
                else:
                    # --- Proceed with Entry ---
                    logger_process_symbol.info(f"   Candle {i}: Entry conditions met for Signal='{signal}'. Attempting entry...")
                    pos_type = "long" if signal == "buy" else "short"
                    next_candle_open = next_candle.get('open')
                    if pd.isna(next_candle_open):
                        logger_process_symbol.error(f"{log_prefix}:    Entry Blocked: Next candle open price is NaN.")
                        continue

                    orig_entry_price = float(next_candle_open)
                    risk_per_trade = getattr(Z_config, 'risk_per_trade', 0.01)
                    # Example Position Size Calculation (Replace with your actual logic)
                    leverage = getattr(Z_config, 'leverage_parameter', 10) # Example leverage
                    balance_for_calc = fixed_start_balance # Use fixed balance per trade
                    trade_value_usd = balance_for_calc * leverage
                    pos_size = trade_value_usd / orig_entry_price if orig_entry_price > 0 else 0
                    #pos_size = (fixed_start_balance * risk_per_trade) / orig_entry_price if orig_entry_price > 0 else 0

                    if pos_size <= 1e-9:
                        logger_process_symbol.warning(f"{log_prefix}:    Entry Blocked: Calculated position size is too small or zero ({pos_size}). Price: {orig_entry_price}")
                        continue

                    slip_factor = getattr(Z_config, 'slippage', 0.0005) # Example 0.05%
                    entry_price_slip = orig_entry_price * (1 + slip_factor if pos_type == "long" else 1 - slip_factor)
                    slip_cost = abs(entry_price_slip - orig_entry_price) * pos_size
                    entry_timestamp = current_time # Time of the signal candle

                    new_pos = tracker.open_position(symbol, entry_price_slip, pos_type, pos_size, entry_timestamp)
                    if new_pos:
                        logger_process_symbol.info(f"--> {pos_type.upper()} POSITION OPENED LOCALLY for {symbol} (ID: {new_pos.get('position_id', 'N/A')}) at {entry_price_slip:.8f}")
                        daily_trades_count[current_date] += 1
                        if symbol in tracker.positions: tracker.positions[symbol]['slippage'] = slip_cost
                        logger_process_symbol.debug(f"    ENTRY DEBUG: PosSize={pos_size:.8f}, InitialStop={tracker.positions[symbol].get('current_stop', 'N/A')}, TP1={tracker.positions[symbol].get('take_profit_levels',[0.0])[0]:.8f}")

                        # Update GLOBAL position_tracking
                        logger_process_symbol.debug(f"    Updating GLOBAL position_tracking after entry...")
                        position_tracking[symbol]['position_open'] = True
                        position_tracking[symbol]['observation_ended'] = False # New position starts new observation context
                        position_tracking[symbol]['position_data'] = tracker.get_position(symbol).copy() # Save current local state globally
                        logger_process_symbol.debug(f"    Global Dict State (After Entry): Open={position_tracking[symbol].get('position_open')}, ObsEnded={position_tracking[symbol].get('observation_ended')}")

                        try: update_symbol_position_status(symbol, True, pos_type)
                        except NameError: logger_process_symbol.debug("update_symbol_position_status not available.")
                        except Exception as e_usps: logger_process_symbol.error(f"Error update_symbol_position_status: {e_usps}")

                        # Phase 1 Check (Entry Candle Conflict)
                        pos_after_open = tracker.get_position(symbol)
                        if pos_after_open and pos_after_open.get("is_active"):
                             sl_ph1, tp_ph1 = None, None; pt_ph1 = pos_after_open.get("position_type")
                             entry_actual_ph1 = pos_after_open.get("entry_price"); qty_actual_ph1 = pos_after_open.get("quantity"); pid_actual_ph1 = pos_after_open.get("position_id")
                             if tracker.use_standard_sl_tp: sl_ph1, tp_ph1 = pos_after_open.get("standard_sl"), pos_after_open.get("standard_tp")
                             else: sl_ph1 = pos_after_open.get("initial_stop"); tp_levels_ph1 = pos_after_open.get("take_profit_levels", []); tp_ph1 = tp_levels_ph1[0] if tp_levels_ph1 else None

                             if sl_ph1 is not None and tp_ph1 is not None and pt_ph1 is not None:
                                 low_entry_candle, high_entry_candle = current_candle.get('low'), current_candle.get('high'); conflict_entry = False
                                 if pd.notna(low_entry_candle) and pd.notna(high_entry_candle):
                                     low_entry_candle, high_entry_candle = float(low_entry_candle), float(high_entry_candle)
                                     if pt_ph1 == "long" and low_entry_candle <= sl_ph1 and high_entry_candle >= tp_ph1: conflict_entry = True
                                     elif pt_ph1 == "short" and high_entry_candle >= sl_ph1 and low_entry_candle <= tp_ph1: conflict_entry = True

                                 if conflict_entry:
                                     logger_process_symbol.warning(f"{log_prefix}:    Phase 1 Conflict Detected on entry candle! SL={sl_ph1:.8f}, TP={tp_ph1:.8f}, Low={low_entry_candle:.8f}, High={high_entry_candle:.8f}")
                                     resolution, hit_time_internal = tracker._resolve_sl_tp_priority(symbol, sl_ph1, tp_ph1, entry_timestamp, pt_ph1)
                                     exit_p_imm, exit_r_imm = (sl_ph1, "initial_stop_immediate" if not tracker.use_standard_sl_tp else "standard_stop_loss_immediate") if resolution == "sl" else (tp_ph1, "take_profit_1_immediate" if not tracker.use_standard_sl_tp else "standard_take_profit_immediate")
                                     exit_time_imm = hit_time_internal
                                     logger_process_symbol.info(f"     Phase 1 Resolution: {resolution.upper()} hit first at {exit_time_imm}. Closing position immediately.")
                                     exit_res_imm = tracker._close_position(symbol, exit_p_imm, exit_time_imm, exit_r_imm)

                                     if exit_res_imm and isinstance(exit_res_imm, dict) and pd.notna(qty_actual_ph1):
                                         # Calculate PnL etc. for immediate exit
                                         pnl_imm = _calculate_pnl(pt_ph1, entry_actual_ph1, exit_p_imm, qty_actual_ph1, commission_rate)
                                         comm_entry_imm=entry_actual_ph1*qty_actual_ph1*commission_rate; comm_exit_imm=exit_p_imm*qty_actual_ph1*commission_rate; total_comm_imm=comm_entry_imm+comm_exit_imm
                                         pnl_pct_imm=(pnl_imm/fixed_start_balance)*100 if fixed_start_balance>0 else 0.0
                                         if current_date in daily_loss_tracker: daily_loss_tracker[current_date]+=pnl_pct_imm
                                         else: daily_loss_tracker[current_date] = pnl_pct_imm

                                         duration_imm_td=timedelta(0); duration_mins_imm=0.0; duration_hrs_imm=0.0
                                         if entry_timestamp and exit_time_imm: duration_imm_td=exit_time_imm-entry_timestamp; duration_mins_imm=duration_imm_td.total_seconds()/60; duration_hrs_imm=duration_mins_imm/60

                                         trade_imm = {
                                             'symbol': symbol, 'position_id': pid_actual_ph1, 'entry_time': entry_timestamp, 'exit_time': exit_time_imm,
                                             'signal': f"exit_{pt_ph1}", 'trigger': exit_r_imm, 'entry_price': entry_actual_ph1, 'exit_price': exit_p_imm,
                                             'trend_strength': current_candle.get('trend_strength', 0),
                                             'entry_balance': fixed_start_balance, 'exit_balance': fixed_start_balance + (pnl_imm if pd.notna(pnl_imm) else 0.0),
                                             'profit_loss': pnl_imm if pd.notna(pnl_imm) else 0.0, 'commission': total_comm_imm, 'slippage': slip_cost,
                                             'stop_loss_price': sl_ph1, 'take_profit_price': tp_ph1, 'breakeven_activated': False, 'is_partial_exit': False,
                                             'exit_level': 0, 'remaining_quantity': 0, 'resolution_timestamp': hit_time_internal if pd.notna(hit_time_internal) else None,
                                             'return_pct': pnl_pct_imm, 'duration': duration_mins_imm, 'duration_hours': duration_hrs_imm
                                         }
                                         trade_imm = validate_trade_timestamps(trade_imm)
                                         trades.append(trade_imm)
                                         # Append immediate exit to GLOBAL CSV
                                         try: append_to_csv([trade_imm], "24h_signals.csv")
                                         except NameError: logger_process_symbol.error(f"{log_prefix}: append_to_csv not available.")
                                         except Exception as csv_err: logger_process_symbol.error(f"Error appending immediate exit trade to GLOBAL CSV: {csv_err}")

                                         # Update GLOBAL position_tracking -> Position is closed
                                         logger_process_symbol.info(f"{log_prefix}: Resetting GLOBAL state after immediate Phase 1 exit.")
                                         position_tracking[symbol] = {'position_open': False, 'observation_ended': False, 'position_data': {}}
                                         try: update_symbol_position_status(symbol, False)
                                         except NameError: logger_process_symbol.debug("update_symbol_position_status not available.")
                                         except Exception as e_usps: logger_process_symbol.error(f"Error update_symbol_position_status: {e_usps}")

                                         logger_process_symbol.info(f"IMMEDIATE PHASE 1 EXIT recorded for {symbol} (PnL: {pnl_imm:.8f}). Skipping rest of candle.")
                                         continue # Skip rest of this candle's logic
                                     else:
                                          logger_process_symbol.error(f"{log_prefix}: Phase 1 conflict resolved, but failed to get valid exit details from _close_position.")
                        else:
                             logger_process_symbol.error(f"{log_prefix}: Phase 1 Check skipped: Invalid position state after opening.")
                    else:
                         logger_process_symbol.error(f"{log_prefix}: Entry attempt failed for {symbol}. tracker.open_position returned None.")

            elif can_open_new_base and not is_current_candle_in_observation:
                 logger_process_symbol.debug(f"   Candle {i}: Entry Signal '{signal}' ignored (Outside Observation Window: is_observation={is_current_candle_in_observation}).")
            # --- End Entry Logic ---

        # --- End of Main Candle Loop ---
        logger_process_symbol.debug(f"{log_prefix}: Main candle loop finished.")

        # --- Final Position Handling at End of DataFrame Slice ---
        final_position_state = tracker.get_position(symbol)
        if final_position_state and final_position_state.get("is_active"):
            is_absolute_end_of_data = False
            global_end_time_cfg = None
            # Determine global end time (use Backtest.parse_datetime if available)
            if hasattr(Z_config, 'use_custom_backtest_datetime') and Z_config.use_custom_backtest_datetime and hasattr(Z_config, 'backtest_datetime'):
                 try:
                     if isinstance(Z_config.backtest_datetime, datetime): global_end_time_cfg = Z_config.backtest_datetime
                     elif isinstance(Z_config.backtest_datetime, str):
                         try: global_end_time_cfg = Backtest.parse_datetime(Z_config.backtest_datetime) # Use Backtest helper
                         except NameError: global_end_time_cfg = datetime.strptime(Z_config.backtest_datetime, "%Y-%m-%d %H:%M:%S") # Fallback
                     if global_end_time_cfg and global_end_time_cfg.tzinfo is None: global_end_time_cfg = global_end_time_cfg.replace(tzinfo=timezone.utc)
                     elif global_end_time_cfg: global_end_time_cfg = global_end_time_cfg.astimezone(timezone.utc)
                 except Exception as parse_err: global_end_time_cfg = None; logger_process_symbol.error(f"Error parsing Z_config.backtest_datetime: {parse_err}")

            if global_end_time_cfg and not df.empty:
                 last_candle_time_in_df = df.index[-1]
                 if last_candle_time_in_df.tzinfo is None: last_candle_time_in_df = last_candle_time_in_df.tz_localize('UTC')
                 else: last_candle_time_in_df = last_candle_time_in_df.tz_convert('UTC')
                 try: interval_minutes_end = Backtest.parse_interval_to_minutes(Z_config.interval)
                 except NameError: interval_minutes_end = 15 # Fallback
                 except Exception: interval_minutes_end = 15 # Fallback
                 time_diff_threshold = timedelta(minutes=interval_minutes_end * 2) # Allow some buffer
                 if last_candle_time_in_df >= global_end_time_cfg - time_diff_threshold:
                      is_absolute_end_of_data = True
                      logger_process_symbol.info(f"{log_prefix}: Detected end of absolute backtest period based on Z_config.backtest_datetime.")

            if close_positions_at_end_run and is_absolute_end_of_data:
                 logger_process_symbol.info(f"{log_prefix}: Closing open position at absolute end of backtest data.")
                 try:
                      last_candle_in_df = df.iloc[-1]; last_candle_time = df.index[-1]
                      if last_candle_time.tzinfo is None: last_candle_time = last_candle_time.tz_localize('UTC')
                      else: last_candle_time = last_candle_time.tz_convert('UTC')
                      final_close_price = last_candle_in_df.get('close')
                      if pd.notna(final_close_price):
                          exit_final = tracker._close_position(symbol, float(final_close_price), last_candle_time, "backtest_end")
                          if exit_final and isinstance(exit_final, dict):
                               entry_price_f=final_position_state.get('entry_price'); exit_qty_f=exit_final.get("exit_quantity"); exit_price_f=exit_final.get("exit_price"); pos_type_f=final_position_state.get("position_type"); pid_f=final_position_state.get("position_id"); entry_time_f=final_position_state.get("entry_time")
                               if None not in [entry_price_f, exit_qty_f, exit_price_f, pos_type_f, pid_f, entry_time_f]:
                                   if entry_time_f.tzinfo is None: entry_time_f=entry_time_f.replace(tzinfo=timezone.utc)
                                   pnl_f=_calculate_pnl(pos_type_f, float(entry_price_f), float(exit_price_f), float(exit_qty_f), commission_rate)
                                   comm_entry_f=float(entry_price_f)*float(exit_qty_f)*commission_rate; comm_exit_f=float(exit_price_f)*float(exit_qty_f)*commission_rate; total_comm_f=comm_entry_f+comm_exit_f
                                   pnl_pct_f=(pnl_f/fixed_start_balance)*100 if fixed_start_balance>0 else 0.0; duration_f=timedelta(0)
                                   if entry_time_f and exit_final.get("exit_time"):
                                        exit_time_f=exit_final["exit_time"];
                                        if exit_time_f.tzinfo is None: exit_time_f=exit_time_f.replace(tzinfo=timezone.utc)
                                        duration_f=exit_time_f-entry_time_f
                                   trade_f = {
                                       'symbol': symbol, 'position_id': pid_f, 'entry_time': entry_time_f,
                                       'exit_time': exit_final.get("exit_time"), 'signal': f"exit_{pos_type_f}",
                                       'trigger': exit_final.get("exit_reason", "backtest_end"),
                                       'entry_price': entry_price_f, 'exit_price': exit_price_f,
                                       'trend_strength': last_candle_in_df.get('trend_strength', 0),
                                       'entry_balance': fixed_start_balance, 'exit_balance': fixed_start_balance + (pnl_f if pd.notna(pnl_f) else 0.0),
                                       'profit_loss': pnl_f if pd.notna(pnl_f) else 0.0, 'commission': total_comm_f,
                                       'slippage': final_position_state.get('slippage', 0),
                                       'stop_loss_price': final_position_state.get("current_stop", 0),
                                       'take_profit_price': final_position_state.get("take_profit_levels", [0.0])[0],
                                       'breakeven_activated': final_position_state.get("breakeven_activated", False),
                                       'is_partial_exit': False, 'exit_level': 0, 'remaining_quantity': 0,
                                       'resolution_timestamp': pd.NaT,
                                       'return_pct': pnl_pct_f, 'duration': duration_f.total_seconds()/60,
                                       'duration_hours': duration_f.total_seconds()/3600
                                   }
                                   trade_f = validate_trade_timestamps(trade_f)
                                   trades.append(trade_f)
                                   try: append_to_csv([trade_f], "24h_signals.csv")
                                   except NameError: logger_process_symbol.error(f"{log_prefix}: append_to_csv not available.")
                                   except Exception as csv_err: logger_process_symbol.error(f"Error appending final trade to GLOBAL CSV: {csv_err}")
                                   # Update GLOBAL state
                                   logger_process_symbol.info(f"{log_prefix}: Resetting GLOBAL state after final backtest_end close.")
                                   position_tracking[symbol] = {'position_open': False, 'observation_ended': False, 'position_data': {}}
                                   try: update_symbol_position_status(symbol, False)
                                   except NameError: logger_process_symbol.debug("update_symbol_position_status not available.")
                                   except Exception as e_usps: logger_process_symbol.error(f"Error update_symbol_position_status: {e_usps}")
                               else: logger_process_symbol.error(f"{log_prefix}: Cannot record final trade for {symbol}: Missing data.")
                          else: logger_process_symbol.error(f"{log_prefix}: Failed to close final position {symbol} locally.")
                      else: logger_process_symbol.error(f"{log_prefix}: Cannot close final position {symbol}: Last candle close NaN.")
                 except Exception as final_close_err: logger_process_symbol.error(f"{log_prefix}: Error closing final position for {symbol}: {final_close_err}", exc_info=True)
            elif final_position_state and final_position_state.get("is_active"):
                 # Position remains open (not end of absolute backtest or close_at_end=False)
                 logger_process_symbol.info(f"{log_prefix}: Position remains open at end of current DataFrame slice.")
                 # Save current state globally ONLY IF close_position is False
                 if not close_position_on_timeout: # close_position_on_timeout mirrors Z_config.close_position
                     logger_process_symbol.info(f"   Saving state globally (close_position=False).")
                     position_tracking[symbol]['position_open'] = True
                     position_tracking[symbol]['observation_ended'] = True # Mark observation ended for this slice
                     position_tracking[symbol]['position_data'] = final_position_state.copy()
                     # Ensure timestamps are TZ-aware for storage
                     if 'entry_time' in position_tracking[symbol]['position_data'] and isinstance(position_tracking[symbol]['position_data']['entry_time'], datetime) and position_tracking[symbol]['position_data']['entry_time'].tzinfo is None:
                         position_tracking[symbol]['position_data']['entry_time'] = position_tracking[symbol]['position_data']['entry_time'].replace(tzinfo=timezone.utc)
                     logger_process_symbol.debug(f"   Saved State: Open={position_tracking[symbol]['position_open']}, ObsEnded={position_tracking[symbol]['observation_ended']}")
                 else: # close_position is True
                     logger_process_symbol.error(f"{log_prefix}: Position still marked active locally at end of slice, but close_position is TRUE. Potential logic error!")
                     position_tracking[symbol] = {'position_open': False, 'observation_ended': False, 'position_data': {}}


        # --- Performance Calculation & Trade Report Generation ---
        performance = {}
        if trades:
             try:
                 # Ensure necessary functions are available
                 from utils.backtest_integration import analyze_backtest_results, generate_trade_report
                 performance = calculate_performance_metrics(trades, symbol, fixed_start_balance)
                 if performance and 'error' not in performance:
                     detailed_results = {
                         "initial_balance": fixed_start_balance,
                         "final_balance": fixed_start_balance + sum(t.get('profit_loss', 0) for t in trades),
                         **performance,
                         "trades": trades
                     }
                     try:
                         analyze_backtest_results(detailed_results)
                         report_path = os.path.join(trades_dir, f"{symbol}_trade_report.csv")
                         generate_trade_report(detailed_results, report_path)
                         logger_process_symbol.info(f"{log_prefix}: Generated/Updated trade report for {symbol}: {report_path}")
                     except NameError: logger_process_symbol.error(f"{log_prefix}: analyze_backtest_results or generate_trade_report not found.")
                     except Exception as report_err: logger_process_symbol.error(f"{log_prefix}: Error generating analysis/report for {symbol}: {report_err}", exc_info=True)
             except NameError:
                  logger_process_symbol.error(f"{log_prefix}: calculate_performance_metrics not found.")
                  performance = {'error': 'Performance function missing', 'symbol': symbol, 'total_trades': len(trades)}
             except Exception as e:
                  logger_process_symbol.error(f"{log_prefix}: Error during performance calculation/report for {symbol}: {e}", exc_info=True)
                  performance = {'error': str(e), 'symbol': symbol, 'total_trades': len(trades)}
        else: # No trades
             logger_process_symbol.info(f"{log_prefix}: No trades generated in this run.")
             try:
                 # from utils.backtest_strategy import calculate_performance_metrics (assuming imported)
                 performance = calculate_performance_metrics([], symbol, fixed_start_balance)
             except NameError:
                 logger_process_symbol.error(f"{log_prefix}: calculate_performance_metrics not found.")
                 performance = {'error': 'Performance function missing', 'symbol': symbol, 'total_trades': 0}
             except Exception as e:
                  logger_process_symbol.error(f"{log_prefix}: Error calculating zero-trade performance for {symbol}: {e}")
                  performance = {'error': 'Zero trades calc error', 'symbol': symbol, 'total_trades': 0}

        if performance is None or not isinstance(performance, dict):
             performance = {'error': 'Performance calculation failed unexpectedly', 'symbol': symbol, 'total_trades': len(trades)}
             logger_process_symbol.error(f"{log_prefix}: Performance calculation resulted in non-dict or None.")
        elif 'symbol' not in performance:
              performance['symbol'] = symbol

        # --- Final Logging & Return ---
        logger_process_symbol.debug(f"{log_prefix}: Finished processing. Returning {len(trades)} trade events.")
        return trades, performance

    # --- Global Error Handling for process_symbol ---
    except Exception as e:
        logger_process_symbol.error(f"--- FATAL ERROR processing symbol {symbol} in process_symbol: {e} ---", exc_info=True)
        if position_tracking is not None and isinstance(position_tracking, dict) and symbol in position_tracking:
             logger_process_symbol.error(f"Resetting global state for {symbol} due to fatal error in process_symbol.")
             position_tracking[symbol] = {'position_open': False, 'observation_ended': False, 'position_data': {}}
        return [], {'error': f"Fatal error processing {symbol}: {e}", 'symbol': symbol, 'total_trades': 0}

    
def track_observation_period(symbol, action, period_data=None, has_position=None):
    """
    Zentrale Funktion zur Verwaltung von Beobachtungszeiträumen und ihrem Status.
    
    Args:
        symbol: Trading-Symbol
        action: 'register' (neuen Zeitraum registrieren), 'update' (Status aktualisieren),
               'check' (Status prüfen), 'remove' (Zeitraum entfernen)
        period_data: Bei 'register' Dictionary mit start_time, end_time, price_change_pct
        has_position: Bei 'update' Boolean, ob eine Position geöffnet wurde
    
    Returns:
        Je nach action: bei 'check' dict mit Statusinformationen, bei anderen True/False
    """
    global active_observation_periods
    
    # Initialisiere für dieses Symbol, falls noch nicht vorhanden
    if symbol not in active_observation_periods and action != 'remove':
        active_observation_periods[symbol] = {
            'active_period': None,
            'has_position': False
        }
    
    if action == 'register':
        # Neuen Beobachtungszeitraum registrieren
        if period_data and isinstance(period_data, dict):
            active_observation_periods[symbol]['active_period'] = {
                'start_time': period_data.get('start_time', period_data.get('start')),
                'end_time': period_data.get('end_time', period_data.get('end')),
                'price_change_pct': period_data.get('price_change_pct', 0)
            }
            logging.info(f"Neuer Beobachtungszeitraum für {symbol} registriert")
            return True
        return False
        
    elif action == 'update':
        # Positionsstatus aktualisieren
        if has_position is not None:
            active_observation_periods[symbol]['has_position'] = has_position
            logging.info(f"Positionsstatus für {symbol} aktualisiert: {'Position geöffnet' if has_position else 'Keine Position'}")
            return True
        return False
        
    elif action == 'check':
        # Status prüfen und zurückgeben
        if symbol in active_observation_periods:
            return active_observation_periods[symbol]
        return None
        
    elif action == 'remove':
        # Beobachtungszeitraum entfernen
        if symbol in active_observation_periods:
            del active_observation_periods[symbol]
            logging.info(f"Beobachtungszeitraum für {symbol} entfernt")
            return True
        return False
    
    # Unbekannte Aktion
    logging.warning(f"Unbekannte Aktion '{action}' für track_observation_period")
    return False


def calculate_max_drawdown(trades, start_balance):
    """
    Berechnet den maximalen Drawdown basierend auf einer Liste von Trades.
    
    Args:
        trades: Liste von Trade-Dictionaries mit 'profit_loss' Einträgen
        start_balance: Startguthaben
        
    Returns:
        Max Drawdown als Prozentsatz und detaillierte Informationen
    """
    if not trades:
        return 0.0, [], [], 0.0
        
    # Beginne mit dem Startguthaben
    current_balance = start_balance
    balances = [current_balance]
    equity_curve = []
    drawdown_curve = []
    
    # Aktualisiere das Guthaben nach jedem Trade
    for i, trade in enumerate(trades):
        profit_loss = trade.get('profit_loss', 0)
        current_balance += profit_loss
        balances.append(current_balance)
        
        # Speichere Equity-Kurve mit Zeitindex
        trade_time = trade.get('exit_time', f"Trade {i+1}")
        equity_curve.append((trade_time, current_balance))
    
    # Berechne den maximalen Drawdown
    peak_balance = start_balance
    max_drawdown_pct = 0.0
    max_drawdown_start = None
    max_drawdown_end = None
    current_drawdown_start = None
    
    for i, balance in enumerate(balances):
        if balance > peak_balance:
            # Neuer Höchststand erreicht
            peak_balance = balance
            # Beende den aktuellen Drawdown-Zeitraum
            current_drawdown_start = None
        else:
            # Berechne den aktuellen Drawdown als Prozentsatz
            current_drawdown_pct = ((peak_balance - balance) / peak_balance) * 100 if peak_balance > 0 else 0
            
            # Speichere Drawdown-Wert
            trade_index = i-1 if i > 0 else 0
            trade_time = trades[trade_index].get('exit_time', f"Trade {trade_index+1}") if trade_index < len(trades) else f"Trade {i}"
            drawdown_curve.append((trade_time, current_drawdown_pct))
            
            # Markiere den Start eines neuen Drawdown-Zeitraums
            if current_drawdown_start is None and current_drawdown_pct > 0:
                current_drawdown_start = i
            
            # Aktualisiere max_drawdown, wenn aktueller Drawdown größer ist
            if current_drawdown_pct > max_drawdown_pct:
                max_drawdown_pct = current_drawdown_pct
                if current_drawdown_start is not None:
                    max_drawdown_start = current_drawdown_start
                    max_drawdown_end = i
    
    # Berechne die Drawdown-Dauer (in Anzahl der Trades)
    drawdown_duration = 0
    if max_drawdown_start is not None and max_drawdown_end is not None:
        drawdown_duration = max_drawdown_end - max_drawdown_start
    
    return max_drawdown_pct, equity_curve, drawdown_curve, drawdown_duration


logger = logging.getLogger(__name__)

def calculate_performance_metrics(trades, symbol, start_balance, final_balance_ignore=None): # Renamed final_balance to avoid confusion
    """
    Calculate performance metrics based on trade history.
    Handles list of trade event dictionaries, including partial exits.

    Args:
        trades: List of trade dictionaries (each dict represents an exit event).
        symbol: Trading symbol.
        start_balance: Initial balance for context (e.g., for drawdown).
        final_balance_ignore: Final balance passed from process_symbol (can be recalculated here).

    Returns:
        Dictionary with performance metrics.
    """
    # --- Default empty result ---
    empty_result = {
        'symbol': symbol, 'total_trades': 0, 'win_rate': 0.0, 'total_profit': 0.0,
        'max_drawdown': 0.0, 'profit_factor': 0.0, 'avg_profit_win': 0.0, 'avg_loss_loss': 0.0,
        'total_commission': 0.0, 'total_slippage': 0.0, 'exit_reasons': {},
        'immediate_sl_hits': 0, 'immediate_tp_hits': 0,
        # Add other keys with default values matching the CSV header
        'total_exits': 0, 'avg_profit': 0.0, 'avg_slippage': 0.0,
        'stop_loss_hits': 0, 'take_profit_hits': 0, 'signal_exits': 0,
        'backtest_end_exits': 0, 'trailing_stop_hits': 0, 'observation_timeout_exits': 0,
        'partial_exits': 0
    }

    # --- Input Validation ---
    if not isinstance(trades, list):
         logger.error(f"Invalid input for calculate_performance_metrics: 'trades' is not a list for {symbol}.")
         return empty_result
    if trades and not isinstance(trades[0], dict):
         logger.error(f"Invalid input for calculate_performance_metrics: 'trades' list does not contain dictionaries for {symbol}.")
         return empty_result

    # --- Data Cleaning and Conversion ---
    valid_trades = []
    for t in trades:
        try:
            trade_copy = t.copy() # Work on a copy
            # Ensure profit_loss is float
            trade_copy['profit_loss'] = float(trade_copy.get('profit_loss', 0.0))
            # Ensure commission and slippage are floats
            trade_copy['commission'] = float(trade_copy.get('commission', 0.0))
            trade_copy['slippage'] = float(trade_copy.get('slippage', 0.0))
            valid_trades.append(trade_copy)
        except (ValueError, TypeError) as e:
            logger.warning(f"Skipping trade record due to conversion error for {symbol} (ID: {t.get('position_id', 'N/A')}): {e}. Trade: {t}")

    trades = valid_trades # Use only valid trades
    total_trade_events = len(trades) # Total number of exit events (incl. partials)

    if total_trade_events == 0:
        logger.info(f"No valid trade events found for {symbol} to calculate metrics.")
        return empty_result

    logger.debug(f"Calculating metrics for {symbol} based on {total_trade_events} valid trade events.")

    # --- Calculate Core Metrics ---
    total_profit = sum(t['profit_loss'] for t in trades)
    total_commission = sum(t['commission'] for t in trades)
    # Slippage is usually associated with entry, sum the value recorded for each exit event's corresponding entry
    total_slippage = sum(t['slippage'] for t in trades)
    avg_slippage = total_slippage / total_trade_events if total_trade_events > 0 else 0.0

    # --- Win Rate (Based on PnL of each exit event) ---
    winning_trade_events = [t for t in trades if t['profit_loss'] > 0]
    losing_trade_events = [t for t in trades if t['profit_loss'] < 0]
    breakeven_trade_events = [t for t in trades if t['profit_loss'] == 0]

    # Calculate Win Rate based on non-breakeven trades
    non_be_trades_count = len(winning_trade_events) + len(losing_trade_events)
    win_rate = len(winning_trade_events) / non_be_trades_count if non_be_trades_count > 0 else 0.0

    # Debugging Win Rate Calculation
    logger.debug(f"Metrics Debug ({symbol}): Total Events={total_trade_events}, Winning Events={len(winning_trade_events)}, Losing Events={len(losing_trade_events)}, BE Events={len(breakeven_trade_events)}")
    logger.debug(f"Metrics Debug ({symbol}): Non-BE Count={non_be_trades_count}, Calculated Win Rate={win_rate:.4f}")

    # Sanity check for win rate
    if not (0.0 <= win_rate <= 1.0):
         logger.error(f"IMPOSSIBLE WIN RATE calculated ({win_rate:.4f}) for {symbol}. Check PnL values and logic. Setting to 0.")
         win_rate = 0.0

    # --- Profit Factor ---
    total_pnl_wins = sum(t['profit_loss'] for t in winning_trade_events)
    total_pnl_losses_abs = abs(sum(t['profit_loss'] for t in losing_trade_events))

    if total_pnl_losses_abs > 0:
        profit_factor = total_pnl_wins / total_pnl_losses_abs
    else:
        profit_factor = float('inf') if total_pnl_wins > 0 else 0.0 # Assign infinity if no losses but wins, 0 if no wins/losses

    # --- Average Win / Loss per Event ---
    avg_profit_win = total_pnl_wins / len(winning_trade_events) if winning_trade_events else 0.0
    avg_loss_loss = sum(t['profit_loss'] for t in losing_trade_events) / len(losing_trade_events) if losing_trade_events else 0.0 # Keep as negative

    # --- Max Drawdown ---
    max_drawdown_pct = 0.0 # Default
    try:
         # Assuming calculate_max_drawdown exists and works correctly with the trades list
         max_drawdown_pct, equity_curve, drawdown_curve, drawdown_duration = calculate_max_drawdown(trades, start_balance)
         logger.debug(f"Metrics Debug ({symbol}): Max Drawdown Calc returned: {max_drawdown_pct:.2f}%")
    except NameError:
        logger.warning("calculate_max_drawdown function not found. Max Drawdown set to 0.")
    except Exception as dd_error:
         logger.error(f"Error calculating max drawdown for {symbol}: {dd_error}", exc_info=True)
         max_drawdown_pct = -1.0 # Indicate error

    # --- Exit Reason Analysis ---
    exit_triggers = [str(t.get('trigger', 'unknown')).lower() for t in trades] # Ensure trigger is string
    exit_reasons_counts = Counter(exit_triggers) # Now Counter will be defined

    # Consolidate counts for summary reporting
    stop_loss_hits = sum(count for reason, count in exit_reasons_counts.items() if 'stop' in reason and 'immediate' not in reason)
    take_profit_hits = sum(count for reason, count in exit_reasons_counts.items() if 'profit' in reason and 'immediate' not in reason) # Includes trailing TP hit
    trailing_stop_hits = sum(count for reason, count in exit_reasons_counts.items() if reason == 'trailing_stop') # Specific trailing SL
    immediate_sl_hits = sum(count for reason, count in exit_reasons_counts.items() if 'immediate' in reason and 'stop' in reason)
    immediate_tp_hits = sum(count for reason, count in exit_reasons_counts.items() if 'immediate' in reason and 'profit' in reason)
    signal_exits = exit_reasons_counts.get('strategy_exit', 0)
    backtest_end_exits = exit_reasons_counts.get('backtest_end', 0)
    observation_timeout_exits = exit_reasons_counts.get('observation_timeout', 0)
    partial_exits_count = sum(1 for t in trades if t.get('is_partial_exit', False))

    # --- Prepare Performance Dictionary ---
    performance = {
        'symbol': symbol,
        'total_trades': total_trade_events, # Renamed for clarity, represents exit events
        'total_exits': total_trade_events, # Alias for header compatibility
        'win_rate': win_rate,
        'total_profit': total_profit,
        'max_drawdown': max_drawdown_pct,
        'profit_factor': profit_factor,
        'avg_profit': total_profit / total_trade_events if total_trade_events > 0 else 0.0, # Overall average PnL per event
        'avg_profit_win': avg_profit_win, # Average PnL of winning events
        'avg_loss_loss': avg_loss_loss, # Average PnL of losing events (negative)
        'total_commission': total_commission,
        'avg_slippage': avg_slippage,
        'total_slippage': total_slippage,
        'exit_reasons': dict(exit_reasons_counts), # Detailed counts per trigger string
        # Consolidated counts for summary reporting / CSV header
        'stop_loss_hits': stop_loss_hits,
        'take_profit_hits': take_profit_hits, # This includes TP1, TP2, TP3, Trailing TP
        'signal_exits': signal_exits,
        'backtest_end_exits': backtest_end_exits,
        'trailing_stop_hits': trailing_stop_hits, # Specific count for trailing SL
        'observation_timeout_exits': observation_timeout_exits,
        'immediate_sl_hits': immediate_sl_hits, # Included for detail, may overlap stop_loss_hits
        'immediate_tp_hits': immediate_tp_hits, # Included for detail, may overlap take_profit_hits
        'partial_exits': partial_exits_count,
    }

    # --- Log Summary for this Symbol ---
    logger.info(f"--- Performance Metrics Calculated for {symbol} ---")
    logger.info(f"Total Events: {performance['total_trades']}")
    logger.info(f"Win Rate: {performance['win_rate']:.2%}")
    logger.info(f"Total PnL: {performance['total_profit']:.4f}")
    logger.info(f"Max Drawdown: {performance['max_drawdown']:.2f}%")
    logger.info(f"Profit Factor: {performance['profit_factor']:.2f}")
    logger.info(f"Avg Win PnL: {performance['avg_profit_win']:.4f} ({len(winning_trade_events)} events)")
    logger.info(f"Avg Loss PnL: {performance['avg_loss_loss']:.4f} ({len(losing_trade_events)} events)")
    logger.info(f"Exit Reasons: {performance['exit_reasons']}")
    logger.info(f"----------------------------------------------")

    # --- Save to performance_summary.csv ---
    # Ensure the dict keys match the CSV header used in main()
    header_keys = "symbol,total_trades,total_exits,win_rate,total_profit,max_drawdown,profit_factor,avg_profit,total_commission,avg_slippage,stop_loss_hits,take_profit_hits,signal_exits,backtest_end_exits,trailing_stop_hits,observation_timeout_exits,partial_exits".split(',')
    performance_for_csv = {key: performance.get(key, 0) for key in header_keys} # Create dict with exactly header keys

    try:
        # Make sure Backtest.append_to_csv is imported or available
        append_to_csv([performance_for_csv], "performance_summary.csv")
    except NameError:
        logger.error("Backtest.append_to_csv function not found. Cannot save performance summary.")
    except Exception as e:
        logger.error(f"Error saving performance metrics to CSV for {symbol}: {e}")

    return performance

def calculate_required_candles():
    """
    Berechnet die minimal benötigte Anzahl an Kerzen basierend auf Z_config,
    entsprechend der Logik aus dem Live-Tool-Snippet.
    """
    try:
        # Stelle sicher, dass die Variablennamen hier exakt denen in deiner Z_config.py entsprechen!
        required_periods = [
            getattr(Z_config, 'ema_fast_parameter', 11),
            getattr(Z_config, 'ema_slow_parameter', 46),
            getattr(Z_config, 'ema_baseline_parameter', 50),
            getattr(Z_config, 'back_trend_periode', 5),
            getattr(Z_config, 'rsi_period', 8),
            getattr(Z_config, 'volume_sma', 30),
            getattr(Z_config, 'momentum_lookback', 15),
            getattr(Z_config, 'min_trend_duration_parameter', 7),
            getattr(Z_config, 'bb_period', 21),
            getattr(Z_config, 'macd_slow_period', 26), # Längste Periode relevant
            getattr(Z_config, 'adx_period', 21),
            getattr(Z_config, 'obv_period', 11), # OBV Periode weniger relevant
            getattr(Z_config, 'cmf_period', 30),
            getattr(Z_config, 'reversal_lookback_period', 20), # Aus Live
            getattr(Z_config, 'atr_period_parameter', 14)
            # Füge HIER weitere relevante Perioden hinzu!
        ]
        numeric_periods = [p for p in required_periods if isinstance(p, (int, float)) and p is not None and p > 0]
        highest_period = max(numeric_periods) if numeric_periods else 100

        # Verwende den KONSISTENTEN Buffer (z.B. 5 wie im Live-Code ODER lese aus Z_config WENN es dort GLEICH ist)
        candle_buffer = getattr(Z_config, 'indicator_candle_buffer', 5) # Annahme: Ist in Z_config jetzt auf 5 gesetzt!
        required = int(highest_period + candle_buffer)
        logging.info(f"calculate_required_candles: Benötigt={required} (MaxPeriod={highest_period} + Buffer={candle_buffer})")
        return required
    except Exception as e_req:
        logging.error(f"Error in calculate_required_candles: {e_req}. Using default 150.")
        return 150

def calculate_trend_duration(df):
    """
    Calculate the duration of the current trend in consecutive candles
    
    Parameters:
    df (pd.DataFrame): DataFrame with 'trend' column already calculated
    
    Returns:
    pd.Series: Series with trend duration values
    """
    # Initialize trend duration with zeros
    duration = pd.Series(0, index=df.index)
    
    # Calculate duration for each candle
    for i in range(1, len(df)):
        current_trend = df['trend'].iloc[i]
        prev_trend = df['trend'].iloc[i-1]
        
        # If trend direction is the same and not neutral, increment duration
        if current_trend == prev_trend and current_trend != 0:
            duration.iloc[i] = duration.iloc[i-1] + 1
        # If trend direction changed or current trend is neutral (0), reset duration
        else:
            duration.iloc[i] = 1 if current_trend != 0 else 0
    
    return duration

from utils import indicators

def get_multi_timeframe_data(symbol, end_time, initial_start_time, position_tracking=None):
    """
    MODIFIED VERSION 2 (Response #7): Fetches raw data ensuring enough history for BOTH
    indicator warm-up AND the configured lookback_hours_parameter.
    Marks warm-up candles. Calls new fetch_data (which handles batching) correctly.
    Alignment trends are still optional.

    Args:
        symbol (str): Trading symbol.
        end_time (datetime): Logical end time for the data needed (TZ-aware UTC).
        initial_start_time (datetime): Logical start time for the BACKTEST period (TZ-aware UTC).
        position_tracking (dict, optional): Passed state of open positions.

    Returns:
        pd.DataFrame or None: DataFrame with OHLCV, is_complete, is_warmup, symbol,
                              and potentially alignment trends, or None on failure.
    """
    log_prefix = f"get_multi (Backtest - {symbol})"
    logger_get_multi.debug(f"{log_prefix}: Starting function call.")
    try:
        # 1. Zeitstempel validieren
        if not all(isinstance(t, datetime) and t.tzinfo is not None for t in [end_time, initial_start_time]):
             logger_get_multi.error(f"{log_prefix}: Provided start/end times must be timezone-aware datetimes.")
             if isinstance(end_time, datetime) and end_time.tzinfo is None: end_time = end_time.replace(tzinfo=timezone.utc)
             if isinstance(initial_start_time, datetime) and initial_start_time.tzinfo is None: initial_start_time = initial_start_time.replace(tzinfo=timezone.utc)
             if not all(isinstance(t, datetime) and t.tzinfo is not None for t in [end_time, initial_start_time]):
                 logger_get_multi.error(f"{log_prefix}: Could not ensure timezone-aware datetimes. Returning None.")
                 return None
             logger_get_multi.warning(f"{log_prefix}: Corrected naive start/end times to UTC.")

        end_time_utc = end_time.astimezone(timezone.utc)
        initial_start_time_utc_ref = initial_start_time.astimezone(timezone.utc) # Referenzpunkt
        logger_get_multi.debug(f"{log_prefix}: Logical Backtest Time Range Requested: {initial_start_time_utc_ref} -> {end_time_utc}")

        # 2. Benötigte GESAMT-Historie berechnen
        required_warmup_candles = calculate_required_candles()
        interval_str = getattr(Z_config, 'interval', '5m')
        interval_minutes = Backtest.parse_interval_to_minutes(interval_str)
        if interval_minutes is None or interval_minutes <= 0:
             logger_get_multi.error(f"{log_prefix}: Invalid base interval '{interval_str}' parsed to {interval_minutes} minutes.")
             return None

        warmup_duration = timedelta(minutes=required_warmup_candles * interval_minutes)
        logger_get_multi.debug(f"{log_prefix}: Required indicator WARMUP duration: {warmup_duration} ({required_warmup_candles} candles * {interval_minutes} min)")

        lookback_hours = getattr(Z_config, 'lookback_hours_parameter', 24) # Default 24h
        # Sanity Check
        if not isinstance(lookback_hours, (int, float)) or lookback_hours <= 0:
            logger_get_multi.error(f"{log_prefix}: Invalid value read for lookback_hours_parameter: {lookback_hours}. Using fallback 24 hours.")
            lookback_hours = 24
        logger_get_multi.info(f"{log_prefix}: Using lookback_hours_parameter = {lookback_hours}") # Log den verwendeten Wert

        backtest_duration = timedelta(hours=lookback_hours)
        logger_get_multi.debug(f"{log_prefix}: Requested BACKTEST duration: {backtest_duration} ({lookback_hours} hours)")

        total_fetch_duration = backtest_duration + warmup_duration
        logger_get_multi.debug(f"{log_prefix}: TOTAL fetch duration needed: {total_fetch_duration}")

        # 3. Fetch-Fenster bestimmen
        fetch_end = end_time_utc

        close_position_setting = getattr(Z_config, 'close_position', False)
        if not close_position_setting and position_tracking and symbol in position_tracking and position_tracking[symbol].get('position_open'):
            pos_data = position_tracking[symbol].get('position_data', {})
            entry_time_data = pos_data.get('entry_time')
            if entry_time_data and isinstance(entry_time_data, datetime):
                buffer_hours_ext = getattr(Z_config, 'fetch_extension_hours', 24)
                potential_extended_end = end_time_utc + timedelta(hours=buffer_hours_ext)
                if potential_extended_end > fetch_end:
                    logger_get_multi.info(f"{log_prefix}: Extending fetch end to {potential_extended_end} (close_position=False).")
                    fetch_end = potential_extended_end
            else:
                 logger_get_multi.warning(f"{log_prefix}: Open pos detected for {symbol}, but entry_time invalid/missing. Cannot extend fetch end accurately.")

        fetch_start_raw = fetch_end - total_fetch_duration
        logger_get_multi.debug(f"{log_prefix}: Raw fetch_start calculated based on total duration: {fetch_start_raw}")

        fetch_start = None
        if interval_minutes > 0:
            if fetch_start_raw.tzinfo is not None: fetch_start_naive = fetch_start_raw.replace(tzinfo=None)
            else: fetch_start_naive = fetch_start_raw
            minutes_since_midnight = fetch_start_naive.hour * 60 + fetch_start_naive.minute
            minutes_into_interval = minutes_since_midnight % interval_minutes
            fetch_start_aligned_naive = fetch_start_naive - timedelta(minutes=minutes_into_interval)
            fetch_start_aligned_naive = fetch_start_aligned_naive.replace(second=0, microsecond=0)
            fetch_start = fetch_start_aligned_naive.replace(tzinfo=timezone.utc)
            if fetch_start > fetch_start_raw:
                 logger_get_multi.warning(f"{log_prefix}: fetch_start alignment resulted in later time? {fetch_start} > {fetch_start_raw}. Reverting to raw.")
                 fetch_start = fetch_start_raw.replace(second=0, microsecond=0)
            logger_get_multi.debug(f"{log_prefix}: Aligned fetch_start (rounded down): {fetch_start}")
        else:
            logger_get_multi.warning(f"{log_prefix}: Invalid interval_minutes ({interval_minutes}), cannot align fetch_start.")
            fetch_start = fetch_start_raw.replace(second=0, microsecond=0)

        logger_get_multi.info(f"{log_prefix}: Final Fetch Window for API call: {fetch_start} -> {fetch_end}")

        # 4. Basisdaten holen (Aufruf der NEUEN Batch-fähigen fetch_data)
        # Es wird KEIN limit_force mehr übergeben!
        logger_get_multi.debug(f"{log_prefix}: Calling Backtest.fetch_data (Batching handled internally)...")
        try:
            df_base = Backtest.fetch_data(
                symbol=symbol,
                interval=interval_str,
                end_time=fetch_end,
                start_time_force=fetch_start
            )
        except AttributeError:
             logger_get_multi.critical(f"{log_prefix}: Backtest.fetch_data function not found or import failed!")
             return None
        except Exception as fetch_err:
             logger_get_multi.error(f"{log_prefix}: Error during Backtest.fetch_data call: {fetch_err}", exc_info=True)
             return None

        if df_base is None or df_base.empty:
            logger_get_multi.warning(f"{log_prefix}: No base data ({interval_str}) returned from Backtest.fetch_data call.")
            return None

        # --- Mark Warm-up Candles ---
        first_backtest_candle_time = fetch_start + warmup_duration
        logger_get_multi.debug(f"{log_prefix}: First non-warmup candle expected around: {first_backtest_candle_time}")
        if 'is_warmup' not in df_base.columns: df_base['is_warmup'] = False
        df_base.loc[df_base.index < first_backtest_candle_time, 'is_warmup'] = True
        warmup_count = df_base['is_warmup'].sum()
        non_warmup_count = len(df_base) - warmup_count
        logger_get_multi.info(f"{log_prefix}: Marked {warmup_count} candles as WARMUP (before {first_backtest_candle_time}). {non_warmup_count} candles available for backtest period.")
        # Überprüfe, ob genügend Nicht-Warmup-Kerzen vorhanden sind
        expected_backtest_candles = math.ceil(backtest_duration.total_seconds() / (interval_minutes * 60)) if interval_minutes > 0 else 0
        if non_warmup_count < expected_backtest_candles * 0.9: # Toleranz für leichte Abweichungen/Lücken
             logger_get_multi.warning(f"{log_prefix}: Fetched significantly fewer non-warmup candles ({non_warmup_count}) than expected ({expected_backtest_candles}). Potential data gaps?")


        # --- Mark 'is_complete' ---
        if 'is_complete' not in df_base.columns: df_base['is_complete'] = True
        current_utc_time_for_flag = datetime.now(timezone.utc)
        if not df_base.empty and fetch_end >= current_utc_time_for_flag - timedelta(minutes=interval_minutes):
             try:
                  last_idx = df_base.index[-1]
                  if last_idx >= fetch_end - timedelta(minutes=interval_minutes):
                       df_base.loc[last_idx, 'is_complete'] = False
                       logger_get_multi.info(f"Marked last fetched candle at {last_idx} as potentially incomplete.")
                  else:
                       logger_get_multi.debug(f"Last candle {last_idx} ends before incomplete threshold, not marked.")
             except Exception as e_flag:
                  logger_get_multi.error(f"Error marking last candle incomplete: {e_flag}")

        # 5. HÖHERE TIMEFRAMES & ALIGNMENT (Optional)
        df_merged = df_base.copy()
        require_alignment = getattr(Z_config, 'require_timeframe_alignment', False)
        tf2_interval = getattr(Z_config, 'interval_int_2', None)
        tf3_interval = getattr(Z_config, 'interval_int_3', None)
        tf2_trend_col = f'trend_{tf2_interval}' if tf2_interval else None
        tf3_trend_col = f'trend_{tf3_interval}' if tf3_interval else None

        if tf2_trend_col: df_merged[tf2_trend_col] = 0
        if tf3_trend_col: df_merged[tf3_trend_col] = 0
        df_merged['all_trends_aligned'] = True

        if require_alignment:
             logger_get_multi.debug(f"{log_prefix}: Fetching/Calculating alignment trends...")
             # TF2
             if tf2_interval:
                  logger_get_multi.debug(f"{log_prefix}: Fetching data for TF2 ({tf2_interval})...")
                  # Verwende denselben Start/End wie Basisdaten, fetch_data batcht intern
                  df_tf2_raw = Backtest.fetch_data(symbol=symbol, interval=tf2_interval, end_time=fetch_end, start_time_force=fetch_start)
                  if df_tf2_raw is not None and not df_tf2_raw.empty:
                      try:
                          df_tf2_indicators = indicators.calculate_indicators(df_tf2_raw, calculate_for_alignment_only=True)
                          if df_tf2_indicators is not None and 'trend' in df_tf2_indicators:
                               resampled_trend_tf2 = df_tf2_indicators['trend'].resample(interval_str).ffill()
                               df_merged = df_merged.join(resampled_trend_tf2.rename(tf2_trend_col), how='left')
                               # Fülle NaNs VOR dem Cast zu int
                               df_merged[tf2_trend_col] = df_merged[tf2_trend_col].ffill().fillna(0).astype(int)
                               logger_get_multi.debug(f"{log_prefix}: TF2 trend merged.")
                          else: logger_get_multi.warning(f"{log_prefix}: Alignment trend calculation failed for {tf2_interval}.")
                      except AttributeError: logger_get_multi.error(f"{log_prefix}: indicators.calculate_indicators not found/imported.")
                      except Exception as indi_err: logger_get_multi.error(f"{log_prefix}: Error calculating TF2 indicators: {indi_err}")
                  else: logger_get_multi.warning(f"{log_prefix}: No data fetched for {tf2_interval}.")

             # TF3
             if tf3_interval:
                  logger_get_multi.debug(f"{log_prefix}: Fetching data for TF3 ({tf3_interval})...")
                  df_tf3_raw = Backtest.fetch_data(symbol=symbol, interval=tf3_interval, end_time=fetch_end, start_time_force=fetch_start)
                  if df_tf3_raw is not None and not df_tf3_raw.empty:
                       try:
                          df_tf3_indicators = indicators.calculate_indicators(df_tf3_raw, calculate_for_alignment_only=True)
                          if df_tf3_indicators is not None and 'trend' in df_tf3_indicators.columns:
                              resampled_trend_tf3 = df_tf3_indicators['trend'].resample(interval_str).ffill()
                              df_merged = df_merged.join(resampled_trend_tf3.rename(tf3_trend_col), how='left')
                              df_merged[tf3_trend_col] = df_merged[tf3_trend_col].ffill().fillna(0).astype(int)
                              logger_get_multi.debug(f"{log_prefix}: TF3 trend merged.")
                          else: logger_get_multi.warning(f"{log_prefix}: Alignment trend calculation failed for {tf3_interval}.")
                       except AttributeError: logger_get_multi.error(f"{log_prefix}: indicators.calculate_indicators not found/imported.")
                       except Exception as indi_err: logger_get_multi.error(f"{log_prefix}: Error calculating TF3 indicators: {indi_err}")
                  else: logger_get_multi.warning(f"{log_prefix}: No data fetched for {tf3_interval}.")

             # Alignment Flag Calculation
             if 'trend' not in df_merged.columns:
                 logger_get_multi.error(f"{log_prefix}: Base 'trend' column missing. Cannot calculate alignment.")
                 df_merged['all_trends_aligned'] = False
             elif tf2_trend_col and tf3_trend_col and all(c in df_merged for c in ['trend', tf2_trend_col, tf3_trend_col]):
                  try:
                      # Sicherstellen, dass Spalten existieren vor Zugriff
                      if all(col in df_merged.columns for col in ['trend', tf2_trend_col, tf3_trend_col]):
                          df_merged['trend'] = pd.to_numeric(df_merged['trend'], errors='coerce').fillna(0).astype(int)
                          df_merged[tf2_trend_col] = pd.to_numeric(df_merged[tf2_trend_col], errors='coerce').fillna(0).astype(int)
                          df_merged[tf3_trend_col] = pd.to_numeric(df_merged[tf3_trend_col], errors='coerce').fillna(0).astype(int)
                          trend_cols_align = ['trend', tf2_trend_col, tf3_trend_col]
                          all_equal = df_merged[trend_cols_align].apply(lambda x: x.nunique(dropna=False) == 1, axis=1)
                          trend_not_zero = df_merged['trend'] != 0
                          df_merged['all_trends_aligned'] = (all_equal & trend_not_zero)
                          logger_get_multi.debug(f"{log_prefix}: Alignment flag calculated (TF1 & TF2 & TF3).")
                      else:
                           logger_get_multi.warning(f"{log_prefix}: One or more trend columns missing for alignment check.")
                           df_merged['all_trends_aligned'] = False
                  except Exception as align_calc_err:
                       logger_get_multi.error(f"{log_prefix}: Error calculating alignment flag: {align_calc_err}")
                       df_merged['all_trends_aligned'] = False
             elif tf2_trend_col and tf2_trend_col in df_merged and tf3_interval is None: # Nur TF1 und TF2
                 try:
                     if all(col in df_merged.columns for col in ['trend', tf2_trend_col]):
                         df_merged['trend'] = pd.to_numeric(df_merged['trend'], errors='coerce').fillna(0).astype(int)
                         df_merged[tf2_trend_col] = pd.to_numeric(df_merged[tf2_trend_col], errors='coerce').fillna(0).astype(int)
                         trend_cols_align = ['trend', tf2_trend_col]
                         all_equal = df_merged[trend_cols_align].apply(lambda x: x.nunique(dropna=False) == 1, axis=1)
                         trend_not_zero = df_merged['trend'] != 0
                         df_merged['all_trends_aligned'] = (all_equal & trend_not_zero)
                         logger_get_multi.debug(f"{log_prefix}: Alignment flag calculated (TF1 & TF2).")
                     else:
                          logger_get_multi.warning(f"{log_prefix}: Trend or TF2 column missing for alignment check.")
                          df_merged['all_trends_aligned'] = False
                 except Exception as align_calc_err:
                      logger_get_multi.error(f"{log_prefix}: Error calculating TF1/TF2 alignment flag: {align_calc_err}")
                      df_merged['all_trends_aligned'] = False
             else: # Nicht alle benötigten TFs für Alignment vorhanden
                 logger_get_multi.info(f"{log_prefix}: Setting 'all_trends_aligned' to False (insufficient TFs for configured check).")
                 df_merged['all_trends_aligned'] = False


        # 6. Rückgabe des DataFrames
        logger_get_multi.info(f"{log_prefix}: Successfully prepared RAW data including WARMUP ({warmup_count}) and backtest ({non_warmup_count}) candles. Final shape: {df_merged.shape}.")

        # Gib alle relevanten Spalten zurück
        final_cols = ['open', 'high', 'low', 'close', 'volume', 'is_complete', 'is_warmup', 'symbol']
        if tf2_trend_col and tf2_trend_col in df_merged: final_cols.append(tf2_trend_col)
        if tf3_trend_col and tf3_trend_col in df_merged: final_cols.append(tf3_trend_col)
        if 'all_trends_aligned' in df_merged: final_cols.append('all_trends_aligned')

        cols_to_return = [col for col in final_cols if col in df_merged.columns]
        if df_merged.empty:
             logger_get_multi.warning(f"{log_prefix}: df_merged became empty before final column selection.")
             return None
        # Check for critical NaNs after warmup
        ohlcv_check = df_merged[['open', 'high', 'low', 'close', 'volume']].iloc[max(0, warmup_count):]
        if ohlcv_check.isnull().all().any():
            logger_get_multi.error(f"{log_prefix}: Critical NaNs detected in OHLCV data after warmup period. Returning None.")
            return None

        return df_merged[cols_to_return]

    except Exception as e:
        logger_get_multi.error(f"--- FATAL ERROR in get_multi_timeframe_data ({symbol}): {e} ---", exc_info=True)
        return None


# Ensure calculate_optimal_context_hours is defined
def calculate_optimal_context_hours() -> float:
    """
    Berechnet die empfohlenen Kontextstunden basierend auf den längsten
    Lookback-Perioden, die in Z_config für die aktivierten Indikatoren definiert sind.

    Dies wird im Backtest hauptsächlich verwendet, um sicherzustellen, dass beim
    Rekonstruieren einer offenen Position genügend historische Daten *vor*
    der Entry-Zeit geholt werden, damit die Indikatoren um diesen Zeitpunkt
    herum korrekt berechnet werden können.

    Die primäre Konsistenz der Indikatorhistorie für *laufende* Signalprüfungen
    wird durch die dynamische Berechnung von `required_candles` in der
    angepassten `get_multi_timeframe_data`-Funktion sichergestellt.

    Returns:
        float: Empfohlene Kontextdauer in Stunden (Minimum ist konfigurierbar,
               Standard-Fallback ist 24.0).
    """
    log_prefix = "calculate_optimal_context_hours"
    try:
        # --- Sammle alle relevanten Perioden aus Z_config ---
        periods = []
        logging.debug(f"{log_prefix}: Sammle Indikatorperioden aus Z_config...")

        # EMAs (Baseline ist typischerweise am längsten)
        periods.append(getattr(Z_config, 'ema_baseline_parameter', 50))

        # RSI
        if hasattr(Z_config, 'rsi_period'): periods.append(Z_config.rsi_period)

        # ATR
        if hasattr(Z_config, 'atr_period_parameter'): periods.append(Z_config.atr_period_parameter)

        # Bollinger Bands
        if getattr(Z_config, 'use_bb', False) and hasattr(Z_config, 'bb_period'):
             periods.append(Z_config.bb_period)

        # MACD (Slow Periode ist relevant für den Lookback)
        if getattr(Z_config, 'use_macd', False) and hasattr(Z_config, 'macd_slow_period'):
             periods.append(Z_config.macd_slow_period)

        # ADX
        if getattr(Z_config, 'use_adx', False) and hasattr(Z_config, 'adx_period'):
             periods.append(Z_config.adx_period)

        # Volume SMA
        if hasattr(Z_config, 'volume_sma'): periods.append(Z_config.volume_sma)

        # Momentum
        if hasattr(Z_config, 'momentum_lookback'): periods.append(Z_config.momentum_lookback)

        # Chaikin Money Flow
        if hasattr(Z_config, 'cmf_period'): periods.append(Z_config.cmf_period)

        # Advanced VWAP Periods (nimm die längste)
        if getattr(Z_config, 'use_advanced_vwap', False):
             vwap_periods = getattr(Z_config, 'advanced_vwap_periods', []) # Default leere Liste
             if vwap_periods and isinstance(vwap_periods, list):
                  numeric_vwap = [p for p in vwap_periods if isinstance(p, (int, float)) and p > 0]
                  if numeric_vwap:
                      periods.append(max(numeric_vwap))

        # Filtere ungültige Werte (None, nicht-numerisch, <= 0)
        valid_periods = [p for p in periods if isinstance(p, (int, float)) and p > 0]
        logging.debug(f"{log_prefix}: Gefundene valide Perioden: {valid_periods}")

        if not valid_periods:
            # Sicherer Standardwert, falls keine gültigen Perioden gefunden wurden
            # (z.B. weil alle Indikatoren deaktiviert sind oder Config fehlt)
            longest_indicator_period = 50
            logging.warning(f"{log_prefix}: Keine gültigen Indikatorperioden (>0) in Z_config gefunden. Verwende Standard-Maximalperiode: {longest_indicator_period}")
        else:
            longest_indicator_period = max(valid_periods)
            logging.debug(f"{log_prefix}: Längste relevante Indikatorperiode: {longest_indicator_period}")

        # --- Berechnung des Kontexts ---
        # Empfehlung: Mindestens 2x oder 3x die längste Periode für stabile Berechnungen
        # Mache den Faktor konfigurierbar mit Fallback
        context_multiplier = getattr(Z_config, 'context_multiplier', 3.0)
        min_context_periods = longest_indicator_period * context_multiplier
        logging.debug(f"{log_prefix}: Benötigte Mindest-Kontext-Perioden: {min_context_periods} ({longest_indicator_period} * {context_multiplier})")

        # Konvertiere Perioden (Anzahl Kerzen) in Stunden
        interval_str = getattr(Z_config, 'interval', '5m') # Basisintervall holen
        interval_minutes = Backtest.parse_interval_to_minutes(interval_str)

        if interval_minutes is None or interval_minutes <= 0:
             default_hours = getattr(Z_config, 'minimum_context_fallback_hours', 24.0)
             logging.warning(f"{log_prefix}: Intervall '{interval_str}' konnte nicht geparst werden. Verwende Fallback-Kontext: {default_hours} Stunden.")
             return float(default_hours)

        min_context_hours = (min_context_periods * interval_minutes) / 60.0

        # Stelle einen Mindestkontext sicher (konfigurierbar mit Fallback)
        minimum_context_fallback_hours = getattr(Z_config, 'minimum_context_fallback_hours', 24.0)
        recommended_hours = max(minimum_context_fallback_hours, min_context_hours)

        logging.info(f"{log_prefix}: Berechneter empfohlener Kontext: {recommended_hours:.2f} Stunden (basiert auf längster Periode {longest_indicator_period}, Faktor {context_multiplier}, min {minimum_context_fallback_hours}h)")
        # Gib immer einen Float zurück
        return float(recommended_hours)

    except AttributeError as ae:
         # Fängt Fehler ab, wenn ein erwartetes Attribut in Z_config fehlt
         default_hours = getattr(Z_config, 'minimum_context_fallback_hours', 24.0)
         logging.error(f"{log_prefix}: Fehler - Fehlendes Attribut in Z_config: {ae}. Verwende Fallback-Kontext: {default_hours} Stunden.")
         return float(default_hours)
    except Exception as e:
        default_hours = getattr(Z_config, 'minimum_context_fallback_hours', 24.0)
        logging.error(f"{log_prefix}: Allgemeiner Fehler bei der Berechnung der Kontextstunden: {e}. Verwende Fallback-Kontext: {default_hours} Stunden.", exc_info=True)
        return float(default_hours)
    

def debug_print_last_indicators(df: pd.DataFrame, symbol: str, label: str = "Indicator Snapshot"):
    """
    Gibt die Werte aller bekannten Indikatoren für die letzte vollständige Kerze aus,
    wenn Z_config.debug_indicators = True ist.

    Args:
        df (pd.DataFrame): DataFrame mit Kerzendaten und allen berechneten Indikatoren.
        symbol (str): Das Handelssymbol für den Kontext.
        label (str): Ein Label zur Identifizierung des Zeitpunkts des Snapshots (z.B. "Nach Berechnung").
    """
    # Prüfe den globalen Debug-Schalter in Z_config
    if not getattr(Z_config, 'debug_indicators', False):
        return # Debugging ist global deaktiviert

    # --- Validierung des DataFrames ---
    if df is None or df.empty:
        logging.warning(f"DEBUG SKIPPED ({symbol} - {label}): DataFrame ist None oder leer.")
        # Optional: Direkte Konsolenausgabe für den User, wenn Debugging an ist
        print(f"\nDEBUG SKIPPED ({symbol} - {label}): DataFrame ist None oder leer.\n")
        return

    # --- Finde die letzte vollständige Kerze ---
    last_complete_row = None
    timestamp_str = "N/A" # String für die Ausgabe

    # Prüfe, ob 'is_complete' Spalte existiert
    if 'is_complete' in df.columns:
        # Bevorzuge explizit als vollständig markierte Kerzen (True)
        complete_df = df[df['is_complete'] == True]
        if not complete_df.empty:
            last_complete_row = complete_df.iloc[-1]
            timestamp_str = str(last_complete_row.name) # Index (Timestamp) als String
        else:
            # Fallback: Nutze die letzte verfügbare Zeile, wenn keine als 'complete' markiert ist
            if not df.empty:
                last_complete_row = df.iloc[-1]
                timestamp_str = f"{str(last_complete_row.name)} (Fallback: Letzte Zeile, keine 'is_complete=True')"
                logging.warning(f"DEBUG ({symbol} - {label}): Keine 'is_complete=True' Kerze gefunden, nutze letzte Zeile ({timestamp_str}) für Debugging.")
                print(f"DEBUG WARNUNG ({symbol} - {label}): Keine 'is_complete=True' Kerze gefunden, nutze letzte Zeile ({timestamp_str}).")
            # else: df war initial nicht leer, aber complete_df schon -> sehr unwahrscheinlich
    elif not df.empty:
         # Wenn es keine 'is_complete'-Spalte gibt, nimm die letzte Zeile an
         last_complete_row = df.iloc[-1]
         timestamp_str = f"{str(last_complete_row.name)} (Fallback: Keine 'is_complete' Spalte)"
         logging.debug(f"DEBUG ({symbol} - {label}): Keine 'is_complete' Spalte gefunden, nutze letzte Zeile ({timestamp_str}) für Debugging.")

    # Wenn keine Zeile gefunden wurde (z.B. wenn df initial leer war)
    if last_complete_row is None:
        logging.warning(f"DEBUG SKIPPED ({symbol} - {label}): Konnte keine Zeile für die Debug-Ausgabe finden.")
        print(f"\nDEBUG SKIPPED ({symbol} - {label}): Konnte keine Zeile für die Debug-Ausgabe finden.\n")
        return

    # --- Beginne die Ausgabe ---
    print(f"\n{'='*20} DEBUG Indikatoren: {symbol} [{label}] @ {timestamp_str} {'='*20}")

    # --- Liste ALLER potenziell vorhandenen Indikator-Spalten ---
    # Diese Liste sollte alle Spalten enthalten, die von calculate_indicators generiert werden KÖNNTEN.
    all_possible_indicator_columns = [
        # Basis OHLCV etc.
        'open', 'high', 'low', 'close', 'volume', 'is_complete',
        # Basis Indikatoren
        'ema_fast', 'ema_slow', 'ema_baseline', 'rsi', 'atr', 'momentum',
        'volume_sma', 'volume_multiplier', 'high_volume', 'trend', 'trend_strength', 'trend_duration',
        'vwap', 'vwap_trend',
        'bb_upper', 'bb_middle', 'bb_lower', 'bb_signal',
        'macd_line', 'macd_signal_line', 'macd_histogram', 'macd_signal', 'macd_crossover',
        # Multi-Timeframe (dynamische Namen basierend auf Config)
         f'trend_{getattr(Z_config, "interval", "UNKNOWN")}',
         f'trend_{getattr(Z_config, "interval_int_2", "UNKNOWN")}',
         f'trend_{getattr(Z_config, "interval_int_3", "UNKNOWN")}',
        'all_trends_aligned',
        # Advanced Indikatoren
        'adx', 'adx_signal',
         *(f'vwap_{p}' for p in getattr(Z_config, 'advanced_vwap_periods', [20, 50, 100])), # Dynamische VWAP-Spalten
        'vwap_std', 'advanced_vwap_signal',
        'obv', 'obv_trend', 'cmf', 'cmf_signal',
        # Strategie-Signale & Kontext-Flags
        'signal', 'trigger',
        'is_context', 'is_observation', 'is_post_observation'
    ]

    output_lines = []
    max_key_len = 0 # Für schönere Ausrichtung der Ausgabe

    # Sammle nur die Indikatoren, die tatsächlich in der Zeile vorhanden sind
    found_indicators = {}
    for col in all_possible_indicator_columns:
        if col in last_complete_row.index: # Prüfe, ob die Spalte in der Series existiert
            value = last_complete_row.get(col) # Sicherer Zugriff auf den Wert
            found_indicators[col] = value
            if len(col) > max_key_len:
                max_key_len = len(col) # Finde die maximale Länge für die Ausrichtung

    # Gib die gefundenen Indikatoren formatiert aus
    if not found_indicators:
        print("  Keine der gesuchten Indikator-Spalten in der letzten Zeile gefunden.")
    else:
        for col, value in found_indicators.items():
            # Grundlegendes Typ-Formatierung für bessere Lesbarkeit
            if isinstance(value, (float, np.floating)):
                # Entscheide Präzision basierend auf Spaltennamen oder Wertgröße
                if 'price' in col or col in ['open','high','low','close','vwap','bb_upper','bb_middle','bb_lower','ema_fast','ema_slow','ema_baseline','macd_line','macd_signal_line','stop_loss_price','take_profit_price'] or abs(value) < 0.001 and value != 0:
                    formatted_value = f"{value:.6f}" # Höhere Präzision für Preise/Level/kleine Werte
                elif col in ['volume_sma', 'obv']:
                     formatted_value = f"{value:_.2f}".replace('_', ' ') # Tausendertrennzeichen, 2 Dezimalen
                elif col in ['rsi', 'adx', 'trend_strength', 'vwap_std']:
                    formatted_value = f"{value:.2f}" # 2 Dezimalen für % oder Standardabweichung
                else:
                    formatted_value = f"{value:.4f}" # Standard-Float-Präzision
            elif isinstance(value, (int, np.integer)):
                 # Tausendertrennzeichen für große Integer (z.B. Volumen)
                 if 'volume' in col or 'obv' in col:
                     formatted_value = f"{value:_}".replace('_', ' ')
                 else:
                     formatted_value = str(value) # Normale Integer-Darstellung
            elif isinstance(value, (bool, np.bool_)):
                formatted_value = str(value) # True / False
            elif pd.isna(value):
                formatted_value = "NaN" # Explizit NaN anzeigen
            else: # Strings oder andere Typen
                formatted_value = str(value)

            # Füge zur Ausgabeliste hinzu (linksbündig ausgerichtet)
            output_lines.append(f"  {col:<{max_key_len}} : {formatted_value}")

    # Drucke die gesammelten Zeilen auf die Konsole
    for line in output_lines:
        print(line)

    print(f"{'='*20} ENDE DEBUG: {symbol} [{label}] {'='*20}\n")