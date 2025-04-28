# indicators.py
import pandas as pd
import numpy as np
import logging
import Z_config # Annahme: Z_config.py ist im Root oder PYTHONPATH
from datetime import datetime, timedelta, timezone
import pytz # Importiere pytz, falls für Zeitzonenkonvertierung benötigt
import traceback

# Logger für dieses Modul holen
logger = logging.getLogger(__name__)

# ==============================================================================
# Advanced Indikatoren (mit potentiellem Debugging)
# ==============================================================================


from tqdm import tqdm # Optional für Fortschrittsbalken

logger_iter_ema = logging.getLogger(__name__)

def _calculate_ema_on_slice(series, span):
    """
    Hilfsfunktion: Berechnet EMA auf einer Series (Slice).
    Verwendet jetzt min_periods=span für konsistentere Initialisierung.
    """
    # Behandle leere Eingabe-Series
    if series.empty:
        return pd.Series(dtype='float64')

    # Stelle sicher, dass span ein gültiger Integer >= 1 ist für min_periods
    safe_span = max(1, int(span))

    # Prüfe, ob genügend gültige (nicht-NaN) Datenpunkte für eine stabile EMA vorhanden sind
    if len(series.dropna()) < safe_span:
        # Wenn nicht genügend Daten, gib eine Series mit NaNs zurück
        return pd.Series(np.nan, index=series.index, dtype='float64')

    # Berechne die EMA mit adjust=False und min_periods=safe_span
    # adjust=False: Kompatibilität mit Standard-TA-Bibliotheken
    # min_periods=safe_span: Stellt sicher, dass die Berechnung erst beginnt,
    #                        wenn mindestens 'span' Datenpunkte verfügbar sind.
    return series.ewm(span=span, adjust=False, min_periods=safe_span).mean()

def calculate_ema_iterative(df, required_candles):
    """
    Berechnet NUR EMAs iterativ für jede Kerze im Backtest-Zeitraum
    mit einem festen Lookback-Fenster. Fügt *_iter Spalten hinzu.

    WARNUNG: Diese Methode ist DEUTLICH langsamer als die vollständig
             vektorisierte EMA-Berechnung!

    Args:
        df (pd.DataFrame): DataFrame, der bereits Standard-Indikatoren
                           und die Spalten 'close', 'is_warmup' enthält.
        required_candles (int): Die Anzahl der Kerzen, die für die
                                 Berechnung jeder Kerze zurückgeschaut werden soll.

    Returns:
        pd.DataFrame: Der ursprüngliche DataFrame mit zusätzlichen
                      *_iter EMA-Spalten oder None bei Fehlern.
    """
    log_prefix = "calculate_ema_iterative"
    logger_iter_ema.warning(f"{log_prefix}: STARTE ITERATIVE EMA BERECHNUNG - DIES WIRD LANGSAM SEIN!")

    if df is None or df.empty:
        logger_iter_ema.error(f"{log_prefix}: Input DataFrame ist leer.")
        return None
    if 'is_warmup' not in df.columns or 'close' not in df.columns:
        logger_iter_ema.error(f"{log_prefix}: Input DataFrame fehlen 'is_warmup' oder 'close' Spalten.")
        return None
    if required_candles <= 0:
        logger_iter_ema.error(f"{log_prefix}: Ungültige required_candles: {required_candles}")
        return None

    # Kopie, um Original nicht zu ändern (optional, je nach Aufrufkontext)
    df_results = df.copy()

    # Parameter holen
    ema_fast_p = getattr(Z_config, 'ema_fast_parameter', 7)
    ema_slow_p = getattr(Z_config, 'ema_slow_parameter', 60)
    ema_base_p = getattr(Z_config, 'ema_baseline_parameter', 50)

    # Neue Spalten initialisieren
    df_results['ema_fast_iter'] = np.nan
    df_results['ema_slow_iter'] = np.nan
    df_results['ema_baseline_iter'] = np.nan

    # Finde ersten Backtest-Index (wo is_warmup False ist)
    non_warmup_indices = df_results[~df_results['is_warmup']].index
    if non_warmup_indices.empty:
        logger_iter_ema.warning(f"{log_prefix}: Keine Nicht-Warmup Kerzen gefunden. Keine iterative Berechnung durchgeführt.")
        return df_results # Gib DF unverändert zurück

    first_backtest_iloc = df_results.index.get_loc(non_warmup_indices[0])
    logger_iter_ema.info(f"{log_prefix}: Starte iterative EMA-Schleife bei Index-Position {first_backtest_iloc} (Timestamp: {non_warmup_indices[0]})")

    # --- Iteriere über Backtest-Kerzen ---
    # tqdm fügt einen Fortschrittsbalken hinzu
    for i in tqdm(range(first_backtest_iloc, len(df_results)), desc=f"{log_prefix} EMA", unit="candle", leave=False):
        current_ts = df_results.index[i]

        # Slice definieren: N Kerzen zurück + aktuelle Kerze i
        # N = required_candles
        start_slice_iloc = max(0, i - required_candles)
        end_slice_iloc = i + 1 # Python slicing ist exklusiv am Ende
        current_slice_close = df_results.iloc[start_slice_iloc:end_slice_iloc]['close']

        # Berechne EMAs auf dem Slice und nimm den letzten Wert
        if len(current_slice_close.dropna()) >= 2: # Braucht min. 2 Punkte für EMA
            ema_fast_slice = _calculate_ema_on_slice(current_slice_close, ema_fast_p)
            if not ema_fast_slice.empty:
                 df_results.loc[current_ts, 'ema_fast_iter'] = ema_fast_slice.iloc[-1]

            ema_slow_slice = _calculate_ema_on_slice(current_slice_close, ema_slow_p)
            if not ema_slow_slice.empty:
                 df_results.loc[current_ts, 'ema_slow_iter'] = ema_slow_slice.iloc[-1]

            ema_base_slice = _calculate_ema_on_slice(current_slice_close, ema_base_p)
            if not ema_base_slice.empty:
                 df_results.loc[current_ts, 'ema_baseline_iter'] = ema_base_slice.iloc[-1]

    logger_iter_ema.warning(f"{log_prefix}: ITERATIVE EMA BERECHNUNG ABGESCHLOSSEN.")
    return df_results


def calculate_adx(high, low, close, period=21):
    """Calculate ADX with optional debug logging."""
    log_prefix = "calculate_adx"
    # Input validation
    if not isinstance(high, pd.Series) or not isinstance(low, pd.Series) or not isinstance(close, pd.Series):
        logger.error(f"{log_prefix}: Invalid input Series (must be pd.Series).")
        valid_index = next((s.index for s in [high, low, close] if isinstance(s, pd.Series) and not s.empty), None)
        return pd.Series(0.0, index=valid_index)
    if high.empty or low.empty or close.empty:
        logger.warning(f"{log_prefix}: Received empty Series.")
        return pd.Series(0.0, index=high.index)
    if not (len(high) == len(low) == len(close)):
        logger.warning(f"{log_prefix}: Input Series have different lengths. Aligning indices.")
        common_index = high.index.intersection(low.index).intersection(close.index)
        if common_index.empty:
             logger.error(f"{log_prefix}: No common index found after alignment.")
             valid_index = next((s.index for s in [high, low, close] if isinstance(s, pd.Series) and not s.empty), None)
             return pd.Series(0.0, index=valid_index)
        high = high.loc[common_index]
        low = low.loc[common_index]
        close = close.loc[common_index]
        if high.empty:
            logger.error(f"{log_prefix}: DataFrame empty after index alignment.")
            valid_index = next((s.index for s in [high, low, close] if isinstance(s, pd.Series) and not s.empty), None)
            return pd.Series(0.0, index=valid_index)

    logger.debug(f"{log_prefix}: Calculating ADX with period={period} on {len(high)} rows.")
    try:
        high_n = pd.to_numeric(high, errors='coerce').ffill()
        low_n = pd.to_numeric(low, errors='coerce').ffill()
        close_n = pd.to_numeric(close, errors='coerce').ffill()
        if high_n.isnull().any() or low_n.isnull().any() or close_n.isnull().any():
            logger.error(f"{log_prefix}: NaNs remain after ffill.")
            return pd.Series(0.0, index=high.index)

        close_shifted = close_n.shift(1).ffill()
        tr1 = high_n - low_n
        tr2 = abs(high_n - close_shifted)
        tr3 = abs(low_n - close_shifted)
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1, skipna=False).ffill()
        if true_range.isnull().any():
            logger.warning(f"{log_prefix}: NaNs found in True Range, filling with 0.")
            true_range.fillna(0, inplace=True)

        high_shifted = high_n.shift(1).ffill()
        low_shifted = low_n.shift(1).ffill()
        up_move = high_n - high_shifted
        down_move = low_shifted - low_n
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)

        min_p = max(1, period)
        smoothed_tr = true_range.ewm(span=period, min_periods=min_p, adjust=False).mean()
        smoothed_plus_dm = plus_dm.ewm(span=period, min_periods=min_p, adjust=False).mean()
        smoothed_minus_dm = minus_dm.ewm(span=period, min_periods=min_p, adjust=False).mean()

        smoothed_tr_safe = smoothed_tr.replace(0, np.nan)
        plus_di = (100 * smoothed_plus_dm / smoothed_tr_safe).fillna(0)
        minus_di = (100 * smoothed_minus_dm / smoothed_tr_safe).fillna(0)

        dx_sum = (plus_di + minus_di).replace(0, np.nan)
        dx = (100 * abs(plus_di - minus_di) / dx_sum).fillna(0)

        adx = dx.ewm(span=period, min_periods=min_p, adjust=False).mean()
        adx = adx.fillna(0)

        logger.debug(f"{log_prefix}: ADX calculation finished. Last value: {adx.iloc[-1]:.2f}" if not adx.empty else "N/A")
        return adx
    except Exception as e:
        logger.error(f"{log_prefix}: Error calculating ADX: {e}", exc_info=True)
        return pd.Series(0.0, index=high.index)

def advanced_vwap(data, periods=[20, 50, 100]):
    """Advanced VWAP calculation with optional debug logging."""
    log_prefix = "advanced_vwap"
    default_index = data.index if data is not None and isinstance(data.index, pd.Index) else None
    required_cols = ['high', 'low', 'close', 'volume']
    default_result = {
        **{f'vwap_{p}': pd.Series(0.0, index=default_index) for p in periods},
        'vwap_std': pd.Series(0.0, index=default_index)
    }

    if data is None or data.empty:
         logger.warning(f"{log_prefix}: Received empty DataFrame.")
         return default_result
    if not all(col in data.columns for col in required_cols):
        logger.error(f"{log_prefix}: Missing required columns: {required_cols}")
        return default_result

    logger.debug(f"{log_prefix}: Calculating VWAPs for periods={periods} on {len(data)} rows.")
    try:
        df = data.copy()
        for col in required_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
        df[['high', 'low', 'close']] = df[['high', 'low', 'close']].ffill()
        df['volume'] = df['volume'].ffill().fillna(0)
        if df[['high', 'low', 'close']].isnull().any().any():
             logger.error(f"{log_prefix}: NaNs remain after ffill.")
             return default_result

        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwaps = {}
        for period in periods:
             min_p = max(1, period)
             tp_vol = typical_price * df['volume']
             cumulative_tp_vol = tp_vol.rolling(window=period, min_periods=min_p).sum()
             cumulative_volume = df['volume'].rolling(window=period, min_periods=min_p).sum()
             vwap_series = cumulative_tp_vol / cumulative_volume.replace(0, np.nan)
             vwap_series = vwap_series.bfill().ffill().fillna(df['close'])
             vwaps[f'vwap_{period}'] = vwap_series
             logger.debug(f"{log_prefix}: VWAP_{period} calculated. Last value: {vwap_series.iloc[-1]:.6f}" if not vwap_series.empty else "N/A")

        primary_vwap_key = f'vwap_{periods[0]}'
        if primary_vwap_key in vwaps:
            vwaps['vwap_std'] = vwaps[primary_vwap_key].rolling(window=periods[0], min_periods=max(1, periods[0])).std().fillna(0)
            logger.debug(f"{log_prefix}: VWAP_std calculated. Last value: {vwaps['vwap_std'].iloc[-1]:.6f}" if not vwaps['vwap_std'].empty else "N/A")
        else:
             logger.warning(f"{log_prefix}: Primary VWAP '{primary_vwap_key}' not found for STD calculation.")
             vwaps['vwap_std'] = pd.Series(0.0, index=df.index)

        return vwaps
    except Exception as e:
        logger.error(f"{log_prefix}: Error: {e}", exc_info=True)
        default_index_error = data.index if data is not None else None
        return {
            **{f'vwap_{p}': pd.Series(data['close'] if 'close' in data and not data['close'].empty else 0.0, index=default_index_error) for p in periods},
            'vwap_std': pd.Series(0.0, index=default_index_error)
        }

def on_balance_volume(close, volume, period=11, symbol=""):
    """Calculate OBV with optional debug logging."""
    log_prefix = f"on_balance_volume ({symbol})"
    # --- Input Validation ---
    if not isinstance(close, pd.Series) or not isinstance(volume, pd.Series):
        logger.error(f"{log_prefix}: Inputs must be pandas Series.")
        return pd.Series(0.0, index=close.index if isinstance(close, pd.Series) else None)
    if close.empty or volume.empty:
        logger.warning(f"{log_prefix}: Received empty Series.")
        return pd.Series(0.0, index=close.index)

    # --- Index Alignment ---
    common_index = close.index.intersection(volume.index)
    if len(common_index) != len(close.index) or len(common_index) != len(volume.index):
         logger.warning(f"{log_prefix}: Indices differ, aligning.")
         close = close.reindex(common_index)
         volume = volume.reindex(common_index)
         if close.empty or volume.empty:
              logger.error(f"{log_prefix}: Empty series after index alignment.")
              return pd.Series(0.0, index=common_index)

    logger.debug(f"{log_prefix}: Calculating OBV on {len(close)} rows.")
    if not close.empty: # Sicherstellen, dass der Index existiert
        logger.debug(f"{log_prefix}: Calculation starts with first timestamp: {close.index[0]} using {len(close)} candles")
    try:
        # --- Data Preparation ---
        close_numeric = pd.to_numeric(close, errors='coerce').ffill()
        volume_numeric = pd.to_numeric(volume, errors='coerce').ffill().fillna(0)

        if close_numeric.isnull().any() or volume_numeric.isnull().any():
             logger.error(f"{log_prefix}: Unfillable NaNs remain after ffill.")
             valid_index = close_numeric.first_valid_index()
             if valid_index is None: valid_index = volume_numeric.first_valid_index()
             index_to_use = close.index if valid_index is None else close.loc[valid_index:].index
             return pd.Series(0.0, index=index_to_use)

        # --- OBV Calculation (Optimized) ---
        price_diff = close_numeric.diff()
        obv_diff = volume_numeric * np.sign(price_diff).fillna(0)
        obv = obv_diff.cumsum()
        obv = obv - obv.iloc[0] if not obv.empty else obv

        logger.debug(f"{log_prefix}: OBV calculation finished. Last value: {obv.iloc[-1]:.2f}" if not obv.empty else "N/A")
        if getattr(Z_config, 'debug_obv', False) and not obv.empty:
             # Detailliertes Logging auf Wunsch
             logger.info(f"\n===== OBV DEBUG ({symbol}) =====")
             logger.info(f"Data length: {len(close_numeric)} candles")
             logger.info(f"Period: {close_numeric.index[0]} to {close_numeric.index[-1]}")
             logger.info(f"First 3 OBV values: {obv.head(3).tolist()}")
             logger.info(f"Last 3 OBV values: {obv.tail(3).tolist()}")
             if len(obv) >= 2:
                  last_diff = obv.iloc[-1] - obv.iloc[-2]
                  obv_trend_debug = np.sign(last_diff)
                  logger.info(f"Last OBV diff: {last_diff:.2f}, Trend: {int(obv_trend_debug)}")
             else: logger.info("Not enough data for OBV trend calculation.")
             logger.info(f"===== END OBV DEBUG ({symbol}) =====\n")

        return obv

    except Exception as e:
        logger.error(f"{log_prefix}: Error: {e}", exc_info=True)
        return pd.Series(0.0, index=close.index)

def chaikin_money_flow(high, low, close, volume, period=20):
    """Calculate CMF with optional debug logging."""
    log_prefix = "chaikin_money_flow"
    # --- Input Validation ---
    required_series = [high, low, close, volume]
    valid_indices = [s.index for s in required_series if isinstance(s, pd.Series) and not s.empty]
    if not valid_indices:
        logger.error(f"{log_prefix}: No valid input Series found.")
        return pd.Series(dtype='float64')
    base_index = valid_indices[0]

    if not all(isinstance(s, pd.Series) for s in required_series):
        logger.error(f"{log_prefix}: All inputs must be pandas Series.")
        return pd.Series(0.0, index=base_index)
    if any(s.empty for s in required_series):
        logger.warning(f"{log_prefix}: Received one or more empty Series.")
        return pd.Series(0.0, index=base_index)

    # Align indices
    common_index = base_index
    for s in required_series[1:]: common_index = common_index.intersection(s.index)
    if common_index.empty:
        logger.error(f"{log_prefix}: No common index found among input Series.")
        return pd.Series(0.0, index=base_index)
    if len(common_index) < len(base_index):
        logger.warning(f"{log_prefix}: Input Series indices differed. Aligning.")
        high, low, close, volume = (s.loc[common_index] for s in required_series)
        if high.empty:
            logger.error(f"{log_prefix}: Empty series after index alignment.")
            return pd.Series(0.0, index=base_index)

    logger.debug(f"{log_prefix}: Calculating CMF with period={period} on {len(high)} rows.")
    try:
        # --- Data Preparation ---
        high_n = pd.to_numeric(high, errors='coerce').ffill()
        low_n = pd.to_numeric(low, errors='coerce').ffill()
        close_n = pd.to_numeric(close, errors='coerce').ffill()
        volume_n = pd.to_numeric(volume, errors='coerce').ffill().fillna(0)

        if any(s.isnull().any() for s in [high_n, low_n, close_n]):
             logger.error(f"{log_prefix}: Unfillable NaNs remain in HLC series after ffill.")
             return pd.Series(0.0, index=high.index)

        # --- CMF Calculation ---
        mfm_numerator = ((close_n - low_n) - (high_n - close_n))
        hl_range_safe = (high_n - low_n).replace(0, np.nan)
        money_flow_multiplier = (mfm_numerator / hl_range_safe).fillna(0)
        money_flow_volume = money_flow_multiplier * volume_n

        min_p = max(1, period)
        sum_mfv = money_flow_volume.rolling(window=period, min_periods=min_p).sum()
        sum_vol = volume_n.rolling(window=period, min_periods=min_p).sum()
        sum_vol_safe = sum_vol.replace(0, np.nan)
        cmf = (sum_mfv / sum_vol_safe).fillna(0)

        logger.debug(f"{log_prefix}: CMF calculation finished. Last value: {cmf.iloc[-1]:.4f}" if not cmf.empty else "N/A")
        return cmf
    except Exception as e:
        logger.error(f"{log_prefix}: Error: {e}", exc_info=True)
        return pd.Series(0.0, index=high.index)

def calculate_advanced_indicators_core(df):
    """Core function calculating advanced indicators with debug logs."""
    log_prefix = "calculate_adv_indicators"
    try:
        if df is None or df.empty:
            logger.warning(f"{log_prefix}: Received None or empty DataFrame.")
            return df

        df_copy = df.copy()
        # Symbol extrahieren für spezifischere Logs, falls vorhanden
        symbol_log = df_copy['symbol'].iloc[0] if 'symbol' in df_copy.columns and not df_copy.empty else "UNKNOWN"
        log_prefix = f"calculate_adv_indicators ({symbol_log} - DF ends {df_copy.index[-1] if not df_copy.empty else 'N/A'})"
        logger.debug(f"{log_prefix}: Starting calculation on {len(df_copy)} rows.")

        # --- Spalten validieren & NaN Handling ---
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df_copy.columns]
        if missing_cols:
            logger.error(f"{log_prefix}: Missing required columns {missing_cols}. Returning original.")
            return df
        for col in required_cols: df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
        df_copy[required_cols] = df_copy[required_cols].ffill()
        df_copy['volume'] = df_copy['volume'].fillna(0)
        if df_copy[required_cols].isnull().any().any():
            logger.error(f"{log_prefix}: NaNs remain after fill. Returning original.")
            return df

        # --- Berechnungen ---
        # ADX
        if getattr(Z_config, 'use_adx', True):
            adx_period = getattr(Z_config, 'adx_period', 21)
            df_copy['adx'] = calculate_adx(df_copy['high'], df_copy['low'], df_copy['close'], period=adx_period)
            adx_threshold = getattr(Z_config, 'adx_threshold', 25)
            df_copy['adx_signal'] = (df_copy['adx'] > adx_threshold).astype(int)
        else: df_copy['adx'], df_copy['adx_signal'] = 0.0, 0

        # Advanced VWAP
        if getattr(Z_config, 'use_advanced_vwap', True):
            vwap_periods = getattr(Z_config, 'advanced_vwap_periods', [20, 50, 100])
            vwap_results = advanced_vwap(df_copy, periods=vwap_periods)
            for period_name, vwap_series in vwap_results.items(): df_copy[period_name] = vwap_series
            std_threshold = getattr(Z_config, 'advanced_vwap_std_threshold', 0.5)
            primary_vwap_col = f'vwap_{vwap_periods[0]}'
            if all(c in df_copy for c in ['close', primary_vwap_col, 'vwap_std']):
                df_copy['advanced_vwap_signal'] = ((df_copy['close'] > df_copy[primary_vwap_col]) & (df_copy['vwap_std'] < std_threshold)).astype(int)
            else: df_copy['advanced_vwap_signal'] = 0
        else:
             vwap_periods = getattr(Z_config, 'advanced_vwap_periods', [20, 50, 100])
             for p in vwap_periods: df_copy[f'vwap_{p}'] = df_copy['close']
             df_copy['vwap_std'] = 0.0
             df_copy['advanced_vwap_signal'] = 0

        # On-Balance Volume
    #   if getattr(Z_config, 'use_obv', True):
     #       obv_period = getattr(Z_config, 'obv_period', 11)
      #      df_copy['obv'] = on_balance_volume(df_copy['close'], df_copy['volume'], period=obv_period, symbol=symbol_log)
       #     if 'obv' in df_copy.columns: df_copy['obv_trend'] = np.sign(df_copy['obv'].diff().fillna(0)).astype(int)
        #    else: df_copy['obv_trend'] = 0
        #else: df_copy['obv'], df_copy['obv_trend'] = 0.0, 0

        # Chaikin Money Flow
        if getattr(Z_config, 'use_chaikin_money_flow', True):
            cmf_period = getattr(Z_config, 'cmf_period', 20)
            df_copy['cmf'] = chaikin_money_flow(df_copy['high'], df_copy['low'], df_copy['close'], df_copy['volume'], period=cmf_period)
            cmf_threshold = getattr(Z_config, 'cmf_threshold', 0.0)
            df_copy['cmf_signal'] = 0
            if 'cmf' in df_copy.columns:
                 df_copy.loc[df_copy['cmf'] > cmf_threshold, 'cmf_signal'] = 1
                 df_copy.loc[df_copy['cmf'] < -cmf_threshold, 'cmf_signal'] = -1
        else: df_copy['cmf'], df_copy['cmf_signal'] = 0.0, 0

        # --- Logging der letzten Werte ---
        if getattr(Z_config, 'debug_indicators', False):
            last_row_to_log = None
            log_time = "N/A"
            is_complete_available = 'is_complete' in df_copy.columns

            if not df_copy.empty:
                 if is_complete_available:
                     complete_candles = df_copy[df_copy['is_complete'] == True]
                     if not complete_candles.empty: last_row_to_log = complete_candles.iloc[-1]
                     else: last_row_to_log = df_copy.iloc[-1] # Fallback
                 else: last_row_to_log = df_copy.iloc[-1] # Fallback
                 if last_row_to_log is not None: log_time = last_row_to_log.name

            if last_row_to_log is not None:
                # Verwende bereits berechnete Variablen für Logs
                adx_threshold_log = getattr(Z_config, 'adx_threshold', 25)
                std_threshold_log = getattr(Z_config, 'advanced_vwap_std_threshold', 0.5)
                cmf_threshold_log = getattr(Z_config, 'cmf_threshold', 0.0)
                primary_vwap_col_log = f'vwap_{getattr(Z_config, "advanced_vwap_periods", [20])[0]}'

                logger.info(f"--- Advanced Indicator Values @ {log_time} ---")
                if 'adx' in df_copy.columns: logger.info(f"  ADX ({adx_period}): {last_row_to_log.get('adx', 'N/A'):.2f} (Threshold: {adx_threshold_log}, Signal: {last_row_to_log.get('adx_signal', 'N/A')})")
                if primary_vwap_col_log in df_copy.columns: logger.info(f"  Adv. VWAP ({primary_vwap_col_log}): {last_row_to_log.get(primary_vwap_col_log, 'N/A'):.6f} (STD: {last_row_to_log.get('vwap_std', 'N/A'):.6f}, STD Thr: {std_threshold_log}, Signal: {last_row_to_log.get('advanced_vwap_signal', 'N/A')})")
                if 'obv' in df_copy.columns: logger.info(f"  OBV: {last_row_to_log.get('obv', 'N/A'):_.2f} (Trend: {last_row_to_log.get('obv_trend', 'N/A')})".replace('_',' '))
                if 'cmf' in df_copy.columns: logger.info(f"  CMF ({cmf_period}): {last_row_to_log.get('cmf', 'N/A'):.4f} (Threshold: {cmf_threshold_log}, Signal: {last_row_to_log.get('cmf_signal', 'N/A')})")
                logger.info(f"--- End Advanced Indicator Log ---")
            else: logger.warning(f"{log_prefix}: Could not find row for logging.")

        logger.debug(f"{log_prefix}: Advanced calculation finished.")
        return df_copy

    except KeyError as ke:
         logger.error(f"{log_prefix}: Missing column: {ke}", exc_info=True)
         return df
    except Exception as e:
        logger.error(f"{log_prefix}: General error: {e}", exc_info=True)
        return df

# ==============================================================================
# Haupt-Indikatorfunktion (konsolidiert) mit DEBUG Logging
# ==============================================================================

def calculate_indicators(data, calculate_for_alignment_only=False):
    """
    Berechnet ALLE Indikatoren (Basis + Erweitert) für das gegebene DataFrame
    mit detailliertem Debug-Logging.
    """
    log_prefix = "calculate_indicators"
    try:
        if data is None or data.empty:
            logger.warning(f"{log_prefix}: Received None or empty data. Returning None.")
            return None

        df = data.copy()
        symbol_log = df['symbol'].iloc[0] if 'symbol' in df.columns and not df.empty else "UNKNOWN"
        log_prefix = f"calculate_indicators ({symbol_log})"
        df_start_log = df.index[0] if not df.empty else 'N/A'
        df_end_log = df.index[-1] if not df.empty else 'N/A'
        logger.debug(f"{log_prefix}: Starting. Data shape: {df.shape}. Time range: {df_start_log} to {df_end_log}. Alignment only: {calculate_for_alignment_only}")

        # --- Index und Daten-Validierung ---
        try:
            if not isinstance(df.index, pd.DatetimeIndex): df.index = pd.to_datetime(df.index)
            if df.index.tz is None: df.index = df.index.tz_localize('UTC')
            elif str(df.index.tz) != 'UTC': df.index = df.index.tz_convert('UTC')
        except Exception as e:
            logger.error(f"{log_prefix}: Failed index conversion/validation: {e}. Returning None.")
            return None

        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in numeric_columns if col not in df.columns]
        if missing_cols:
            logger.error(f"{log_prefix}: Missing required columns {missing_cols}. Returning None.")
            return None
        for col in numeric_columns: df[col] = pd.to_numeric(df[col], errors='coerce')

        nan_before = df[numeric_columns].isnull().sum().sum()
        if nan_before > 0:
            logger.debug(f"{log_prefix}: Filling {nan_before} NaNs in OHLCV using ffill.")
            df[numeric_columns] = df[numeric_columns].ffill()
            # Prüfung auf verbleibende NaNs nach ffill (ausser am Anfang)
            if df.iloc[1:][numeric_columns].isnull().any().any():
                 logger.error(f"{log_prefix}: Unfillable NaN values found in OHLCV columns after ffill (beyond first row). Cannot proceed.")
                 return None
        # --- Ende Validierung ---

        # --- Basis Indikatoren (Immer für Trend/Alignment) ---
        ema_fast_p = getattr(Z_config, 'ema_fast_parameter', 11)
        ema_slow_p = getattr(Z_config, 'ema_slow_parameter', 46)
        ema_base_p = getattr(Z_config, 'ema_baseline_parameter', 50)
        logger.debug(f"{log_prefix}: Calculating EMAs (Fast={ema_fast_p}, Slow={ema_slow_p}, Base={ema_base_p}).")
        # Verwende min_periods >= span für stabilere EMAs, fülle NaNs danach
        df['ema_fast'] = df['close'].ewm(span=ema_fast_p, min_periods=ema_fast_p, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=ema_slow_p, min_periods=ema_slow_p, adjust=False).mean()
        df['ema_baseline'] = df['close'].ewm(span=ema_base_p, min_periods=ema_base_p, adjust=False).mean()
        # Fülle initiale NaNs (bfill zuerst, dann ffill)
        for col in ['ema_fast', 'ema_slow', 'ema_baseline']: df[col] = df[col].bfill().ffill()
        logger.debug(f"{log_prefix}: EMAs calculated. Last values: Fast={df['ema_fast'].iloc[-1]:.6f}, Slow={df['ema_slow'].iloc[-1]:.6f}, Base={df['ema_baseline'].iloc[-1]:.6f}" if not df.empty else "N/A")

        df['trend'] = 0
        # Stelle sicher, dass EMAs nicht NaN sind vor dem Vergleich
        long_trend_cond = (df['ema_fast'] > df['ema_slow']) & (df['close'] > df['ema_baseline']) & df['ema_fast'].notna() & df['ema_slow'].notna() & df['ema_baseline'].notna()
        short_trend_cond = (df['ema_fast'] < df['ema_slow']) & (df['close'] < df['ema_baseline']) & df['ema_fast'].notna() & df['ema_slow'].notna() & df['ema_baseline'].notna()
        df.loc[long_trend_cond, 'trend'] = 1
        df.loc[short_trend_cond, 'trend'] = -1
        df['trend'] = df['trend'].astype(int)
        logger.debug(f"{log_prefix}: Trend calculated. Last value: {df['trend'].iloc[-1]}" if not df.empty else "N/A")

        # --- Volle Indikatoren (Nur wenn angefordert) ---
        if not calculate_for_alignment_only:
            logger.debug(f"{log_prefix}: Calculating FULL set of indicators...")

            # Trend Strength & Duration
            df['trend_strength'] = (abs(df['ema_fast'] - df['ema_slow']) / df['ema_slow'].replace(0, np.nan) * 100).fillna(0)
            trend_changes = df['trend'].diff().fillna(0) != 0
            trend_groups = trend_changes.cumsum()
            df['trend_duration'] = df.groupby(trend_groups).cumcount() + 1
            df.loc[df['trend'] == 0, 'trend_duration'] = 0
            logger.debug(f"{log_prefix}: Trend Strength/Duration calculated.")

            # RSI
            rsi_p = getattr(Z_config, 'rsi_period', 8)
            logger.debug(f"{log_prefix}: Calculating RSI (Period={rsi_p}).")
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).fillna(0)
            loss = -delta.where(delta < 0, 0).fillna(0)
            # Verwende EWM für Glättung (alternative RSI-Variante, oft als 'wilder' bezeichnet)
            # avg_gain = gain.ewm(alpha=1/rsi_p, min_periods=rsi_p, adjust=False).mean()
            # avg_loss = loss.ewm(alpha=1/rsi_p, min_periods=rsi_p, adjust=False).mean()
            # ODER bleibe bei SMA (rolling mean)
            avg_gain = gain.rolling(window=rsi_p, min_periods=rsi_p).mean()
            avg_loss = loss.rolling(window=rsi_p, min_periods=rsi_p).mean()
            rs = avg_gain / avg_loss.replace(0, np.inf)
            df['rsi'] = (100 - (100 / (1 + rs))).fillna(50)
            logger.debug(f"{log_prefix}: RSI calculated. Last value: {df['rsi'].iloc[-1]:.2f}" if not df.empty else "N/A")

            # ATR
            atr_p = getattr(Z_config, 'atr_period_parameter', 14)
            logger.debug(f"{log_prefix}: Calculating ATR (Period={atr_p}).")
            high_low = df['high'] - df['low']
            high_close_prev = abs(df['high'] - df['close'].shift(1))
            low_close_prev = abs(df['low'] - df['close'].shift(1))
            tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1, skipna=False)
            df['atr'] = tr.rolling(window=atr_p, min_periods=atr_p).mean().bfill().fillna(0)
            logger.debug(f"{log_prefix}: ATR calculated. Last value: {df['atr'].iloc[-1]:.6f}" if not df.empty else "N/A")

            # Volume Indicators
            vol_sma_p = getattr(Z_config, 'volume_sma', 30)
            logger.debug(f"{log_prefix}: Calculating Volume Indicators (SMA Period={vol_sma_p}).")
            df['volume_sma'] = df['volume'].rolling(window=vol_sma_p, min_periods=vol_sma_p).mean()
            df['volume_multiplier'] = (df['volume'] / df['volume_sma'].replace(0, np.nan)).fillna(0)
            df['high_volume'] = ((df['volume'] >= Z_config.min_volume) & (df['volume_multiplier'] >= Z_config.volume_multiplier / 1.2) & (df['volume'] > df['volume_sma'] * Z_config.volume_entry_multiplier)).astype(bool)
            logger.debug(f"{log_prefix}: Volume indicators calculated.")

            # Momentum
            momentum_p = getattr(Z_config, 'momentum_lookback', 15)
            logger.debug(f"{log_prefix}: Calculating Momentum (Period={momentum_p}).")
            df['momentum'] = df['close'].pct_change(periods=momentum_p).fillna(0)
            logger.debug(f"{log_prefix}: Momentum calculated.")

            # VWAP (Cumulative)
            logger.debug(f"{log_prefix}: Calculating Cumulative VWAP.")
            tp = (df['high'] + df['low'] + df['close']) / 3
            cumulative_volume = df['volume'].cumsum()
            df['vwap'] = ((tp * df['volume']).cumsum() / cumulative_volume.replace(0, np.nan)).bfill().fillna(df['close'])
            df['vwap_trend'] = np.sign(df['close'] - df['vwap']).astype(int)
            logger.debug(f"{log_prefix}: VWAP calculated.")

            # Bollinger Bands
            bb_p = getattr(Z_config, 'bb_period', 21)
            bb_dev_p = getattr(Z_config, 'bb_deviation', 2.0)
            logger.debug(f"{log_prefix}: Calculating Bollinger Bands (Period={bb_p}, Dev={bb_dev_p}).")
            bb_middle = df['close'].rolling(window=bb_p, min_periods=bb_p).mean()
            bb_std = df['close'].rolling(window=bb_p, min_periods=bb_p).std().fillna(0)
            df['bb_middle'] = bb_middle.bfill().fillna(df['close'])
            df['bb_upper'] = (df['bb_middle'] + (bb_std * bb_dev_p)).bfill().fillna(df['close'])
            df['bb_lower'] = (df['bb_middle'] - (bb_std * bb_dev_p)).bfill().fillna(df['close'])
            df['bb_signal'] = 0
            df.loc[df['close'] < df['bb_lower'], 'bb_signal'] = 1
            df.loc[df['close'] > df['bb_upper'], 'bb_signal'] = -1
            logger.debug(f"{log_prefix}: BBands calculated.")

            # MACD
            macd_f = getattr(Z_config, 'macd_fast_period', 12)
            macd_s = getattr(Z_config, 'macd_slow_period', 26)
            macd_sig = getattr(Z_config, 'macd_signal_period', 9)
            logger.debug(f"{log_prefix}: Calculating MACD (Periods={macd_f}/{macd_s}/{macd_sig}).")
            ema_fast_macd = df['close'].ewm(span=macd_f, adjust=False).mean()
            ema_slow_macd = df['close'].ewm(span=macd_s, adjust=False).mean()
            df['macd_line'] = ema_fast_macd - ema_slow_macd
            df['macd_signal_line'] = df['macd_line'].ewm(span=macd_sig, adjust=False).mean()
            df['macd_histogram'] = (df['macd_line'] - df['macd_signal_line']).fillna(0)
            df['macd_signal'] = np.sign(df['macd_histogram']).astype(int) # Histogramm-basiertes Signal
            prev_macd_line = df['macd_line'].shift(1)
            prev_macd_signal_line = df['macd_signal_line'].shift(1)
            crossed_above = (prev_macd_line < prev_macd_signal_line) & (df['macd_line'] > df['macd_signal_line'])
            crossed_below = (prev_macd_line > prev_macd_signal_line) & (df['macd_line'] < df['macd_signal_line'])
            df['macd_crossover'] = 0
            df.loc[crossed_above, 'macd_crossover'] = 1
            df.loc[crossed_below, 'macd_crossover'] = -1
            logger.debug(f"{log_prefix}: MACD calculated.")

            # --- Erweiterte Indikatoren ---
            logger.debug(f"{log_prefix}: Calling calculate_advanced_indicators_core.")
            df = calculate_advanced_indicators_core(df)
            if df is None: return None
            logger.debug(f"{log_prefix}: Finished calculate_advanced_indicators_core.")

        # --- Sicherstellen, dass alle erwarteten Spalten existieren ---
        # (Code bleibt wie in vorheriger Version)
        expected_cols = [
            'ema_fast', 'ema_slow', 'ema_baseline', 'trend', 'trend_strength', 'trend_duration',
            'rsi', 'atr', 'volume_sma', 'volume_multiplier', 'high_volume', 'momentum', 'vwap',
            'vwap_trend', 'bb_middle', 'bb_upper', 'bb_lower', 'bb_signal', 'macd_line',
            'macd_signal_line', 'macd_histogram', 'macd_signal', 'macd_crossover',
            'adx', 'adx_signal', *(f'vwap_{p}' for p in getattr(Z_config, 'advanced_vwap_periods', [20, 50, 100])),
            'vwap_std', 'advanced_vwap_signal', 'obv', 'obv_trend', 'cmf', 'cmf_signal' ]
        added_cols_count = 0
        for col in expected_cols:
            if col not in df.columns:
                added_cols_count += 1
                if col in ['trend','trend_duration','vwap_trend','bb_signal','macd_signal','macd_crossover','adx_signal','advanced_vwap_signal','obv_trend','cmf_signal']: df[col] = 0
                elif col == 'high_volume': df[col] = False
                else: df[col] = 0.0 # Default 0.0 für numerische
        if added_cols_count > 0: logger.debug(f"{log_prefix}: Added {added_cols_count} missing indicator columns with default values.")

        # --- Debug Logging für die letzte Kerze ---
        # (Kann hier bleiben oder durch externe Funktion wie debug_print_last_indicators ersetzt werden)
        if getattr(Z_config, 'debug_indicators', False) and not calculate_for_alignment_only:
            # (Code für Logging der letzten Kerze)
            pass # Dein bestehender Logging-Code kann hier stehen bleiben

        logger.debug(f"{log_prefix}: Finished. Returning DataFrame with shape {df.shape}")
        return df

    except KeyError as ke:
        logger.error(f"{log_prefix}: Missing column: {ke}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"{log_prefix}: General error: {str(e)}", exc_info=True)
        return None