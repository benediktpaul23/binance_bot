
#Backtest.py
import os
import logging
import pandas as pd
import numpy as np
from binance.client import Client
from datetime import datetime, timedelta, timezone
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm 
import math
from utils.symbol_filter import should_observe_symbol
import time
import sys
from symbols import symbol_liste
from Z_config import (
    lookback_hours_parameter, interval_int, 
    ema_fast_parameter, ema_slow_parameter, ema_baseline_parameter,
    min_trend_strength_parameter,
    rsi_period, volume_sma, momentum_lookback,
)




# Initialize Binance Client
client = Client()

resultat = "resultat.csv"

# Füge am Anfang der Datei Backtest.py hinzu:
import threading
csv_write_lock = threading.Lock()

def append_to_csv(results, file_name="resultat.csv"):
    """Append results to a CSV file with improved error handling, path creation, and thread safety"""
    try:
        # Thread-Lock zur Vermeidung von Race Conditions
        with csv_write_lock:
            # Create directory if it doesn't exist
            directory = os.path.dirname(os.path.abspath(file_name))
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                print(f"Created directory: {directory}")
            
            # Überprüfe die Daten vor dem Schreiben
            if not results:
                print(f"Warning: Empty results passed to append_to_csv for {file_name}")
                return
                
            # Debugging-Ausgabe für Trades
            if file_name == "24h_signals.csv":
                for trade in results:
                    print(f"Writing trade to CSV: {trade['symbol']} - {trade['signal']} - {trade['trigger']} - P&L: {trade.get('profit_loss', 'N/A')}")
            
            # Create DataFrame and save to CSV
            df_results = pd.DataFrame(results)
            df_results.to_csv(file_name, mode='a', index=False, header=not os.path.exists(file_name))
            
            # Verify file was written
            if os.path.exists(file_name):
                file_size = os.path.getsize(file_name)
                print(f"✅ Successfully wrote to {file_name} ({file_size} bytes)")
                
                # Validiere die geschriebenen Daten
                if file_name == "24h_signals.csv" and file_size > 0:
                    try:
                        df_check = pd.read_csv(file_name)
                        print(f"CSV contains {len(df_check)} entries")
                    except Exception as e:
                        print(f"Warning: Could not validate CSV content: {e}")
            else:
                print(f"❌ Failed to create {file_name}")
    except Exception as e:
        error_msg = f"Error appending results to CSV {file_name}: {e}"
        print(f"ERROR: {error_msg}")
        logging.error(error_msg)
        import traceback
        traceback.print_exc()

# In Backtest.py (ERSETZE die alte calculate_trends Funktion hiermit)
import utils.Backtest as Backtest # Stelle sicher, dass Z_config hier verfügbar ist oder übergebe Parameter
import Z_config # Importiere Z_config für die Parameter
# Importiere auch die Funktion für erweiterte Indikatoren
try:
    from utils.indicators import calculate_advanced_indicators_core
    has_advanced_indicators = True
except ImportError:
    has_advanced_indicators = False
    print("WARNUNG: advanced_indicators_file.py nicht gefunden, erweiterte Indikatoren werden übersprungen.")

# Importiere pandas und numpy, falls noch nicht geschehen
import pandas as pd
import numpy as np

from collections import deque

def check_rate_limit(weight=1):
    """
    Überwacht sowohl das Weight-Limit als auch das Verbindungslimit der Binance API.
    
    Args:
        weight: Das Gewicht der Anfrage (Standard: 1)
    """
    global minute_usage, connection_attempts, minute_limit, connection_limit
    
    # Initialisierung beim ersten Aufruf
    if 'minute_usage' not in globals():
        minute_usage = deque()         # Speichert (timestamp, weight)
        connection_attempts = deque()  # Speichert timestamps der Verbindungen
        minute_limit = 6000            # Binance Limit: 6000 weight pro Minute
        connection_limit = 300         # Binance Limit: 300 Verbindungen pro 5 Minuten
        print(f"Rate Limiter initialisiert: {minute_limit} Weight pro Minute, {connection_limit} Verbindungen pro 5 Minuten")
    
    now = datetime.now()
    
    # 1. Prüfe Weight-Limit (pro Minute)
    # Entferne Einträge, die älter als 1 Minute sind
    while minute_usage and (now - minute_usage[0][0]) > timedelta(minutes=1):
        minute_usage.popleft()
    
    # Berechne aktuelle Weight-Nutzung
    current_weight_usage = sum(w for _, w in minute_usage)
    
    # 2. Prüfe Verbindungslimit (pro 5 Minuten)
    # Entferne Verbindungsversuche, die älter als 5 Minuten sind
    while connection_attempts and (now - connection_attempts[0]) > timedelta(minutes=5):
        connection_attempts.popleft()
    
    # Berechne aktuelle Verbindungsnutzung
    current_connection_count = len(connection_attempts)
    
    # Sicherheitsmargen (95% des Limits)
    safe_weight_limit = int(minute_limit * 0.95)
    safe_connection_limit = int(connection_limit * 0.95)
    
    # Prüfe, ob eines der Limits überschritten würde
    weight_limited = current_weight_usage + weight > safe_weight_limit
    connection_limited = current_connection_count + 1 > safe_connection_limit
    
    if weight_limited or connection_limited:
        if weight_limited and connection_limited:
            limit_type = "Weight- und Verbindungslimit"
        elif weight_limited:
            limit_type = "Weight-Limit"
        else:
            limit_type = "Verbindungslimit"
        
        # Berechne Wartezeit basierend auf dem kritischeren Limit
        if weight_limited and minute_usage:
            oldest_weight = minute_usage[0][0]
            weight_wait_time = 60 - (now - oldest_weight).total_seconds() + 0.5  # +0.5s Sicherheitspuffer
        else:
            weight_wait_time = 0
            
        if connection_limited and connection_attempts:
            oldest_connection = connection_attempts[0]
            connection_wait_time = 300 - (now - oldest_connection).total_seconds() + 0.5  # +0.5s Sicherheitspuffer
        else:
            connection_wait_time = 0
        
        # Wähle die längere Wartezeit
        wait_time = max(weight_wait_time, connection_wait_time)
        
        if wait_time > 0:
            if current_weight_usage > safe_weight_limit * 0.8 or current_connection_count > safe_connection_limit * 0.8:
                print(f"{limit_type} fast erreicht. Weight: {current_weight_usage}/{safe_weight_limit}, "
                      f"Connections: {current_connection_count}/{safe_connection_limit}. "
                      f"Warte {wait_time:.1f}s...")
            time.sleep(wait_time)
            return check_rate_limit(weight)  # Rekursiver Aufruf nach dem Warten
    
    # Füge aktuelle Anfrage hinzu
    minute_usage.append((now, weight))
    connection_attempts.append(now)
    return True



logger_fetch = logging.getLogger(__name__) # Eigener Logger für diese Funktion


# --- Caching Konfiguration ---
CACHE_DIR = "data_cache"
CACHE_ENABLED = True # Schalter zum Aktivieren/Deaktivieren des Caching
CACHE_TTL_SECONDS = 6 * 3600 # 6 Stunden Gültigkeit für Cache-Dateien

# Stelle sicher, dass das Cache-Verzeichnis existiert
if CACHE_ENABLED and not os.path.exists(CACHE_DIR):
    try:
        os.makedirs(CACHE_DIR)
        logger_fetch.info(f"Cache directory created: {CACHE_DIR}")
    except OSError as e:
        logger_fetch.error(f"Could not create cache directory {CACHE_DIR}: {e}. Caching disabled.")
        CACHE_ENABLED = False


def _get_cache_key(symbol, interval, start_time, end_time):
    """Erstellt einen eindeutigen Schlüssel für die Cache-Datei."""
    # Verwende nur Jahr-Monat-Tag-Stunde für Start/Ende, um Schlüssel stabil zu halten
    start_str = start_time.strftime('%Y%m%d%H')
    end_str = end_time.strftime('%Y%m%d%H')
    # Füge eine Hash-Komponente hinzu, um Kollisionen weiter zu reduzieren (optional)
    # key_hash = hashlib.md5(f"{symbol}_{interval}_{start_str}_{end_str}".encode()).hexdigest()[:8]
    return f"{symbol}_{interval}_{start_str}_{end_str}" # Einfacher Schlüssel

def fetch_data(symbol, interval=None, lookback_hours=None, end_time=None, start_time_force=None, limit_force=None):
    """
    MODIFIED V3: Fetches historical data, integrates file-based caching.
    Handles API limits by fetching data in batches if necessary.

    Args:
        symbol (str): Trading symbol (e.g., 'BTCUSDT').
        interval (str, optional): Candlestick interval (e.g., '5m', '1h'). Defaults to Z_config.interval.
        lookback_hours (float, optional): DEPRECATED if start_time_force is used.
        end_time (datetime, optional): End time for data fetching (timezone-aware). Defaults to now(utc).
        start_time_force (datetime, optional): Specific start time (timezone-aware). Overrides lookback_hours.
        limit_force (int, optional): DEPRECATED. Limit calculation is now internal.

    Returns:
        pd.DataFrame or None: DataFrame with OHLCV data, 'is_complete', 'is_warmup', 'symbol', or None on failure.
    """
    global client # Zugriff auf den globalen Client
    if client is None:
        logger_fetch.error("fetch_data: Binance client not initialized.")
        return None

    log_prefix = f"fetch_data ({symbol}/{interval})"

    # --- Parameter Defaults and Validation ---
    if interval is None: interval = getattr(Z_config, 'interval', '5m')
    interval_minutes = parse_interval_to_minutes(interval) # Stelle sicher, dass diese Funktion verfügbar ist
    if interval_minutes is None or interval_minutes <= 0:
        logger_fetch.error(f"{log_prefix}: Invalid interval '{interval}'.")
        return None

    max_retries = 3
    retry_delay = 2
    max_api_limit = 1499 # Sicherheitsmarge

    # --- Timezone Handling ---
    if end_time is None: end_time = datetime.now(timezone.utc)
    elif end_time.tzinfo is None: end_time = end_time.replace(tzinfo=timezone.utc)
    else: end_time = end_time.astimezone(timezone.utc)

    # --- Determine Start Time and Total Candles ---
    start_time = None
    total_candles_needed = 0

    if start_time_force:
        if start_time_force.tzinfo is None: start_time = start_time_force.replace(tzinfo=timezone.utc)
        else: start_time = start_time_force.astimezone(timezone.utc)
        # logger_fetch.debug(f"{log_prefix}: Using forced start time: {start_time}")
        fetch_duration_seconds = (end_time - start_time).total_seconds()
        if fetch_duration_seconds > 0:
             total_candles_needed = math.ceil(fetch_duration_seconds / (interval_minutes * 60))
        else:
             total_candles_needed = 1
    elif lookback_hours is not None and lookback_hours > 0:
         start_time = end_time - timedelta(hours=lookback_hours)
         total_candles_needed = math.ceil(lookback_hours * 60 / interval_minutes)
         logger_fetch.warning(f"{log_prefix}: Using legacy lookback_hours={lookback_hours}. Expected candles: {total_candles_needed}")
    else:
        logger_fetch.error(f"{log_prefix}: No valid start_time_force or lookback_hours provided.")
        return None

    logger_fetch.info(f"{log_prefix}: Requesting data: {start_time} -> {end_time} ({total_candles_needed} candles est.)")

    # --- Caching Logic ---
    cache_key = None
    cache_file_path = None
    if CACHE_ENABLED:
        try:
            cache_key = _get_cache_key(symbol, interval, start_time, end_time)
            cache_file_path = os.path.join(CACHE_DIR, f"{cache_key}.pkl")

            if os.path.exists(cache_file_path):
                file_mod_time = os.path.getmtime(cache_file_path)
                file_age_seconds = time.time() - file_mod_time

                if file_age_seconds < CACHE_TTL_SECONDS:
                    logger_fetch.info(f"Cache HIT for key '{cache_key}'. Loading from {cache_file_path}")
                    try:
                        with open(cache_file_path, 'rb') as f:
                            cached_df = pickle.load(f)
                        # Basic validation of cached data
                        required_cols_cache = ['open', 'high', 'low', 'close', 'volume', 'is_complete', 'is_warmup', 'symbol']
                        if isinstance(cached_df, pd.DataFrame) and not cached_df.empty and all(col in cached_df.columns for col in required_cols_cache):
                            # Filter again for exact time range just in case cache is slightly broader
                            cached_df = cached_df[(cached_df.index >= start_time) & (cached_df.index <= end_time)]
                            logger_fetch.info(f"Loaded {len(cached_df)} candles from cache.")
                            return cached_df # Return cached data
                        else:
                            logger_fetch.warning(f"Invalid cache data found for key '{cache_key}'. Fetching fresh data.")
                            os.remove(cache_file_path) # Remove invalid cache file
                    except Exception as load_err:
                        logger_fetch.error(f"Error loading cache file {cache_file_path}: {load_err}. Fetching fresh data.")
                        try: os.remove(cache_file_path)
                        except OSError: pass
                else:
                    logger_fetch.info(f"Cache STALE for key '{cache_key}' (Age: {file_age_seconds:.0f}s > TTL: {CACHE_TTL_SECONDS}s). Fetching fresh data.")
                    try: os.remove(cache_file_path)
                    except OSError: pass
            else:
                 logger_fetch.info(f"Cache MISS for key '{cache_key}'. Fetching fresh data.")

        except Exception as cache_init_err:
             logger_fetch.error(f"Error during cache check for {symbol}: {cache_init_err}. Proceeding without cache.")

    # --- Fetching Logic (If Cache Miss or Stale) ---
    logger_fetch.info(f"{log_prefix}: Fetching from API...")
    all_klines = []
    current_fetch_end_time = end_time
    fetched_count = 0
    request_count = 0
    candles_to_fetch_total = total_candles_needed # Use calculated total

    while True: # Loop until enough candles are fetched or no more data
        request_count += 1
        limit_for_this_batch = max_api_limit # Maximize batch size
        end_ms = int(current_fetch_end_time.timestamp() * 1000)
        # Start time for the API call (Binance expects ms timestamp)
        # We don't need a start time if using limit correctly, but it can help avoid extra data
        # Let's fetch backwards from end_time using the limit
        start_ms = None # Let Binance determine start based on end and limit initially

        logger_fetch.debug(f"{log_prefix}: Batch {request_count}: Fetching up to {limit_for_this_batch} candles ending <= {current_fetch_end_time}")

        klines = None
        for attempt in range(max_retries):
            try:
                # check_rate_limit(weight=1) # Add rate limit check if necessary per batch
                klines = client.futures_klines( # Or client.get_klines depending on market
                    symbol=symbol, interval=interval, # startTime=start_ms, # Omit start initially
                    endTime=end_ms, limit=limit_for_this_batch
                )
                if klines:
                    logger_fetch.debug(f"{log_prefix}: Batch {request_count} attempt {attempt+1}: Fetched {len(klines)} klines.")
                else:
                     logger_fetch.debug(f"{log_prefix}: Batch {request_count} attempt {attempt+1}: Fetched 0 klines.")
                break # Success, exit retry loop
            except Exception as e:
                logger_fetch.warning(f"{log_prefix}: Batch {request_count} attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    logger_fetch.error(f"{log_prefix}: All attempts failed for batch {request_count}.")
                    # Return None if fetching fails critically
                    return None

        if not klines:
            logger_fetch.warning(f"{log_prefix}: Batch {request_count} returned no klines. Stopping fetch.")
            break # Stop fetching if API returns empty

        # Convert fetched klines timestamps immediately for checking
        fetched_times = [datetime.fromtimestamp(k[0]/1000, tz=timezone.utc) for k in klines]
        # Filter klines that are before or equal to the requested start time
        klines_in_range = [k for k, t in zip(klines, fetched_times) if t >= start_time]
        fetched_in_range_count = len(klines_in_range)

        # Prepend the valid klines
        all_klines = klines_in_range + all_klines
        fetched_count += fetched_in_range_count

        # Get the timestamp of the *earliest* kline fetched IN THIS BATCH
        if not klines_in_range: # If no klines were in the desired range this batch
            # This might happen if the API returns data slightly outside the end_ms constraint
            # Or if we've fetched past the start_time significantly.
            # Check the timestamp of the very first kline returned by API
            if klines:
                 earliest_api_time = datetime.fromtimestamp(klines[0][0]/1000, tz=timezone.utc)
                 if earliest_api_time < start_time:
                      logger_fetch.info(f"{log_prefix}: Earliest API kline ({earliest_api_time}) is before target start ({start_time}). Stopping fetch.")
                      break # We've fetched past our target start time
                 else:
                      # Continue fetching from before this batch's earliest kline
                      current_fetch_end_time = earliest_api_time - timedelta(milliseconds=1)
            else: # Should not happen if break above worked
                  break
        else: # klines_in_range is not empty
             earliest_fetched_time_in_range = datetime.fromtimestamp(klines_in_range[0][0]/1000, tz=timezone.utc)
             if earliest_fetched_time_in_range <= start_time:
                 logger_fetch.info(f"{log_prefix}: Earliest fetched candle in range ({earliest_fetched_time_in_range}) reached target start time ({start_time}). Stopping fetch.")
                 break
             # Update end_time for the next batch to be just before the earliest candle we just got
             current_fetch_end_time = earliest_fetched_time_in_range - timedelta(milliseconds=1)

        # Check if we have enough (consider a small buffer for safety)
        # This condition might be less reliable if there are large gaps in data.
        # The start_time check is the primary stop condition.
        # if fetched_count >= candles_to_fetch_total:
        #     logger_fetch.info(f"{log_prefix}: Fetched {fetched_count} candles, reaching target {candles_to_fetch_total}. Stopping fetch.")
        #     break

        # Small delay between batches
        time.sleep(0.1)

    # --- DataFrame Creation and Processing (after fetching) ---
    if not all_klines:
         logger_fetch.warning(f"{log_prefix}: No klines fetched in total from API.")
         # Try to load from cache one last time if path exists (maybe TTL was borderline)
         if CACHE_ENABLED and cache_file_path and os.path.exists(cache_file_path):
              logger_fetch.warning(f"{log_prefix}: Attempting recovery load from cache: {cache_file_path}")
              try:
                   with open(cache_file_path, 'rb') as f: cached_df = pickle.load(f)
                   if isinstance(cached_df, pd.DataFrame) and not cached_df.empty:
                        cached_df = cached_df[(cached_df.index >= start_time) & (cached_df.index <= end_time)]
                        if not cached_df.empty:
                             logger_fetch.info(f"Successfully recovered {len(cached_df)} candles from cache.")
                             return cached_df
              except Exception: pass # Ignore errors during recovery load
         return None # Return None if fetch and recovery failed

    logger_fetch.info(f"{log_prefix}: API Fetch completed. Total {len(all_klines)} klines fetched in {request_count} batches.")

    df = pd.DataFrame(all_klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'closeTime', 'quoteAssetVolume', 'numberOfTrades',
        'takerBuyBaseVol', 'takerBuyQuoteVol', 'ignore' ])
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='first')] # Remove duplicates
    df = df.sort_index()

    # Final precise filter for the requested range
    df = df[(df.index >= start_time) & (df.index <= end_time)]

    if df.empty:
         logger_fetch.warning(f"{log_prefix}: Data empty after final filtering {start_time} to {end_time}")
         return None

    final_candle_count = len(df)
    logger_fetch.info(f"{log_prefix}: Final DataFrame: {final_candle_count} candles from {df.index.min()} to {df.index.max()}.")
    # Check difference (optional)
    # if abs(final_candle_count - total_candles_needed) > max(10, total_candles_needed * 0.1):
    #      logger_fetch.warning(f"{log_prefix}: Difference between estimated ({total_candles_needed}) and fetched ({final_candle_count}) candles. Check gaps.")

    # Add 'is_complete', 'is_warmup', 'symbol' (as before)
    df['is_complete'] = True
    df['is_warmup'] = False # Will be set later if needed by calling function
    df['symbol'] = symbol
    # Mark last candle incomplete logic (as before)
    current_utc_time_for_flag = datetime.now(timezone.utc)
    if not df.empty:
        original_end_time_for_check = end_time
        interval_minutes_for_check = parse_interval_to_minutes(interval)
        if interval_minutes_for_check and original_end_time_for_check >= current_utc_time_for_flag - timedelta(minutes=interval_minutes_for_check):
             try:
                  last_idx = df.index[-1]
                  if last_idx >= original_end_time_for_check - timedelta(minutes=interval_minutes_for_check):
                       df.loc[last_idx, 'is_complete'] = False
                       logger_fetch.debug(f"{log_prefix}: Marked last candle at {last_idx} as potentially incomplete.")
             except Exception as e_flag: logger_fetch.error(f"{log_prefix}: Error marking last candle incomplete: {e_flag}")

    # --- Save to Cache ---
    if CACHE_ENABLED and cache_file_path and not df.empty:
        logger_fetch.info(f"Saving {len(df)} fetched candles to cache: {cache_file_path}")
        try:
            with open(cache_file_path, 'wb') as f:
                pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL) # Use highest protocol
        except Exception as save_err:
            logger_fetch.error(f"Error saving data to cache file {cache_file_path}: {save_err}")

    # Return relevant columns
    cols_to_return = ['open', 'high', 'low', 'close', 'volume', 'is_complete', 'is_warmup', 'symbol']
    missing_return_cols = [c for c in cols_to_return if c not in df.columns]
    if missing_return_cols:
        logger_fetch.error(f"{log_prefix}: Final DataFrame missing expected columns: {missing_return_cols}")
        for col in missing_return_cols: df[col] = np.nan # Add missing cols with NaN
        # Return None or the df with NaNs? Returning df might be better for structure.
        # return None
    return df[cols_to_return].copy()


def filter_historical_candles_by_trading_time(df):
    """
    Filtert einen DataFrame mit Candle-Daten, um nur die Candles zu behalten,
    die innerhalb des konfigurierten Handelszeitfensters liegen.
    
    Args:
        df: DataFrame mit Candle-Daten und Zeitindex
        
    Returns:
        DataFrame mit nur den Candles, die im Handelszeitfenster liegen
    """
    if df is None or df.empty:
        return df
    
    # Prüfe, ob Zeitfilterung aktiv ist
    time_filter_active = Z_config.time_filter_active if hasattr(Z_config, 'time_filter_active') else True
    
    # Wenn keine Zeitfilterung aktiv, gib den ursprünglichen DataFrame zurück
    if not time_filter_active:
        return df
    
    # Count total candles before filtering
    initial_count = len(df)
    
    # Konfigurierte Handelsparameter holen
    trading_days = Z_config.trading_days if hasattr(Z_config, 'trading_days') else [0, 1, 2, 3, 4]
    start_hour = Z_config.trading_start_hour if hasattr(Z_config, 'trading_start_hour') else 0
    start_minute = Z_config.trading_start_minute if hasattr(Z_config, 'trading_start_minute') else 0
    end_hour = Z_config.trading_end_hour if hasattr(Z_config, 'trading_end_hour') else 23
    end_minute = Z_config.trading_end_minute if hasattr(Z_config, 'trading_end_minute') else 59
    
    # Erstelle eine Maske für die Filterung nach Wochentag
    day_mask = df.index.weekday.isin(trading_days)
    
    # Erstelle eine Maske für die Filterung nach Uhrzeit
    # Füge diese Logik für Übernacht-Fenster hinzu
    if start_hour < end_hour:
        # Normaler Fall: Start bis Ende am selben Tag
        time_mask = (
            ((df.index.hour > start_hour) | 
            ((df.index.hour == start_hour) & (df.index.minute >= start_minute))) &
            ((df.index.hour < end_hour) |
            ((df.index.hour == end_hour) & (df.index.minute <= end_minute)))
        )
    else:
        # Übernacht-Fall: Start bis Ende über Mitternacht
        time_mask = (
            ((df.index.hour > start_hour) | 
            ((df.index.hour == start_hour) & (df.index.minute >= start_minute))) |
            ((df.index.hour < end_hour) |
            ((df.index.hour == end_hour) & (df.index.minute <= end_minute)))
        )
    
    # Kombiniere die Masken und filtern
    filtered_df = df[day_mask & time_mask].copy()
    
    # Protokolliere die Ergebnisse
    filtered_count = len(filtered_df)
    day_names = ['Montag', 'Dienstag', 'Mittwoch', 'Donnerstag', 'Freitag', 'Samstag', 'Sonntag']
    trading_days_names = [day_names[d] for d in trading_days]
    
    if initial_count > 0:
        kept_percent = (filtered_count / initial_count) * 100
        logging.info(f"Zeitfilterung: {filtered_count} von {initial_count} Candles ({kept_percent:.1f}%) liegen im Handelszeitfenster")
    
    logging.info(f"Handelszeitfenster: {trading_days_names}, {start_hour:02d}:{start_minute:02d}-{end_hour:02d}:{end_minute:02d}")
    
    if filtered_count == 0 and initial_count > 0:
        logging.warning(f"ACHTUNG: Keine Candles im konfigurierten Handelszeitfenster gefunden!")
    
    return filtered_df
    
#Backtest.py
# Hilfsfunktion zum Bestimmen des Startpunkts der aktuellen Candle
def get_current_candle_start(interval):
    now = datetime.now(timezone.utc)
    now = now.replace(second=0, microsecond=0)
    
    if interval == "15m":
        return now - timedelta(minutes=now.minute % 15)
    elif interval == "5m":
        return now - timedelta(minutes=now.minute % 5)
    elif interval == "1h":
        return now.replace(minute=0)
    else:
        return now


import pytz

def save_strategy_data(data, conditions_met, symbol, base_path="strategy_results"):
    """
    Speichert Strategiedaten mit allen Indikatoren in CSV-Dateien.
    Der Zeitstempel in full_data.csv wird nach Europe/Berlin (MESZ/MEZ) konvertiert.
    """
    # Stelle sicher, dass das Verzeichnis existiert
    os.makedirs(base_path, exist_ok=True)

    # Dateinamen definieren
    strategy_details_file = os.path.join(base_path, "strategy_details.csv")
    trade_log_file = os.path.join(base_path, "trade_log.csv")
    full_data_file = os.path.join(base_path, "full_data.csv")

    try:
        # --- Strategy Details vorbereiten und speichern ---
        # Optional: Zeitstempel hier auch schon in CEST?
        # Verwende pytz für korrekte Zeitzonenbehandlung
        cest_timezone = pytz.timezone('Europe/Berlin')
        now_cest = datetime.now(cest_timezone)

        details = {
            'timestamp': now_cest.strftime('%Y-%m-%d %H:%M:%S %Z%z'), # Zeitstempel in CEST mit Offset
            'symbol': symbol,
            # ... (Rest der 'details'-Felder wie in deinem Code) ...
             'trend_conditions_met': int(conditions_met.get('trend', 0)),
             'strength_conditions_met': int(conditions_met.get('strength', 0)),
             'duration_conditions_met': int(conditions_met.get('duration', 0)),
             'rsi_conditions_met': int(conditions_met.get('rsi', 0)),
             'volume_conditions_met': int(conditions_met.get('volume', 0)),
             'ut_bot_signals_met': int(conditions_met.get('ut_bot', 0)),
             'total_signals': int(conditions_met.get('total_signals', 0)),
             'min_rsi': round(float(min(conditions_met.get('rsi_values', [0]))), 2) if conditions_met.get('rsi_values') else 0,
             'max_rsi': round(float(max(conditions_met.get('rsi_values', [0]))), 2) if conditions_met.get('rsi_values') else 0,
             'mean_rsi': round(float(sum(conditions_met.get('rsi_values', [0]))/max(len(conditions_met.get('rsi_values', [1])), 1)), 2) if conditions_met.get('rsi_values') else 0,
             'buy_signals': int(len(data[data['signal'] == 'buy'])) if not data.empty else 0, # Check if data is empty
             'sell_signals': int(len(data[data['signal'] == 'sell'])) if not data.empty else 0,
             'exit_long_signals': int(len(data[data['signal'] == 'exit_long'])) if not data.empty else 0,
             'exit_short_signals': int(len(data[data['signal'] == 'exit_short'])) if not data.empty else 0,
             'stop_loss_triggers': int(len(data[data['trigger'] == 'stop_loss'])) if not data.empty and 'trigger' in data.columns else 0,
             'take_profit_triggers': int(len(data[data['trigger'] == 'take_profit'])) if not data.empty and 'trigger' in data.columns else 0,
             'strategy_triggers': int(len(data[data['trigger'] == 'strategy'])) if not data.empty and 'trigger' in data.columns else 0,
             'ut_bot_buy_signals': int(data['ut_buy'].sum() if not data.empty and 'ut_buy' in data.columns else 0),
             'ut_bot_sell_signals': int(data['ut_sell'].sum() if not data.empty and 'ut_sell' in data.columns else 0),
             'ut_bot_combined_signals': int(len(data[(data['ut_buy'] | data['ut_sell'])]) if not data.empty and 'ut_buy' in data.columns and 'ut_sell' in data.columns else 0),
             # Neue Metriken
             'adx_signals': int(data['adx_signal'].sum()) if not data.empty and 'adx_signal' in data.columns else 0,
             'vwap_signals': int(data['advanced_vwap_signal'].sum()) if not data.empty and 'advanced_vwap_signal' in data.columns else 0,
             'obv_positive_trends': int(len(data[data['obv_trend'] > 0])) if not data.empty and 'obv_trend' in data.columns else 0,
             'obv_negative_trends': int(len(data[data['obv_trend'] < 0])) if not data.empty and 'obv_trend' in data.columns else 0,
             'cmf_positive_signals': int(len(data[data['cmf_signal'] > 0])) if not data.empty and 'cmf_signal' in data.columns else 0,
             'cmf_negative_signals': int(len(data[data['cmf_signal'] < 0])) if not data.empty and 'cmf_signal' in data.columns else 0,
             # Alignment Metriken
             'aligned_5m_15m_count': int(len(data[data['trend_aligned'] == True])) if not data.empty and 'trend_aligned' in data.columns else 0,
             'aligned_all_timeframes_count': int(len(data[data['all_trends_aligned'] == True])) if not data.empty and 'all_trends_aligned' in data.columns else 0,
        }
        details_df = pd.DataFrame([details])
        details_df.to_csv(strategy_details_file, mode='a', header=not os.path.exists(strategy_details_file), index=False)

        # --- Trade Log speichern (Zeitstempel bleiben UTC für Berechnungen) ---
        if not data.empty:
            # Filtere Trades mit gültigem Signal
            trade_log = data[data['signal'].isin(['buy', 'sell', 'exit_long', 'exit_short'])].copy()
            if not trade_log.empty:
                # Füge Zeitstempel als Spalte hinzu (ist bereits UTC-informiert vom Index)
                trade_log['timestamp'] = trade_log.index
                trade_log['symbol'] = symbol

                # Trigger-Spalte sicherstellen
                if 'trigger' not in trade_log.columns:
                    trade_log['trigger'] = 'unknown'

                # Spaltenreihenfolge definieren
                essential_columns = ['timestamp', 'symbol', 'signal', 'trigger', 'open', 'high', 'low', 'close']
                # Füge optional vorhandene Preis-Spalten hinzu
                for col in ['entry_price', 'stop_loss_price', 'take_profit_price']:
                     if col in trade_log.columns:
                         essential_columns.append(col)

                other_columns = [col for col in trade_log.columns if col not in essential_columns]
                # Filtere nur existierende Spalten für die finale Liste
                final_columns = [col for col in essential_columns + other_columns if col in trade_log.columns]

                # Speichere Trade Log
                trade_log.to_csv(trade_log_file, mode='a',
                                 header=not os.path.exists(trade_log_file),
                                 index=False, # Index nicht speichern, Zeitstempel ist Spalte
                                 columns=final_columns) # Nur definierte Spalten speichern

        # --- Volle Daten speichern mit konvertiertem Zeitstempel ---
        if not data.empty:
            data_copy = data.copy()
            data_copy['symbol'] = symbol # Symbol als Spalte hinzufügen

            # --- Konvertiere Zeitstempel-Index nach 'Europe/Berlin' (CEST/CET) ---
            original_index_name = data_copy.index.name if data_copy.index.name else 'timestamp_utc' # Ursprünglichen Namen merken oder Standard
            try:
                cest_timezone = pytz.timezone('Europe/Berlin')

                # 1. Stelle sicher, dass der Index ein DatetimeIndex und UTC ist
                if isinstance(data_copy.index, pd.DatetimeIndex):
                    if data_copy.index.tz is None:
                        logging.warning(f"Index für {symbol} ist naiv. Nehme UTC an und lokalisiere.")
                        # Versuche vorsichtig zu lokalisieren, vermeide Fehler bei Duplikaten etc.
                        try:
                            data_copy.index = data_copy.index.tz_localize('UTC', ambiguous='infer', nonexistent='shift_forward')
                        except Exception as loc_err:
                             logging.error(f"Fehler beim Lokalisieren des Index für {symbol} als UTC: {loc_err}. Konvertierung übersprungen.")
                             # Fahre fort ohne Konvertierung, speichere ursprünglichen Index als Spalte
                             data_copy.reset_index(inplace=True)
                             data_copy.rename(columns={'index': original_index_name}, inplace=True)

                    elif str(data_copy.index.tz) != str(pytz.utc) and str(data_copy.index.tz) != 'UTC':
                         # Konvertiere zuerst nach UTC, falls er eine andere Zeitzone hat
                         logging.warning(f"Index für {symbol} ist in {data_copy.index.tz}, konvertiere zuerst nach UTC.")
                         data_copy.index = data_copy.index.tz_convert(pytz.utc)

                    # 2. Konvertiere von UTC nach Europe/Berlin (nur wenn vorherige Schritte erfolgreich waren)
                    if data_copy.index.tz is not None: # Prüfe erneut, ob Index jetzt Zeitzonen-informiert ist
                        data_copy.index = data_copy.index.tz_convert(cest_timezone)
                        logging.debug(f"Timestamp-Index für {symbol} erfolgreich nach {cest_timezone} konvertiert.")

                        # 3. Setze den konvertierten Index als Spalte zurück
                        data_copy.reset_index(inplace=True)
                        # Benenne die neue Spalte um (verwende _cest Suffix)
                        new_timestamp_col = 'timestamp_cest'
                        data_copy.rename(columns={'index': new_timestamp_col, original_index_name: new_timestamp_col}, inplace=True)


                        # 4. Optional: Zeitstempel als String formatieren (Pandas macht das oft gut)
                        # data_copy[new_timestamp_col] = data_copy[new_timestamp_col].dt.strftime('%Y-%m-%d %H:%M:%S %Z%z')

                    # (Else-Block für fehlgeschlagene Lokalisierung oben behandelt)

                else:
                     logging.warning(f"Index für {symbol} ist kein DatetimeIndex. Zeitstempel-Konvertierung für Index übersprungen.")
                     # Kein reset_index hier, da der Index kein Datum ist.

            except Exception as tz_error:
                print(f"FEHLER bei der Zeitstempel-Konvertierung nach CEST für {symbol}: {tz_error}")
                logging.error(f"FEHLER bei der Zeitstempel-Konvertierung nach CEST für {symbol}: {tz_error}", exc_info=True)
                # Fallback: Setze Index zurück ohne Konvertierung, wenn es ein DatetimeIndex war
                if isinstance(data_copy.index, pd.DatetimeIndex):
                     data_copy.reset_index(inplace=True)
                     # Versuche, die neue Spalte sinnvoll zu benennen
                     if 'index' in data_copy.columns:
                          data_copy.rename(columns={'index': original_index_name}, inplace=True)

            # --- Speichern ---
            # Debug-Ausgabe der Spalten vor dem Speichern
            # print(f"DEBUG: Spalten vor dem Speichern von {full_data_file}: {data_copy.columns.tolist()}")

            # Speichere alle Spalten. 'index=False', da der (potenziell konvertierte) Zeitstempel jetzt eine Spalte ist.
            data_copy.to_csv(full_data_file, mode='a',
                             header=not os.path.exists(full_data_file),
                             index=False)

            print(f"Daten gespeichert in {full_data_file} mit {len(data_copy.columns)} Spalten (Zeitstempel versucht nach CEST zu konvertieren)")

    except Exception as e:
        print(f"Fehler beim Speichern der Strategiedaten für {symbol}: {str(e)}")
        logging.error(f"Fehler beim Speichern der Strategiedaten für {symbol}: {str(e)}", exc_info=True)
        # Optional: Schreibe Fehler in eine separate Log-Datei
        try:
            with open(os.path.join(base_path, 'error_log.txt'), 'a') as f:
                f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Fehler save_strategy_data für {symbol} - {str(e)}\n")
        except Exception:
            pass # Vermeide Fehler im Fehler-Handler


def verify_dataframe(df, symbol):
    """Verify DataFrame structure and content"""
    if df is None or df.empty:
        print(f"DataFrame for {symbol} is empty or None")
        return False
        
    required_columns = ['close', 'high', 'low', 'volume', 'signal', 'is_complete']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Missing columns for {symbol}: {missing_columns}")
        print("Available columns:", df.columns.tolist())
        return False
    
    # Check for trades
    if 'signal' in df.columns:
        buy_signals = len(df[df['signal'] == 'buy'])
        sell_signals = len(df[df['signal'] == 'sell'])
        print(f"Buy signals: {buy_signals}, Sell signals: {sell_signals}")
        
        if buy_signals == 0 and sell_signals == 0:
            print("Warning: No buy or sell signals found in DataFrame")
            
    # Check for complete candles
    if 'is_complete' in df.columns:
        complete_candles = len(df[df['is_complete'] == True])
        incomplete_candles = len(df[df['is_complete'] == False])
        print(f"Complete candles: {complete_candles}, Incomplete candles: {incomplete_candles}")
    
    return True

"""

def setup_logging():
    log_filename = 'log.txt'  # Fixed filename as requested
    log_file = open(log_filename, 'w', encoding='utf-8')
    original_stdout = sys.stdout
    sys.stdout = Tee(sys.stdout, log_file)
    
    # Setup basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('backtest.log'),
            logging.StreamHandler()
        ]
    )
    
    return original_stdout, log_file
"""


import warnings
warnings.filterwarnings('ignore')

class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for file in self.files:
            file.write(obj)
            file.flush()

    def flush(self):
        for file in self.files:
            file.flush()

"""
def cleanup_logging(original_stdout, log_file):
    sys.stdout = original_stdout
    log_file.close()"""

def save_detailed_statistics(performance_df, backtest_start_time, current_time, 
                           lookback_hours_parameter, min_trend_strength_parameter, 
                           SYMBOLS, total_exits):
    """
    Save detailed backtest statistics to file with accurate exit type counting
    """
    stats_file = "backtest_statistics.txt"
    with open(stats_file, 'w') as f:
        f.write("Backtest Statistics\n")
        f.write("="*50 + "\n")
        f.write(f"Period: {backtest_start_time} to {current_time}\n")
        f.write(f"Lookback Hours: {lookback_hours_parameter}\n")
        f.write(f"Minimum Trend Strength: {min_trend_strength_parameter}%\n")
        f.write(f"Symbols analyzed: {len(SYMBOLS)}\n")
        f.write(f"Symbols traded: {len(performance_df)}\n")
        f.write(f"Total trades: {performance_df['total_trades'].sum()}\n")
        f.write(f"Average win rate: {performance_df['win_rate'].mean():.2%}\n")
        f.write(f"Total profit: {performance_df['total_profit'].sum():.2f}\n")
        f.write(f"Average profit per trade: {performance_df['avg_profit'].mean():.4f}\n")
        f.write(f"Total commission: {performance_df['total_commission'].sum():.4f}\n")
        f.write(f"Average slippage: {performance_df['avg_slippage'].mean():.6f}\n")
        
        # Write exit type statistics
        f.write("\nExit Types:\n")
        total_trades = performance_df['total_trades'].sum()
        
        # Calculate total exits for each type directly from performance_df
        exit_types = {
            'Stop Loss': performance_df['stop_loss_hits'].sum(),
            'Take Profit': performance_df['take_profit_hits'].sum(),
            'Signal/Strategy': performance_df['signal_exits'].sum(),
            'Backtest End': performance_df['backtest_end_exits'].sum() if 'backtest_end_exits' in performance_df.columns else 0
        }
        
        for exit_type, count in exit_types.items():
            percentage = (count / total_trades) * 100 if total_trades > 0 else 0
            f.write(f"{exit_type}: {count} ({percentage:.1f}%)\n")
        
        # Add section for detailed performance metrics
        f.write("\nDetailed Performance Metrics:\n")
        f.write("-"*50 + "\n")
        
        # Calculate aggregate metrics
        total_stop_loss_percentage = (exit_types['Stop Loss'] / total_trades) * 100 if total_trades > 0 else 0
        total_take_profit_percentage = (exit_types['Take Profit'] / total_trades) * 100 if total_trades > 0 else 0
        
        f.write(f"Stop Loss Hit Rate: {total_stop_loss_percentage:.2f}%\n")
        f.write(f"Take Profit Hit Rate: {total_take_profit_percentage:.2f}%\n")
        
        # Calculate Risk-Reward metrics
        avg_profit = performance_df[performance_df['total_profit'] > 0]['avg_profit'].mean() if not performance_df[performance_df['total_profit'] > 0].empty else 0
        avg_loss = performance_df[performance_df['total_profit'] < 0]['avg_profit'].mean() if not performance_df[performance_df['total_profit'] < 0].empty else 0
        
        if avg_loss != 0:
            risk_reward_ratio = abs(avg_profit / avg_loss)
            f.write(f"Risk-Reward Ratio: {risk_reward_ratio:.2f}\n")
        
        # Calculate average win rate across all symbols
        f.write(f"Average Win Rate: {performance_df['win_rate'].mean():.2%}\n")
        
        # Calculate average profit factor
        avg_profit_factor = performance_df['profit_factor'].replace([float('inf')], np.nan).mean()
        f.write(f"Average Profit Factor: {avg_profit_factor:.2f}\n")
        
        # Calculate average maximum drawdown
        f.write(f"Average Maximum Drawdown: {performance_df['max_drawdown'].mean():.2f}%\n")
        
        # Add additional statistics about SL/TP configuration
        f.write("\nPosition Management Configuration:\n")
        f.write("-"*50 + "\n")
        f.write(f"Standard SL/TP used: {Z_config.use_standard_sl_tp}\n")
        
        if Z_config.use_standard_sl_tp:
            f.write(f"Standard Stop Loss: {Z_config.stop_loss_parameter*100:.2f}%\n")
            f.write(f"Standard Take Profit: {Z_config.take_profit_parameter*100:.2f}%\n")
        else:
            f.write(f"Trailing Stop Settings:\n")
            f.write(f"  Activation Threshold: {Z_config.activation_threshold}%\n")
            f.write(f"  Trailing Distance: {Z_config.trailing_distance}%\n")
            f.write(f"  Take Profit Levels: {Z_config.take_profit_levels}\n")
            f.write(f"  Take Profit Size Percentages: {Z_config.take_profit_size_percentages}\n")
            f.write(f"  Breakeven Enabled: {Z_config.enable_breakeven}\n")
            f.write(f"  Trailing Take Profit Enabled: {Z_config.enable_trailing_take_profit}\n")


def validate_and_fix_config():
    """
    Überprüft und korrigiert die Konfiguration, um sicherzustellen, dass
    wenn use_standard_sl_tp = True ist, die erweiterten Position Management
    Parameter nicht aktiv sind.
    """
    if hasattr(Z_config, 'use_standard_sl_tp') and Z_config.use_standard_sl_tp:
        # Wenn Standard SL/TP aktiv ist, müssen die erweiterten Parameter deaktiviert werden
        if hasattr(Z_config, 'enable_breakeven') and Z_config.enable_breakeven:
            print("WARNUNG: use_standard_sl_tp=True, setze enable_breakeven=False")
            Z_config.enable_breakeven = False
            
        if hasattr(Z_config, 'enable_trailing_take_profit') and Z_config.enable_trailing_take_profit:
            print("WARNUNG: use_standard_sl_tp=True, setze enable_trailing_take_profit=False")
            Z_config.enable_trailing_take_profit = False
            
        print("Konfiguration validiert: Standard SL/TP aktiv, erweiterte Parameter deaktiviert")
        
        # Überprüfe, ob die binance parameter gesetzt sind
        if not hasattr(Z_config, 'stop_loss_parameter_binance') or not hasattr(Z_config, 'take_profit_parameter_binance'):
            print("WARNUNG: Standard SL/TP aktiv, aber stop_loss_parameter_binance oder take_profit_parameter_binance fehlen!")
            # Verwende Standardwerte, falls nicht konfiguriert
            if not hasattr(Z_config, 'stop_loss_parameter_binance'):
                Z_config.stop_loss_parameter_binance = 0.05
                print(f"  stop_loss_parameter_binance auf Standardwert {Z_config.stop_loss_parameter_binance} gesetzt")
            if not hasattr(Z_config, 'take_profit_parameter_binance'):
                Z_config.take_profit_parameter_binance = 0.05
                print(f"  take_profit_parameter_binance auf Standardwert {Z_config.take_profit_parameter_binance} gesetzt")
    else:
        # Wenn erweiterte Position Management verwendet wird, überprüfe die Konfiguration
        if not hasattr(Z_config, 'activation_threshold') or not hasattr(Z_config, 'trailing_distance'):
            print("WARNUNG: Erweitertes Position Management aktiv, aber einige Parameter fehlen!")
        
        # Protokolliere die aktiven erweiterten Features
        breakeven_active = Z_config.enable_breakeven if hasattr(Z_config, 'enable_breakeven') else False
        trailing_tp_active = Z_config.enable_trailing_take_profit if hasattr(Z_config, 'enable_trailing_take_profit') else False
        
        print(f"Konfiguration validiert: Verwende erweitertes Position Management")
        print(f"  - Breakeven aktiviert: {breakeven_active}")
        print(f"  - Trailing Take Profit aktiviert: {trailing_tp_active}")

import Z_config
import utils.backtest_strategy as backtest_strategy

def parse_interval(interval_str):
    """Extrahiert Zahl und Einheit aus einem Intervall-String wie '5m', '1h', '1d'"""
    number = int(''.join(filter(str.isdigit, interval_str)))
    unit = ''.join(filter(str.isalpha, interval_str))
    
    # Konvertiere in Minuten
    if unit == 'm':
        return number
    elif unit == 'h':
        return number * 60
    elif unit == 'd':
        return number * 60 * 24
    else:
        raise ValueError(f"Unbekannte Intervall-Einheit: {unit}")

# Beispiel
interval = "5m"
interval_int = parse_interval(interval)  # Ergibt 5 Minuten

# Oder für "1h"
interval = "1h"
interval_int = parse_interval(interval)  # Ergibt 60 Minuten

# Oder für "1d"
interval = "1d"
interval_int = parse_interval(interval)  # Ergibt 1440 Minuten

def parse_datetime(datetime_str):
    """
    Parse a datetime string into a timezone-aware datetime object.
    
    Args:
        datetime_str (str): Date string in format "YYYY-MM-DD HH:MM:SS"
        
    Returns:
        datetime: Timezone-aware datetime object
    """
    try:
        # Try to parse with various formats
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d",
            "%d-%m-%Y %H:%M:%S",
            "%d-%m-%Y %H:%M",
            "%d-%m-%Y"
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(datetime_str, fmt)
                # Make timezone aware
                dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                continue
                
        raise ValueError(f"Could not parse datetime string: {datetime_str}")
    except Exception as e:
        print(f"Error parsing datetime: {e}")
        return None
    


def parse_interval_to_minutes(interval_str):
    """
    Parse any standard interval string (like '1m', '5m', '1h', '1d', '1w', '1M', '1y')
    and convert it to minutes.
    
    Args:
        interval_str: A string in the format of a number followed by a unit 
                     (s=seconds, m=minutes, h=hours, d=days, w=weeks, M=months, y=years)
    
    Returns:
        Number of minutes corresponding to the interval, or None if parsing fails
    """
    import re
    
    # Default return if parsing fails
    default_minutes = 15
    
    try:
        # Extract number and unit from the interval string
        match = re.match(r'^(\d+)([smhdwMy])$', interval_str)
        if not match:
            print(f"Warning: Could not parse interval '{interval_str}', using default {default_minutes}m")
            return default_minutes
            
        value = int(match.group(1))
        unit = match.group(2)
        
        # Convert to minutes based on unit
        if unit == 's':
            return value / 60  # seconds to minutes (can be fractional)
        elif unit == 'm':
            return value  # already in minutes
        elif unit == 'h':
            return value * 60  # hours to minutes
        elif unit == 'd':
            return value * 24 * 60  # days to minutes
        elif unit == 'w':
            return value * 7 * 24 * 60  # weeks to minutes
        elif unit == 'M':
            return value * 30.44 * 24 * 60  # months to minutes (approximate)
        elif unit == 'y':
            return value * 365.25 * 24 * 60  # years to minutes (approximate)
        else:
            print(f"Warning: Unknown interval unit '{unit}', using default {default_minutes}m")
            return default_minutes
            
    except Exception as e:
        print(f"Error parsing interval '{interval_str}': {e}. Using default {default_minutes}m")
        return default_minutes


def get_data_with_context(symbol, observation_start, observation_end, context_hours=24):
    """
    Holt Daten für einen Beobachtungszeitraum mit zusätzlichem Kontext für die Indikatorberechnung
    
    Args:
        symbol: Trading-Symbol
        observation_start: Startzeit des Beobachtungszeitraums
        observation_end: Endzeit des Beobachtungszeitraums
        context_hours: Stunden an zusätzlichem Kontext vor dem Beobachtungszeitraum
    
    Returns:
        DataFrame mit zwei Markierungsspalten:
        - 'is_context': True für Kontext-Candles, False für Beobachtungszeitraum-Candles
        - 'is_observation': True für Candles im Beobachtungszeitraum, False für Kontext-Candles
    """
    # Berechne den erweiterten Startzeitpunkt mit Kontext
    extended_start = observation_start - timedelta(hours=context_hours)
    
    # Hole Daten für den erweiterten Zeitraum
    df = backtest_strategy.get_multi_timeframe_data(
        symbol, 
        start_time=extended_start,
        end_time=observation_end
    )
    
    if df is None or df.empty:
        return None
    
    # Markiere Kontext- und Beobachtungszeitraum-Candles
    df['is_context'] = df.index < observation_start
    df['is_observation'] = (df.index >= observation_start) & (df.index <= observation_end)
    
    return df

# Globales Dictionary zur Verfolgung aktiver Beobachtungszeiträume und Positionsstatus
active_observation_periods = {}  # Format: {symbol: {'active_period': period_data, 'has_position': bool}}

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
            logging.info(f"Neuer Beobachtungszeitraum für {symbol} registriert: "
                         f"{active_observation_periods[symbol]['active_period']['start_time']} bis "
                         f"{active_observation_periods[symbol]['active_period']['end_time']}")
            return True
        return False
        
    elif action == 'update':
        # Positionsstatus aktualisieren
        if has_position is not None:
            active_observation_periods[symbol]['has_position'] = has_position
            logging.info(f"Positionsstatus für {symbol} aktualisiert: "
                         f"{'Position geöffnet' if has_position else 'Keine Position'}")
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

import pickle

def get_cached_data(symbol, interval, lookback_hours, end_time=None):
    """
    Holt Daten aus dem Cache, wenn verfügbar, sonst von der API und speichert sie
    """
    # Cache-Verzeichnis erstellen, falls es nicht existiert
    cache_dir = 'data_cache'
    os.makedirs(cache_dir, exist_ok=True)
    
    # Cache-Schlüssel basierend auf den Parametern erstellen
    if end_time is None:
        end_time = datetime.now(timezone.utc)
    elif not end_time.tzinfo:
        end_time = end_time.replace(tzinfo=timezone.utc)
    
    # Nutze nur das Datum für den Cache-Schlüssel
    cache_key = f"{symbol}_{interval}_{lookback_hours}_{end_time.strftime('%Y%m%d')}"
    cache_file = f"{cache_dir}/{cache_key}.pkl"
    
    # Prüfe, ob Cache-Datei existiert und nicht zu alt ist
    if os.path.exists(cache_file):
        file_age = time.time() - os.path.getmtime(cache_file)
        cache_ttl = 6 * 3600  # 6 Stunden in Sekunden
        
        if file_age < cache_ttl:
            try:
                print(f"Lade Daten für {symbol} aus dem Cache")
                with open(cache_file, 'rb') as f:
                    df = pickle.load(f)
                
                # Prüfe, ob der Cache die erforderlichen Spalten hat
                required_cols = ['open', 'high', 'low', 'close', 'volume', 'is_complete']
                if all(col in df.columns for col in required_cols):
                    return df
                else:
                    print(f"Cache für {symbol} fehlen erforderliche Spalten, hole neu...")
            except Exception as e:
                print(f"Fehler beim Laden des Caches für {symbol}: {e}")
    
    # Daten von der API holen
    print(f"Hole neue Daten für {symbol}")
    df = fetch_data(symbol, interval, lookback_hours, end_time)
    
    # Daten cachen, wenn erfolgreich
    if df is not None and not df.empty:
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(df, f)
            print(f"{len(df)} Candles für {symbol} gecached")
        except Exception as e:
            print(f"Fehler beim Cachen der Daten für {symbol}: {e}")
    
    return df


from utils.symbol_filter import verify_observation_indicators, manage_observation_periods
from utils.parallel_processing import process_all_symbols_parallel

def main(custom_start_time=None):
    """
    Optimierte Hauptfunktion mit expliziter Dictionary-Übergabe und integrierter Dateilöschung.
    """
    original_stdout, log_file = setup_logging() # Logging zuerst

    try:
        total_start_time = time.time()
        logging.info("Starting main backtest function...")



        # === Dictionaries LOKAL initialisieren ===
        position_tracking = {}
        active_observation_periods = {}
        observed_symbols = {}
        logging.info("State dictionaries initialized locally in main.")
        # ========================================

        # --- Konfiguration und Setup ---
        logging.debug("Validating configuration...")
        validate_and_fix_config() # Deine Funktion

        SYMBOLS = list(set(symbol_liste)) # Lade deine Symbole
        logging.info(f"Loaded {len(SYMBOLS)} unique symbols.")

        # --- Zeitrahmen bestimmen ---
        logging.debug("Determining backtest timeframe...")
        if custom_start_time:
             # Stelle sicher, dass custom_start_time timezone-aware ist (UTC)
             if isinstance(custom_start_time, datetime):
                  if custom_start_time.tzinfo is None:
                       current_time = custom_start_time.replace(tzinfo=timezone.utc)
                       logging.warning("Custom start time was naive. Assuming UTC.")
                  else:
                       current_time = custom_start_time.astimezone(timezone.utc)
             else:
                  logging.error("Invalid custom_start_time provided. Using current time.")
                  current_time = datetime.now(timezone.utc)
             print(f"Using provided custom end time (as current_time): {current_time}")
             logging.info(f"Using provided custom end time (as current_time): {current_time}")
        elif Z_config.use_custom_backtest_datetime and Z_config.backtest_datetime:
            try:
                if isinstance(Z_config.backtest_datetime, datetime):
                     custom_time = Z_config.backtest_datetime
                     if not custom_time.tzinfo: custom_time = custom_time.replace(tzinfo=timezone.utc)
                else:
                     custom_time = parse_datetime(Z_config.backtest_datetime) # Nutze deine Parse-Funktion

                if custom_time:
                     current_time = custom_time.astimezone(timezone.utc) # Stelle UTC sicher
                     print(f"Using custom backtest datetime from Z_config: {current_time}")
                     logging.info(f"Using custom backtest datetime from Z_config: {current_time}")
                else:
                     current_time = datetime.now(timezone.utc)
                     print(f"Warning: Could not parse custom backtest datetime from Z_config. Using current time.")
                     logging.warning("Could not parse custom backtest datetime from Z_config. Using current time.")
            except Exception as e:
                 current_time = datetime.now(timezone.utc)
                 print(f"Error processing custom backtest datetime: {e}. Falling back to current time.")
                 logging.error(f"Error processing custom backtest datetime: {e}. Falling back to current time.")
        else:
            current_time = datetime.now(timezone.utc)
            print(f"Using current time as end time: {current_time}")
            logging.info(f"Using current time as end time: {current_time}")

        lookback_hours_parameter = Z_config.lookback_hours_parameter
        backtest_start_time = current_time - timedelta(hours=lookback_hours_parameter)

        # --- Konfiguration anzeigen ---
        print(f"\n{'='*50}")
        print(f"BACKTEST CONFIGURATION")
        print(f"{'='*50}")
        # ... (Dein Code zum Anzeigen der Konfiguration) ...
        print(f"Start Time: {backtest_start_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"End Time:   {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"Lookback:   {lookback_hours_parameter} hours")
        print(f"Symbols:    {len(SYMBOLS)}")
        # ... (Weitere Konfigurationsdetails) ...
        print(f"{'='*50}\n")
        logging.info("Backtest configuration displayed.")


        # --- CSV-Header initialisieren ---
        logging.debug("Initializing CSV headers...")
        initialize_csv_headers() # Deine Funktion


        # --- Früher Zeitfilter-Check ---
        logging.debug("Performing early time filter check...")
        early_time_filter_check(current_time) # Deine Funktion


        # --- Parallele Verarbeitung ---
        parallel_start_time = time.time()
        logging.info(f"Starting parallel processing for {len(SYMBOLS)} symbols...")

        # ***** HIER IST DIE KORREKTUR *****
        # Übergib die drei LOKAL initialisierten Dictionaries als Argumente
        all_trades_results = process_all_symbols_parallel(
            SYMBOLS,
            current_time,
            position_tracking,            # <-- Übergeben
            active_observation_periods,   # <-- Übergeben
            observed_symbols              # <-- Übergeben
        )
        # **********************************

        parallel_time = time.time() - parallel_start_time
        logging.info(f"Parallel processing finished in {parallel_time:.2f} seconds.")

        # --- Nachbearbeitung und Analyse ---
        # Der Zustand ist jetzt in den lokalen Dictionaries aktuell.

        # --- Finalen Zustand anzeigen ---
        print("\n" + "="*50)
        print("FINAL STATE AFTER BACKTEST")
        print("="*50)
        # Offene Positionen (aus lokalem dict)
        print("\n--- Open Positions ---")
        open_positions_found = False
        for symbol, status in position_tracking.items(): # Nutze lokales dict
            if status.get('position_open', False):
                open_positions_found = True
                pos_data = status.get('position_data', {})
                print(f"  {symbol}: Type={pos_data.get('position_type', '?')}, Entry={pos_data.get('entry_time', '?')}, Qty={pos_data.get('remaining_quantity', '?')}")
        if not open_positions_found: print("  None")
        # Aktive Beobachtungsperioden (aus lokalem dict)
        print("\n--- Active Observation Periods ---")
        active_periods_found = False
        for symbol, status in active_observation_periods.items():
             if status.get('active_period') or status.get('has_position'):
                 active_periods_found = True
                 period_info = status.get('active_period')
                 period_str = f"Start: {period_info['start_time']}, End: {period_info['end_time']}" if period_info else "No period data"
                 print(f"  {symbol}: Has Position: {status.get('has_position', False)}, Period: {period_str}")
        if not active_periods_found: print("  None")
        # Beobachtete Symbole (aus lokalem dict)
        print("\n--- Observed Symbols Log ---")
        observed_count = sum(1 for periods in observed_symbols.values() if periods)
        print(f"  {observed_count} symbols triggered observation periods.")
        print("="*50)
        logging.info("Final state dictionaries checked and displayed.")

        # --- Finale Performance-Zusammenfassung ---
        logging.info("Generating final performance summary...")
        # --- Gesamtausführungszeit ---
        total_time = time.time() - total_start_time
        print(f"\nTotal execution time: {total_time:.2f} seconds")
        if total_time > 0: print(f"  Parallel processing part: {parallel_time:.2f}s ({parallel_time/total_time*100:.1f}%)")
        logging.info(f"Total execution time: {total_time:.2f} seconds")

    except KeyboardInterrupt:
        print("\nScript interrupted by user.")
        logging.warning("Script interrupted by user.")
    except Exception as e:
        # Fange hier allgemeine Fehler ab, die während des Setups oder der Hauptlogik auftreten
        logging.error("An unexpected error occurred in main", exc_info=True)
        print(f"\nAn error occurred: {e}") # Gib den Fehler aus, der zum Abbruch führte
    finally:
        logging.info("Cleaning up logging...")
        cleanup_logging(original_stdout, log_file) # Deine Cleanup-Funktion



def setup_logging():
    # Your implementation here
    logging.basicConfig(
    level=logging.DEBUG,  # <--- ÄNDERE DIES AUF DEBUG
    format='%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s] - %(message)s', # Format ggf. anpassen (funcName hinzugefügt)
    handlers=[
        logging.FileHandler('backtest.log', mode='w'), # 'w' überschreibt bei jedem Lauf
        logging.StreamHandler() # Gibt auch auf Konsole aus
    ]
    )
    print("Logging setup.")
    return sys.stdout, None # Dummy return for example

def cleanup_logging(original_stdout, log_file):
    # Your implementation here
    print("Logging cleaned up.")
    pass
# --- End Logging Setup Placeholder ---


def initialize_csv_headers():
    # Deine Logik zum Schreiben der Header
    print("CSV headers initialized.")
    pass

def early_time_filter_check(current_time):
    # Deine Logik für den frühen Zeitfilter
    print("Early time filter check passed.")
    pass
                                           
if __name__ == "__main__":
    main()                        
                    
