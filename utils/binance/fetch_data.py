
#Backtest.py
import os
import logging
import pandas as pd
import numpy as np
from binance.client import Client
from datetime import datetime, timedelta, timezone
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm 
import sys
from collections import deque

client = Client()

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


#Backtest.py
def fetch_data(symbol, interval=None, lookback_hours=None, end_time=None):
    """Fetch historical candlestick data with proper time handling and support for large datasets"""
    max_retries = 2
    retry_delay = 1
    max_candles_per_request = 1400  # Binance erlaubt max. 1500 Candles pro Anfrage
    
    for attempt in range(max_retries):
        try:
            if end_time is None:
                # Wenn keine Zeit angegeben, aktuelle Zeit verwenden
                end_time = datetime.now(timezone.utc)
            elif not end_time.tzinfo:
                end_time = end_time.replace(tzinfo=timezone.utc)
            
            # WICHTIGE ÄNDERUNG: Statt zu runden, fügen wir ein vollständiges Intervall hinzu
            # um sicherzustellen, dass wir die aktuelle Candle bekommen
            original_end_time = end_time.replace(second=0, microsecond=0)
            
            # Parse the interval string to get the value and unit
            import re
            interval_match = re.match(r'(\d+)([mhdwM])', interval)
            
            if not interval_match:
                print(f"Unsupported interval format: {interval}")
                return None
                
            interval_value = int(interval_match.group(1))
            interval_unit = interval_match.group(2)
            
            # Calculate interval in minutes for expected candle count
            if interval_unit == 'm':
                interval_minutes = interval_value
                # Runde zur aktuellen Candle
                current_candle_start = original_end_time - timedelta(minutes=original_end_time.minute % interval_value)
                
                # Check if this candle is still in progress
                now = datetime.now(timezone.utc)
                if now < current_candle_start + timedelta(minutes=interval_value):
                    # We're still in this candle, so go back to the previous completed one
                    end_time = current_candle_start
                else:
                    # Current candle is complete, use it
                    end_time = current_candle_start + timedelta(minutes=interval_value)
            elif interval_unit == 'h':
                interval_minutes = interval_value * 60
                # Round to the beginning of the current hour-based interval
                rounded_end_time = original_end_time.replace(
                    hour=original_end_time.hour - original_end_time.hour % interval_value,
                    minute=0
                )
                # Add the interval to get the current candle
                end_time = rounded_end_time + timedelta(hours=interval_value)
            elif interval_unit == 'd':
                interval_minutes = interval_value * 24 * 60
                # For daily intervals, round to beginning of day
                rounded_end_time = original_end_time.replace(hour=0, minute=0)
                # Add days to get current candle
                end_time = rounded_end_time + timedelta(days=interval_value)
            elif interval_unit == 'w':
                interval_minutes = interval_value * 7 * 24 * 60
                # For weekly intervals, find the beginning of the week
                weekday = original_end_time.weekday()  # 0 is Monday, 6 is Sunday
                days_to_subtract = weekday
                rounded_end_time = (original_end_time - timedelta(days=days_to_subtract)).replace(hour=0, minute=0)
                # Add weeks to get current candle
                end_time = rounded_end_time + timedelta(weeks=interval_value)
            else:
                print(f"Unsupported interval unit: {interval_unit} in interval {interval}")
                return None
            
            # Berechne die ungefähre Anzahl der benötigten Candles
            start_time = end_time - timedelta(hours=lookback_hours)
            expected_candles = int(lookback_hours * 60 / interval_minutes)
            
            # Debug-Ausgabe für die Zeitberechnung
            print(f"Original time: {original_end_time}, Adjusted end time: {end_time}")
            print(f"Expected candles: ~{expected_candles} for {lookback_hours}h of {interval} data")
            
            # Wenn wir mehr als max_candles_per_request benötigen, müssen wir mehrere Anfragen stellen
            all_candles = []
            current_end_time = end_time
            
            while len(all_candles) < expected_candles:
                # For subsequent batches, use the oldest timestamp from previous batch
                if len(all_candles) > 0:
                    # Sort all_candles by timestamp to find the oldest one
                    sorted_candles = sorted(all_candles, key=lambda x: x[0])
                    oldest_timestamp = sorted_candles[0][0]  # First element is timestamp
                    
                    # Convert to datetime and subtract 1ms to avoid overlap
                    oldest_time = pd.to_datetime(oldest_timestamp, unit='ms', utc=True)
                    current_end_time = oldest_time - timedelta(milliseconds=1)
                    current_end_ms = int(current_end_time.timestamp() * 1000)
                    
                    # Calculate new start time based on current end time
                    current_start_time = current_end_time - timedelta(minutes=interval_minutes * max_candles_per_request)
                    current_start_ms = int(current_start_time.timestamp() * 1000)
                    
                    print(f"New batch range: {current_start_time} to {current_end_time}")
                else:
                    # For first batch, use the original calculated times
                    current_end_ms = int(current_end_time.timestamp() * 1000)
                    current_start_time = current_end_time - timedelta(minutes=interval_minutes * max_candles_per_request)
                    current_start_ms = int(current_start_time.timestamp() * 1000)
                
                # Vor jedem API-Aufruf - überprüfe Rate Limit (Weight=1 für klines/candlesticks)
                check_rate_limit(weight=1)
                
                # Make the API request
                batch_candles = client.futures_klines(
                    symbol=symbol,
                    interval=interval,
                    startTime=current_start_ms,
                    endTime=current_end_ms,
                    limit=max_candles_per_request
                )
                
                if not batch_candles:
                    print(f"No candles returned for {symbol} in this batch")
                    break
                
                # Füge die neuen Candles hinzu
                all_candles.extend(batch_candles)
                print(f"Fetched {len(batch_candles)} candles in this batch, total: {len(all_candles)}")
                
                # Wenn wir weniger als das Maximum bekommen haben, haben wir alle verfügbaren Daten
                if len(batch_candles) < max_candles_per_request:
                    break
                
                # Wenn wir die Gesamtstart-Zeit erreicht haben, sind wir fertig
                if current_start_time <= start_time:
                    break
                
                # Kurze Pause zwischen den Anfragen für Stabilität (optional, da der Rate Limiter das übernimmt)
                time.sleep(0.1)
            
            if not all_candles:
                print(f"No candles returned for {symbol}")
                return None
            
            # Erstelle DataFrame aus allen gesammelten Candles
            df = pd.DataFrame(all_candles, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'closeTime', 'quoteAssetVolume', 'numberOfTrades',
                'takerBuyBaseVol', 'takerBuyQuoteVol', 'ignore'
            ])
            
            # Convert numeric columns
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_columns] = df[numeric_columns].astype(float)
            
            # Convert timestamp to timezone-aware datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df['timestamp'] = df['timestamp'].dt.tz_convert('Europe/Berlin')  # Oder Ihre Zeitzone
            df.set_index('timestamp', inplace=True)
            
            # Sortiere nach Zeit (wichtig bei mehreren Anfragen)
            df = df.sort_index()
            
            # Filter to exact timeframe
            df = df[df.index >= start_time]
            df = df[df.index <= end_time]
            
            # Debug-Ausgabe für die erhaltenen Candles
            if not df.empty:
                print(f"Final dataset: {len(df)} candles for {symbol}, time range: {df.index[0]} to {df.index[-1]}")
                if len(df) >= 1:
                    # Prüfe, ob die letzte Candle die aktuelle Candle ist
                    last_candle_time = df.index[-1]
                    current_candle_start = get_current_candle_start(interval)
                    print(f"Last candle time: {last_candle_time}, Current candle start: {current_candle_start}")
                    print(f"Last candle is current: {last_candle_time == current_candle_start}")
            else:
                print(f"No data found for {symbol}")
            
            # CRITICAL FIX: Ensure is_complete flag is correctly set
            df['is_complete'] = True
            
            # KRITISCHE ÄNDERUNG: Korrektes Markieren der aktuellen Candle als unvollständig
            # Verwende die aktuelle Systemzeit als absolute Referenz
            now = datetime.now(timezone.utc)
            print(f"Current system time (UTC): {now}")
            
            # Berechne den Start der aktuellen (laufenden) Candle in UTC
            if interval == "5m":
                current_candle_start = now.replace(minute=now.minute - now.minute % 5, second=0, microsecond=0)
            elif interval == "15m":
                current_candle_start = now.replace(minute=now.minute - now.minute % 15, second=0, microsecond=0)
            elif interval == "1h":
                current_candle_start = now.replace(minute=0, second=0, microsecond=0)
            else:
                # Fallback for other intervals
                current_candle_start = now
                
            print(f"Current candle start (UTC): {current_candle_start}")
            
            # Konvertiere zu lokaler Zeitzone für den Vergleich mit dem DataFrame
            current_candle_start_local = current_candle_start.astimezone(df.index[0].tzinfo if not df.empty else None)
            print(f"Current candle start (local): {current_candle_start_local}")
            
            # VERBESSERTE LOGIK: Finde die aktuelle Candle und markiere sie als unvollständig
            if not df.empty:
                # Erst alle als vollständig markieren
                df['is_complete'] = True
                
                # Versuche, die aktuelle Candle zu finden
                current_candle_found = False
                
                # 1. Versuche exakte Übereinstimmung des Timestamps
                if current_candle_start_local in df.index:
                    df.loc[current_candle_start_local, 'is_complete'] = False
                    print(f"Marked current candle at {current_candle_start_local} as incomplete (exact match)")
                    current_candle_found = True
                else:
                    # 2. Finde die Candle, die den aktuellen Zeitraum abdeckt
                    interval_minutes = int(interval[:-1]) if interval[-1] == 'm' else 60
                    
                    for i in range(len(df)):
                        candle_time = df.index[i]
                        next_candle_time = candle_time + timedelta(minutes=interval_minutes)
                        
                        # Prüfe, ob 'now' zwischen dem Start dieser Candle und dem Start der nächsten Candle liegt
                        if candle_time <= now < next_candle_time:
                            df.iloc[i, df.columns.get_loc('is_complete')] = False
                            print(f"Found and marked current candle at {candle_time} as incomplete (contains current time {now})")
                            current_candle_found = True
                            break
                
                # 3. Wenn keine aktuelle Candle gefunden wurde, prüfe die letzte Candle
                if not current_candle_found:
                    last_candle_time = df.index[-1]
                    candle_age = now - last_candle_time
                    
                    # Nur als unvollständig markieren, wenn die letzte Candle relativ aktuell ist
                    max_age_seconds = interval_minutes * 60  # Maximales Alter in Sekunden
                    
                    if candle_age.total_seconds() <= max_age_seconds:
                        df.iloc[-1, df.columns.get_loc('is_complete')] = False
                        print(f"Last candle at {last_candle_time} is recent (age: {candle_age.total_seconds()}s), marking as incomplete")
                    else:
                        print(f"WARNING: Last candle at {last_candle_time} is too old (age: {candle_age.total_seconds()}s), all candles marked as complete")
            
            # Count complete and incomplete candles for verification
            if 'is_complete' in df.columns:
                complete_count = len(df[df['is_complete'] == True])
                incomplete_count = len(df[df['is_complete'] == False])
                print(f"Complete candles: {complete_count}, Incomplete candles: {incomplete_count}")
                
                # List incomplete candles for debugging
                if incomplete_count > 0:
                    incomplete_candles = df[df['is_complete'] == False].index.tolist()
                    print(f"Incomplete candle times: {incomplete_candles}")
            
            # Füge warmup-Flag hinzu wie im Live-Code
            df['is_warmup'] = df.index < start_time
            
            return df[['open', 'high', 'low', 'close', 'volume', 'is_complete', 'is_warmup']]
            
        except Exception as e:
            logging.warning(f"Attempt {attempt + 1}/{max_retries} failed for {symbol}. Error: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                logging.error(f"All attempts to fetch data for {symbol} failed")
                return None
