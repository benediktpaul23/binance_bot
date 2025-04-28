# parallel_processing.py
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time
import logging
from datetime import timezone
from tqdm import tqdm
from datetime import timedelta
import Z_config
import pandas as pd
import numpy as np
import gc
from utils.symbol_filter import check_price_change_threshold
from utils.Backtest import check_rate_limit, parse_interval_to_minutes

def process_all_symbols_parallel(symbols, current_time, position_tracking, active_observation_periods, observed_symbols):
    """
    Verarbeitet eine konfigurierbare Anzahl von Symbolen gleichzeitig
    und übergibt die Tracking-Dictionaries an die Worker.

    Args:
        symbols: Liste der zu verarbeitenden Symbole
        current_time: Endzeit für Datenanalyse (sollte timezone-aware sein, z.B. UTC)
        position_tracking (dict): Dictionary zur Verfolgung offener Positionen.
        active_observation_periods (dict): Dictionary aktiver Beobachtungsperioden.
        observed_symbols (dict): Dictionary zur Protokollierung beobachteter Symbole.

    Returns:
        list: Aggregierte Liste aller generierten Trades (obwohl Trades jetzt oft direkt gespeichert werden).
              Der Rückgabewert ist hier weniger kritisch, da der Zustand über die Dictionaries geteilt wird.
    """
    max_concurrent_symbols = getattr(Z_config, 'max_concurrent_symbols', 3)
    logging.info(f"Processing {len(symbols)} symbols with max {max_concurrent_symbols} concurrently.")

    start_time = time.time()
    all_trades_aggregated = [] # Kann Ergebnisse sammeln, falls benötigt
    processed_symbols = 0

    # Ensure current_time is timezone-aware (UTC recommended)
    if current_time.tzinfo is None:
        logging.warning("process_all_symbols_parallel received a naive datetime for current_time. Assuming UTC.")
        current_time = current_time.replace(tzinfo=timezone.utc)

    symbol_batches = [symbols[i:i + max_concurrent_symbols] for i in range(0, len(symbols), max_concurrent_symbols)]

    for batch_index, symbol_batch in enumerate(symbol_batches):
        logging.info(f"\nProcessing Symbol Batch {batch_index+1}/{len(symbol_batches)}: {symbol_batch}")

        with ThreadPoolExecutor(max_workers=max_concurrent_symbols) as executor:
            # === CORE CHANGE: Pass dictionaries to worker ===
            futures = {
                executor.submit(
                    process_symbol_parallel, # Name der Worker-Funktion
                    symbol,                   # Argument 1 (symbol)
                    current_time,             # Argument 2 (current_time)
                    position_tracking,        # Argument 3 (position_tracking dict)
                    active_observation_periods, # Argument 4 (active_observation_periods dict)
                    observed_symbols          # Argument 5 (observed_symbols dict)
                ): symbol                     # Key im Future-Dictionary ist das Future-Objekt
                for symbol in symbol_batch
            }
            # ==============================================

            # Progress bar setup
            progress_bar = tqdm(as_completed(futures), total=len(symbol_batch), desc=f"Batch {batch_index+1}", unit="sym")

            for future in progress_bar:
                symbol = futures[future]
                progress_bar.set_postfix_str(f"Processing {symbol}")
                try:
                    # Das Ergebnis von process_symbol_parallel ist jetzt (symbol, trades, performance, df_for_saving)
                    result = future.result()

                    processed_symbols += 1

                    if result:
                        returned_symbol, trades, performance, df_saved = result # df_saved gibt an, ob save_strategy_data aufgerufen wurde

                        if returned_symbol != symbol: # Sanity check
                             logging.warning(f"Mismatch between future key and returned symbol: {symbol} vs {returned_symbol}")

                        if trades: # Check if trades list is not empty
                            all_trades_aggregated.extend(trades) # Aggregiere hier, wenn nötig
                            logging.info(f"✓ {symbol}: {len(trades)} trade events processed.")
                        else:
                            logging.info(f"✓ {symbol}: Processing complete (no new trade events).")

                        # Display performance summary immediately
                        if performance and isinstance(performance, dict) and 'error' not in performance:
                            summarize_symbol_performance(symbol, performance) # Annahme: Funktion existiert
                        elif performance and 'error' in performance:
                             logging.warning(f"✗ {symbol}: Performance calculation failed: {performance['error']}")
                        else:
                             logging.warning(f"✗ {symbol}: No valid performance data returned.")

                        # Logge, ob die Daten gespeichert wurden
                        if df_saved:
                            logging.info(f"✓ {symbol}: Strategy data saving attempted.")
                        else:
                            logging.warning(f"✗ {symbol}: Strategy data saving skipped or failed.")

                    else:
                        logging.error(f"✗ {symbol}: Processing function returned None or invalid result.")

                    # Update progress bar description with overall progress
                    progress_bar.set_description(f"Batch {batch_index+1} ({processed_symbols}/{len(symbols)} Total)")

                except Exception as e:
                    processed_symbols += 1 # Count as processed even if failed
                    logging.error(f"ERROR processing result for {symbol}: {e}", exc_info=True)
                    # Update progress bar description with overall progress
                    progress_bar.set_description(f"Batch {batch_index+1} ({processed_symbols}/{len(symbols)} Total)")


        # Cleanup after batch
        logging.debug(f"Batch {batch_index+1} completed. Running GC...")
        gc.collect()

    end_time = time.time()
    total_time = end_time - start_time
    logging.info(f"\nParallel processing completed in {total_time:.2f} seconds.")
    if len(symbols) > 0:
        logging.info(f"Average time per symbol: {total_time/len(symbols):.2f} seconds.")

    # Call the final summary function (reads from CSVs)
    try:
        update_total_performance() # Annahme: Funktion existiert
    except NameError:
         logging.error("update_total_performance function not found/imported.")
    except Exception as e:
         logging.error(f"Error calling update_total_performance: {e}")

    return all_trades_aggregated

from utils import Backtest
from datetime import datetime
from utils import backtest_strategy
from utils.symbol_filter import verify_observation_indicators, update_symbol_position_status
from utils import indicators

logger_proc_obs = logging.getLogger(__name__)


def process_observation_period_with_full_data(
    symbol,
    full_df_raw, # Erhält Rohdaten + evtl. Alignment-Trends
    period,
    use_ut_bot=False,
    position_tracking=None,
    active_observation_periods=None,
    observed_symbols=None
):
    """
    MODIFIED (Based on Response #11 and User Request): Verarbeitet einen Beobachtungszeitraum.
    1. Berechnet Standard-Indikatoren (VEKTORISIERT, inkl. vorläufigem OBV - wird später entfernt).
    2. Berechnet EMAs ITERATIV und fügt sie als *_iter Spalten hinzu.
    3. Berechnet OBV ITERATIV und fügt obv_iter, obv_trend_iter hinzu.
    4. Wendet Zeitfilter an.
    5. Setzt Kontext-Flags basierend auf 'period'.
    6. Generiert Signale mit apply_base_strategy (nutzt *_iter EMAs und _iter OBV).
    7. Filtert Daten für die Trade-Verarbeitung.
    8. Führt Trendstärkeprüfung durch.
    9. Ruft process_symbol für die Trade-Logik auf.
    10. Aktualisiert den globalen State.

    Args:
        symbol (str): Trading-Symbol
        full_df_raw (pd.DataFrame): Vollständiger DataFrame mit OHLCV, is_complete, is_warmup
                                    und evtl. Alignment-Trends (von get_multi_timeframe_data).
        period (dict): Dictionary mit start_time, end_time, price_change_pct.
        use_ut_bot (bool): Ob UT Bot verwendet werden soll.
        position_tracking (dict): Globaler Positionsstatus.
        active_observation_periods (dict): Aktive Beobachtungsperioden.
        observed_symbols (dict): Protokollierte beobachtete Symbole.

    Returns:
        Tuple mit (df_processed_final, trades, performance) oder (None, [], {}) bei Fehlern.
        df_processed_final enthält Rohdaten, Indikatoren (Standard + iterativ), Flags und Signale.
    """
    log_prefix = f"process_obs ({symbol})"
    logger_proc_obs = logging.getLogger(f"{__name__}.{log_prefix}") # Eigener Logger pro Aufruf/Symbol
    logger_proc_obs.debug(f"Starting processing.")

    # --- Initial Checks ---
    if position_tracking is None or active_observation_periods is None or observed_symbols is None:
        logger_proc_obs.error(f"CRITICAL - Tracking dictionaries not passed!")
        return None, [], {}
    if full_df_raw is None or full_df_raw.empty:
        logger_proc_obs.error(f"✗ No raw data available (full_df_raw).")
        return None, [], {}

    # --- Zeitrahmen validieren ---
    observation_start_time = period.get('start_time')
    observation_end_time = period.get('end_time')
    price_change_pct = period.get('price_change_pct', 0)
    if not all(isinstance(t, datetime) and t.tzinfo is not None for t in [observation_start_time, observation_end_time]):
        logger_proc_obs.error(f"Invalid or non-TZ-aware period start/end times ({observation_start_time}, {observation_end_time}).")
        return None, [], {}
    observation_start_time = observation_start_time.astimezone(timezone.utc)
    observation_end_time = observation_end_time.astimezone(timezone.utc)

    logger_proc_obs.info(f"Processing Period: {observation_start_time} -> {observation_end_time} (Price Change: {price_change_pct:.2f}%)")

    # --- Schritt 1: Standard-Indikatoren berechnen ---
    # WICHTIG: calculate_indicators sollte den Standard-OBV NICHT mehr berechnen (siehe Plan Schritt 3)
    logger_proc_obs.debug(f"Calculating STANDARD indicators (excl. OBV) on raw_df (shape: {full_df_raw.shape})...")
    try:
        df_with_std_indicators = indicators.calculate_indicators(full_df_raw.copy(), calculate_for_alignment_only=False)
    except AttributeError:
        logger_proc_obs.critical(f"Function 'indicators.calculate_indicators' not found!")
        return None, [], {}
    except Exception as e:
        logger_proc_obs.error(f"Error during standard indicator calculation: {e}", exc_info=True)
        return None, [], {}

    if df_with_std_indicators is None:
        logger_proc_obs.error(f"Standard indicator calculation failed (returned None).")
        return None, [], {}
    logger_proc_obs.debug(f"Standard indicators calculated. DF shape: {df_with_std_indicators.shape}")

    # --- Schritt 1b: Iterative EMAs berechnen ---
    logger_proc_obs.debug(f"Calculating ITERATIVE EMAs...")
    required_candles_ema = 0 # Init
    try:
        required_candles_ema = backtest_strategy.calculate_required_candles() # Hole die Anzahl erneut
        logger_proc_obs.info(f"Using required_candles={required_candles_ema} for iterative EMA calculation.")
        df_with_iter_ema = indicators.calculate_ema_iterative(df_with_std_indicators, required_candles_ema)
    except AttributeError:
        logger_proc_obs.critical(f"Function 'indicators.calculate_ema_iterative' or 'calculate_required_candles' not found!")
        return None, [], {}
    except Exception as e:
        logger_proc_obs.error(f"Error during iterative EMA calculation: {e}", exc_info=True)
        return None, [], {}

    if df_with_iter_ema is None:
        logger_proc_obs.error(f"Iterative EMA calculation failed (returned None).")
        # Fallback: Nutze den DataFrame mit Standard-Indikatoren weiter
        df_with_iter_ema = df_with_std_indicators.copy() # Arbeite mit Kopie
        logger_proc_obs.warning(f"Proceeding with standard EMA values due to iterative calculation error.")
        # Füge leere _iter Spalten hinzu, damit apply_base_strategy nicht fehlschlägt
        df_with_iter_ema['ema_fast_iter'] = df_with_iter_ema.get('ema_fast', np.nan)
        df_with_iter_ema['ema_slow_iter'] = df_with_iter_ema.get('ema_slow', np.nan)
        df_with_iter_ema['ema_baseline_iter'] = df_with_iter_ema.get('ema_baseline', np.nan)
    else:
         # Stelle sicher, dass wir mit einer Kopie weiterarbeiten, falls nötig
         df_with_iter_ema = df_with_iter_ema.copy()

    logger_proc_obs.debug(f"Iterative EMAs calculated/handled. Proceeding...")

    # --- Schritt 1c: Iterative OBV-Berechnung Hinzufügen ---
    logger_proc_obs.debug(f"Calculating ITERATIVE OBV...")
    df_with_iter_ema['obv_iter'] = np.nan # Neue Spalte initialisieren
    df_with_iter_ema['obv_trend_iter'] = 0 # Neue Trend-Spalte initialisieren

    # Finde ersten Backtest-Index (wo is_warmup False ist)
    non_warmup_indices_iter = df_with_iter_ema[~df_with_iter_ema['is_warmup']].index
    first_backtest_iloc_iter = -1
    if not non_warmup_indices_iter.empty:
        try:
            first_backtest_iloc_iter = df_with_iter_ema.index.get_loc(non_warmup_indices_iter[0])
        except KeyError:
             logger_proc_obs.error(f"Could not find first non-warmup index for iterative OBV.")

    if first_backtest_iloc_iter != -1:
        required_candles_obv = 0 # Init
        try:
             # Verwende die GLEICHE Funktion/Logik wie für EMA, um Konsistenz zu wahren
             required_candles_obv = backtest_strategy.calculate_required_candles()
             if required_candles_obv <= 0: # Sicherheitscheck
                 logger_proc_obs.error(f"calculate_required_candles returned invalid value: {required_candles_obv}. Using fallback 100.")
                 required_candles_obv = 100
        except NameError:
             logger_proc_obs.error(f"backtest_strategy.calculate_required_candles not found for OBV iter! Using fallback 100.")
             required_candles_obv = 100 # Fallback, anpassen!
        except Exception as req_err:
             logger_proc_obs.error(f"Error getting required_candles for OBV: {req_err}. Using fallback 100.")
             required_candles_obv = 100

        logger_proc_obs.info(f"Starting iterative OBV loop from index {first_backtest_iloc_iter} using required_candles={required_candles_obv}")

        # --- Iterative OBV Schleife ---
        for i in tqdm(range(first_backtest_iloc_iter, len(df_with_iter_ema)), desc=f"{log_prefix} OBV Iter", unit="candle", leave=False):
            current_ts_iter = df_with_iter_ema.index[i]

            # Slice: Genau N Kerzen, endend mit der aktuellen Kerze i
            # N = required_candles_obv
            start_slice_iloc_iter = max(0, i - required_candles_obv + 1) # +1 damit der Slice N lang ist
            end_slice_iloc_iter = i + 1 # Aktuelle Kerze einschließen

            # Prüfe, ob der Slice gültig ist (mindestens 2 Punkte für diff())
            if end_slice_iloc_iter - start_slice_iloc_iter < 2:
                 # Für die allererste(n) Berechnung(en) kann OBV 0 sein
                 df_with_iter_ema.loc[current_ts_iter, 'obv_iter'] = 0.0
                 if i % 100 == 0: # Gelegentliches Logging
                      logger_proc_obs.debug(f"OBV Iter {i}: Slice too short ({end_slice_iloc_iter - start_slice_iloc_iter}), setting OBV to 0.")
                 continue

            data_slice_iter = df_with_iter_ema.iloc[start_slice_iloc_iter:end_slice_iloc_iter]

            # Rufe die Standard OBV Funktion auf dem SLICE auf
            try:
                # Stelle sicher, dass close und volume nicht nur NaNs enthalten im Slice
                if data_slice_iter['close'].notna().any() and data_slice_iter['volume'].notna().any():
                    # Wichtig: Symbol mitgeben für interne Logs der OBV Funktion
                    obv_series_slice = indicators.on_balance_volume(data_slice_iter['close'], data_slice_iter['volume'], symbol=f"{symbol}_iter_slice_{i}")
                    if obv_series_slice is not None and not obv_series_slice.empty:
                        obv_value_iter = obv_series_slice.iloc[-1] # Nimm den letzten Wert des Slices
                        df_with_iter_ema.loc[current_ts_iter, 'obv_iter'] = obv_value_iter
                        if i % 100 == 0: # Gelegentliches Logging
                             logger_proc_obs.debug(f"OBV Iter {i} @ {current_ts_iter}: Slice [{start_slice_iloc_iter}:{end_slice_iloc_iter}], Length={len(data_slice_iter)}, OBV={obv_value_iter:.2f}")
                    else:
                         logger_proc_obs.warning(f"OBV Iter {i} @ {current_ts_iter}: on_balance_volume returned empty Series. Setting NaN.")
                         df_with_iter_ema.loc[current_ts_iter, 'obv_iter'] = np.nan # Fallback falls leer
                else:
                     logger_proc_obs.warning(f"OBV Iter {i} @ {current_ts_iter}: Slice contains all NaNs for close or volume. Setting OBV to NaN.")
                     df_with_iter_ema.loc[current_ts_iter, 'obv_iter'] = np.nan

            except Exception as obv_iter_err:
                 logger_proc_obs.error(f"Error during single OBV iteration for {current_ts_iter}: {obv_iter_err}", exc_info=True)
                 df_with_iter_ema.loc[current_ts_iter, 'obv_iter'] = np.nan

        # Berechne den Trend des iterativen OBV NACH der Schleife
        # Fülle NaNs im OBV vor der Trendberechnung (z.B. mit 0 oder ffill)
        df_with_iter_ema['obv_iter'] = df_with_iter_ema['obv_iter'].fillna(0) # Oder ffill() ? Entscheiden! 0 ist oft sicherer.
        df_with_iter_ema['obv_trend_iter'] = np.sign(df_with_iter_ema['obv_iter'].diff().fillna(0)).astype(int)
        logger_proc_obs.debug(f"Iterative OBV calculation finished. Last value: {df_with_iter_ema['obv_iter'].iloc[-1]:.2f}" if not df_with_iter_ema.empty else "N/A")
        # --- Ende Iterative OBV Schleife ---
    else:
         logger_proc_obs.warning(f"Skipping iterative OBV calculation as no non-warmup index found.")
         # Stelle sicher, dass die Spalten existieren, auch wenn sie leer sind
         df_with_iter_ema['obv_iter'] = 0.0
         df_with_iter_ema['obv_trend_iter'] = 0
    # --- Ende Schritt 1c ---

    # --- Schritt 1.5: Zeitfilterung ANWENDEN ---
    # Wende Filter auf den DataFrame an, der JETZT iterative EMAs UND OBV enthält
    logger_proc_obs.debug(f"Applying trading time filter...")
    try:
        df_processed = Backtest.filter_historical_candles_by_trading_time(df_with_iter_ema) # Verwende df_with_iter_ema
    except AttributeError:
        logger_proc_obs.error(f"Function 'Backtest.filter_historical_candles_by_trading_time' not found!")
        df_processed = df_with_iter_ema # Fallback: Keine Filterung
    except Exception as e:
        logger_proc_obs.error(f"Error during time filtering: {e}", exc_info=True)
        df_processed = df_with_iter_ema # Fallback

    if df_processed is None or df_processed.empty:
        logger_proc_obs.warning(f"DataFrame empty after applying trading time filter. Saving original indicators DataFrame (with iter values).")
        # Gib den DF mit allen Indikatoren (Standard + Iterativ) zurück, falls keine Kerzen im Handelsfenster
        return df_with_iter_ema, [], {} # Gib den nicht-gefilterten zurück
    logger_proc_obs.debug(f"Trading time filter applied. Shape after filter: {df_processed.shape}")
    df_processed = df_processed.copy() # Sicherstellen, dass wir mit einer Kopie arbeiten

    # --- Schritt 2: Kontext-Flags setzen ---
    logger_proc_obs.debug(f"Setting context flags based on period: {observation_start_time} -> {observation_end_time}")
    try:
        # Stelle Zeitzone sicher (wichtig für Vergleiche)
        if not isinstance(df_processed.index, pd.DatetimeIndex):
             logger_proc_obs.error("Index is not DatetimeIndex after filtering. Cannot set context flags reliably.")
             return df_processed, [], {}
        if df_processed.index.tz is None: df_processed.index = df_processed.index.tz_localize('UTC')
        elif str(df_processed.index.tz) != 'UTC': df_processed.index = df_processed.index.tz_convert('UTC')
    except Exception as tz_err:
         logger_proc_obs.error(f"Error ensuring index timezone awareness before flag setting: {tz_err}", exc_info=True)
         return df_processed, [], {}

    # Setze Flags basierend auf dem übergebenen 'period' dict
    df_processed['is_context'] = df_processed.index < observation_start_time
    df_processed['is_observation'] = (df_processed.index >= observation_start_time) & (df_processed.index <= observation_end_time)
    df_processed['is_post_observation'] = df_processed.index > observation_end_time
    # Korrektur: is_context sollte False sein, wenn is_observation oder is_post_observation True ist
    df_processed.loc[df_processed['is_observation'] | df_processed['is_post_observation'], 'is_context'] = False
    logger_proc_obs.info(f"Flags set: Context={df_processed['is_context'].sum()}, Obs={df_processed['is_observation'].sum()}, PostObs={df_processed['is_post_observation'].sum()}")

    # --- Schritt 3: Base Strategy anwenden (Signalgenerierung) ---
    # Diese Funktion muss jetzt die *_iter Spalten verwenden! (OBV Anpassung ist dort nötig)
    logger_proc_obs.debug(f"Calling apply_base_strategy (should use iterative EMAs and OBV)...")
    try:
        df_with_signals = backtest_strategy.apply_base_strategy(df_processed, symbol, use_ut_bot=use_ut_bot)
    except AttributeError:
        logger_proc_obs.critical(f"Function 'backtest_strategy.apply_base_strategy' not found!")
        return df_processed, [], {} # Gib zumindest den DF mit Flags zurück
    except Exception as e:
        logger_proc_obs.error(f"Error during apply_base_strategy: {e}", exc_info=True)
        return df_processed, [], {}

    if df_with_signals is None:
        logger_proc_obs.error(f"apply_base_strategy returned None.")
        return df_processed, [], {} # Gib zumindest den DF mit Flags zurück
    logger_proc_obs.debug(f"apply_base_strategy finished. Signal counts:\n{df_with_signals['signal'].value_counts()}")

    # --- Schritt 4: Signale außerhalb der Observation zurücksetzen ---
    # (Dies ist wichtig, damit Trades nur im Beobachtungszeitraum oder Post-Obs (wenn close_position=False) ausgelöst werden)
    logger_proc_obs.debug(f"Resetting signals outside 'is_observation' (and 'is_post_observation' if close_position=True)...")
    close_position_setting_signals = getattr(Z_config, 'close_position', False)
    if 'is_observation' in df_with_signals.columns and 'is_post_observation' in df_with_signals.columns:
         if close_position_setting_signals:
             # Wenn Positionen am Ende der Obs geschlossen werden, ignoriere Post-Obs Signale
             reset_mask = ~df_with_signals['is_observation']
             logger_proc_obs.debug("close_position=True: Resetting signals where is_observation=False.")
         else:
             # Wenn Positionen offen bleiben können, erlaube Signale auch in Post-Obs
             reset_mask = ~(df_with_signals['is_observation'] | df_with_signals['is_post_observation'])
             logger_proc_obs.debug("close_position=False: Resetting signals where neither is_observation nor is_post_observation is True.")

         df_with_signals.loc[reset_mask, 'signal'] = 'no_signal'
         df_with_signals.loc[reset_mask, 'trigger'] = None
    else:
        logger_proc_obs.warning(f"'is_observation'/'is_post_observation' columns not found after apply_base_strategy. Cannot reset signals based on period.")

    # --- Schritt 5: Verify Indicators (Optional) ---
    try:
        # Stelle sicher, dass verify_observation_indicators existiert
        verify_observation_indicators(df_with_signals, observation_start_time, symbol)
    except NameError:
        logger_proc_obs.debug(f"verify_observation_indicators not available.")
    except Exception as verif_err:
        logger_proc_obs.error(f"Error during verify_observation_indicators: {verif_err}", exc_info=True)


    # --- Schritt 6: Relevanten DataFrame für Trade Processing extrahieren ---
    logger_proc_obs.debug(f"Extracting DataFrame slice for trade processing...")
    close_position_setting_trades = getattr(Z_config, 'close_position', False)
    observation_df_for_trades = pd.DataFrame() # Initialisiere leer

    has_obs_flags_trades = 'is_observation' in df_with_signals.columns and 'is_post_observation' in df_with_signals.columns

    if has_obs_flags_trades:
        if close_position_setting_trades:
            # Nur Observation verwenden, wenn Positionen am Ende geschlossen werden
            observation_df_for_trades = df_with_signals[df_with_signals['is_observation'] == True].copy()
            logger_proc_obs.info(f"close_position=True: Using {len(observation_df_for_trades)} observation candles for trade processing.")
        else:
            # Observation UND Post-Observation verwenden, wenn Positionen offen bleiben können
            trade_processing_mask = (df_with_signals['is_observation'] == True) | (df_with_signals['is_post_observation'] == True)
            observation_df_for_trades = df_with_signals[trade_processing_mask].copy()
            obs_count = (observation_df_for_trades['is_observation'] == True).sum()
            post_obs_count = (observation_df_for_trades['is_post_observation'] == True).sum()
            logger_proc_obs.info(f"close_position=False: Using {len(observation_df_for_trades)} candles ({obs_count} Obs + {post_obs_count} Post-Obs) for trade processing.")
    else:
        # Fallback, wenn Flags fehlen (sollte nicht passieren, aber sicherheitshalber)
        logger_proc_obs.warning(f"Observation flags missing, using entire df_with_signals for trade processing.")
        observation_df_for_trades = df_with_signals.copy()

    # Prüfung auf leeren DataFrame für Trades
    if observation_df_for_trades.empty:
        logger_proc_obs.warning(f"observation_df_for_trades is empty after filtering. No trades possible.")
        # Gib den DataFrame mit Signalen zurück, aber keine Trades/Performance
        return df_with_signals, [], {}

    # --- Schritt 7: Trend Strength Check ---
    logger_proc_obs.debug(f"Starting Trend Strength Check...")
    # Verwende den DataFrame, der für das Trade Processing bestimmt ist
    trend_strength_check_df = observation_df_for_trades[observation_df_for_trades.get('is_complete', True)].copy() # Prüfe nur auf kompletten Kerzen

    meets_strength_criteria = False
    strongest_trend_value = 0.0
    strongest_trend_direction = 0
    strongest_trend_time = None
    min_trend_strength_cfg = getattr(Z_config, 'min_trend_strength_parameter', 1.0)

    # Verwende ITERATIVE EMA-Werte für den Check, falls verfügbar
    trend_strength_col = 'trend_strength_iter' if 'trend_strength_iter' in trend_strength_check_df.columns else 'trend_strength'
    trend_col = 'trend_iter' if 'trend_iter' in trend_strength_check_df.columns else 'trend'

    if not trend_strength_check_df.empty and trend_strength_col in trend_strength_check_df.columns and trend_col in trend_strength_check_df.columns:
        logger_proc_obs.debug(f"Checking trend strength in {len(trend_strength_check_df)} complete trade processing candles (using {trend_strength_col})...")
        # Finde den stärksten Trend im relevanten Zeitraum
        strongest_trend_idx = trend_strength_check_df[trend_strength_col].abs().idxmax() # Finde Index der max. absoluten Stärke
        if pd.notna(strongest_trend_idx):
            strongest_candle = trend_strength_check_df.loc[strongest_trend_idx]
            strongest_trend_value = float(strongest_candle.get(trend_strength_col, 0.0))
            strongest_trend_direction = int(strongest_candle.get(trend_col, 0))
            strongest_trend_time = strongest_trend_idx
            # Prüfe, ob IRGENDEINE Kerze das Kriterium erfüllt
            meets_strength_criteria = (trend_strength_check_df[trend_strength_col].abs() >= min_trend_strength_cfg).any()

        if meets_strength_criteria:
            logger_proc_obs.info(f"✓ Trend strength criterion met somewhere in period. Strongest observed: {strongest_trend_value:.1f}% (Dir: {strongest_trend_direction}) @ {strongest_trend_time}")
        else:
             logger_proc_obs.info(f"✗ Trend strength criterion ({min_trend_strength_cfg}%) NOT met in trade processing period. Max observed: {strongest_trend_value:.1f}%")
    else:
        logger_proc_obs.warning(f"No complete candles or required columns ({trend_strength_col}, {trend_col}) for trend strength check in trade processing data.")

    # --- Behandlung offener Positionen ---
    is_tracking_open_position = position_tracking.get(symbol, {}).get('position_open', False)
    if is_tracking_open_position:
        logger_proc_obs.info(f"ℹ️ Skipping trend strength requirement check as open position is tracked.")
        meets_strength_criteria = True # Überschreibt den Check
        if strongest_trend_time is None and not observation_df_for_trades.empty:
            strongest_trend_time = observation_df_for_trades.index[0] # Fallback auf Start des Trade-Zeitraums

    # --- Abbruchbedingung ---
    if not meets_strength_criteria:
        logger_proc_obs.info(f"Skipping trade processing: Unmet trend strength and no open position tracked.")
        # Gib den finalen DF mit allen Indikatoren/Signalen zurück, aber keine Trades
        return df_with_signals, [], {}

    # --- Schritt 8: Trade Processing Call ---
    symbol_data_for_process = {
        'symbol': symbol,
        # Übergib hier ggf. relevante Daten, die process_symbol braucht
        # 'trend_strength': strongest_trend_value, # Optional, wenn process_symbol es braucht
        # 'trend_direction': strongest_trend_direction, # Optional
        'current_time': strongest_trend_time if strongest_trend_time else observation_start_time # Zeit des stärksten Trends oder Start
    }
    logger_proc_obs.info(f"Calling process_symbol with {len(observation_df_for_trades)} relevant candles...")
    try:
        # Übergib den DataFrame, der für Trades gefiltert wurde UND alle Indikatoren enthält
        result = backtest_strategy.process_symbol(
            symbol=symbol,
            df=observation_df_for_trades, # Der gefilterte Slice für die Trade-Logik
            symbol_data=symbol_data_for_process,
            start_time=observation_start_time, # Start der ursprünglichen Beobachtung
            position_tracking=position_tracking # Zustand übergeben
        )
    except AttributeError:
        logger_proc_obs.critical(f"Function 'backtest_strategy.process_symbol' not found!")
        return df_with_signals, [], {} # Gib den DF mit Signalen zurück
    except Exception as e:
        logger_proc_obs.error(f"Error calling process_symbol: {e}", exc_info=True)
        return df_with_signals, [], {} # Gib den DF mit Signalen zurück

    # --- Schritt 9: Ergebnisse verarbeiten und State Update ---
    trades = []
    performance = {}
    if result and isinstance(result, tuple) and len(result) == 2:
        trades, performance = result
        logger_proc_obs.info(f"process_symbol resulted in {len(trades)} trade events.")
        # Stelle sicher, dass 'symbol' im Performance-Dict ist
        if isinstance(performance, dict) and 'symbol' not in performance:
            performance['symbol'] = symbol
    else:
        logger_proc_obs.error(f"✗ process_symbol returned invalid results! Type: {type(result)}")
        # Erzeuge leere Performance, falls None zurückgegeben wurde oder Format falsch ist
        performance = {'symbol': symbol, 'total_trades': 0, 'error': 'process_symbol returned invalid result'}

    # --- State Update Logik (bleibt wie zuvor) ---
    logger_proc_obs.debug(f"Starting post-processing state update...")
    try:
        position_still_open_after = position_tracking.get(symbol, {}).get('position_open', False)

        if symbol not in active_observation_periods:
             active_observation_periods[symbol] = {'active_period': None, 'has_position': False}

        if position_still_open_after:
            logger_proc_obs.info(f"⚠️ Position remains OPEN. Updating active_observation_periods.")
            active_observation_periods[symbol]['has_position'] = True
            # Setze aktive Periode nur, wenn sie noch nicht gesetzt ist oder wenn sie für die offene Pos relevant ist
            if not active_observation_periods[symbol].get('active_period'):
                 active_observation_periods[symbol]['active_period'] = period # Verwende das Original 'period' dict
                 logger_proc_obs.info(f"Setting active_period for open position.")
            # Debugging für close_position=False vs True bleibt wie in deinem Originalcode
            close_position_setting_state = getattr(Z_config, 'close_position', False)
            if not close_position_setting_state:
                if position_tracking.get(symbol, {}).get('observation_ended') == True:
                     logger_proc_obs.info(f"    observation_ended=True confirmed in position_tracking.")
                else:
                     logger_proc_obs.debug(f"    observation_ended is False/None in position_tracking for OPEN position ({symbol}) - Expected if opened within period and close_position=False.")
            elif close_position_setting_state:
                 logger_proc_obs.error(f"CRITICAL INCONSISTENCY: Position open, but close_position=True! Check timeout logic in process_symbol.")

        else: # Position ist geschlossen
            logger_proc_obs.info(f"✓ Position is CLOSED. Updating active_observation_periods.")
            active_observation_periods[symbol]['has_position'] = False
            active_observation_periods[symbol]['active_period'] = None # Keine aktive Periode mehr zugeordnet

            if position_tracking.get(symbol, {}).get('position_open'):
                 logger_proc_obs.error(f"CRITICAL INCONSISTENCY: Position closed, but position_tracking still shows open for {symbol}! Resetting global state manually.")
                 position_tracking[symbol] = {'position_open': False, 'observation_ended': False, 'position_data': {}}

            try:
                update_symbol_position_status(symbol, False)
                logger_proc_obs.info(f"Notified symbol_filter: Status updated to 'no position'.")
            except NameError: logger_proc_obs.debug("update_symbol_position_status not imported/available.")
            except Exception as e_usps: logger_proc_obs.error(f"Error calling update_symbol_position_status: {e_usps}")

    except Exception as post_process_err:
        logger_proc_obs.error(f"Error during post-processing state update: {post_process_err}", exc_info=True)
    logger_proc_obs.debug(f"Post-processing state update finished.")
    # --- Ende State Update ---

    # --- Schritt 10: Rückgabe ---
    # Gib den DataFrame zurück, der ALLES enthält (Rohdaten, alle Indikatoren [Std+Iter], Flags, Signale)
    # df_with_signals enthält bereits alle iterativen Werte.
    logger_proc_obs.info(f"Finished processing period. Returning DataFrame (shape {df_with_signals.shape}) and results.")
    return df_with_signals, trades, performance

        

def summarize_symbol_performance(symbol, performance_data):
    """
    Erstellt eine Zusammenfassung der Performance für ein einzelnes Symbol nach Abschluss der Verarbeitung.
    
    Args:
        symbol: Das verarbeitete Symbol
        performance_data: Performance-Daten des Symbols
    """
    if not performance_data:
        print(f"\n{'='*60}")
        print(f"PERFORMANCE ZUSAMMENFASSUNG FÜR {symbol}")
        print(f"{'='*60}")
        print(f"Keine Performance-Daten verfügbar für {symbol}")
        print(f"{'='*60}")
        return
    
    # Formatierte Ausgabe der Performance-Daten
    print(f"\n{'='*60}")
    print(f"PERFORMANCE ZUSAMMENFASSUNG FÜR {symbol}")
    print(f"{'='*60}")
    print(f"Gesamtzahl Trades: {performance_data.get('total_trades', 0)}")
    print(f"Erfolgsrate: {performance_data.get('win_rate', 0):.2%}")
    print(f"Gesamtgewinn: {performance_data.get('total_profit', 0):.4f}")
    print(f"Max. Drawdown: {performance_data.get('max_drawdown', 0):.2f}%")
    
    # Zeige Profit-Faktor
    profit_factor = performance_data.get('profit_factor', 0)
    if profit_factor == float('inf'):
        print(f"Profit-Faktor: ∞ (keine Verluste)")
    else:
        print(f"Profit-Faktor: {profit_factor:.2f}")
    
    print(f"Durchschnittlicher Gewinn pro Trade: {performance_data.get('avg_profit', 0):.4f}")
    print(f"Gesamtkommission: {performance_data.get('total_commission', 0):.4f}")
    

    # Exit-Analyse
    total_exits = (
        performance_data.get('stop_loss_hits', 0) + 
        performance_data.get('take_profit_hits', 0) + 
        performance_data.get('signal_exits', 0) + 
        performance_data.get('trailing_stop_hits', 0) +
        performance_data.get('observation_timeout_exits', 0) +
        performance_data.get('backtest_end_exits', 0)
    )
    
    if total_exits > 0:
        print(f"\nExit-Analyse:")
        print(f"  Stop Loss: {performance_data.get('stop_loss_hits', 0)} ({performance_data.get('stop_loss_hits', 0)/total_exits*100:.1f}%)")
        print(f"  Take Profit: {performance_data.get('take_profit_hits', 0)} ({performance_data.get('take_profit_hits', 0)/total_exits*100:.1f}%)")
        print(f"  Signal/Strategie: {performance_data.get('signal_exits', 0)} ({performance_data.get('signal_exits', 0)/total_exits*100:.1f}%)")
        print(f"  Trailing Stop: {performance_data.get('trailing_stop_hits', 0)} ({performance_data.get('trailing_stop_hits', 0)/total_exits*100:.1f}%)")
        print(f"  Beobachtungszeitraum-Ende: {performance_data.get('observation_timeout_exits', 0)} ({performance_data.get('observation_timeout_exits', 0)/total_exits*100:.1f}%)")
        print(f"  Backtest-Ende: {performance_data.get('backtest_end_exits', 0)} ({performance_data.get('backtest_end_exits', 0)/total_exits*100:.1f}%)")
    
    print(f"{'='*60}")

def update_total_performance():
    """
    Erstellt eine Zusammenfassung der Performance pro Symbol aus der performance_summary.csv
    und speichert diese in total_symbol_performance.csv
    """
    import os
    import numpy as np
    
    performance_file = "performance_summary.csv"
    total_performance_file = "total_symbol_performance.csv"
    trades_dir = "./trades"
    
    if not os.path.exists(performance_file):
        print(f"Keine Performance-Daten gefunden in {performance_file}")
        return
    
    try:
        # Lade Performance-Daten
        df = pd.read_csv(performance_file)
        
        if df.empty:
            print("Keine Performance-Daten zum Analysieren verfügbar")
            return
        
        # Gruppiere nach Symbol und aggregiere die Daten
        aggregated_data = []
        
        # Gruppiere nach Symbol
        grouped = df.groupby('symbol')
        
        for symbol, group in grouped:
            # Basismetriken summieren
            total_trades = group['total_trades'].sum()
            
            # Berechne die Gesamtzahl der gewinnenden Trades
            if 'win_rate' in group.columns:
                # Wenn win_rate als Dezimalzahl (0-1) angegeben ist
                winning_trades = sum(group['win_rate'] * group['total_trades'])
                win_rate = winning_trades / total_trades if total_trades > 0 else 0
            else:
                win_rate = 0
            
            total_profit = group['total_profit'].sum()
            
            # Für echten symbolübergreifenden Max Drawdown müssen wir die Trade-Details laden
            # Suche nach dem Trade-Report für dieses Symbol
            trade_report_path = os.path.join(trades_dir, f"{symbol}_trade_report.csv")
            
            if os.path.exists(trade_report_path):
                try:
                    # Lade alle Trades für dieses Symbol
                    trades_df = pd.read_csv(trade_report_path)
                    
                    # Sortiere nach Entry-Zeit, um die chronologische Reihenfolge zu gewährleisten
                    if 'entry_time' in trades_df.columns:
                        trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
                        trades_df = trades_df.sort_values('entry_time')
                    
                    # Berechne symbolübergreifenden Max Drawdown
                    if not trades_df.empty and 'profit_loss' in trades_df.columns:
                        # Konstante Startbilanz für den gesamten Symbolzeitraum
                        start_balance = 25.0  # Typischerweise 25 USDT
                        
                        # Equity-Kurve erstellen
                        current_balance = start_balance
                        balances = [current_balance]
                        
                        for _, trade in trades_df.iterrows():
                            profit_loss = trade.get('profit_loss', 0)
                            current_balance += profit_loss
                            balances.append(current_balance)
                        
                        # Max Drawdown berechnen
                        peak_balance = start_balance
                        max_drawdown_pct = 0.0
                        
                        for balance in balances:
                            if balance > peak_balance:
                                peak_balance = balance
                            else:
                                current_drawdown_pct = ((peak_balance - balance) / peak_balance) * 100 if peak_balance > 0 else 0
                                max_drawdown_pct = max(max_drawdown_pct, current_drawdown_pct)
                        
                        print(f"Symbolübergreifender Max Drawdown für {symbol}: {max_drawdown_pct:.2f}%")
                    else:
                        # Fallback auf den größten Drawdown aus der Performance-Zusammenfassung
                        max_drawdown_pct = group['max_drawdown'].max() if 'max_drawdown' in group.columns else 0
                        print(f"Verwende Max Drawdown aus Zusammenfassung für {symbol}: {max_drawdown_pct:.2f}%")
                except Exception as e:
                    print(f"Fehler beim Berechnen des symbolübergreifenden Drawdowns für {symbol}: {e}")
                    # Fallback auf den größten Drawdown aus der Performance-Zusammenfassung
                    max_drawdown_pct = group['max_drawdown'].max() if 'max_drawdown' in group.columns else 0
            else:
                # Wenn keine Trade-Details verfügbar sind, verwende den größten Drawdown aus der Zusammenfassung
                max_drawdown_pct = group['max_drawdown'].max() if 'max_drawdown' in group.columns else 0
                print(f"Keine Trade-Details für {symbol} gefunden, verwende Max Drawdown: {max_drawdown_pct:.2f}%")
            
            # Rest der Aggregationen wie zuvor
            group_with_pf = group.copy()
            group_with_pf.loc[group_with_pf['profit_factor'] == float('inf'), 'profit_factor'] = 1000
            weighted_profit_factor = (group_with_pf['profit_factor'] * group_with_pf['total_trades']).sum() / total_trades if total_trades > 0 else 0
            
            avg_profit = total_profit / total_trades if total_trades > 0 else 0
            
            total_commission = group['total_commission'].sum() if 'total_commission' in group.columns else 0
            avg_slippage = group['avg_slippage'].mean() if 'avg_slippage' in group.columns else 0
            total_slippage = group['total_slippage'].sum() if 'total_slippage' in group.columns else 0
            
            # Exit-Statistiken
            stop_loss_hits = group['stop_loss_hits'].sum() if 'stop_loss_hits' in group.columns else 0
            take_profit_hits = group['take_profit_hits'].sum() if 'take_profit_hits' in group.columns else 0
            signal_exits = group['signal_exits'].sum() if 'signal_exits' in group.columns else 0
            trailing_stop_hits = group['trailing_stop_hits'].sum() if 'trailing_stop_hits' in group.columns else 0
            observation_timeout_exits = group['observation_timeout_exits'].sum() if 'observation_timeout_exits' in group.columns else 0
            backtest_end_exits = group['backtest_end_exits'].sum() if 'backtest_end_exits' in group.columns else 0
            
            # Berechne die Gesamtzahl aller Exits
            total_exits = (
                stop_loss_hits + 
                take_profit_hits + 
                signal_exits + 
                trailing_stop_hits +
                observation_timeout_exits +
                backtest_end_exits
            )
            
            # Berechne Prozentsätze für Exit-Typen
            stop_loss_percent = (stop_loss_hits / total_exits) * 100 if total_exits > 0 else 0
            take_profit_percent = (take_profit_hits / total_exits) * 100 if total_exits > 0 else 0
            signal_exits_percent = (signal_exits / total_exits) * 100 if total_exits > 0 else 0
            trailing_stop_percent = (trailing_stop_hits / total_exits) * 100 if total_exits > 0 else 0
            observation_timeout_percent = (observation_timeout_exits / total_exits) * 100 if total_exits > 0 else 0
            backtest_end_percent = (backtest_end_exits / total_exits) * 100 if total_exits > 0 else 0
            
            # Erstelle Datensatz für dieses Symbol
            symbol_data = {
                'symbol': symbol,
                'total_trades': total_trades,
                'win_rate': win_rate,
                'total_profit': total_profit,
                'max_drawdown': max_drawdown_pct,  # Neu berechneter symbolübergreifender Max Drawdown
                'profit_factor': weighted_profit_factor,
                'avg_profit': avg_profit,
                'total_commission': total_commission,
                'avg_slippage': avg_slippage,
                'total_slippage': total_slippage,
                'stop_loss_hits': stop_loss_hits,
                'take_profit_hits': take_profit_hits,
                'signal_exits': signal_exits,
                'trailing_stop_hits': trailing_stop_hits,
                'observation_timeout_exits': observation_timeout_exits,
                'backtest_end_exits': backtest_end_exits,
                'stop_loss_percent': stop_loss_percent,
                'take_profit_percent': take_profit_percent,
                'signal_exits_percent': signal_exits_percent,
                'trailing_stop_percent': trailing_stop_percent,
                'observation_timeout_percent': observation_timeout_percent,
                'backtest_end_percent': backtest_end_percent
            }
            
            aggregated_data.append(symbol_data)
        
        # Erstelle DataFrame aus den aggregierten Daten
        aggregated_df = pd.DataFrame(aggregated_data)
        
        # Sortiere nach Gesamtgewinn (absteigend)
        aggregated_df = aggregated_df.sort_values(by='total_profit', ascending=False)
        
        # Speichere in CSV
        aggregated_df.to_csv(total_performance_file, index=False)
        
        # Ausgabe der Gesamtstatistik pro Symbol
        print(f"\n{'='*80}")
        print(f"PERFORMANCE PRO SYMBOL (Top 10)")
        print(f"{'='*80}")
        print(f"{'Symbol':<15} {'Trades':>8} {'Win Rate':>10} {'Total Profit':>15} {'Avg Profit':>12} {'Max DD%':>10} {'PF':>8}")
        print(f"{'-'*80}")
        
        # Zeige die Top 10 Symbole
        for i, row in aggregated_df.head(10).iterrows():
            print(f"{row['symbol']:<15} {int(row['total_trades']):>8} {row['win_rate']*100:>9.2f}% {row['total_profit']:>15.4f} {row['avg_profit']:>12.4f} {row['max_drawdown']:>10.2f} {row['profit_factor']:>8.2f}")
        
        # Falls es mehr als 10 Symbole gibt, zeige auch die schlechtesten
        if len(aggregated_df) > 10:
            print(f"\n{'='*80}")
            print(f"PERFORMANCE PRO SYMBOL (Bottom 10)")
            print(f"{'='*80}")
            print(f"{'Symbol':<15} {'Trades':>8} {'Win Rate':>10} {'Total Profit':>15} {'Avg Profit':>12} {'Max DD%':>10} {'PF':>8}")
            print(f"{'-'*80}")
            
            # Zeige die schlechtesten 10 Symbole
            for i, row in aggregated_df.tail(10).iterrows():
                print(f"{row['symbol']:<15} {int(row['total_trades']):>8} {row['win_rate']*100:>9.2f}% {row['total_profit']:>15.4f} {row['avg_profit']:>12.4f} {row['max_drawdown']:>10.2f} {row['profit_factor']:>8.2f}")
        
        # Zeige auch allgemeine Statistiken
        total_trades = aggregated_df['total_trades'].sum()
        total_profit = aggregated_df['total_profit'].sum()
        avg_win_rate = (aggregated_df['win_rate'] * aggregated_df['total_trades']).sum() / total_trades if total_trades > 0 else 0
        
        print(f"\n{'='*80}")
        print(f"ZUSAMMENFASSUNG ALLER SYMBOLE")
        print(f"{'='*80}")
        print(f"Anzahl der Symbole: {len(aggregated_df)}")
        print(f"Gesamtzahl Trades: {total_trades}")
        print(f"Gesamtgewinn: {total_profit:.4f}")
        print(f"Durchschnittliche Win Rate: {avg_win_rate*100:.2f}%")
        print(f"Durchschnittlicher Gewinn pro Trade: {total_profit/total_trades if total_trades > 0 else 0:.4f}")
        print(f"{'='*80}")
        print(f"Detaillierte Statistiken gespeichert in {total_performance_file}")
        
    except Exception as e:
        print(f"Fehler beim Erstellen der Statistik pro Symbol: {e}")
        import traceback
        traceback.print_exc()

def process_symbol_parallel(symbol, current_time, position_tracking, active_observation_periods, observed_symbols):
    """
    Verarbeitet ein einzelnes Symbol, holt Daten, führt die Strategie aus,
    speichert die vollständigen Daten und gibt Ergebnisse zurück.
    Diese Funktion ist der "Worker" für den ThreadPoolExecutor.

    Args:
        symbol (str): Symbol zum Verarbeiten
        current_time (datetime): Endzeit für Datenanalyse (timezone-aware UTC)
        position_tracking (dict): Dictionary zur Verfolgung offener Positionen.
        active_observation_periods (dict): Dictionary aktiver Beobachtungsperioden.
        observed_symbols (dict): Dictionary zur Protokollierung beobachteter Symbole.

    Returns:
        Tuple aus (symbol, trades, performance, data_saved_flag) oder None bei Fehler.
        data_saved_flag (bool): True, wenn save_strategy_data erfolgreich aufgerufen wurde.
    """
    # --- Initial Checks ---
    if None in [position_tracking, active_observation_periods, observed_symbols]:
        logging.error(f"CRITICAL ({symbol}): One or more tracking dictionaries are None in process_symbol_parallel!")
        return None
    if not isinstance(current_time, datetime) or current_time.tzinfo is None:
        logging.error(f"CRITICAL ({symbol}): current_time is not a timezone-aware datetime object!")
        if isinstance(current_time, datetime) and current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)
        else:
            return None

    try:
        # --- Imports ---
        from datetime import timedelta
        import Z_config
        # Importiere Backtest-Strategie Funktionen und andere Helfer
        # Stelle sicher, dass diese Pfade korrekt sind!
        import utils.backtest_strategy as backtest_strategy
        from utils.Backtest import check_rate_limit, parse_interval_to_minutes, save_strategy_data # save_strategy_data hier importieren
        from utils.symbol_filter import check_price_change_threshold, verify_observation_indicators, update_symbol_position_status
        # Importiere die spezifische Funktion für Beobachtungsperioden
        from utils.parallel_processing import process_observation_period_with_full_data

        # --- Start Processing ---
        logging.info(f"\n{'-'*15} Starting processing for: {symbol} {'-'*15}")

        # --- Check for Existing Open Position ---
        has_open_position = False
        if symbol in position_tracking and position_tracking[symbol].get('position_open', False):
            has_open_position = True
            logging.info(f"[{symbol}] Found existing open position from passed state.")
            pos_data_debug = position_tracking[symbol].get('position_data', {})
            logging.debug(f"  Details: Entry={pos_data_debug.get('entry_time','?')}, Type={pos_data_debug.get('position_type','?')}, Qty={pos_data_debug.get('remaining_quantity','?')}")

        # --- Fetch Data ---
        check_rate_limit(weight=1) # Rate limit check
        logging.debug(f"[{symbol}] Calling get_multi_timeframe_data...")
        # Verwende die Funktion aus dem backtest_strategy Modul
        overall_lookback_hours = getattr(Z_config, 'lookback_hours_parameter', 24*7) # Dein globaler Lookback
        logical_start_time = current_time - timedelta(hours=overall_lookback_hours)

        # 2. Rufe die *neue* get_multi_timeframe_data mit der korrekten Signatur auf
        logging.debug(f"[{symbol}] Calling get_multi_timeframe_data with logical start: {logical_start_time}, end: {current_time}")
        full_df = backtest_strategy.get_multi_timeframe_data(
            symbol=symbol,
            end_time=current_time,             # Das logische Ende des Backtest-Zeitraums
            initial_start_time=logical_start_time, # Der logische Beginn des Backtest-Zeitraums
            position_tracking=position_tracking  # Das Tracking-Dictionary übergeben
            # KEIN lookback_hours Argument mehr hier!
        )

        if full_df is None or full_df.empty:
            logging.warning(f"[{symbol}] No data returned from get_multi_timeframe_data.")
            return None # Kein None zurückgeben, sondern Standard-Performance für das Symbol

        # --- Konfigurationsparameter holen ---
        filtering_active = getattr(Z_config, 'filtering_active', True)
        beobachten_active = getattr(Z_config, 'beobachten_active', True)
        observation_hours = getattr(Z_config, 'symbol_observation_hours', 48)
        min_pct = getattr(Z_config, 'min_price_change_pct_min', 2.0)
        max_pct = getattr(Z_config, 'min_price_change_pct_max', 50.0)
        lookback_minutes = getattr(Z_config, 'price_change_lookback_minutes', 60*12)
        direction = getattr(Z_config, 'seite', 'both')
        close_position_setting = getattr(Z_config, 'close_position', False)
        use_ut_bot_setting = getattr(Z_config, 'ut_bot', False)

        # --- Initialisierung für Ergebnisse ---
        all_trades_for_symbol = []
        final_performance = {}
        last_processed_df = None # DataFrame zum Speichern

        # --- Daten filtern und vorbereiten ---
        full_df = full_df[full_df.index <= current_time] # Ensure data doesn't exceed current time
        complete_candles = full_df[full_df.get('is_complete', True)]
        if complete_candles.empty:
            logging.warning(f"[{symbol}] No complete candles found in the fetched data.")
            # Return mit leerer Performance, damit das Symbol gezählt wird
            return symbol, [], {'symbol': symbol, 'total_trades': 0, 'error': 'No complete candles'}, False

        logging.info(f"[{symbol}] Initial data: {len(full_df)} rows ({full_df.index.min()} to {full_df.index.max()}). Complete candles: {len(complete_candles)}")

        if symbol not in observed_symbols:
            observed_symbols[symbol] = []

        # --- Verarbeitung (Offene Position oder Suche nach Perioden) ---
        processed_a_period = False # Flag, um zu sehen, ob eine Periode verarbeitet wurde

        # 1. Offene Position verarbeiten
        if has_open_position:
            logging.info(f"[{symbol}] Processing existing open position first...")
            position_data_global = position_tracking[symbol].get('position_data', {})
            entry_time_global = position_data_global.get('entry_time')
            if entry_time_global and isinstance(entry_time_global, datetime):
                if entry_time_global.tzinfo is None: entry_time_utc = entry_time_global.replace(tzinfo=timezone.utc)
                else: entry_time_utc = entry_time_global.astimezone(timezone.utc)

                try: from utils.backtest_strategy import calculate_optimal_context_hours; context_hours = calculate_optimal_context_hours()
                except (ImportError, NameError): context_hours = 24; logging.warning(f"[{symbol}] Using default context {context_hours}h.")

                processing_start_with_context = max(entry_time_utc - timedelta(hours=context_hours), full_df.index.min())
                period_for_open_pos = {'start_time': processing_start_with_context, 'entry_time': entry_time_utc, 'end_time': full_df.index.max(), 'price_change_pct': 0}

                logging.info(f"[{symbol}] Calling process_observation_period for open pos (Context Start: {processing_start_with_context}, End: {period_for_open_pos['end_time']})")
                df_processed, trades_open_pos, perf_open_pos = process_observation_period_with_full_data(
                    symbol=symbol, full_df_raw=full_df.copy(), period=period_for_open_pos, use_ut_bot=use_ut_bot_setting,
                    position_tracking=position_tracking, active_observation_periods=active_observation_periods, observed_symbols=observed_symbols
                )
                processed_a_period = True # Markieren, dass etwas verarbeitet wurde
                if df_processed is not None: last_processed_df = df_processed # Speichere den DF zum späteren Sichern
                if trades_open_pos: all_trades_for_symbol.extend(trades_open_pos)
                if perf_open_pos and isinstance(perf_open_pos, dict) and 'error' not in perf_open_pos: final_performance = perf_open_pos

                has_open_position = position_tracking.get(symbol, {}).get('position_open', False) # Status nach Verarbeitung prüfen
                if not has_open_position: logging.info(f"[{symbol}] Open position was closed during processing.")
                else:
                    logging.info(f"[{symbol}] Open position remains open after processing.")
                    if not close_position_setting:
                        logging.info(f"[{symbol}] Skipping search for new periods (Position open and close_position=False).")
                        # KEIN return hier, stattdessen weiter zum Speichern gehen
            else: # Ungültige Entry-Zeit
                logging.warning(f"[{symbol}] Cannot process open pos: Invalid entry_time. Resetting global state.")
                position_tracking[symbol] = {'position_open': False, 'observation_ended': False, 'position_data': {}}
                if symbol in active_observation_periods: active_observation_periods[symbol] = {'active_period': None, 'has_position': False}
                has_open_position = False

        # 2. Neue Perioden suchen (nur wenn keine offene Pos mehr ODER close_position=True)
        #    ODER wenn Filterung nicht aktiv ist (Fallback)
        if (not has_open_position or close_position_setting):
            if filtering_active and beobachten_active:
                # ... (Logik zum Finden und Verarbeiten von Beobachtungsperioden, wie zuvor) ...
                 logging.info(f"[{symbol}] Searching for new observation periods (Price Change Filter Active)...")
                 interval_minutes = parse_interval_to_minutes(Z_config.interval)
                 lookback_candles = lookback_minutes // interval_minutes if interval_minutes and interval_minutes > 0 else 0

                 if lookback_candles <= 0: logging.error(f"[{symbol}] Invalid lookback_candles ({lookback_candles}).")
                 else:
                     current_observation_end_tracker = None
                     # Finde alle qualifizierenden Perioden
                     found_periods_data = []
                     for i in range(lookback_candles, len(complete_candles)):
                         current_candle_time = complete_candles.index[i]
                         if current_observation_end_tracker and current_candle_time <= current_observation_end_tracker: continue
                         start_slice_idx = max(0, i - lookback_candles); lookback_slice = complete_candles.iloc[start_slice_idx : i + 1]
                         threshold_reached, price_change_pct, _, _ = check_price_change_threshold(
                             lookback_slice, min_pct, max_pct, lookback_minutes, direction )
                         if threshold_reached:
                              logging.info(f"[{symbol}] ✓ Price change threshold met @ {current_candle_time}: {price_change_pct:.2f}%")
                              observation_start = current_candle_time
                              if hasattr(Z_config, 'buy_delay_1_candle_spaeter') and Z_config.buy_delay_1_candle_spaeter:
                                  delay_minutes = interval_minutes; observation_start += timedelta(minutes=delay_minutes)
                              observation_end = observation_start + timedelta(hours=observation_hours)
                              observation_end = min(observation_end, full_df.index.max())
                              current_observation_end_tracker = observation_end # Verhindere Überlappung im Fund
                              found_periods_data.append({'start_time': observation_start, 'end_time': observation_end, 'price_change_pct': price_change_pct})
                     # Verarbeite alle gefundenen Perioden
                     logging.info(f"[{symbol}] Found {len(found_periods_data)} potential observation periods to process.")
                     for period_data in found_periods_data:
                         logging.info(f"[{symbol}] Processing period: {period_data['start_time']} -> {period_data['end_time']}")
                         df_obs, trades_obs, perf_obs = process_observation_period_with_full_data(
                        symbol=symbol,
                        full_df_raw=full_df.copy(), # <<< Argumentname korrigiert
                        period=period_data,
                        use_ut_bot=use_ut_bot_setting,
                        position_tracking=position_tracking,
                        active_observation_periods=active_observation_periods,
                        observed_symbols=observed_symbols
                    )
                         processed_a_period = True
                         if df_obs is not None: last_processed_df = df_obs # Überschreibe mit letztem DF
                         if trades_obs: all_trades_for_symbol.extend(trades_obs)
                         if perf_obs and isinstance(perf_obs, dict) and 'error' not in perf_obs: final_performance = perf_obs # Nimm letzte Performance
                         if symbol in observed_symbols: observed_symbols[symbol].append({'start': period_data['start_time'], 'end': period_data['end_time'], 'price_change_pct': period_data['price_change_pct']})
                         # Prüfe nach jeder Periode, ob Position offen ist und Suche abgebrochen werden soll
                         position_opened_in_period = position_tracking.get(symbol, {}).get('position_open', False)
                         if position_opened_in_period and not close_position_setting:
                             logging.info(f"[{symbol}] Position opened and close_position=False. Stopping processing further periods.")
                             break # Verlasse die Schleife der gefundenen Perioden

            elif not (filtering_active and beobachten_active): # Fallback: Keine Filterung -> Verarbeite Gesamtdaten
                 logging.info(f"[{symbol}] Filtering/Observation skipped. Processing full dataset...")
                 period_data = {'start_time': full_df.index.min(), 'end_time': full_df.index.max(), 'price_change_pct': 0}
                 df_full, trades_full, perf_full = process_observation_period_with_full_data(
                     symbol=symbol, full_df_raw=full_df.copy(), period=period_data, use_ut_bot=use_ut_bot_setting,
                     position_tracking=position_tracking, active_observation_periods=active_observation_periods, observed_symbols=observed_symbols )
                 processed_a_period = True
                 if df_full is not None: last_processed_df = df_full # Speichere den DF
                 if trades_full: all_trades_for_symbol.extend(trades_full)
                 if perf_full and isinstance(perf_full, dict) and 'error' not in perf_full: final_performance = perf_full


        # --- Nachbearbeitung & Speichern von full_data.csv ---
        logging.debug(f"[{symbol}] Checkpoint G: Preparing data for saving.")
        df_to_save = None
        data_saved_flag = False # Flag für Rückgabewert

        if last_processed_df is not None and not last_processed_df.empty:
            df_to_save = last_processed_df
            logging.debug(f"[{symbol}] Using last processed DataFrame (shape: {df_to_save.shape}) for saving.")
        elif full_df is not None and not full_df.empty and not processed_a_period:
            # Fallback: Wenn keine Periode explizit verarbeitet wurde (z.B. nur offene Pos. behalten, aber close_pos=False)
            # UND wir den full_df haben, dann berechne Signale darauf, um zumindest Indikatoren zu speichern.
            logging.warning(f"[{symbol}] No specific period processed, applying base strategy to original full_df before saving.")
            df_to_save = backtest_strategy.apply_base_strategy(full_df.copy(), symbol, use_ut_bot=use_ut_bot_setting)
        else:
            logging.error(f"[{symbol}] Cannot save strategy data: No valid DataFrame available at the end.")

        # Speichern, wenn DataFrame vorhanden ist
        if df_to_save is not None and not df_to_save.empty:
            try:
                # Erstelle einfaches conditions_met dict basierend auf finalem DF
                conditions_met_simple = {
                    'total_signals': len(df_to_save[df_to_save['signal'].isin(['buy', 'sell', 'exit_long', 'exit_short'])]) if 'signal' in df_to_save.columns else 0,
                    'buy_signals': len(df_to_save[df_to_save['signal'] == 'buy']) if 'signal' in df_to_save.columns else 0,
                    'sell_signals': len(df_to_save[df_to_save['signal'] == 'sell']) if 'signal' in df_to_save.columns else 0,
                    'exit_long_signals': len(df_to_save[df_to_save['signal'] == 'exit_long']) if 'signal' in df_to_save.columns else 0,
                    'exit_short_signals': len(df_to_save[df_to_save['signal'] == 'exit_short']) if 'signal' in df_to_save.columns else 0,
                    # Füge ggf. weitere Zählungen hinzu, wenn die Spalten existieren
                }
                logging.info(f"[{symbol}] Calling save_strategy_data...")
                print(f"Aufruf von save_strategy_data für {symbol}...")
                # Stelle sicher, dass save_strategy_data aus utils.Backtest importiert wurde
                save_strategy_data(df_to_save, conditions_met_simple, symbol)
                logging.info(f"[{symbol}] save_strategy_data finished.")
                data_saved_flag = True # Setze Flag auf True
            except ImportError:
                logging.error(f"[{symbol}] Cannot save strategy data: save_strategy_data function not found/imported.")
                print(f"FEHLER: save_strategy_data nicht gefunden für {symbol}")
            except Exception as e:
                logging.error(f"[{symbol}] Error calling save_strategy_data: {e}", exc_info=True)
                print(f"FEHLER beim Aufruf von save_strategy_data für {symbol}: {e}")
        else:
             logging.warning(f"[{symbol}] Skipping save_strategy_data because no valid DataFrame was available.")


        # --- Konsistenzcheck und Finalisierung ---
        tracking_has_pos = position_tracking.get(symbol, {}).get('position_open', False)
        active_obs_has_pos = active_observation_periods.get(symbol, {}).get('has_position', False)
        if tracking_has_pos != active_obs_has_pos:
            logging.warning(f"[{symbol}] INCONSISTENCY DETECTED at end: pos_track={tracking_has_pos} vs active_obs={active_obs_has_pos}. Syncing active_obs to pos_track.")
            if symbol not in active_observation_periods: active_observation_periods[symbol] = {}
            active_observation_periods[symbol]['has_position'] = tracking_has_pos
            if not tracking_has_pos: active_observation_periods[symbol]['active_period'] = None # Clear period if pos closed

        # --- Performance-Dictionary vervollständigen (falls leer) ---
        if not final_performance or 'symbol' not in final_performance:
            logging.warning(f"[{symbol}] No performance data generated, creating default entry.")
            if not isinstance(final_performance, dict): final_performance = {}
            final_performance.update({
                'symbol': symbol,
                'total_trades': len(all_trades_for_symbol),
                'win_rate': 0.0, 'total_profit': 0.0, 'max_drawdown': 0.0, 'profit_factor': 0.0,
                'avg_profit_win': 0.0, 'avg_loss_loss': 0.0, 'total_commission': 0.0,
                'total_slippage': 0.0, 'exit_reasons': {}, 'immediate_sl_hits': 0, 'immediate_tp_hits': 0,
                'total_exits': 0, 'avg_profit': 0.0, 'avg_slippage': 0.0, 'stop_loss_hits': 0,
                'take_profit_hits': 0, 'signal_exits': 0, 'backtest_end_exits': 0,
                'trailing_stop_hits': 0, 'observation_timeout_exits': 0, 'partial_exits': 0
            })


        # --- Rückgabe ---
        logging.info(f"Finished processing {symbol}. Returning {len(all_trades_for_symbol)} trade events. Data saved: {data_saved_flag}")
        return symbol, all_trades_for_symbol, final_performance, data_saved_flag # <-- Füge data_saved_flag hinzu

    except Exception as e:
        # --- Fehlerbehandlung ---
        logging.error(f"--- ERROR in process_symbol_parallel for {symbol}: {e} ---", exc_info=True)
        # Versuche, globalen Status zurückzusetzen
        if position_tracking is not None and symbol in position_tracking:
             position_tracking[symbol] = {'position_open': False, 'observation_ended': False, 'position_data': {}}
        if active_observation_periods is not None and symbol in active_observation_periods:
             active_observation_periods[symbol] = {'has_position': False, 'active_period': None}
        # Gib None zurück, um Fehler anzuzeigen
        return None