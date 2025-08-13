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
    precalculated_full_df, # Erhält DataFrame mit ALLEN vorab berechneten Indikatoren
    period,
    use_ut_bot=False,
    position_tracking=None,
    active_observation_periods=None,
    observed_symbols=None
):
    """
    MODIFIED: Verarbeitet einen Beobachtungszeitraum mit vorab berechneten Indikatoren.
    1. Wendet Zeitfilter an.
    2. Setzt Kontext-Flags basierend auf 'period'.
    3. Generiert Signale mit apply_base_strategy (nutzt *_iter EMAs und _iter OBV aus precalculated_full_df).
    4. Filtert Daten für die Trade-Verarbeitung.
    5. Führt Trendstärkeprüfung durch.
    6. Ruft process_symbol für die Trade-Logik auf.
    7. Aktualisiert den globalen State.

    Args:
        symbol (str): Trading-Symbol
        precalculated_full_df (pd.DataFrame): Vollständiger DataFrame mit OHLCV, is_complete, is_warmup
                                             UND ALLEN vorab berechneten Indikatoren (Standard + Iterativ).
        period (dict): Dictionary mit start_time, end_time, price_change_pct.
        use_ut_bot (bool): Ob UT Bot verwendet werden soll.
        position_tracking (dict): Globaler Positionsstatus.
        active_observation_periods (dict): Aktive Beobachtungsperioden.
        observed_symbols (dict): Protokollierte beobachtete Symbole.

    Returns:
        Tuple mit (df_processed_final, trades, performance) oder (precalculated_full_df, [], {}) bei Fehlern
        vor der Signalgenerierung.
        df_processed_final enthält den (zeitgefilterten) precalculated_full_df mit zusätzlichen Flags und Signalen.
    """
    log_prefix = f"process_obs ({symbol})"
    logger_proc_obs = logging.getLogger(f"{__name__}.{log_prefix}")
    logger_proc_obs.debug(f"Starting processing with precalculated indicators.")

    # --- Initial Checks ---
    if position_tracking is None or active_observation_periods is None or observed_symbols is None:
        logger_proc_obs.error(f"CRITICAL - Tracking dictionaries not passed!")
        # Gebe den unveränderten precalculated_full_df zurück, falls er existiert, sonst None
        return precalculated_full_df if precalculated_full_df is not None else None, [], {}
    if precalculated_full_df is None or precalculated_full_df.empty:
        logger_proc_obs.error(f"✗ No precalculated data available (precalculated_full_df).")
        return None, [], {}

    # --- Zeitrahmen validieren ---
    observation_start_time = period.get('start_time')
    observation_end_time = period.get('end_time')
    price_change_pct = period.get('price_change_pct', 0)
    if not all(isinstance(t, datetime) and t.tzinfo is not None for t in [observation_start_time, observation_end_time]):
        logger_proc_obs.error(f"Invalid or non-TZ-aware period start/end times ({observation_start_time}, {observation_end_time}).")
        return precalculated_full_df, [], {}
    observation_start_time = observation_start_time.astimezone(timezone.utc)
    observation_end_time = observation_end_time.astimezone(timezone.utc)

    logger_proc_obs.info(f"Processing Period: {observation_start_time} -> {observation_end_time} (Price Change: {price_change_pct:.2f}%)")

    # --- Schritt 1, 1b, 1c (Indikatorberechnung) SIND ENTFERNT ---
    # Die Indikatoren (Standard, iterative EMA, iterativer OBV) sind bereits in precalculated_full_df enthalten.

    # --- Schritt 1.5: Zeitfilterung ANWENDEN ---
    logger_proc_obs.debug(f"Applying trading time filter to precalculated data (shape: {precalculated_full_df.shape})...")
    df_processed = None # Initialisierung
    try:
        # Wichtig: .copy() verwenden, um den originalen precalculated_full_df nicht zu verändern
        df_processed = Backtest.filter_historical_candles_by_trading_time(precalculated_full_df.copy())
    except AttributeError:
        logger_proc_obs.error(f"Function 'Backtest.filter_historical_candles_by_trading_time' not found!")
        df_processed = precalculated_full_df.copy() # Fallback: Keine Filterung, aber Kopie erstellen
    except Exception as e:
        logger_proc_obs.error(f"Error during time filtering: {e}", exc_info=True)
        df_processed = precalculated_full_df.copy() # Fallback, aber Kopie erstellen

    if df_processed is None or df_processed.empty:
        logger_proc_obs.warning(f"DataFrame empty after applying trading time filter for period. No trades possible in this window.")
        # Gib den *ursprünglichen* precalculated_full_df zurück, da nach Filterung keine Daten mehr für diese Periode übrig sind.
        # Dieser DF könnte für das Speichern der Gesamtstrategiedaten noch relevant sein.
        return precalculated_full_df, [], {}

    logger_proc_obs.debug(f"Trading time filter applied. Shape after filter: {df_processed.shape}")
    # Sicherstellen, dass wir mit einer Kopie arbeiten, falls filter_historical_candles_by_trading_time
    # unter bestimmten Umständen das Original zurückgibt (sollte nicht, aber zur Sicherheit).
    # Wenn oben .copy() verwendet wurde, ist dies redundant, schadet aber nicht.
    df_processed = df_processed.copy()

    # --- Schritt 2: Kontext-Flags setzen ---
    logger_proc_obs.debug(f"Setting context flags based on period: {observation_start_time} -> {observation_end_time}")
    try:
        if not isinstance(df_processed.index, pd.DatetimeIndex):
             logger_proc_obs.error("Index is not DatetimeIndex after filtering. Cannot set context flags reliably.")
             return df_processed, [], {} # df_processed ist hier der zeitgefilterte DataFrame
        if df_processed.index.tz is None: df_processed.index = df_processed.index.tz_localize('UTC')
        elif str(df_processed.index.tz) != 'UTC': df_processed.index = df_processed.index.tz_convert('UTC')
    except Exception as tz_err:
         logger_proc_obs.error(f"Error ensuring index timezone awareness before flag setting: {tz_err}", exc_info=True)
         return df_processed, [], {}

    df_processed['is_context'] = df_processed.index < observation_start_time
    df_processed['is_observation'] = (df_processed.index >= observation_start_time) & (df_processed.index <= observation_end_time)
    df_processed['is_post_observation'] = df_processed.index > observation_end_time
    df_processed.loc[df_processed['is_observation'] | df_processed['is_post_observation'], 'is_context'] = False
    logger_proc_obs.info(f"Flags set: Context={df_processed['is_context'].sum()}, Obs={df_processed['is_observation'].sum()}, PostObs={df_processed['is_post_observation'].sum()}")

    # --- Schritt 3: Base Strategy anwenden (Signalgenerierung) ---
    logger_proc_obs.debug(f"Calling apply_base_strategy (uses precalculated iterative EMAs and OBV)...")
    df_with_signals = None # Initialisierung
    try:
        # apply_base_strategy muss die Spaltennamen der iterativen Indikatoren kennen (z.B. ema_fast_iter, obv_iter)
        df_with_signals = backtest_strategy.apply_base_strategy(df_processed, symbol, use_ut_bot=use_ut_bot)
    except AttributeError:
        logger_proc_obs.critical(f"Function 'backtest_strategy.apply_base_strategy' not found!")
        return df_processed, [], {}
    except Exception as e:
        logger_proc_obs.error(f"Error during apply_base_strategy: {e}", exc_info=True)
        return df_processed, [], {}

    if df_with_signals is None:
        logger_proc_obs.error(f"apply_base_strategy returned None.")
        return df_processed, [], {}
    logger_proc_obs.debug(f"apply_base_strategy finished. Signal counts in df_with_signals:\n{df_with_signals['signal'].value_counts(dropna=False)}")

    # --- Schritt 4: Signale außerhalb der Observation zurücksetzen ---
    logger_proc_obs.debug(f"Resetting signals outside 'is_observation' (and 'is_post_observation' if close_position=True)...")
    close_position_setting_signals = getattr(Z_config, 'close_position', False)
    if 'is_observation' in df_with_signals.columns and 'is_post_observation' in df_with_signals.columns:
         if close_position_setting_signals:
             reset_mask = ~df_with_signals['is_observation']
             logger_proc_obs.debug("close_position=True: Resetting signals where is_observation=False.")
         else:
             reset_mask = ~(df_with_signals['is_observation'] | df_with_signals['is_post_observation'])
             logger_proc_obs.debug("close_position=False: Resetting signals where neither is_observation nor is_post_observation is True.")

         df_with_signals.loc[reset_mask, 'signal'] = 'no_signal'
         df_with_signals.loc[reset_mask, 'trigger'] = None
    else:
        logger_proc_obs.warning(f"'is_observation'/'is_post_observation' columns not found in df_with_signals. Cannot reset signals based on period.")

    # --- Schritt 5: Verify Indicators (Optional) ---
    try:
        verify_observation_indicators(df_with_signals, observation_start_time, symbol)
    except NameError:
        logger_proc_obs.debug(f"verify_observation_indicators not available.")
    except Exception as verif_err:
        logger_proc_obs.error(f"Error during verify_observation_indicators: {verif_err}", exc_info=True)

    # --- Schritt 6: Relevanten DataFrame für Trade Processing extrahieren ---
    logger_proc_obs.debug(f"Extracting DataFrame slice for trade processing...")
    close_position_setting_trades = getattr(Z_config, 'close_position', False)
    observation_df_for_trades = pd.DataFrame()

    has_obs_flags_trades = 'is_observation' in df_with_signals.columns and 'is_post_observation' in df_with_signals.columns

    if has_obs_flags_trades:
        if close_position_setting_trades:
            observation_df_for_trades = df_with_signals[df_with_signals['is_observation'] == True].copy()
            logger_proc_obs.info(f"close_position=True: Using {len(observation_df_for_trades)} observation candles for trade processing.")
        else:
            trade_processing_mask = (df_with_signals['is_observation'] == True) | (df_with_signals['is_post_observation'] == True)
            observation_df_for_trades = df_with_signals[trade_processing_mask].copy()
            obs_count = (observation_df_for_trades['is_observation'] == True).sum()
            post_obs_count = (observation_df_for_trades['is_post_observation'] == True).sum()
            logger_proc_obs.info(f"close_position=False: Using {len(observation_df_for_trades)} candles ({obs_count} Obs + {post_obs_count} Post-Obs) for trade processing.")
    else:
        logger_proc_obs.warning(f"Observation flags missing in df_with_signals, using entire df_with_signals for trade processing.")
        observation_df_for_trades = df_with_signals.copy()

    if observation_df_for_trades.empty:
        logger_proc_obs.warning(f"observation_df_for_trades is empty after filtering for this period. No trades possible.")
        return df_with_signals, [], {}

    # --- Schritt 7: Trend Strength Check ---
    logger_proc_obs.debug(f"Starting Trend Strength Check...")
    trend_strength_check_df = observation_df_for_trades[observation_df_for_trades.get('is_complete', True)].copy()

    meets_strength_criteria = False
    strongest_trend_value = 0.0
    strongest_trend_direction = 0
    strongest_trend_time = None
    min_trend_strength_cfg = getattr(Z_config, 'min_trend_strength_parameter', 1.0)

    trend_strength_col = 'trend_strength_iter' if 'trend_strength_iter' in trend_strength_check_df.columns else 'trend_strength'
    trend_col = 'trend_iter' if 'trend_iter' in trend_strength_check_df.columns else 'trend'

    if not trend_strength_check_df.empty and trend_strength_col in trend_strength_check_df.columns and trend_col in trend_strength_check_df.columns:
        logger_proc_obs.debug(f"Checking trend strength in {len(trend_strength_check_df)} complete trade processing candles (using {trend_strength_col})...")
        # idxmax() kann fehlschlagen, wenn die Spalte nur NaNs enthält
        if trend_strength_check_df[trend_strength_col].notna().any():
            strongest_trend_idx = trend_strength_check_df[trend_strength_col].abs().idxmax()
            if pd.notna(strongest_trend_idx):
                strongest_candle = trend_strength_check_df.loc[strongest_trend_idx]
                strongest_trend_value = float(strongest_candle.get(trend_strength_col, 0.0))
                strongest_trend_direction = int(strongest_candle.get(trend_col, 0))
                strongest_trend_time = strongest_trend_idx
                meets_strength_criteria = (trend_strength_check_df[trend_strength_col].abs() >= min_trend_strength_cfg).any()
        else:
            logger_proc_obs.warning(f"Trend strength column '{trend_strength_col}' contains all NaNs. Cannot determine strongest trend.")


        if meets_strength_criteria:
            logger_proc_obs.info(f"✓ Trend strength criterion met somewhere in period. Strongest observed: {strongest_trend_value:.1f}% (Dir: {strongest_trend_direction}) @ {strongest_trend_time}")
        else:
             logger_proc_obs.info(f"✗ Trend strength criterion ({min_trend_strength_cfg}%) NOT met in trade processing period. Max observed: {strongest_trend_value:.1f}%")
    else:
        logger_proc_obs.warning(f"No complete candles or required columns ({trend_strength_col}, {trend_col}) for trend strength check in trade processing data.")

    is_tracking_open_position = position_tracking.get(symbol, {}).get('position_open', False)
    if is_tracking_open_position:
        logger_proc_obs.info(f"ℹ️ Skipping trend strength requirement check as open position is tracked.")
        meets_strength_criteria = True
        if strongest_trend_time is None and not observation_df_for_trades.empty:
            strongest_trend_time = observation_df_for_trades.index[0]

    if not meets_strength_criteria:
        logger_proc_obs.info(f"Skipping trade processing: Unmet trend strength and no open position tracked.")
        return df_with_signals, [], {}

    # --- Schritt 8: Trade Processing Call ---
    symbol_data_for_process = {
        'symbol': symbol,
        'current_time': strongest_trend_time if strongest_trend_time else observation_start_time
    }
    logger_proc_obs.info(f"Calling process_symbol with {len(observation_df_for_trades)} relevant candles...")
    result = None # Initialisierung
    try:
        result = backtest_strategy.process_symbol(
        symbol=symbol,
        df=observation_df_for_trades,
        symbol_data=symbol_data_for_process,
        start_time=observation_start_time,
        position_tracking=position_tracking,
        active_observation_periods_dict=active_observation_periods, # HINZUGEFÜGT
        observed_symbols_dict=observed_symbols                      # HINZUGEFÜGT
    )
    except AttributeError:
        logger_proc_obs.critical(f"Function 'backtest_strategy.process_symbol' not found!")
        return df_with_signals, [], {}
    except Exception as e:
        logger_proc_obs.error(f"Error calling process_symbol: {e}", exc_info=True)
        return df_with_signals, [], {}

    # --- Schritt 9: Ergebnisse verarbeiten und State Update ---
    trades = []
    performance = {}
    if result and isinstance(result, tuple) and len(result) == 2:
        trades, performance = result
        logger_proc_obs.info(f"process_symbol resulted in {len(trades)} trade events.")
        if isinstance(performance, dict) and 'symbol' not in performance:
            performance['symbol'] = symbol
    else:
        logger_proc_obs.error(f"✗ process_symbol returned invalid results! Type: {type(result)}")
        performance = {'symbol': symbol, 'total_trades': 0, 'error': 'process_symbol returned invalid result'}

    logger_proc_obs.debug(f"Starting post-processing state update...")
    try:
        position_still_open_after = position_tracking.get(symbol, {}).get('position_open', False)

        if symbol not in active_observation_periods: # active_observation_periods ist hier der übergebene Parameter
            active_observation_periods[symbol] = {'active_period': None, 'has_position': False}

        current_pos_type = None # Hole den Positionstyp, falls vorhanden
        if position_still_open_after and position_tracking.get(symbol, {}).get('position_data'):
            current_pos_type = position_tracking[symbol]['position_data'].get('position_type')


        if position_still_open_after:
            logger_proc_obs.info(f"⚠️ Position remains OPEN for {symbol}. Updating active_observation_periods.")
            active_observation_periods[symbol]['has_position'] = True
            if not active_observation_periods[symbol].get('active_period'):
                active_observation_periods[symbol]['active_period'] = period # 'period' ist die aktuelle Periode dieser Funktion
                logger_proc_obs.info(f"Setting active_period for open position: {period.get('start_time')} to {period.get('end_time')}")
            # ... restliche Logik für offene Positionen ...
            try:
                # HIER DIE ANPASSUNG: Dictionaries übergeben
                update_symbol_position_status(
                    symbol,
                    True, # has_position
                    active_observation_periods, # das übergebene dict
                    observed_symbols,           # das übergebene dict
                    position_type=current_pos_type
                )
                logger_proc_obs.info(f"Notified symbol_filter: Status updated to 'position open'.")
            except NameError: logger_proc_obs.debug("update_symbol_position_status not imported/available.")
            except Exception as e_usps: logger_proc_obs.error(f"Error calling update_symbol_position_status for open pos: {e_usps}")

        else: # Position is CLOSED
            logger_proc_obs.info(f"✓ Position is CLOSED for {symbol}. Updating active_observation_periods.")
            active_observation_periods[symbol]['has_position'] = False
            # Wenn die Periode die aktuelle war, und die Position geschlossen wurde, kann die Periode auch weg
            if active_observation_periods[symbol].get('active_period') == period:
                 active_observation_periods[symbol]['active_period'] = None
                 logger_proc_obs.info(f"Active period for {symbol} cleared as position closed within it.")

            if position_tracking.get(symbol, {}).get('position_open'):
                logger_proc_obs.error(f"CRITICAL INCONSISTENCY: Position closed, but position_tracking still shows open for {symbol}! Resetting global state manually.")
                position_tracking[symbol] = {'position_open': False, 'observation_ended': False, 'position_data': {}}
            try:
                # HIER DIE ANPASSUNG: Dictionaries übergeben
                update_symbol_position_status(
                    symbol,
                    False, # has_position
                    active_observation_periods, # das übergebene dict
                    observed_symbols,           # das übergebene dict
                    position_type=None # Kein Typ, da keine Position
                )
                logger_proc_obs.info(f"Notified symbol_filter: Status updated to 'no position'.")
            except NameError: logger_proc_obs.debug("update_symbol_position_status not imported/available.")
            except Exception as e_usps: logger_proc_obs.error(f"Error calling update_symbol_position_status for closed pos: {e_usps}")
    except Exception as post_process_err:
        logger_proc_obs.error(f"Error during post-processing state update: {post_process_err}", exc_info=True)
    logger_proc_obs.debug(f"Post-processing state update finished.")

    # --- Schritt 10: Rückgabe ---
    # df_with_signals ist der (zeitgefilterte) DataFrame mit allen vorab berechneten Indikatoren, Flags und Signalen.
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
        # Im Fehlerfall ein Tupel zurückgeben, das dem erwarteten Format entspricht, um Abstürze in der aufrufenden Schleife zu vermeiden
        return symbol, [], {'symbol': symbol, 'total_trades': 0, 'error': 'Tracking dictionary None'}, False
        
    if not isinstance(current_time, datetime) or current_time.tzinfo is None:
        logging.error(f"CRITICAL ({symbol}): current_time is not a timezone-aware datetime object!")
        if isinstance(current_time, datetime) and current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)
        else:
            return symbol, [], {'symbol': symbol, 'total_trades': 0, 'error': 'Invalid current_time'}, False

    # Erstelle einen spezifischen Logger für diese Funktion und dieses Symbol
    logger_psp = logging.getLogger(f"process_symbol_parallel.{symbol}")

    try:
        # --- Imports innerhalb der Funktion (um Kapselung zu gewährleisten, falls als eigenständige Einheit betrachtet) ---
        # In einer typischen Struktur wären diese eher auf Modulebene.
        from datetime import timedelta
        import Z_config
        import utils.backtest_strategy as backtest_strategy
        from utils.Backtest import check_rate_limit, parse_interval_to_minutes, save_strategy_data
        from utils.symbol_filter import check_price_change_threshold # verify_observation_indicators, update_symbol_position_status (falls hier direkt genutzt)
        from utils.parallel_processing import process_observation_period_with_full_data
        from utils import indicators # Für die einmalige Indikatorberechnung

        # --- Start Processing ---
        logger_psp.info(f"\n{'-'*15} Starting processing for: {symbol} {'-'*15}")

        # --- Check for Existing Open Position ---
        has_open_position_initial = False
        if symbol in position_tracking and position_tracking[symbol].get('position_open', False):
            has_open_position_initial = True
            logger_psp.info(f"Found existing open position from passed state.")
            pos_data_debug = position_tracking[symbol].get('position_data', {})
            logger_psp.debug(f"  Details: Entry={pos_data_debug.get('entry_time','?')}, Type={pos_data_debug.get('position_type','?')}, Qty={pos_data_debug.get('remaining_quantity','?')}")

        # --- Fetch Data ---
        check_rate_limit(weight=1)
        logger_psp.debug(f"Calling get_multi_timeframe_data...")
        overall_lookback_hours = getattr(Z_config, 'lookback_hours_parameter', 24*7)
        logical_start_time = current_time - timedelta(hours=overall_lookback_hours)

        logger_psp.debug(f"Calling get_multi_timeframe_data with logical start: {logical_start_time}, end: {current_time}")
        full_df = backtest_strategy.get_multi_timeframe_data(
            symbol=symbol,
            end_time=current_time,
            initial_start_time=logical_start_time,
            position_tracking=position_tracking
        )

        if full_df is None or full_df.empty:
            logger_psp.warning(f"No data returned from get_multi_timeframe_data.")
            return symbol, [], {'symbol': symbol, 'total_trades': 0, 'error': 'No base data from get_multi_timeframe_data'}, False

        # --- BEGINN EINMALIGE INDIKATORBERECHNUNG ---
        df_with_all_indicators = None
        logger_psp.info(f"Starting ONE-TIME full indicator calculation for the entire dataset (Shape: {full_df.shape})...")
        try:
            # 1. Standard Indikatoren (OHNE iterative EMA/OBV, falls die separat kommen)
            logger_psp.debug(f"Calculating standard indicators (one-time)...")
            # Annahme: calculate_indicators berechnet KEINE iterativen EMAs/OBV, wenn calculate_for_alignment_only=False
            # oder eine andere Logik stellt dies sicher.
            df_std_ind_once = indicators.calculate_indicators(full_df.copy(), calculate_for_alignment_only=False)
            if df_std_ind_once is None:
                raise ValueError("Standard indicator calculation (once) returned None.")
            logger_psp.debug(f"Standard indicators calculated. Shape: {df_std_ind_once.shape}")

            # 2. Iterative EMAs (einmalig auf dem gesamten Datensatz)
            logger_psp.debug(f"Calculating iterative EMAs (one-time)...")
            required_candles_ema_once = backtest_strategy.calculate_required_candles()
            df_iter_ema_once = indicators.calculate_ema_iterative(df_std_ind_once, required_candles_ema_once)
            if df_iter_ema_once is None:
                raise ValueError("Iterative EMA calculation (once) returned None.")
            logger_psp.debug(f"Iterative EMAs calculated. Shape: {df_iter_ema_once.shape}")

            # 3. Iterative OBV (einmalig auf dem gesamten Datensatz)
            logger_psp.debug(f"Calculating iterative OBV (one-time)...")
            # Initialisiere Spalten im DataFrame, der bereits iterative EMAs enthält
            df_with_obv_calc = df_iter_ema_once.copy() # Arbeite mit einer Kopie für die OBV-Berechnung
            df_with_obv_calc['obv_iter'] = np.nan
            df_with_obv_calc['obv_trend_iter'] = 0

            non_warmup_indices_obv_once = df_with_obv_calc[~df_with_obv_calc['is_warmup']].index
            first_backtest_iloc_obv_once = -1
            if not non_warmup_indices_obv_once.empty:
                try:
                    first_backtest_iloc_obv_once = df_with_obv_calc.index.get_loc(non_warmup_indices_obv_once[0])
                except KeyError:
                    logger_psp.error(f"Could not find first non-warmup index for one-time iterative OBV.")

            if first_backtest_iloc_obv_once != -1:
                required_candles_obv_once = backtest_strategy.calculate_required_candles()
                if required_candles_obv_once <= 0:
                    logger_psp.warning(f"calculate_required_candles for OBV returned {required_candles_obv_once}. Using fallback 100.")
                    required_candles_obv_once = 100

                logger_psp.info(f"Starting one-time iterative OBV loop from index {first_backtest_iloc_obv_once} using {required_candles_obv_once} required candles.")
                # Iterative OBV Schleife (EINMALIGER DURCHLAUF)
                for i_once in tqdm(range(first_backtest_iloc_obv_once, len(df_with_obv_calc)), desc=f"[{symbol}] One-Time OBV Iter", unit="c", leave=False, mininterval=10.0, ncols=80):
                    current_ts_iter_once = df_with_obv_calc.index[i_once]
                    start_slice_iloc_iter_once = max(0, i_once - required_candles_obv_once + 1)
                    end_slice_iloc_iter_once = i_once + 1

                    if end_slice_iloc_iter_once - start_slice_iloc_iter_once < 2: # Benötigt min. 2 Punkte für diff() in on_balance_volume
                        df_with_obv_calc.loc[current_ts_iter_once, 'obv_iter'] = 0.0 # Oder np.nan, je nach gewünschtem Startverhalten
                        continue

                    data_slice_iter_once = df_with_obv_calc.iloc[start_slice_iloc_iter_once:end_slice_iloc_iter_once]

                    if data_slice_iter_once['close'].notna().any() and data_slice_iter_once['volume'].notna().any():
                        # Annahme: indicators.on_balance_volume kann mit einer Series umgehen und gibt eine Series zurück
                        obv_series_slice_once = indicators.on_balance_volume(data_slice_iter_once['close'], data_slice_iter_once['volume'], symbol=f"{symbol}_onetime_slice_{i_once}")
                        if obv_series_slice_once is not None and not obv_series_slice_once.empty:
                            df_with_obv_calc.loc[current_ts_iter_once, 'obv_iter'] = obv_series_slice_once.iloc[-1]
                        else:
                            df_with_obv_calc.loc[current_ts_iter_once, 'obv_iter'] = np.nan
                    else:
                        df_with_obv_calc.loc[current_ts_iter_once, 'obv_iter'] = np.nan
                
                df_with_obv_calc['obv_iter'] = df_with_obv_calc['obv_iter'].fillna(0) # Oder eine andere geeignete Füllmethode
                df_with_obv_calc['obv_trend_iter'] = np.sign(df_with_obv_calc['obv_iter'].diff().fillna(0)).astype(int)
                logger_psp.debug(f"One-time iterative OBV calculation finished.")
            else:
                logger_psp.warning(f"Skipping one-time iterative OBV as no non-warmup index found. OBV columns will be default.")
                df_with_obv_calc['obv_iter'] = 0.0 # Default-Werte sicherstellen
                df_with_obv_calc['obv_trend_iter'] = 0
            
            df_with_all_indicators = df_with_obv_calc # Dies ist der DataFrame mit allen Indikatoren

        except Exception as e_one_time_calc:
            logger_psp.error(f"CRITICAL ERROR during ONE-TIME full indicator calculation: {e_one_time_calc}", exc_info=True)
            return symbol, [], {'symbol': symbol, 'total_trades': 0, 'error': f'One-time indicator calc failed: {e_one_time_calc}'}, False
        
        if df_with_all_indicators is None: # Zusätzlicher Check
             logger_psp.error(f"CRITICAL: df_with_all_indicators is None after one-time calculation block for {symbol}.")
             return symbol, [], {'symbol': symbol, 'total_trades': 0, 'error': 'df_with_all_indicators is None'}, False

        logger_psp.info(f"ONE-TIME full indicator calculation completed. Final DF shape: {df_with_all_indicators.shape}")
        # --- ENDE EINMALIGE INDIKATORBERECHNUNG ---

        # --- Konfigurationsparameter holen (bleibt gleich) ---
        filtering_active = getattr(Z_config, 'filtering_active', True)
        beobachten_active = getattr(Z_config, 'beobachten_active', True)
        observation_hours = getattr(Z_config, 'symbol_observation_hours', 48)
        min_pct = getattr(Z_config, 'min_price_change_pct_min', 2.0)
        max_pct = getattr(Z_config, 'min_price_change_pct_max', 50.0)
        lookback_minutes = getattr(Z_config, 'price_change_lookback_minutes', 60*12)
        direction = getattr(Z_config, 'seite', 'both')
        close_position_setting = getattr(Z_config, 'close_position', False)
        use_ut_bot_setting = getattr(Z_config, 'ut_bot', False) # Annahme: ut_bot ist ein boolscher Schalter

        # --- Initialisierung für Ergebnisse ---
        all_trades_for_symbol = []
        final_performance = {}
        last_processed_df_for_saving = None # DataFrame zum Speichern am Ende

        # --- Daten filtern und vorbereiten (auf Basis des ursprünglichen full_df für die Periodenfindung) ---
        # Die Indikatoren sind jetzt in df_with_all_indicators. Die Periodenfindung basiert weiterhin auf Rohpreisen.
        temp_full_df_for_period_finding = full_df[full_df.index <= current_time]
        complete_candles_for_period_finding = temp_full_df_for_period_finding[temp_full_df_for_period_finding.get('is_complete', True)]

        if complete_candles_for_period_finding.empty:
            logger_psp.warning(f"No complete candles found in the fetched data for period finding.")
            # Auch wenn keine Perioden gefunden werden, könnten offene Positionen verarbeitet werden
            # oder ein Fallback-Processing des gesamten Datensatzes stattfinden.
            # Das Speichern von df_with_all_indicators könnte immer noch sinnvoll sein.
            # Wir setzen hier fort und lassen die spätere Logik entscheiden.
        
        logger_psp.info(f"Data for period finding: {len(temp_full_df_for_period_finding)} rows. Complete candles: {len(complete_candles_for_period_finding)}")

        if symbol not in observed_symbols:
            observed_symbols[symbol] = {}

        # --- Verarbeitung (Offene Position oder Suche nach Perioden) ---
        processed_a_period_flag = False
        # Die Variable has_open_position wurde bereits initial gesetzt (has_open_position_initial)
        # Wir müssen sie nach der Verarbeitung einer offenen Position ggf. aktualisieren.
        current_has_open_position = has_open_position_initial


        # 1. Offene Position verarbeiten
        if current_has_open_position:
            logger_psp.info(f"Processing existing open position first...")
            position_data_global = position_tracking[symbol].get('position_data', {})
            entry_time_global = position_data_global.get('entry_time')
            if entry_time_global and isinstance(entry_time_global, datetime):
                if entry_time_global.tzinfo is None: entry_time_utc = entry_time_global.replace(tzinfo=timezone.utc)
                else: entry_time_utc = entry_time_global.astimezone(timezone.utc)

                try: from utils.backtest_strategy import calculate_optimal_context_hours; context_hours = calculate_optimal_context_hours()
                except (ImportError, NameError): context_hours = 24; logger_psp.warning(f"Using default context {context_hours}h for open position.")

                # Der Start der Periode für eine offene Position sollte den Kontext für Indikatoren berücksichtigen.
                # df_with_all_indicators enthält bereits den vollen Kontext.
                # Die 'period' definiert den relevanten Ausschnitt für die Trade-Logik.
                # Der Start der Periode sollte der Entry-Zeitpunkt sein, minus Kontext für die Signale, die zum Entry führten.
                # Da wir jetzt den *gesamten* df_with_all_indicators haben, ist der "Start" der Periode hier
                # eher der Beginn des relevanten Ausschnitts für die *Fortführung* der Position.
                # Der 'start_time' des period-Dictionary für process_observation_period_with_full_data
                # wird verwendet, um is_context, is_observation zu setzen.
                # Für eine offene Position ist alles ab Entry 'is_observation' oder 'is_post_observation'.
                # Der Kontext davor ist bereits in df_with_all_indicators enthalten.
                
                # Der Start der "Beobachtung" für die offene Position ist die Entry-Zeit.
                # Der "Kontext" davor ist bereits im df_with_all_indicators.
                # Die Endzeit ist das Ende des gesamten Datenframes.
                period_start_for_open_pos = entry_time_utc # Die Beobachtung beginnt mit dem Entry
                period_end_for_open_pos = df_with_all_indicators.index.max() # Bis zum Ende der verfügbaren Daten

                period_for_open_pos = {
                    'start_time': period_start_for_open_pos, # Ab hier ist 'is_observation'
                    'entry_time': entry_time_utc, # Info für die Funktion
                    'end_time': period_end_for_open_pos, # Ende der Periode
                    'price_change_pct': 0 # Nicht relevant für offene Positionen
                }
                
                logger_psp.info(f"Calling process_observation_period for open pos (Period Start: {period_for_open_pos['start_time']}, End: {period_for_open_pos['end_time']})")
                # Wichtig: precalculated_full_df übergeben!
                df_processed_open, trades_open_pos, perf_open_pos = process_observation_period_with_full_data(
                    symbol=symbol,
                    precalculated_full_df=df_with_all_indicators.copy(), # Die vorab berechneten Daten
                    period=period_for_open_pos,
                    use_ut_bot=use_ut_bot_setting,
                    position_tracking=position_tracking,
                    active_observation_periods=active_observation_periods,
                    observed_symbols=observed_symbols
                )
                processed_a_period_flag = True
                if df_processed_open is not None: last_processed_df_for_saving = df_processed_open
                if trades_open_pos: all_trades_for_symbol.extend(trades_open_pos)
                if perf_open_pos and isinstance(perf_open_pos, dict) and 'error' not in perf_open_pos: final_performance = perf_open_pos

                current_has_open_position = position_tracking.get(symbol, {}).get('position_open', False) # Status aktualisieren
                if not current_has_open_position: logger_psp.info(f"Open position was closed during processing.")
                else:
                    logger_psp.info(f"Open position remains open after processing.")
                    if not close_position_setting: # Z_config.close_position
                        logger_psp.info(f"Skipping search for new periods (Position open and close_position=False).")
                        # Nicht returnen, da df_with_all_indicators noch gespeichert werden soll
            else:
                logger_psp.warning(f"Cannot process open pos: Invalid entry_time in global state. Resetting global state.")
                position_tracking[symbol] = {'position_open': False, 'observation_ended': False, 'position_data': {}}
                if symbol in active_observation_periods: active_observation_periods[symbol] = {'active_period': None, 'has_position': False}
                current_has_open_position = False

        # 2. Neue Perioden suchen (nur wenn keine offene Pos mehr ODER close_position_setting=True)
        if not current_has_open_position or close_position_setting:
            if filtering_active and beobachten_active:
                 logger_psp.info(f"Searching for new observation periods (Price Change Filter Active)...")
                 interval_minutes_cfg = parse_interval_to_minutes(Z_config.interval)
                 lookback_candles_cfg = lookback_minutes // interval_minutes_cfg if interval_minutes_cfg and interval_minutes_cfg > 0 else 0

                 if lookback_candles_cfg <= 0:
                     logger_psp.error(f"Invalid lookback_candles ({lookback_candles_cfg}) for period finding.")
                 else:
                     current_observation_end_tracker = None
                     found_periods_data = []
                     # Verwende complete_candles_for_period_finding (basiert auf Roh-full_df)
                     for i_period in range(lookback_candles_cfg, len(complete_candles_for_period_finding)):
                         current_candle_time_period = complete_candles_for_period_finding.index[i_period]
                         if current_observation_end_tracker and current_candle_time_period <= current_observation_end_tracker:
                             continue
                         
                         start_slice_idx_period = max(0, i_period - lookback_candles_cfg)
                         lookback_slice_period = complete_candles_for_period_finding.iloc[start_slice_idx_period : i_period + 1]
                         
                         threshold_reached, price_change_pct_period, _, _ = check_price_change_threshold(
                             lookback_slice_period, min_pct, max_pct, lookback_minutes, direction
                         )
                         if threshold_reached:
                              logger_psp.info(f"✓ Price change threshold met @ {current_candle_time_period}: {price_change_pct_period:.2f}%")
                              observation_start_period = current_candle_time_period
                              if hasattr(Z_config, 'buy_delay_1_candle_spaeter') and Z_config.buy_delay_1_candle_spaeter:
                                  delay_minutes_period = interval_minutes_cfg
                                  observation_start_period += timedelta(minutes=delay_minutes_period)
                              
                              observation_end_period = observation_start_period + timedelta(hours=observation_hours)
                              # Stelle sicher, dass das Ende nicht über die vorhandenen Daten in df_with_all_indicators hinausgeht
                              observation_end_period = min(observation_end_period, df_with_all_indicators.index.max())
                              
                              current_observation_end_tracker = observation_end_period
                              found_periods_data.append({
                                  'start_time': observation_start_period,
                                  'end_time': observation_end_period,
                                  'price_change_pct': price_change_pct_period
                              })
                     
                     logger_psp.info(f"Found {len(found_periods_data)} potential new observation periods to process.")
                     for period_data_item in found_periods_data:
                         logger_psp.info(f"Processing period: {period_data_item['start_time']} -> {period_data_item['end_time']}")
                         # Wichtig: precalculated_full_df übergeben!
                         df_obs_new, trades_obs_new, perf_obs_new = process_observation_period_with_full_data(
                            symbol=symbol,
                            precalculated_full_df=df_with_all_indicators.copy(), # Die vorab berechneten Daten
                            period=period_data_item,
                            use_ut_bot=use_ut_bot_setting,
                            position_tracking=position_tracking,
                            active_observation_periods=active_observation_periods,
                            observed_symbols=observed_symbols
                         )
                         processed_a_period_flag = True
                         if df_obs_new is not None: last_processed_df_for_saving = df_obs_new
                         if trades_obs_new: all_trades_for_symbol.extend(trades_obs_new)
                         if perf_obs_new and isinstance(perf_obs_new, dict) and 'error' not in perf_obs_new: final_performance = perf_obs_new
                         
                         if symbol in observed_symbols: # Sicherstellen, dass observed_symbols[symbol] eine Liste ist
                            if not isinstance(observed_symbols[symbol], list): observed_symbols[symbol] = []
                            observed_symbols[symbol].append({
                                 'start': period_data_item['start_time'],
                                 'end': period_data_item['end_time'],
                                 'price_change_pct': period_data_item['price_change_pct']
                            })

                         position_opened_in_this_period = position_tracking.get(symbol, {}).get('position_open', False)
                         if position_opened_in_this_period and not close_position_setting:
                             logger_psp.info(f"Position opened during period processing and close_position=False. Stopping search for further new periods.")
                             break # Verlasse die Schleife der gefundenen Perioden
            
            elif not (filtering_active and beobachten_active): # Fallback: Keine Filterung
                 logger_psp.info(f"Filtering/Observation skipped. Processing full dataset as a single period...")
                 # Definiere eine Periode, die den gesamten precalculated_full_df abdeckt
                 # Der "Kontext" ist bereits enthalten. is_observation beginnt mit der ersten Kerze.
                 period_data_full_fallback = {
                     'start_time': df_with_all_indicators.index.min(), # Ab hier ist is_observation
                     'end_time': df_with_all_indicators.index.max(),
                     'price_change_pct': 0
                 }
                 df_full_fallback, trades_full_fallback, perf_full_fallback = process_observation_period_with_full_data(
                     symbol=symbol,
                     precalculated_full_df=df_with_all_indicators.copy(),
                     period=period_data_full_fallback,
                     use_ut_bot=use_ut_bot_setting,
                     position_tracking=position_tracking,
                     active_observation_periods=active_observation_periods,
                     observed_symbols=observed_symbols
                 )
                 processed_a_period_flag = True
                 if df_full_fallback is not None: last_processed_df_for_saving = df_full_fallback
                 if trades_full_fallback: all_trades_for_symbol.extend(trades_full_fallback)
                 if perf_full_fallback and isinstance(perf_full_fallback, dict) and 'error' not in perf_full_fallback: final_performance = perf_full_fallback

        # --- Nachbearbeitung & Speichern von full_data.csv ---
        logger_psp.debug(f"Preparing data for saving (save_strategy_data call).")
        df_to_save_final = None
        data_saved_flag = False

        if last_processed_df_for_saving is not None and not last_processed_df_for_saving.empty:
            # Dies ist der DataFrame, der aus dem letzten Aufruf von process_observation_period_with_full_data stammt.
            # Er sollte bereits alle Indikatoren, Flags und Signale für die letzte verarbeitete Periode enthalten.
            # Für das Speichern des "gesamten" Laufs ist df_with_all_indicators relevanter, wenn keine Periode verarbeitet wurde.
            df_to_save_final = last_processed_df_for_saving
            logger_psp.debug(f"Using last_processed_df_for_saving (shape: {df_to_save_final.shape}).")
        elif df_with_all_indicators is not None and not df_with_all_indicators.empty and not processed_a_period_flag :
            # Fallback: Wenn keine spezifische Periode verarbeitet wurde, aber wir den df_with_all_indicators haben,
            # dann berechne Signale darauf, um zumindest den voll indizierten Datensatz zu speichern.
            logger_psp.warning(f"No specific period was processed. Applying base strategy to df_with_all_indicators before saving.")
            # apply_base_strategy benötigt möglicherweise Parameter, die aus Z_config stammen
            # oder als Argumente übergeben werden. Hier verwenden wir die Defaults oder Z_config.
            df_to_save_final = backtest_strategy.apply_base_strategy(df_with_all_indicators.copy(), symbol, use_ut_bot=use_ut_bot_setting)
            if df_to_save_final is None: # Sicherstellen, dass es nicht None ist
                logger_psp.error(f"apply_base_strategy on df_with_all_indicators returned None for {symbol}.")
                df_to_save_final = df_with_all_indicators.copy() # Fallback zum Speichern des Roh-Indikator-DFs
        elif df_with_all_indicators is not None and not df_with_all_indicators.empty:
            # Wenn Perioden verarbeitet wurden, aber last_processed_df_for_saving leer/None ist,
            # nimm den df_with_all_indicators als Basis, auf dem dann die Signale der letzten Periode wären,
            # aber das ist komplex. Sicherer ist es, df_with_all_indicators zu nehmen, auf dem die Signale für
            # die *gesamte* Zeitspanne berechnet werden müssten.
            # Da last_processed_df_for_saving bereits Signale enthält, ist dieser Fall weniger wahrscheinlich,
            # außer process_observation_period_with_full_data gibt manchmal None zurück.
            # Für Konsistenz: Wenn etwas verarbeitet wurde, sollte last_processed_df_for_saving gesetzt sein.
            # Wenn nicht, und wir wollen trotzdem speichern, dann ist der Fall oben (not processed_a_period_flag) relevanter.
            # Wenn processed_a_period_flag True ist, aber last_processed_df_for_saving None ist, ist das ein Fehler in der Logik.
            # Hier nehmen wir an, wir wollen den voll indizierten Datensatz speichern, wenn nichts Besseres da ist.
            if df_to_save_final is None : # Nur wenn noch nicht gesetzt
                 logger_psp.warning(f"last_processed_df_for_saving was not set, but indicators were calculated. Saving df_with_all_indicators (possibly without signals from a specific period). Applying base strategy globally.")
                 df_to_save_final = backtest_strategy.apply_base_strategy(df_with_all_indicators.copy(), symbol, use_ut_bot=use_ut_bot_setting)


        if df_to_save_final is not None and not df_to_save_final.empty:
            try:
                conditions_met_simple = {
                    'total_signals': len(df_to_save_final[df_to_save_final['signal'].isin(['buy', 'sell', 'exit_long', 'exit_short'])]) if 'signal' in df_to_save_final.columns else 0,
                    'buy_signals': len(df_to_save_final[df_to_save_final['signal'] == 'buy']) if 'signal' in df_to_save_final.columns else 0,
                    'sell_signals': len(df_to_save_final[df_to_save_final['signal'] == 'sell']) if 'signal' in df_to_save_final.columns else 0,
                }
                logger_psp.info(f"Calling save_strategy_data for {symbol}...")
                save_strategy_data(df_to_save_final, conditions_met_simple, symbol)
                logger_psp.info(f"save_strategy_data finished for {symbol}.")
                data_saved_flag = True
            except ImportError:
                logger_psp.error(f"Cannot save strategy data: save_strategy_data function not found/imported.")
            except Exception as e_save:
                logger_psp.error(f"Error calling save_strategy_data for {symbol}: {e_save}", exc_info=True)
        else:
             logger_psp.warning(f"Skipping save_strategy_data for {symbol} because no valid DataFrame was available (df_to_save_final is None or empty).")

        # --- Konsistenzcheck und Finalisierung ---
        tracking_has_pos_final = position_tracking.get(symbol, {}).get('position_open', False)
        active_obs_has_pos_final = active_observation_periods.get(symbol, {}).get('has_position', False)
        if tracking_has_pos_final != active_obs_has_pos_final:
            logger_psp.warning(f"INCONSISTENCY DETECTED at end of process_symbol_parallel for {symbol}: pos_track={tracking_has_pos_final} vs active_obs={active_obs_has_pos_final}. Syncing active_obs to pos_track.")
            if symbol not in active_observation_periods: active_observation_periods[symbol] = {} # Sicherstellen, dass der Key existiert
            active_observation_periods[symbol]['has_position'] = tracking_has_pos_final
            if not tracking_has_pos_final:
                active_observation_periods[symbol]['active_period'] = None

        # --- Performance-Dictionary vervollständigen (falls leer) ---
        if not final_performance or 'symbol' not in final_performance: # Wenn final_performance leer ist oder das Symbol fehlt
            logger_psp.warning(f"No specific performance data generated from period processing for {symbol}, creating default performance entry.")
            if not isinstance(final_performance, dict): final_performance = {} # Sicherstellen, dass es ein Dict ist
            # Fülle mit Standardwerten, falls keine Trades/Performance aus Periodenverarbeitung kam
            final_performance.update({
                'symbol': symbol, 'total_trades': len(all_trades_for_symbol),
                'win_rate': 0.0, 'total_profit': 0.0, 'max_drawdown': 0.0, 'profit_factor': 0.0,
                'avg_profit_win': 0.0, 'avg_loss_loss': 0.0, 'total_commission': 0.0,
                'total_slippage': 0.0, 'exit_reasons': {}, 'immediate_sl_hits': 0, 'immediate_tp_hits': 0,
                'total_exits': len(all_trades_for_symbol), 'avg_profit': 0.0, 'avg_slippage': 0.0,
                'stop_loss_hits': 0, 'take_profit_hits': 0, 'signal_exits': 0,
                'backtest_end_exits': 0, 'trailing_stop_hits': 0, 'observation_timeout_exits': 0,
                'partial_exits': 0
            })
            # Wenn es Trades gab, aber keine Performance, versuche Performance neu zu berechnen
            if all_trades_for_symbol and 'error' in final_performance : # Nur wenn ein Fehler vorlag
                try:
                    from utils.backtest_strategy import calculate_performance_metrics # Import hier, falls nicht global
                    recalculated_perf = calculate_performance_metrics(all_trades_for_symbol, symbol, getattr(Z_config, 'start_balance_parameter', 25.0))
                    if recalculated_perf and 'error' not in recalculated_perf:
                        logger_psp.info(f"Successfully recalculated performance for {symbol} as fallback.")
                        final_performance = recalculated_perf
                    else:
                         logger_psp.warning(f"Fallback performance recalculation for {symbol} failed or also resulted in error.")
                except Exception as e_recalc:
                    logger_psp.error(f"Error during fallback performance recalculation for {symbol}: {e_recalc}")


        # --- Rückgabe ---
        logger_psp.info(f"Finished processing {symbol}. Returning {len(all_trades_for_symbol)} trade events. Data saved: {data_saved_flag}")
        return symbol, all_trades_for_symbol, final_performance, data_saved_flag

    except Exception as e_outer:
        # --- Äußere Fehlerbehandlung für die gesamte Funktion ---
        logger_psp.error(f"--- FATAL ERROR in process_symbol_parallel for {symbol}: {e_outer} ---", exc_info=True)
        # Versuche, globalen Status zurückzusetzen, falls die Dictionaries übergeben wurden und gültig sind
        if position_tracking is not None and isinstance(position_tracking, dict) and symbol in position_tracking:
             position_tracking[symbol] = {'position_open': False, 'observation_ended': False, 'position_data': {}}
        if active_observation_periods is not None and isinstance(active_observation_periods, dict) and symbol in active_observation_periods:
             active_observation_periods[symbol] = {'has_position': False, 'active_period': None}
        # Gib ein Fehler-Tupel zurück, das dem erwarteten Format entspricht
        return symbol, [], {'symbol': symbol, 'total_trades': 0, 'error': f'Outer fatal error: {e_outer}'}, False