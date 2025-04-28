# run_optimization.py

import Z_config
# Stelle sicher, dass der Import von Backtest klappt (ggf. Pfad anpassen)
# Wenn Backtest.py im gleichen Ordner ist:
from utils import Backtest
# Wenn es in einem Unterordner 'utils' liegt:
# from utils import Backtest
import pandas as pd
import os
import random
import time
import logging
import copy
from datetime import datetime
import numpy as np
import shutil # Import für Verzeichnisoperationen

# --- Logging Setup ---
log_filename_opt = 'optimization_run.log'
logger_opt = logging.getLogger("OptimizationRun")
if not logger_opt.hasHandlers():
    logger_opt.setLevel(logging.INFO)
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s')
    file_handler = logging.FileHandler(log_filename_opt, mode='w')
    file_handler.setFormatter(log_formatter)
    logger_opt.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)
    logger_opt.addHandler(stream_handler)

# --- Globale Einstellungen ---
NUM_TRIALS = 10
ALL_RUNS_CSV = "all_runs_optimization_results.csv"
PERFORMANCE_CSV = "total_symbol_performance.csv"

# --- Parameter Suchraum Definition ---
# (HIER DEINEN VOLLSTÄNDIGEN param_space EINFÜGEN)
param_space = {
    # Standard SL/TP oder Trailing
    'use_standard_sl_tp': [True, False],
    'stop_loss_parameter_binance': lambda: round(random.uniform(0.015, 0.07), 4),
    'take_profit_parameter_binance': lambda: round(random.uniform(0.015, 0.07), 4),
    # Trailing Stop Parameter
    'activation_threshold': lambda: round(random.uniform(0.1, 1.0), 2),
    'trailing_distance': lambda: round(random.uniform(0.2, 6.0), 2),
    'adjustment_step': lambda: round(random.uniform(0.05, 1.5), 2),
    'take_profit_levels': lambda: sorted([round(random.uniform(0.5, 5.0), 1) for _ in range(random.randint(1, 3))]),
    'take_profit_size_percentages': lambda: [100],
    'third_level_trailing_distance': lambda: round(random.uniform(0.5, 6.0), 2),
    'enable_breakeven': [True, False],
    'enable_trailing_take_profit': [True],
    # Filterung & Beobachtung
    'filtering_active': [False],
    'beobachten_active': [False],
    'seite': ['long', 'short', 'both'],
    'min_price_change_pct_min': lambda: round(random.uniform(0.5, 5.0), 2),
    'min_price_change_pct_max': lambda: round(random.uniform(5.1, 50.0), 2),
    'price_change_lookback_minutes': lambda: random.randint(30, 60*12),
    'symbol_observation_hours': lambda: random.randint(1, 12),
    'close_position': [True, False],
    # Zeitfilter (Beispielhaft auskommentiert)
    #'time_filter_active': [True, False],
    # Positionsrichtung
    'allow_short': [True],
    'allow_long': [True],
    # Vortagesfilter
    'filter_previous_day': [False],
    'previous_day_direction': ['bullish', 'bearish'],
    # EMA & Trend
    'require_timeframe_alignment': [False],
    'ema_fast_parameter': lambda: random.randint(3, 30),
    'ema_slow_parameter': lambda: random.randint(31, 150),
    'ema_baseline_parameter': lambda: random.randint(50, 250),
    'back_trend_periode': lambda: random.randint(3, 20),
    'min_trend_strength_parameter': lambda: round(random.uniform(0.0, 5.0), 2),
    'min_trend_duration_parameter': lambda: random.randint(1, 10),
    # RSI
    'rsi_period': lambda: random.randint(5, 30),
    'rsi_buy': lambda: random.randint(55, 85),
    'rsi_sell': lambda: random.randint(15, 45),
    'rsi_exit_overbought': lambda: random.randint(75, 98),
    'rsi_exit_oversold': lambda: random.randint(2, 30),
    # Volume
    'volume_sma': lambda: random.randint(10, 60),
    'volume_multiplier': lambda: round(random.uniform(0.8, 5.0), 2),
    'volume_entry_multiplier': lambda: round(random.uniform(0.4, 3.0), 2),
    'min_volume': lambda: round(random.uniform(0.0, 1.0), 4),
    # Momentum
    'use_momentum_check': [True, False],
    'momentum_lookback': lambda: random.randint(2, 40),
    'momentum': lambda: round(random.uniform(0.01, 1.0), 4),
    # Bollinger Bands
    'use_bb': [True, False],
    'bb_period': lambda: random.randint(10, 50),
    'bb_deviation': lambda: round(random.uniform(1.5, 4.0), 2),
    # MACD
    'use_macd': [True, False],
    'macd_fast_period': lambda: random.randint(5, 25),
    'macd_slow_period': lambda: random.randint(26, 60),
    'macd_signal_period': lambda: random.randint(5, 20),
    # ADX
    'use_adx': [True, False],
    'adx_period': lambda: random.randint(7, 40),
    'adx_threshold': lambda: round(random.uniform(15.0, 40.0), 1),
    # VWAP
    'use_vwap': [True, False],
    'use_advanced_vwap': [True, False],
    'advanced_vwap_periods': lambda: sorted([random.randint(10, 50), random.randint(51, 150), random.randint(151, 300)]),
    'advanced_vwap_std_threshold': lambda: round(random.uniform(0.3, 2.5), 2),
    # OBV
    'use_obv': [True, False],
    'obv_period': lambda: random.randint(5, 30),
    # Chaikin Money Flow
    'use_chaikin_money_flow': [True, False],
    'cmf_period': lambda: random.randint(5, 40),
    'cmf_threshold': lambda: round(random.uniform(-0.3, 0.3), 2),
}


# --- Hilfsfunktionen ---
# (generate_random_params, modify_z_config, restore_z_config unverändert)
def generate_random_params(space):
    params = {}
    #logger_opt.debug("Generating new parameter set...")
    for name, generator in space.items():
        try:
            if callable(generator): params[name] = generator()
            elif isinstance(generator, list): params[name] = random.choice(generator)
            else: params[name] = generator
        except Exception as e:
            logger_opt.error(f"Error generating parameter '{name}': {e}")
            params[name] = None
    #logger_opt.debug("Applying dependency logic...")
    if params.get('use_standard_sl_tp') is True:
        tp_binance_val = params.get('take_profit_parameter_binance', 0.05)
        params['activation_threshold'] = 0.0
        params['trailing_distance'] = 1.0
        params['adjustment_step'] = 0.1
        params['take_profit_levels'] = [round(tp_binance_val * 100, 1)]
        params['take_profit_size_percentages'] = [100.0]
        params['enable_breakeven'] = False
        params['enable_trailing_take_profit'] = False
        params['third_level_trailing_distance'] = 1.0
        #logger_opt.debug("  Standard SL/TP is True. Overriding trailing parameters.")
    else:
        params['stop_loss_parameter_binance'] = 0.99
        params['take_profit_parameter_binance'] = 1.00
        num_levels = len(params.get('take_profit_levels', []))
        if num_levels > 0:
            base_pct = 100.0 / num_levels
            percentages = [base_pct] * num_levels
            total_pct = sum(percentages)
            diff = 100.0 - total_pct
            percentages[-1] += diff
            params['take_profit_size_percentages'] = [round(p, 2) for p in percentages]
        else:
            params['take_profit_size_percentages'] = []
        if num_levels < 3:
             params['enable_trailing_take_profit'] = False
             params['third_level_trailing_distance'] = 1.0
        #logger_opt.debug(f"  Standard SL/TP is False. Using trailing. TP Levels: {params.get('take_profit_levels')}, TP Sizes: {params.get('take_profit_size_percentages')}")
    if params.get('require_timeframe_alignment') is False:
         base_interval = getattr(Z_config, 'interval', '15m')
         params['interval_int_2'] = base_interval
         params['interval_int_3'] = base_interval
         #logger_opt.debug(f"  Timeframe Alignment is False. Setting interval_int_2/3 to '{base_interval}'.")
    else:
         if params.get('interval_int_2') is None: params['interval_int_2'] = '1h'
         if params.get('interval_int_3') is None: params['interval_int_3'] = '4h'

    def set_inactive_indicator_params(indicator_prefix, use_flag, param_names):
        if params.get(use_flag) is False:
            #logger_opt.debug(f"  {use_flag} is False. Setting related params to defaults.")
            for param_name in param_names:
                 if param_name in params:
                     default_value = getattr(Z_config, param_name, None)
                     params[param_name] = default_value
    set_inactive_indicator_params("Momentum", 'use_momentum_check', ['momentum_lookback', 'momentum'])
    set_inactive_indicator_params("BB", 'use_bb', ['bb_period', 'bb_deviation'])
    set_inactive_indicator_params("MACD", 'use_macd', ['macd_fast_period', 'macd_slow_period', 'macd_signal_period'])
    set_inactive_indicator_params("ADX", 'use_adx', ['adx_period', 'adx_threshold'])
    set_inactive_indicator_params("AdvVWAP", 'use_advanced_vwap', ['advanced_vwap_periods', 'advanced_vwap_std_threshold'])
    set_inactive_indicator_params("OBV", 'use_obv', ['obv_period'])
    set_inactive_indicator_params("CMF", 'use_chaikin_money_flow', ['cmf_period', 'cmf_threshold'])
    if 'ema_slow_parameter' in params and 'ema_fast_parameter' in params:
        if params['ema_slow_parameter'] <= params['ema_fast_parameter']: params['ema_slow_parameter'] = params['ema_fast_parameter'] + random.randint(5, 50)
    if 'macd_slow_period' in params and 'macd_fast_period' in params:
        if params['macd_slow_period'] <= params['macd_fast_period']: params['macd_slow_period'] = params['macd_fast_period'] + random.randint(5, 20)
    if 'rsi_buy' in params and 'rsi_sell' in params:
        if params['rsi_buy'] <= params['rsi_sell']: params['rsi_buy'] = params['rsi_sell'] + random.randint(10, 30)
    if 'rsi_exit_overbought' in params and 'rsi_buy' in params:
         if params['rsi_exit_overbought'] <= params['rsi_buy']: params['rsi_exit_overbought'] = params['rsi_buy'] + random.randint(5, 15)
    if 'rsi_exit_oversold' in params and 'rsi_sell' in params:
        if params['rsi_exit_oversold'] >= params['rsi_sell']: params['rsi_exit_oversold'] = max(1, params['rsi_sell'] - random.randint(5, 15))
    if 'min_price_change_pct_max' in params and 'min_price_change_pct_min' in params:
        if params['min_price_change_pct_max'] <= params['min_price_change_pct_min']: params['min_price_change_pct_max'] = params['min_price_change_pct_min'] + random.uniform(0.1, 10.0)
    #logger_opt.debug("Parameter generation and dependency check complete.")
    return params

def modify_z_config(params_to_set):
    logger_opt.info("Modifying Z_config attributes for the current trial...")
    modified_count = 0
    skipped_count = 0
    for key, value in params_to_set.items():
        if hasattr(Z_config, key):
            try: setattr(Z_config, key, value); modified_count += 1
            except Exception as e: logger_opt.error(f"  Error setting Z_config.{key} = {value}: {e}")
        else: skipped_count += 1
    logger_opt.info(f"Attempted to modify {len(params_to_set)} parameters. Modified: {modified_count}, Skipped: {skipped_count}.")

def restore_z_config(original_config_vars):
    logger_opt.info("Restoring original Z_config attributes...")
    restored_count = 0
    if not isinstance(original_config_vars, dict): logger_opt.error("Cannot restore Z_config: original_config_vars is not a dictionary."); return
    for key, value in original_config_vars.items():
        if not key.startswith('__'):
            try: setattr(Z_config, key, value); restored_count += 1
            except Exception as e: logger_opt.error(f"  Error restoring Z_config.{key} = {value}: {e}")
    logger_opt.info(f"Restored {restored_count} original attributes in Z_config.")

# --- BEREINIGUNGSFUNKTION (wird NACH jedem Trial aufgerufen) ---
def cleanup_backtest_outputs():
    """
    Löscht die temporären Ausgabedateien und Verzeichnisse von Backtest.main
    NACHDEM die Ergebnisse eines Trials verarbeitet wurden.
    WICHTIG: Backtest.py darf diese Dateien NICHT mehr selbst löschen!
    """
    logger_opt.info("Cleaning up temporary backtest output files/dirs...")
    files_to_delete = [
        "performance_summary.csv",
        "24h_signals.csv",
        "total_symbol_performance.csv",

        os.path.join("strategy_results", "strategy_details.csv"),
        os.path.join("strategy_results", "trade_log.csv"),
        os.path.join("strategy_results", "full_data.csv"),
    ]
    dirs_to_clear = ["trades", "strategy_results"]

    # Dateien löschen
    for file_path in files_to_delete:
        abs_path = os.path.abspath(file_path)
        try:
            if os.path.exists(abs_path):
                os.remove(abs_path)
                logger_opt.info(f"  - Deleted file: {abs_path}")
            else: logger_opt.debug(f"  - File not found (OK to ignore during cleanup): {abs_path}")
        except OSError as e:
            logger_opt.error(f"  - Error deleting file {abs_path}: {e}")

    # Verzeichnisse leeren/neu erstellen
    for dir_rel_path in dirs_to_clear:
        dir_abs_path = os.path.abspath(dir_rel_path)
        try:
            if os.path.exists(dir_abs_path):
                shutil.rmtree(dir_abs_path) # Löscht Verzeichnis + Inhalt
                logger_opt.info(f"  - Removed directory tree: {dir_abs_path}")
            os.makedirs(dir_abs_path, exist_ok=True) # Erstellt Verzeichnis (neu)
            logger_opt.info(f"  - Ensured directory exists: {dir_abs_path}")
        except OSError as e:
            logger_opt.error(f"  - Error clearing/creating directory {dir_abs_path}: {e}")
    logger_opt.info("Finished cleaning up temporary outputs for this trial.")
# ***** ENDE BEREINIGUNGSFUNKTION *****


# --- Haupt Optimierungs-Schleife ---
if __name__ == "__main__":
    logger_opt.info(f"=================================================")
    logger_opt.info(f" Starting Optimization Run: {NUM_TRIALS} Trials")
    logger_opt.info(f" Results will be appended to: {ALL_RUNS_CSV}")
    logger_opt.info(f" Expecting performance file: {PERFORMANCE_CSV}")
    logger_opt.info(f" --- REMINDER: Backtest.py should NOT delete its outputs! ---")
    logger_opt.info(f"=================================================")
    run_start_time = time.time()

    # Speichere die ursprüngliche Konfiguration
    logger_opt.debug("Creating snapshot of original Z_config...")
    original_config_snapshot = {}
    try:
        for k, v in vars(Z_config).items():
             if not k.startswith('__'):
                 original_config_snapshot[k] = copy.deepcopy(v)
        logger_opt.info("Original Z_config snapshot created successfully.")
    except Exception as copy_err:
         logger_opt.critical(f"FATAL: Could not create snapshot of Z_config: {copy_err}. Aborting.", exc_info=True)
         exit()

    # Lösche alte Gesamt-Ergebnisdatei für sauberen Start
    if os.path.exists(ALL_RUNS_CSV):
        try:
            os.remove(ALL_RUNS_CSV)
            logger_opt.info(f"Removed existing overall results file: {ALL_RUNS_CSV}")
        except OSError as e:
            logger_opt.error(f"Could not remove existing results file {ALL_RUNS_CSV}: {e}")

    performance_columns = [] # Zum Speichern der Spaltennamen für Fehlerzeilen

    # --- Schleife über alle Trials ---
    for trial_num in range(NUM_TRIALS):
        trial_start_time = time.time()
        logger_opt.info(f"-------------------------------------------------")
        logger_opt.info(f"--- Starting Trial {trial_num + 1}/{NUM_TRIALS} ---")
        logger_opt.info(f"-------------------------------------------------")

        current_parameters = {}
        performance_data = None
        run_successful = False
        result_summary = "Trial starting"

        # --- HIER KEINE BEREINIGUNG ---
        # Die Bereinigung erfolgt jetzt NACH dem Speichern der Ergebnisse.

        try:
            # 1. Parameter generieren
            current_parameters = generate_random_params(param_space)
            logger_opt.info(f"Trial {trial_num + 1} Parameters: {current_parameters}")

            # 2. Z_config modifizieren
            modify_z_config(current_parameters)

            time.sleep(1)

            # 3. Backtest ausführen
            logger_opt.info(f"Trial {trial_num + 1}: Calling Backtest.main()...")
            Backtest.main() # Annahme: löscht nichts mehr selbst
            run_successful = True
            logger_opt.info(f"Trial {trial_num + 1}: Backtest.main() finished execution.")

            time.sleep(2)

            # 4. Ergebnis extrahieren
            logger_opt.info(f"Trial {trial_num + 1}: Checking and reading {PERFORMANCE_CSV}...")
            if os.path.exists(PERFORMANCE_CSV):
                try:
                    # Kurze Wartezeit, falls das Schreiben der Datei minimal verzögert ist
                    time.sleep(0.2)
                    file_size = os.path.getsize(PERFORMANCE_CSV)
                    logger_opt.info(f"CHECK: {PERFORMANCE_CSV} exists. Size: {file_size} bytes.")
                    if file_size > 0:
                        performance_data = pd.read_csv(PERFORMANCE_CSV)
                        if performance_data.empty:
                            logger_opt.warning(f"Read {PERFORMANCE_CSV}, but empty DataFrame.")
                            performance_data = None; result_summary = f"{PERFORMANCE_CSV} read but empty"
                        else:
                            logger_opt.info(f"Successfully read {len(performance_data)} rows from {PERFORMANCE_CSV}.")
                            result_summary = f"Read {len(performance_data)} performance rows"
                            if not performance_columns: performance_columns = list(performance_data.columns)
                    else:
                        logger_opt.warning(f"{PERFORMANCE_CSV} exists but is empty (size 0).")
                        result_summary = f"{PERFORMANCE_CSV} empty (size 0)"
                except pd.errors.EmptyDataError:
                     logger_opt.warning(f"{PERFORMANCE_CSV} exists but EmptyDataError.")
                     result_summary = f"{PERFORMANCE_CSV} exists but EmptyDataError"
                except Exception as read_err:
                    logger_opt.error(f"Error reading {PERFORMANCE_CSV}: {read_err}", exc_info=True)
                    result_summary = f"Error reading {PERFORMANCE_CSV}"
            else:
                logger_opt.warning(f"CHECK: {PERFORMANCE_CSV} not found after Backtest.main()!")
                result_summary = f"No trades found for this setting"

        except KeyboardInterrupt:
             logger_opt.warning("KeyboardInterrupt received. Stopping.")
             print("\nOptimization stopped by user.")
             result_summary = "Interrupted"
             restore_z_config(original_config_snapshot)
             break
        except Exception as main_err:
            logger_opt.error(f"--- ERROR during Backtest.main() in Trial {trial_num + 1} ---", exc_info=True)
            run_successful = False
            result_summary = f"Backtest.main execution error"

        # --- Ergebnis Verarbeitung und Speicherung ---
        logger_opt.info(f"Trial {trial_num + 1}: Aggregating results. Status: {result_summary}")
        final_rows_for_csv = []
        if performance_data is not None:
            for _, perf_row in performance_data.iterrows():
                row_dict = {'trial': trial_num + 1, **current_parameters, **perf_row.to_dict()}
                row_dict.pop('error', None)
                final_rows_for_csv.append(row_dict)
        else:
            error_row = {'trial': trial_num + 1, **current_parameters, 'error': result_summary}
            cols_to_add = performance_columns if performance_columns else ['symbol', 'total_trades', 'win_rate', 'total_profit', 'max_drawdown', 'profit_factor', 'avg_profit', 'total_commission'] # Fallback
            for col in cols_to_add:
                 if col not in error_row: error_row[col] = np.nan
            final_rows_for_csv.append(error_row)

        # An die globale CSV anhängen
        if final_rows_for_csv:
            combined_df = pd.DataFrame(final_rows_for_csv)
            try:
                file_exists = os.path.exists(ALL_RUNS_CSV)
                all_param_keys = list(current_parameters.keys())
                perf_cols_to_use = performance_columns if performance_columns else ['symbol', 'total_trades', 'win_rate', 'total_profit', 'max_drawdown', 'profit_factor', 'avg_profit', 'total_commission']
                known_meta_cols = ['trial', 'error']
                all_expected_cols = known_meta_cols + all_param_keys + perf_cols_to_use
                # Entferne Duplikate und behalte eine definierte Reihenfolge bei
                final_cols_ordered = []
                seen_cols = set()
                for col in all_expected_cols:
                     if col not in seen_cols:
                          final_cols_ordered.append(col)
                          seen_cols.add(col)

                combined_df = combined_df.reindex(columns=final_cols_ordered, fill_value=np.nan)
                combined_df.to_csv(ALL_RUNS_CSV, mode='a', header=not file_exists, index=False, columns=final_cols_ordered)
                logger_opt.info(f"Trial {trial_num + 1}: Appended {len(combined_df)} row(s) to {ALL_RUNS_CSV}")
            except Exception as e:
                logger_opt.error(f"Trial {trial_num + 1}: Error appending results to {ALL_RUNS_CSV}: {e}", exc_info=True)

        # --- JETZT die temporären Backtest-Ausgaben löschen ---
        try:
             logger_opt.info(f"Trial {trial_num + 1}: Cleaning up temporary backtest output files...")
             cleanup_backtest_outputs() # Aufruf der Cleanup-Funktion HIER
             time.sleep(1)
        except Exception as cleanup_err:
             logger_opt.error(f"Trial {trial_num + 1}: Error during post-trial cleanup: {cleanup_err}", exc_info=True)

        # --- Original Z_config wiederherstellen ---
        logger_opt.info(f"Trial {trial_num + 1}: Restoring original Z_config...")
        restore_z_config(original_config_snapshot)

        trial_duration = time.time() - trial_start_time
        logger_opt.info(f"--- Trial {trial_num + 1} finished in {trial_duration:.2f} seconds. Status: {result_summary} ---")
        time.sleep(1) # Kurze Pause

    # --- Ende der Hauptschleife ---
    run_duration = time.time() - run_start_time
    logger_opt.info(f"=================================================")
    logger_opt.info(f" Optimization Run Finished in {run_duration:.2f} seconds")
    logger_opt.info(f" Results saved in: {ALL_RUNS_CSV}")
    logger_opt.info(f"=================================================")