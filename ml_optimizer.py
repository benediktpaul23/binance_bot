# run_optimization_ml.py
import Z_config
from utils import Backtest # Stelle sicher, dass der Import klappt
import pandas as pd
import os
import random
import time
import logging
import copy
from datetime import datetime
import numpy as np
import shutil
import warnings

# --- ML / SKLearn Imports ---
try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("WARNING: scikit-learn not found. ML features disabled. Run 'pip install scikit-learn'")
    warnings.warn("scikit-learn not found. Optimization will run in purely random mode.")




# --- Logging Setup ---
log_filename_opt = 'optimization_run_ml.log' # Changed log filename
logger_opt = logging.getLogger("OptimizationRunML") # Changed logger name
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
NUM_TRIALS = 50 # Erhöhe die Anzahl für sinnvolles ML
ALL_RUNS_CSV = "all_runs_optimization_results_ml.csv" # Changed results filename
PERFORMANCE_CSV = "total_symbol_performance.csv"

# --- ML Spezifische Einstellungen ---
USE_ML_GUIDANCE = ML_AVAILABLE # Nur nutzen, wenn scikit-learn installiert ist
NUM_INITIAL_RANDOM_TRIALS = 15  # Anzahl der rein zufälligen Läufe zu Beginn
NUM_CANDIDATES_PER_ML_TRIAL = 50 # Wie viele Kandidaten pro ML-Lauf generieren/bewerten?
RETRAIN_EVERY_N_TRIALS = 10    # Wie oft das ML-Modell neu trainieren?
TARGET_METRIC = 'profit_factor' # Welche Spalte aus PERFORMANCE_CSV soll optimiert werden? (Maximieren)
# TARGET_METRIC = 'total_profit' # Alternative
MIN_DATA_FOR_TRAINING = 10     # Mindestanzahl gültiger (!) Ergebnisse für das erste Training


TARGET_METRIC = 'profit_factor' # Welche Spalte aus PERFORMANCE_CSV soll optimiert werden? (Maximieren)
MIN_DATA_FOR_TRAINING = 10     # Mindestanzahl gültiger (!) Ergebnisse für das erste Training

# NEU: Globale Definition der erwarteten Performance-Spalten
# Passe diese Liste ggf. an die tatsächlichen Spalten deiner total_symbol_performance.csv an!
known_performance_cols = [
    'symbol', 'total_trades', 'win_rate', 'total_profit', 'max_drawdown',
    'profit_factor', 'avg_profit', 'total_commission', 'avg_slippage',
    'total_slippage', 'stop_loss_hits', 'take_profit_hits', 'signal_exits',
    'trailing_stop_hits', 'observation_timeout_exits', 'backtest_end_exits',
    'stop_loss_percent', 'take_profit_percent', 'signal_exits_percent',
    'trailing_stop_percent', 'observation_timeout_percent', 'backtest_end_percent'
]

# Globale Variablen für das ML Modell und Preprocessor
ml_model = None
ml_preprocessor = None
ml_feature_names = None # Store feature names used during training

# --- Parameter Suchraum Definition ---
# (HIER DEINEN VOLLSTÄNDIGEN param_space EINFÜGEN - unverändert)
param_space = {
    # Standard SL/TP oder Trailing
    'use_standard_sl_tp': [True, False],
    'stop_loss_parameter_binance': lambda: round(random.uniform(0.015, 0.07), 4),
    'take_profit_parameter_binance': lambda: round(random.uniform(0.015, 0.07), 4),
    # Trailing Stop Parameter
    'activation_threshold': lambda: round(random.uniform(0.1, 1.0), 2),
    'trailing_distance': lambda: round(random.uniform(0.2, 6.0), 2),
    'adjustment_step': lambda: round(random.uniform(0.05, 1.5), 2),
    # --- VEREINFACHUNG FÜR ML: Listen erstmal ignorieren ---
    # 'take_profit_levels': lambda: sorted([round(random.uniform(0.5, 5.0), 1) for _ in range(random.randint(1, 3))]),
    # 'take_profit_size_percentages': lambda: [100], # Wird in generate_random_params dynamisch gesetzt
    'third_level_trailing_distance': lambda: round(random.uniform(0.5, 6.0), 2),
    'enable_breakeven': [True, False],
    'enable_trailing_take_profit': [True], # Wird in generate_random_params ggf. angepasst
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
    'allow_short': [True], # Behalte bei, falls Z_Config sie braucht
    'allow_long': [True],  # Behalte bei, falls Z_Config sie braucht
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
    # --- VEREINFACHUNG FÜR ML: Listen erstmal ignorieren ---
    # 'advanced_vwap_periods': lambda: sorted([random.randint(10, 50), random.randint(51, 150), random.randint(151, 300)]),
    'advanced_vwap_std_threshold': lambda: round(random.uniform(0.3, 2.5), 2),
    # OBV
    'use_obv': [True, False],
    'obv_period': lambda: random.randint(5, 30),
    # Chaikin Money Flow
    'use_chaikin_money_flow': [True, False],
    'cmf_period': lambda: random.randint(5, 40),
    'cmf_threshold': lambda: round(random.uniform(-0.3, 0.3), 2),
    # Intervalle (werden in generate_random_params dynamisch gesetzt)
    # 'interval_int_2', 'interval_int_3' nicht hier definieren
}

# --- Hilfsfunktionen ---
def generate_random_params(space):
    params = {}
    # Temporär Listen-Parameter aus dem Space nehmen für die Hauptgeneration
    space_copy = space.copy()
    list_params_generators = {
        'take_profit_levels': lambda: sorted([round(random.uniform(0.5, 5.0), 1) for _ in range(random.randint(1, 3))]),
        'advanced_vwap_periods': lambda: sorted([random.randint(10, 50), random.randint(51, 150), random.randint(151, 300)])
    }
    # Entferne sie temporär aus dem space, wenn sie dort sind
    space_copy.pop('take_profit_levels', None)
    space_copy.pop('take_profit_size_percentages', None)
    space_copy.pop('advanced_vwap_periods', None)

    for name, generator in space_copy.items():
        try:
            if callable(generator): params[name] = generator()
            elif isinstance(generator, list): params[name] = random.choice(generator)
            else: params[name] = generator # Falls fixe Werte drin sind
        except Exception as e:
            logger_opt.error(f"Error generating parameter '{name}': {e}")
            params[name] = None # Oder einen Standardwert setzen

    # --- Abhängigkeiten und Logik ---
    # (Größtenteils unverändert, aber Listenparameter hinzufügen/anpassen)

    # Generiere Listen-Parameter jetzt
    params['take_profit_levels'] = list_params_generators['take_profit_levels']()
    params['advanced_vwap_periods'] = list_params_generators['advanced_vwap_periods']()

    # Abhängigkeiten basierend auf 'use_standard_sl_tp'
    if params.get('use_standard_sl_tp') is True:
        tp_binance_val = params.get('take_profit_parameter_binance', 0.05) # Hole generierten Wert
        params['activation_threshold'] = 0.0
        params['trailing_distance'] = 1.0
        params['adjustment_step'] = 0.1
        # Überschreibe 'take_profit_levels' passend zu 'take_profit_parameter_binance'
        params['take_profit_levels'] = [round(tp_binance_val * 100, 1)]
        params['take_profit_size_percentages'] = [100.0] # Immer 100% bei Standard TP
        params['enable_breakeven'] = False
        params['enable_trailing_take_profit'] = False
        params['third_level_trailing_distance'] = 1.0
    else: # Trailing Stop Logik
        params['stop_loss_parameter_binance'] = 0.99 # Setze Standardwerte, wenn Trailing aktiv
        params['take_profit_parameter_binance'] = 1.00
        num_levels = len(params.get('take_profit_levels', []))
        if num_levels > 0:
             # Gleichmäßige Verteilung der Prozente
             base_pct = 100.0 / num_levels
             percentages = [base_pct] * num_levels
             # Rundungsdifferenzen dem letzten Level hinzufügen
             total_pct = sum(p for p in percentages) # Direkt summieren
             diff = 100.0 - total_pct
             percentages[-1] += diff
             params['take_profit_size_percentages'] = [round(p, 2) for p in percentages]
        else:
             params['take_profit_size_percentages'] = [] # Leere Liste, wenn keine Level da

        # Enable trailing TP nur wenn mind. 3 Level? (Deine alte Logik)
        if num_levels < 3:
             params['enable_trailing_take_profit'] = False
             params['third_level_trailing_distance'] = 1.0 # Setze Standard, wenn deaktiviert
        else:
             params['enable_trailing_take_profit'] = True # Aktivieren, wenn genug Level


    # Timeframe Alignment
    base_interval = getattr(Z_config, 'interval', '15m') # Hole Basisintervall aus Z_config
    if params.get('require_timeframe_alignment') is False:
        params['interval_int_2'] = base_interval
        params['interval_int_3'] = base_interval
    else:
        # Wenn Alignment aktiv, setze höhere Intervalle (Beispiel)
        # Diese könnten auch Teil des param_space sein!
        params['interval_int_2'] = '1h' # Beispiel
        params['interval_int_3'] = '4h' # Beispiel

    # Inaktive Indikatoren (unverändert)
    def set_inactive_indicator_params(indicator_prefix, use_flag, param_names):
        if params.get(use_flag) is False:
            #logger_opt.debug(f"  {use_flag} is False. Setting related params to defaults.")
            for param_name in param_names:
                 if param_name in params:
                     # Versuche, Standardwert aus Z_config zu holen, sonst None
                     default_value = getattr(Z_config, param_name, None)
                     params[param_name] = default_value
                     #logger_opt.debug(f"    Set {param_name} to default: {default_value}")


    set_inactive_indicator_params("Momentum", 'use_momentum_check', ['momentum_lookback', 'momentum'])
    set_inactive_indicator_params("BB", 'use_bb', ['bb_period', 'bb_deviation'])
    set_inactive_indicator_params("MACD", 'use_macd', ['macd_fast_period', 'macd_slow_period', 'macd_signal_period'])
    set_inactive_indicator_params("ADX", 'use_adx', ['adx_period', 'adx_threshold'])
    # Hier advanced_vwap_periods NICHT zurücksetzen, da es oben generiert wird
    set_inactive_indicator_params("AdvVWAP", 'use_advanced_vwap', ['advanced_vwap_std_threshold']) # Nur std_threshold
    set_inactive_indicator_params("OBV", 'use_obv', ['obv_period'])
    set_inactive_indicator_params("CMF", 'use_chaikin_money_flow', ['cmf_period', 'cmf_threshold'])

    # Konsistenzprüfungen (unverändert)
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

    # Konvertiere Listen in Strings für die Speicherung/Logausgabe (optional, aber gut für Konsistenz)
    for key, value in params.items():
        if isinstance(value, list):
            params[key] = str(value)

    return params

def modify_z_config(params_to_set):
    logger_opt.info("Modifying Z_config attributes for the current trial...")
    modified_count = 0
    skipped_count = 0
    error_count = 0
    for key, value in params_to_set.items():
        if hasattr(Z_config, key):
            try:
                # Spezielle Behandlung für Listen-Strings -> zurück zu Listen
                current_value = value
                if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
                    try:
                        current_value = eval(value) # Vorsicht mit eval! Nur weil wir die Quelle kontrollieren.
                    except:
                        logger_opt.warning(f"Could not eval list string for Z_config.{key}: {value}. Skipping.")
                        error_count +=1
                        continue # Nicht setzen, wenn eval fehlschlägt

                setattr(Z_config, key, current_value)
                modified_count += 1
            except Exception as e:
                logger_opt.error(f"  Error setting Z_config.{key} = {value}: {e}")
                error_count += 1
        else:
            skipped_count += 1
            # logger_opt.debug(f"  Attribute {key} not found in Z_config, skipped.") # Optional: Weniger verbose

    logger_opt.info(f"Attempted to modify {len(params_to_set)} parameters. Modified: {modified_count}, Skipped: {skipped_count}, Errors: {error_count}.")


def restore_z_config(original_config_vars):
    # Unverändert
    logger_opt.info("Restoring original Z_config attributes...")
    restored_count = 0
    error_count = 0
    if not isinstance(original_config_vars, dict):
        logger_opt.error("Cannot restore Z_config: original_config_vars is not a dictionary.")
        return
    for key, value in original_config_vars.items():
        if not key.startswith('__'): # Ignoriere interne Python-Attribute
            try:
                setattr(Z_config, key, value)
                restored_count += 1
            except Exception as e:
                logger_opt.error(f"  Error restoring Z_config.{key} = {value}: {e}")
                error_count += 1
    logger_opt.info(f"Restored {restored_count} original attributes in Z_config. Errors: {error_count}.")


def cleanup_backtest_outputs():
    # Unverändert
    """
    Löscht die temporären Ausgabedateien und Verzeichnisse von Backtest.main
    NACHDEM die Ergebnisse eines Trials verarbeitet wurden.
    WICHTIG: Backtest.py darf diese Dateien NICHT mehr selbst löschen!
    """
    logger_opt.info("Cleaning up temporary backtest output files/dirs...")
    files_to_delete = [
        "performance_summary.csv",
        "24h_signals.csv",
        PERFORMANCE_CSV, # Diese Datei wird jetzt auch gelöscht nach Verarbeitung

        # Pfade relativ zum Skript-Ausführungsort
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
            # else: logger_opt.debug(f"  - File not found (OK during cleanup): {abs_path}") # Weniger verbose
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
            # logger_opt.info(f"  - Ensured directory exists: {dir_abs_path}") # Weniger verbose
        except OSError as e:
            logger_opt.error(f"  - Error clearing/creating directory {dir_abs_path}: {e}")
    logger_opt.info("Finished cleaning up temporary outputs for this trial.")

# --- ML Helper Functions ---

def get_feature_names_and_types(df, exclude_cols):
    """Identifiziert numerische und kategorische Features."""
    features = df.drop(columns=exclude_cols, errors='ignore')
    numeric_features = features.select_dtypes(include=np.number).columns.tolist()
    categorical_features = features.select_dtypes(include=['object', 'category', 'boolean']).columns.tolist()
    # Manuelle Korrektur für boolsche Werte, die oft als numerisch interpretiert werden, aber one-hot encoded werden sollten
    bool_cols = [col for col in numeric_features if df[col].isin([0, 1, True, False]).all()]
    # bool_cols = [col for col in features.columns if df[col].dropna().isin([True, False, 0, 1]).all() and col not in categorical_features]
    # bool_cols += [col for col in categorical_features if df[col].dropna().isin(['True', 'False', 'true', 'false']).all()] # Falls als String


    # Verschiebe boolsche Spalten von numerisch zu kategorisch für OHE
    corrected_numeric = [col for col in numeric_features if col not in bool_cols]
    corrected_categorical = list(set(categorical_features + bool_cols)) # Verwende set für Eindeutigkeit

    # Entferne Listen-artige Spalten (die als String gespeichert sind)
    list_like_cols = [col for col in corrected_categorical if df[col].astype(str).str.startswith('[').any()]
    final_categorical = [col for col in corrected_categorical if col not in list_like_cols]
    final_numeric = corrected_numeric

    logger_opt.debug(f"Identified Numeric Features: {final_numeric}")
    logger_opt.debug(f"Identified Categorical Features: {final_categorical}")
    logger_opt.debug(f"Ignored List-like Features for ML: {list_like_cols}")

    return final_numeric, final_categorical


def build_preprocessor(numeric_features, categorical_features):
    """Baut einen scikit-learn Preprocessor für die Features."""
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), # Fehlende numerische Werte mit Median füllen
        ('scaler', StandardScaler())]) # Numerische Werte skalieren

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), # Fehlende kategorische Werte mit häufigstem Wert füllen
        ('onehot', OneHotEncoder(handle_unknown='ignore'))]) # Kategorische Werte in One-Hot umwandeln

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)],
        remainder='drop') # Ignoriere Spalten, die nicht explizit genannt werden

    return preprocessor


ml_model = None
ml_preprocessor = None # Bleibt Teil der Pipeline, aber die Variable selbst wird nicht direkt genutzt
ml_feature_names = None # Store feature names used during training for prediction alignment
numeric_features_global = None   # NEU: Für Feature Importance
categorical_features_global = None # NEU: Für Feature Importance

# --- In der Funktion train_ml_model ---
def train_ml_model(data_df, target_metric):
    """Trainiert ein ML-Modell auf den gegebenen Daten."""
    # Zugriff auf globale Variablen zum Aktualisieren
    global ml_model, ml_feature_names, numeric_features_global, categorical_features_global

    logger_opt.info(f"Attempting to train ML model on {len(data_df)} data points using target '{target_metric}'...")

    # 1. Daten vorbereiten
    # Performance-Spalten identifizieren (alle außer 'trial', 'error', 'selected_by' und param_space keys)
    # Finde tatsächliche Parameter-Spalten dynamischer
    meta_cols = ['trial', 'error', 'selected_by']
    # Definiere bekannte Performance-Spalten (passe ggf. an deine CSV an)
    performance_cols_in_df = [col for col in known_performance_cols if col in data_df.columns]
    exclude_from_features = meta_cols + performance_cols_in_df

    # Features sind alle Spalten außer den ausgeschlossenen
    features_df = data_df.drop(columns=exclude_from_features, errors='ignore')

    # Target-Spalte extrahieren
    if target_metric not in data_df.columns:
        logger_opt.error(f"Target metric '{target_metric}' not found in the data. Cannot train model.")
        return False
    y = data_df[target_metric]

    # Entferne Zeilen mit fehlendem Target-Wert oder unendlichem Wert
    valid_indices = y.notna() & np.isfinite(y)
    if valid_indices.sum() < MIN_DATA_FOR_TRAINING:
        logger_opt.error(f"Not enough valid data points ({valid_indices.sum()}) with finite target '{target_metric}' to train. Need at least {MIN_DATA_FOR_TRAINING}.")
        return False

    X = features_df[valid_indices].copy() # Arbeite mit einer Kopie
    y = y[valid_indices].copy()
    logger_opt.info(f"Training with {len(X)} valid data points.")

    # 2. Features identifizieren und Preprocessor bauen
    try:
        # Diese Funktion gibt jetzt die Listen zurück
        current_numeric_features, current_categorical_features = get_feature_names_and_types(X, exclude_cols=[]) # Keine expliziten Excludes hier, da schon gefiltert
        # Überprüfe, ob Features gefunden wurden
        if not current_numeric_features and not current_categorical_features:
             logger_opt.error("No numeric or categorical features identified for training.")
             return False
        current_ml_feature_names = current_numeric_features + current_categorical_features
        logger_opt.debug(f"Features identified for this training run: Numeric={len(current_numeric_features)}, Categorical={len(current_categorical_features)}")
        logger_opt.debug(f"Numeric Features: {current_numeric_features}")
        logger_opt.debug(f"Categorical Features: {current_categorical_features}")
    except Exception as e:
        logger_opt.error(f"Error identifying feature types: {e}", exc_info=True)
        return False # Frühzeitiger Ausstieg, wenn Feature-Identifikation fehlschlägt


    # Preprocessor bauen - nur wenn Features vorhanden sind
    transformers_list = []
    if current_numeric_features:
         numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])
         transformers_list.append(('num', numeric_transformer, current_numeric_features))

    if current_categorical_features:
         categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]) # sparse=False kann helfen bei manchen Modellen
         transformers_list.append(('cat', categorical_transformer, current_categorical_features))

    if not transformers_list:
         logger_opt.error("Could not build transformers as no features were identified.")
         return False

    try:
        current_preprocessor = ColumnTransformer(
            transformers=transformers_list,
            remainder='drop') # 'passthrough' würde nicht-genannte Spalten behalten
    except Exception as e:
        logger_opt.error(f"Error building preprocessor: {e}", exc_info=True)
        return False


    # 3. Modell definieren (RandomForestRegressor im Pipeline)
    current_model = Pipeline(steps=[('preprocessor', current_preprocessor),
                                   ('regressor', RandomForestRegressor(n_estimators=100, # Anzahl Bäume
                                                                       random_state=42,   # Für Reproduzierbarkeit
                                                                       n_jobs=-1,         # Nutze alle CPU-Kerne
                                                                       max_depth=15,      # Beispiel: Begrenze Tiefe
                                                                       min_samples_split=5,# Beispiel: Mindestgröße für Split
                                                                       min_samples_leaf=3 # Beispiel: Mindestgröße Blattknoten
                                                                       ))])

    # 4. Modell trainieren
    try:
        start_train_time = time.time()
        logger_opt.info(f"Fitting the model pipeline...")
        current_model.fit(X, y) # X sollte nur die Feature-Spalten enthalten
        train_duration = time.time() - start_train_time
        logger_opt.info(f"ML Model training completed successfully in {train_duration:.2f} seconds.")

        # Speichere das trainierte Modell und die Feature-Listen global
        ml_model = current_model
        ml_feature_names = current_ml_feature_names # Für Prediction Alignment
        numeric_features_global = current_numeric_features # NEU: Global speichern
        categorical_features_global = current_categorical_features # NEU: Global speichern

        return True
    except ValueError as ve:
         logger_opt.error(f"ValueError during model training. Check feature consistency or data types. Error: {ve}", exc_info=True)
         # Detailliertere Fehlersuche, falls möglich
         try:
              logger_opt.debug("Attempting to transform data separately for debugging...")
              transformed_data = current_preprocessor.fit_transform(X)
              logger_opt.debug(f"Data transformed shape: {transformed_data.shape}")
         except Exception as transform_err:
              logger_opt.error(f"Error occurred during preprocessor transform: {transform_err}", exc_info=True)

         ml_model = None; ml_feature_names = None; numeric_features_global = None; categorical_features_global = None
         return False

    except Exception as e:
        logger_opt.error(f"An unexpected error occurred during ML model training: {e}", exc_info=True)
        # Setze alle globalen ML-Variablen zurück bei Fehler
        ml_model = None
        ml_feature_names = None
        numeric_features_global = None
        categorical_features_global = None
        return False

def predict_performance(candidate_params_list):
    """Nutzt das trainierte ML-Modell, um die Performance von Kandidaten vorherzusagen."""
    if ml_model is None or ml_feature_names is None:
        logger_opt.warning("ML model not trained yet or feature names missing. Cannot predict.")
        return None

    # Konvertiere die Liste von Dictionaries in einen DataFrame
    candidates_df = pd.DataFrame(candidate_params_list)

    # Stelle sicher, dass der DataFrame die gleichen Feature-Spalten hat wie beim Training
    # Fehlende Spalten hinzufügen (mit NaN füllen), überzählige entfernen
    cols_to_add = list(set(ml_feature_names) - set(candidates_df.columns))
    for col in cols_to_add:
        candidates_df[col] = np.nan # Fehlende Spalten mit NaN füllen (Imputer kümmert sich drum)

    cols_to_keep = [col for col in ml_feature_names if col in candidates_df.columns]
    candidates_df_aligned = candidates_df[cols_to_keep]

    # Füge Spalten hinzu, die beim Training dabei waren, aber jetzt fehlen (z.B. durch Inaktivierung)
    missing_training_cols = list(set(ml_feature_names) - set(candidates_df_aligned.columns))
    for col in missing_training_cols:
        candidates_df_aligned[col] = np.nan # Imputer sollte das handhaben

    # Ordne Spalten in der Trainingsreihenfolge an (wichtig für manche Preprocessors)
    try:
        candidates_df_aligned = candidates_df_aligned[ml_feature_names]
    except KeyError as e:
        logger_opt.error(f"Column mismatch during prediction alignment: {e}. Available: {candidates_df_aligned.columns}. Needed: {ml_feature_names}")
        return None


    logger_opt.info(f"Predicting performance for {len(candidates_df_aligned)} candidates...")
    try:
        start_pred_time = time.time()
        # Vorhersage mit der Pipeline (die intern transformiert)
        predictions = ml_model.predict(candidates_df_aligned)
        pred_duration = time.time() - start_pred_time
        logger_opt.info(f"Prediction completed in {pred_duration:.2f} seconds.")
        return predictions
    except Exception as e:
        logger_opt.error(f"Error during prediction: {e}", exc_info=True)
        return None

# --- Haupt Optimierungs-Schleife ---
if __name__ == "__main__":
    logger_opt.info(f"=================================================")
    logger_opt.info(f" Starting ML-Guided Optimization Run: {NUM_TRIALS} Trials")
    if USE_ML_GUIDANCE:
        logger_opt.info(f" ML Guidance ENABLED. Initial random: {NUM_INITIAL_RANDOM_TRIALS}, Retrain every: {RETRAIN_EVERY_N_TRIALS}, Target: '{TARGET_METRIC}'")
        logger_opt.info(f" Minimum valid results for training: {MIN_DATA_FOR_TRAINING}")
        logger_opt.info(f" Candidates per ML trial: {NUM_CANDIDATES_PER_ML_TRIAL}")
    else:
        logger_opt.info(f" ML Guidance DISABLED (scikit-learn not found or USE_ML_GUIDANCE=False). Running purely random trials.")
    logger_opt.info(f" Results will be appended to: {ALL_RUNS_CSV}")
    logger_opt.info(f" Backtest performance file expected: {PERFORMANCE_CSV}")
    logger_opt.info(f" --- REMINDER: Backtest.py should NOT delete its own outputs! ---")
    logger_opt.info(f"=================================================")
    run_start_time = time.time()

    # Speichere die ursprüngliche Konfiguration
    logger_opt.debug("Creating snapshot of original Z_config...")
    original_config_snapshot = {}
    try:
        for k, v in vars(Z_config).items():
             if not k.startswith('__'):
                 original_config_snapshot[k] = copy.deepcopy(v) # Deep copy wichtig
        logger_opt.info("Original Z_config snapshot created successfully.")
    except Exception as copy_err:
         logger_opt.critical(f"FATAL: Could not create snapshot of Z_config: {copy_err}. Aborting.", exc_info=True)
         exit(1) # Beende mit Fehlercode

    # Lösche alte Gesamt-Ergebnisdatei für sauberen Start
    if os.path.exists(ALL_RUNS_CSV):
        try:
            os.remove(ALL_RUNS_CSV)
            logger_opt.info(f"Removed existing overall results file: {ALL_RUNS_CSV}")
        except OSError as e:
            logger_opt.error(f"Could not remove existing results file {ALL_RUNS_CSV}: {e}")
            # Entscheiden, ob Abbruch oder Weitermachen sinnvoll ist
            # exit(1)

    performance_columns = [] # Speichert Spaltennamen aus PERFORMANCE_CSV beim ersten Lesen
    all_results_data = [] # Sammelt alle Ergebnisse (als Dictionaries) für ML Training im Speicher

    # --- Schleife über alle Trials ---
    for trial_num in range(NUM_TRIALS):
        trial_start_time = time.time()
        logger_opt.info(f"-------------------------------------------------")
        logger_opt.info(f"--- Starting Trial {trial_num + 1}/{NUM_TRIALS} ---")
        logger_opt.info(f"-------------------------------------------------")

        current_parameters = {}
        performance_data = None # Wird DataFrame oder None
        run_successful = False
        result_summary = "Trial starting"
        selected_by = "Random" # Default Annahme

        # --- Parameter Auswahl ---
        is_ml_phase = USE_ML_GUIDANCE and trial_num >= NUM_INITIAL_RANDOM_TRIALS

        # Prüfe, ob *genug valide Daten* im Speicher sind
        valid_results = [r for r in all_results_data
                         if TARGET_METRIC in r and pd.notna(r[TARGET_METRIC]) and np.isfinite(r[TARGET_METRIC])]
        valid_results_count = len(valid_results)
        can_train = valid_results_count >= MIN_DATA_FOR_TRAINING

        # Modell trainieren/neu trainieren?
        needs_training = ml_model is None # Erstes Training
        needs_retraining = (trial_num - NUM_INITIAL_RANDOM_TRIALS) % RETRAIN_EVERY_N_TRIALS == 0 # Intervall erreicht
        # Trigger nur, wenn in ML-Phase, genug Daten da sind UND Training/Retraining nötig ist
        if is_ml_phase and can_train and (needs_training or needs_retraining):
             logger_opt.info(f"Trial {trial_num + 1}: Triggering ML model training/retraining with {valid_results_count} valid data points...")
             # Verwende nur die validen Ergebnisse für das Training
             training_df = pd.DataFrame(valid_results)

             if not training_df.empty:
                 training_success = train_ml_model(training_df, TARGET_METRIC)
                 if not training_success:
                     logger_opt.warning("ML training failed. Falling back to random parameter selection for this trial.")
                     # WICHTIG: Setze is_ml_phase nicht global zurück, nur für diesen Trial könnte ML ausfallen
                     # Die nächste Retrain-Chance wird wieder versucht.
             else:
                  logger_opt.warning("No valid data found for training despite count. Should not happen. Skipping training.")
                  training_success = False # Kein Training möglich

             # Nach dem Versuch, setze is_ml_phase für *diesen Durchlauf*, falls Training fehlschlug
             if not training_success:
                 is_ml_phase = False

        # Parameter generieren: ML-geführt oder zufällig?
        if is_ml_phase and ml_model is not None: # Prüfe explizit, ob Modell nach Trainingsversuch existiert
            logger_opt.info(f"Trial {trial_num + 1}: Generating {NUM_CANDIDATES_PER_ML_TRIAL} candidates for ML-guided selection...")
            selected_by = "ML-Guided"
            candidate_params_list = [generate_random_params(param_space) for _ in range(NUM_CANDIDATES_PER_ML_TRIAL)]

            # Performance vorhersagen
            predicted_scores = predict_performance(candidate_params_list)

            if predicted_scores is not None and len(predicted_scores) == len(candidate_params_list):
                # Besten Kandidaten auswählen (höchster vorhergesagter Score)
                best_candidate_index = np.argmax(predicted_scores)
                current_parameters = candidate_params_list[best_candidate_index]
                logger_opt.info(f"Selected candidate #{best_candidate_index + 1} (predicted score: {predicted_scores[best_candidate_index]:.4f})")
            else:
                logger_opt.warning("ML prediction failed or returned unexpected results. Falling back to random selection for this trial.")
                current_parameters = generate_random_params(param_space)
                selected_by = "Random (ML Fallback)"
        else:
            # Rein zufällige Auswahl (Initialphase oder ML deaktiviert/fehlgeschlagen/nicht bereit)
            if trial_num < NUM_INITIAL_RANDOM_TRIALS:
                 selected_by = "Random (Initial Phase)"
            elif not USE_ML_GUIDANCE:
                 selected_by = "Random (ML Disabled)"
            elif not can_train:
                 selected_by = f"Random (Waiting for Data: {valid_results_count}/{MIN_DATA_FOR_TRAINING})"
            elif ml_model is None:
                 selected_by = "Random (ML Trained Failed Previously)"
            else: # Sollte nicht vorkommen, aber als Fallback
                 selected_by = "Random (Unknown Reason)"
            logger_opt.info(f"Trial {trial_num + 1}: Generating parameters randomly ({selected_by})...")
            current_parameters = generate_random_params(param_space)

        # Logge die ausgewählten Parameter (ohne die langen Listen für Übersicht)
        log_params = {k: v for k, v in current_parameters.items() if k not in ['take_profit_levels', 'take_profit_size_percentages', 'advanced_vwap_periods']}
        logger_opt.info(f"Trial {trial_num + 1} Parameters ({selected_by}): {log_params}")

        # --- Backtest Ausführung ---
        try:
            # 2. Z_config modifizieren
            modify_z_config(current_parameters)
            time.sleep(0.5) # Kurze Pause, falls nötig

            # 3. Backtest ausführen
            logger_opt.info(f"Trial {trial_num + 1}: Calling Backtest.main()...")
            Backtest.main() # Annahme: löscht Ausgabedateien NICHT selbst
            run_successful = True
            logger_opt.info(f"Trial {trial_num + 1}: Backtest.main() finished.")
            time.sleep(1) # Wartezeit, damit Dateien sicher geschrieben werden

            # 4. Ergebnis extrahieren
            logger_opt.info(f"Trial {trial_num + 1}: Reading performance results from {PERFORMANCE_CSV}...")
            if os.path.exists(PERFORMANCE_CSV):
                try:
                    time.sleep(0.2) # Zusätzliche Wartezeit
                    file_size = os.path.getsize(PERFORMANCE_CSV)
                    # logger_opt.debug(f"CHECK: {PERFORMANCE_CSV} exists. Size: {file_size} bytes.")
                    if file_size > 0:
                        # Lese CSV, behandle mögliche Fehler bei der Konvertierung später
                        performance_data_read = pd.read_csv(PERFORMANCE_CSV)
                        if not performance_data_read.empty:
                            performance_data = performance_data_read # Jetzt ist es ein DataFrame
                            logger_opt.info(f"Successfully read {len(performance_data)} performance row(s) from {PERFORMANCE_CSV}.")
                            result_summary = f"Success ({len(performance_data)} perf rows)"
                            # Speichere Spaltennamen beim ersten erfolgreichen Lesen
                            if not performance_columns and performance_data is not None:
                                performance_columns = list(performance_data.columns)
                        else:
                            logger_opt.warning(f"Read {PERFORMANCE_CSV}, but it resulted in an empty DataFrame.")
                            result_summary = f"{PERFORMANCE_CSV} read but empty"
                            performance_data = None # Setze explizit auf None
                    else:
                        logger_opt.warning(f"{PERFORMANCE_CSV} exists but is empty (size 0). No trades recorded?")
                        result_summary = f"{PERFORMANCE_CSV} empty (size 0)"
                        performance_data = None
                except pd.errors.EmptyDataError:
                     logger_opt.warning(f"{PERFORMANCE_CSV} could not be parsed (Pandas EmptyDataError).")
                     result_summary = f"{PERFORMANCE_CSV} parse error (EmptyDataError)"
                     performance_data = None
                except Exception as read_err:
                    logger_opt.error(f"Error reading or processing {PERFORMANCE_CSV}: {read_err}", exc_info=True)
                    result_summary = f"Error reading {PERFORMANCE_CSV}"
                    performance_data = None
            else:
                logger_opt.warning(f"CHECK: {PERFORMANCE_CSV} not found after Backtest.main() execution!")
                result_summary = f"No trades or {PERFORMANCE_CSV} not generated"
                performance_data = None

        except KeyboardInterrupt:
             logger_opt.warning("KeyboardInterrupt received during backtest phase. Stopping optimization.")
             print("\nOptimization stopped by user.")
             result_summary = "Interrupted"
             restore_z_config(original_config_snapshot)
             break # Verlasse die for-Schleife
        except Exception as main_err:
            logger_opt.error(f"--- ERROR during Backtest.main() execution or result processing in Trial {trial_num + 1} ---", exc_info=True)
            run_successful = False # redundant, aber klar
            result_summary = f"Backtest/Result Error: {str(main_err)[:100]}" # Kurze Fehlermeldung
            performance_data = None # Stelle sicher, dass keine alten/falschen Daten verwendet werden

        # --- Ergebnis Verarbeitung und Speicherung ---
        logger_opt.info(f"Trial {trial_num + 1}: Aggregating results. Status: {result_summary}")
        trial_results_list_for_append = [] # Nur die Ergebnisse dieses EINEN Trials für CSV/ML-Daten

        # Funktion zum sicheren Konvertieren zu Float
        def safe_float_convert(value):
            try:
                return float(value)
            except (ValueError, TypeError):
                return np.nan # Konvertiere ungültige Werte zu NaN

        if performance_data is not None and not performance_data.empty:
            # Es gibt Performance-Daten (können mehrere Zeilen sein)
            for _, perf_row in performance_data.iterrows():
                 row_dict = {
                     'trial': trial_num + 1,
                     'selected_by': selected_by,
                     'error': None, # Kein Fehler hier, da Performance Daten vorhanden sind
                     **current_parameters, # Füge Parameter hinzu
                 }
                 # Füge Performance-Metriken hinzu, versuche Konvertierung zu float
                 for col, val in perf_row.items():
                      if col in known_performance_cols: # Nur bekannte Performance-Spalten versuchen zu konvertieren
                           row_dict[col] = safe_float_convert(val)
                      elif col not in row_dict: # Andere Spalten (z.B. 'symbol') direkt übernehmen
                           row_dict[col] = val
                 trial_results_list_for_append.append(row_dict)
        else:
            # Keine Performance-Daten vorhanden (Fehler, keine Trades etc.)
            error_row = {
                'trial': trial_num + 1,
                'selected_by': selected_by,
                **current_parameters,
                'error': result_summary # Haupt-Fehler-/Statusmeldung
            }
            # Füge leere Performance-Spalten mit NaN hinzu
            cols_to_add = performance_columns if performance_columns else known_performance_cols # Fallback auf bekannte Liste
            for col in cols_to_add:
                 if col not in error_row:
                     error_row[col] = np.nan
            trial_results_list_for_append.append(error_row)

        # Füge Ergebnisse dieses Trials zur globalen Liste für ML hinzu
        # WICHTIG: Hier nur Dictionaries hinzufügen
        all_results_data.extend(trial_results_list_for_append)

        # An die globale CSV anhängen
        if trial_results_list_for_append:
            combined_df = pd.DataFrame(trial_results_list_for_append) # DataFrame nur für diesen Trial
            try:
                file_exists = os.path.exists(ALL_RUNS_CSV) and os.path.getsize(ALL_RUNS_CSV) > 0

                # Spaltenreihenfolge definieren (nur einmal bestimmen oder dynamisch anpassen)
                # Nimm alle Spalten aus dem aktuellen DataFrame
                current_cols = list(combined_df.columns)
                # Definiere eine sinnvolle Reihenfolge
                preferred_order = ['trial', 'selected_by', 'error'] + \
                                  list(param_space.keys()) + \
                                  ['interval_int_2', 'interval_int_3'] + \
                                  [col for col in known_performance_cols if col in current_cols] # Nur vorhandene Perf.-Spalten

                # Finale Liste: Bevorzugte zuerst, dann der Rest
                final_cols_ordered = []
                seen_cols = set()
                for col in preferred_order:
                    if col in current_cols and col not in seen_cols:
                        final_cols_ordered.append(col)
                        seen_cols.add(col)
                remaining_cols = sorted([col for col in current_cols if col not in seen_cols])
                final_cols_ordered.extend(remaining_cols)

                # Stelle sicher, dass der DataFrame diese Spalten hat (sollte der Fall sein)
                combined_df = combined_df.reindex(columns=final_cols_ordered, fill_value=np.nan)

                # Schreibe in CSV
                combined_df.to_csv(ALL_RUNS_CSV, mode='a', header=not file_exists, index=False, columns=final_cols_ordered)
                logger_opt.info(f"Trial {trial_num + 1}: Appended {len(combined_df)} row(s) to {ALL_RUNS_CSV}")

            except Exception as e:
                logger_opt.error(f"Trial {trial_num + 1}: Error appending results to {ALL_RUNS_CSV}: {e}", exc_info=True)

        # --- Temporäre Backtest-Ausgaben löschen ---
        try:
             logger_opt.info(f"Trial {trial_num + 1}: Cleaning up temporary backtest output files...")
             cleanup_backtest_outputs() # Bereinigt PERFORMANCE_CSV etc.
             time.sleep(0.5) # Kurze Pause
        except Exception as cleanup_err:
             logger_opt.error(f"Trial {trial_num + 1}: Error during post-trial cleanup: {cleanup_err}", exc_info=True)

        # --- Original Z_config wiederherstellen ---
        try:
            logger_opt.info(f"Trial {trial_num + 1}: Restoring original Z_config...")
            restore_z_config(original_config_snapshot)
        except Exception as restore_err:
             logger_opt.error(f"Trial {trial_num + 1}: Error restoring Z_config: {restore_err}", exc_info=True)
             # Eventuell kritisch? Entscheiden, ob Abbruch nötig.

        trial_duration = time.time() - trial_start_time
        logger_opt.info(f"--- Trial {trial_num + 1} finished in {trial_duration:.2f} seconds. Status: {result_summary} ---")
        time.sleep(0.5) # Kurze Pause zwischen Trials

    # --- Ende der Hauptschleife ---
    run_duration = time.time() - run_start_time
    logger_opt.info(f"=================================================")
    logger_opt.info(f" Optimization Run Finished in {run_duration:.2f} seconds")
    logger_opt.info(f" Results saved in: {ALL_RUNS_CSV}")
    if USE_ML_GUIDANCE:
        logger_opt.info(f" ML model was used for guidance after trial {NUM_INITIAL_RANDOM_TRIALS}.")
        # Optional: Feature Importances ausgeben, wenn das Modell und die Feature-Listen existieren
        if ml_model is not None and numeric_features_global is not None and categorical_features_global is not None:
            logger_opt.info("Attempting to calculate Feature Importances from the last trained model...")
            try:
                # Zugriff auf Pipeline-Schritte
                regressor = ml_model.named_steps['regressor']
                preprocessor_step = ml_model.named_steps['preprocessor']

                # Feature-Namen aus dem Preprocessor extrahieren
                transformed_feature_names = []
                # Numerische zuerst (Reihenfolge wie im ColumnTransformer)
                if 'num' in preprocessor_step.transformers_:
                     transformed_feature_names.extend(numeric_features_global)

                # Dann kategorische (OneHotEncoded)
                if 'cat' in preprocessor_step.transformers_:
                     cat_pipeline = preprocessor_step.named_transformers_['cat']
                     ohe_step = cat_pipeline.named_steps['onehot']
                     # Stelle sicher, dass categorical_features_global nicht leer ist
                     if categorical_features_global:
                          ohe_feature_names = ohe_step.get_feature_names_out(categorical_features_global)
                          transformed_feature_names.extend(list(ohe_feature_names))
                     else:
                          logger_opt.debug("No categorical features were processed by OHE.")


                importances = regressor.feature_importances_

                # Stelle sicher, dass die Anzahl der Namen mit der Anzahl der Importances übereinstimmt
                if len(transformed_feature_names) == len(importances):
                    feature_importance_df = pd.DataFrame({'feature': transformed_feature_names, 'importance': importances})
                    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
                    # Logge nur die Top N Features
                    top_n = 20
                    logger_opt.info(f"Top {top_n} Feature Importances (higher is better):")
                    # Verwende repr() für eine saubere Darstellung im Log, falls .to_string() Probleme macht
                    try:
                        importance_string = feature_importance_df.head(top_n).to_string(index=False)
                        logger_opt.info("\n" + importance_string)
                    except Exception as str_err:
                         logger_opt.error(f"Could not format feature importance to string: {str_err}")
                         logger_opt.info(repr(feature_importance_df.head(top_n))) # Fallback auf repr

                else:
                    logger_opt.error(f"Feature name count ({len(transformed_feature_names)}) mismatch with importance score count ({len(importances)}). Skipping importance display.")
                    logger_opt.debug(f"Transformed Feature Names ({len(transformed_feature_names)}): {transformed_feature_names}")
                    logger_opt.debug(f"Importance Scores ({len(importances)}): {importances}")


            except AttributeError as ae:
                 logger_opt.error(f"Could not access expected parts of the scikit-learn pipeline ('regressor', 'preprocessor', 'cat', 'onehot' etc.). Model structure might be different than expected or training failed. Error: {ae}", exc_info=True)
            except Exception as fe_err:
                logger_opt.error(f"Could not extract or display feature importances due to an unexpected error: {fe_err}", exc_info=True)
        elif USE_ML_GUIDANCE:
             logger_opt.warning("ML model or feature lists were not available/trained successfully. Skipping feature importance calculation.")


    logger_opt.info(f"=================================================")