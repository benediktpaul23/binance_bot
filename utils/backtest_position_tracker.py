# utils/backtest_position_tracker.py
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone # Ensure timezone is imported
import Z_config
import utils.Backtest as Backtest # Import for fetch_data
import copy

# Configure logging (if not already done globally)
logger = logging.getLogger(__name__) # Use __name__ for logger hierarchy
# logger.setLevel(logging.INFO) # Set level as needed
# handler = logging.StreamHandler()
# handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
# if not logger.handlers: # Avoid adding multiple handlers if imported multiple times
#     logger.addHandler(handler)

# utils/backtest_position_tracker.py
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import Z_config # Make sure this is accessible
# Use an alias for Backtest module to avoid potential naming conflicts
import utils.Backtest as BacktestModule 
import copy

logger = logging.getLogger(__name__)

class BacktestPositionTracker:
    def __init__(self):
        # --- Basis-Konfiguration lesen ---
        self.use_standard_sl_tp = getattr(Z_config, 'use_standard_sl_tp', False)
        self.commission_rate = getattr(Z_config, 'taker_fee_parameter', 0.00045)

        # --- Stufe 3 Simulationsparameter ---
        self.tsl_simulation_ticks = getattr(Z_config, 'intra_1m_tsl_simulation_ticks', 60) 
        self.conflict_simulation_ticks_per_segment = getattr(Z_config, 'intra_1m_conflict_simulation_ticks', 20)

        # --- Interne Defaults (können unten überschrieben werden) ---
        self.take_profit_parameter = 0.03
        self.stop_loss_parameter = 0.02

        # --- Logik basierend auf dem Modus (Standard oder Advanced) ---
        if self.use_standard_sl_tp:
            # --- Standard SL/TP Modus ---
            tp_val_from_config = getattr(Z_config, 'take_profit_parameter_binance', self.take_profit_parameter)
            sl_val_from_config = getattr(Z_config, 'stop_loss_parameter_binance', self.stop_loss_parameter)
            
            if tp_val_from_config is None:
                logging.error("Z_config.take_profit_parameter_binance is None! Using default 0.03")
                self.take_profit_parameter = 0.03
            else:
                try: self.take_profit_parameter = float(tp_val_from_config)
                except (ValueError, TypeError):
                    logging.error(f"Invalid value for Z_config.take_profit_parameter_binance: '{tp_val_from_config}'. Using default 0.03")
                    self.take_profit_parameter = 0.03

            if sl_val_from_config is None:
                logging.error("Z_config.stop_loss_parameter_binance is None! Using default 0.02")
                self.stop_loss_parameter = 0.02
            else:
                try: self.stop_loss_parameter = float(sl_val_from_config)
                except (ValueError, TypeError):
                     logging.error(f"Invalid value for Z_config.stop_loss_parameter_binance: '{sl_val_from_config}'. Using default 0.02")
                     self.stop_loss_parameter = 0.02

            if isinstance(self.stop_loss_parameter, (int, float)) and isinstance(self.take_profit_parameter, (int, float)):
                 logging.info(f"Tracker Init (Standard): SL={self.stop_loss_parameter*100:.2f}%, TP={self.take_profit_parameter*100:.2f}%")
            else:
                 logging.error(f"Tracker Init (Standard): SL/TP values invalid after processing. SL={self.stop_loss_parameter}, TP={self.take_profit_parameter}")
        else:
             # --- Advanced Parameter Modus (Trailing, Multi-TP) ---
             self.trailing_activation_threshold = getattr(Z_config, 'activation_threshold', 0.5)
             self.trailing_distance = getattr(Z_config, 'trailing_distance', 1.9)

             tp_levels_val = getattr(Z_config, 'take_profit_levels', [1.8, 3.7, 4.0])
             if isinstance(tp_levels_val, str):
                 try:
                     self.take_profit_levels = eval(tp_levels_val)
                     if not isinstance(self.take_profit_levels, list): raise ValueError("Eval did not return a list for take_profit_levels")
                 except Exception as e:
                     logging.warning(f"Could not eval take_profit_levels string '{tp_levels_val}': {e}. Using default.")
                     self.take_profit_levels = [1.8, 3.7, 4.0]
             elif isinstance(tp_levels_val, list): self.take_profit_levels = tp_levels_val
             else:
                 logging.warning(f"Invalid type for take_profit_levels ('{type(tp_levels_val)}'). Using default.")
                 self.take_profit_levels = [1.8, 3.7, 4.0]

             tp_size_val = getattr(Z_config, 'take_profit_size_percentages', [35, 35, 30])
             if isinstance(tp_size_val, str):
                 try:
                     self.take_profit_size_percentages = eval(tp_size_val)
                     if not isinstance(self.take_profit_size_percentages, list): raise ValueError("Eval did not return a list for take_profit_size_percentages")
                 except Exception as e:
                     logging.warning(f"Could not eval take_profit_size_percentages string '{tp_size_val}': {e}. Using default.")
                     self.take_profit_size_percentages = [35, 35, 30]
             elif isinstance(tp_size_val, list): self.take_profit_size_percentages = tp_size_val
             else:
                 logging.warning(f"Invalid type for take_profit_size_percentages ('{type(tp_size_val)}'). Using default.")
                 self.take_profit_size_percentages = [35, 35, 30]

             self.third_level_trailing_distance = getattr(Z_config, 'third_level_trailing_distance', 2.5)
             self.enable_breakeven = getattr(Z_config, 'enable_breakeven', False)
             self.enable_trailing_take_profit = getattr(Z_config, 'enable_trailing_take_profit', True)
             logging.info("Tracker Init: Using Advanced Position Management (Trailing/Multi-TP)")

        self.positions = {}

    def open_position(self, symbol, entry_price, position_type, quantity, entry_time):
        if symbol in self.positions and self.positions[symbol].get("is_active", False):
            logging.warning(f"Attempted to open a new position for {symbol} while one is already active. Ignoring.")
            return None

        if not isinstance(entry_time, datetime): entry_time = pd.to_datetime(entry_time)
        if entry_time.tzinfo is None: entry_time = entry_time.replace(tzinfo=timezone.utc)
        else: entry_time = entry_time.astimezone(timezone.utc)

        position_id = f"{symbol}_{entry_time.strftime('%Y%m%d%H%M%S')}_{position_type}"
        position_data = {
            "symbol": symbol, "position_id": position_id, "entry_price": float(entry_price),
            "position_type": position_type, "quantity": float(quantity), "entry_time": entry_time,
            "remaining_quantity": float(quantity), "is_active": True, "partial_exits": [],
            "exit_price": None, "exit_time": None, "exit_reason": None, "slippage": 0.0,
            "standard_sl": None, "standard_tp": None, "initial_stop": None, "current_stop": None,
            "activation_price": None, "trailing_activated": False, "best_price": float(entry_price),
            "take_profit_levels": [], "take_profit_quantities": [], "current_tp_level": 0,
            "reached_first": False, "reached_second": False, "reached_third": False,
            "tp_trailing_activated": False, "breakeven_activated": False, "breakeven_level": None,
            "best_tp_price": float(entry_price),
        }

        if self.use_standard_sl_tp:
            if position_type == "long":
                position_data["standard_sl"] = position_data["entry_price"] * (1 - self.stop_loss_parameter)
                position_data["standard_tp"] = position_data["entry_price"] * (1 + self.take_profit_parameter)
            else: # short
                position_data["standard_sl"] = position_data["entry_price"] * (1 + self.stop_loss_parameter)
                position_data["standard_tp"] = position_data["entry_price"] * (1 - self.take_profit_parameter)
            position_data["current_stop"] = position_data["standard_sl"]
            position_data["take_profit_levels"] = [position_data["standard_tp"]]
            position_data["take_profit_quantities"] = [position_data["quantity"]] # For standard, 100% at the single TP
        else:
            if position_type == "long":
                position_data["initial_stop"] = position_data["entry_price"] * (1 - self.trailing_distance / 100) 
                position_data["activation_price"] = position_data["entry_price"] * (1 + self.trailing_activation_threshold / 100)
            else: # short
                position_data["initial_stop"] = position_data["entry_price"] * (1 + self.trailing_distance / 100)
                position_data["activation_price"] = position_data["entry_price"] * (1 - self.trailing_activation_threshold / 100)
            position_data["current_stop"] = position_data["initial_stop"]
            position_data["take_profit_levels"] = self._calculate_tp_levels(position_data["entry_price"], position_type)
            position_data["take_profit_quantities"] = self._calculate_tp_quantities(position_data["quantity"])

        self.positions[symbol] = position_data
        logging.info(f"Position opened for {symbol} (ID: {position_id}) at {position_data['entry_price']:.5f}. Initial Stop: {position_data['current_stop']:.5f}")
        return copy.deepcopy(position_data)

    def _calculate_tp_levels(self, entry_price, position_type):
        tp_levels_prices = []
        for level_pct in self.take_profit_levels: # Uses the list from __init__
            if position_type == "long":
                tp_levels_prices.append(entry_price * (1 + float(level_pct) / 100))
            else:
                tp_levels_prices.append(entry_price * (1 - float(level_pct) / 100))
        return tp_levels_prices

    def _calculate_tp_quantities(self, total_quantity):
        tp_quantities = []
        remaining_qty_for_calc = float(total_quantity)
        # Ensure percentages sum to 100 or normalize
        current_sum_pct = sum(self.take_profit_size_percentages) # Uses the list from __init__
        normalized_percentages = list(self.take_profit_size_percentages) # Start with a copy

        if not np.isclose(current_sum_pct, 100.0) and current_sum_pct > 0:
            logging.warning(f"TP size percentages {self.take_profit_size_percentages} do not sum to 100. Normalizing.")
            normalized_percentages = [(p / current_sum_pct) * 100 for p in self.take_profit_size_percentages]
        elif current_sum_pct == 0 and len(self.take_profit_size_percentages) > 0 :
             equal_share = 100.0 / len(self.take_profit_size_percentages)
             normalized_percentages = [equal_share] * len(self.take_profit_size_percentages)
             logging.warning(f"TP size percentages sum to 0. Distributing equally: {normalized_percentages}")
        
        if not normalized_percentages and total_quantity > 0 : # if no TP sizes defined, but TP levels exist, assume 100% at first TP
            if self.take_profit_levels :
                normalized_percentages = [100.0] + [0.0] * (len(self.take_profit_levels) -1)


        for i, pct in enumerate(normalized_percentages):
            if i == len(normalized_percentages) - 1: # Last level gets all remaining
                qty_to_exit = remaining_qty_for_calc
            else:
                qty_to_exit = float(total_quantity) * (float(pct) / 100.0)
            
            qty_to_exit = min(qty_to_exit, remaining_qty_for_calc) 
            tp_quantities.append(qty_to_exit)
            remaining_qty_for_calc -= qty_to_exit
            if remaining_qty_for_calc < 1e-9: 
                tp_quantities.extend([0.0] * (len(normalized_percentages) - (i + 1))) 
                break
        return tp_quantities

    def _get_current_fixed_sl_and_next_tp(self, position_data):
        sl_price = position_data["current_stop"] 
        active_tp_price = None
        if self.use_standard_sl_tp:
            active_tp_price = position_data.get("standard_tp")
        else:
            for i in range(len(position_data.get("take_profit_levels", []))):
                level_flag = f"reached_{['first', 'second', 'third'][i]}" if i < 3 else f"reached_{i+1}"
                # Ensure take_profit_quantities has an entry for this level
                if i < len(position_data.get("take_profit_quantities", [])):
                    if not position_data.get(level_flag, False) and position_data["take_profit_quantities"][i] > 1e-9:
                        active_tp_price = position_data["take_profit_levels"][i]
                        break
                else: # Should not happen if initialized correctly
                    logging.warning(f"Missing take_profit_quantities for TP level {i+1} for {position_data['symbol']}")
                    break 
        return sl_price, active_tp_price

    def _calculate_new_tsl_level(self, position_data, best_price_for_tsl):
        if position_data["position_type"] == "long":
            return best_price_for_tsl * (1 - self.trailing_distance / 100)
        else:
            return best_price_for_tsl * (1 + self.trailing_distance / 100)

    def _update_tsl_stop_if_better(self, position_data, proposed_stop_level):
        stop_changed = False
        original_stop = position_data["current_stop"]
        if position_data["position_type"] == "long":
            if proposed_stop_level > original_stop:
                position_data["current_stop"] = proposed_stop_level
                stop_changed = True
        else: # short
            if proposed_stop_level < original_stop:
                position_data["current_stop"] = proposed_stop_level
                stop_changed = True
        if stop_changed:
             logging.debug(f"TSL UPDATED for {position_data['symbol']} from {original_stop:.5f} to {proposed_stop_level:.5f} based on best_price {position_data['best_price']:.5f}")
        return stop_changed

    def _resolve_1m_candle_sl_tp_ambiguity(
        self, one_m_candle_ohlc, sl_level, tp_level, position_type
    ):
        o,h,l,c = one_m_candle_ohlc['open'], one_m_candle_ohlc['high'], one_m_candle_ohlc['low'], one_m_candle_ohlc['close']
        logger.debug(
            f"Stufe 3.2 SL/TP Conflict Sim: O={o:.5f}, H={h:.5f}, L={l:.5f} | SL={sl_level:.5f}, TP={tp_level:.5f}, Type={position_type}"
        )
        num_ticks = self.conflict_simulation_ticks_per_segment

        if position_type == "long":
            if o != l:
                path_ol = np.linspace(o, l, num_ticks)
                for price_tick in path_ol:
                    if price_tick <= sl_level: return "sl"
                    if price_tick >= tp_level: return "tp"
            elif o <= sl_level: return "sl"
            elif o >= tp_level: return "tp"
            if l != h:
                path_lh = np.linspace(l, h, num_ticks)[1:]
                for price_tick in path_lh:
                    if price_tick <= sl_level: return "sl" 
                    if price_tick >= tp_level: return "tp"
        elif position_type == "short":
            if o != h:
                path_oh = np.linspace(o, h, num_ticks)
                for price_tick in path_oh:
                    if price_tick >= sl_level: return "sl"
                    if price_tick <= tp_level: return "tp"
            elif o >= sl_level: return "sl"
            elif o <= tp_level: return "tp"
            if h != l:
                path_hl = np.linspace(h, l, num_ticks)[1:]
                for price_tick in path_hl:
                    if price_tick >= sl_level: return "sl"
                    if price_tick <= tp_level: return "tp"
        
        logger.warning(f"Stufe 3.2 Sim: Path inconclusive. Defaulting to 'sl'. O={o},H={h},L={l},SL={sl_level},TP={tp_level}")
        return "sl"

    def _simulate_seconds_for_trailing_stop(
        self, position_data_sim, one_m_candle_ohlc, one_m_candle_time_utc
    ):
        o,h,l,c = one_m_candle_ohlc['open'], one_m_candle_ohlc['high'], one_m_candle_ohlc['low'], one_m_candle_ohlc['close']
        pos_type = position_data_sim["position_type"]
        
        logger.debug(f"Stufe 3.1 TSL Sim: {position_data_sim['symbol']} 1m O={o:.5f} H={h:.5f} L={l:.5f} C={c:.5f} | Current TSL={position_data_sim['current_stop']:.5f}, BestPrice={position_data_sim['best_price']:.5f}")

        path_segments_combined = []
        num_ticks_per_segment = max(1, self.tsl_simulation_ticks // 3) 

        current_path_point = o
        path_segments_combined.append(current_path_point)

        target_points_long = [l, h, c]
        target_points_short = [h, l, c]
        targets = target_points_long if pos_type == "long" else target_points_short

        for target_point in targets:
            if abs(current_path_point - target_point) > 1e-9: # If not already at target
                segment = np.linspace(current_path_point, target_point, num_ticks_per_segment + 1) # +1 to include end point
                path_segments_combined.extend(segment[1:]) # Add all but the first (already have current_path_point)
                current_path_point = target_point
            elif target_point != path_segments_combined[-1]: # If target is same as current but not last in list
                 path_segments_combined.append(target_point)


        path = [val for i, val in enumerate(path_segments_combined) if i == 0 or abs(val - path_segments_combined[i-1]) > 1e-9] # Remove consecutive duplicates due to float precision
        if not path: path = [o,l,h,c] 

        num_total_ticks_in_path = len(path)
        if num_total_ticks_in_path <= 1: 
             logger.warning(f"Stufe 3.1 TSL Sim: Path for {position_data_sim['symbol']} resulted in {num_total_ticks_in_path} tick(s). Skipping detailed sim for this candle, will use 1m OHLC.")
             return None, position_data_sim # Fallback to 1m OHLC check if path is too simple

        for i, tick_price in enumerate(path):
            if pos_type == "long":
                if tick_price > position_data_sim["best_price"]:
                    position_data_sim["best_price"] = tick_price
                    if not position_data_sim["trailing_activated"]:
                        if position_data_sim["entry_price"] > 0:
                             price_mv_pct = ((position_data_sim["best_price"] - position_data_sim["entry_price"]) / position_data_sim["entry_price"]) * 100
                             if price_mv_pct >= self.trailing_activation_threshold:
                                position_data_sim["trailing_activated"] = True
                                logging.debug(f"Stufe 3.1 Sim: TSL ACTIVATED for {position_data_sim['symbol']} at tick_price {tick_price:.5f}")
                    if position_data_sim["trailing_activated"]:
                        new_tsl = self._calculate_new_tsl_level(position_data_sim, position_data_sim["best_price"])
                        self._update_tsl_stop_if_better(position_data_sim, new_tsl)
            else: # short
                if tick_price < position_data_sim["best_price"]:
                    position_data_sim["best_price"] = tick_price
                    if not position_data_sim["trailing_activated"]:
                        if position_data_sim["entry_price"] > 0:
                            price_mv_pct = ((position_data_sim["entry_price"] - position_data_sim["best_price"]) / position_data_sim["entry_price"]) * 100
                            if price_mv_pct >= self.trailing_activation_threshold:
                                position_data_sim["trailing_activated"] = True
                                logging.debug(f"Stufe 3.1 Sim: TSL ACTIVATED for {position_data_sim['symbol']} at tick_price {tick_price:.5f}")
                    if position_data_sim["trailing_activated"]:
                        new_tsl = self._calculate_new_tsl_level(position_data_sim, position_data_sim["best_price"])
                        self._update_tsl_stop_if_better(position_data_sim, new_tsl)
            
            current_tsl_to_check = position_data_sim["current_stop"]
            tsl_hit_this_tick = False
            # Check if TSL is active OR if it's the initial stop and breakeven isn't active yet.
            is_tsl_or_initial_stop_relevant = position_data_sim["trailing_activated"] or \
                                             (not position_data_sim.get("breakeven_activated", False) and \
                                              current_tsl_to_check == position_data_sim.get("initial_stop"))

            if is_tsl_or_initial_stop_relevant:
                 if pos_type == "long" and tick_price <= current_tsl_to_check: tsl_hit_this_tick = True
                 elif pos_type == "short" and tick_price >= current_tsl_to_check: tsl_hit_this_tick = True

            if tsl_hit_this_tick:
                exit_price_tsl = current_tsl_to_check
                fraction = (i + 1) / num_total_ticks_in_path if num_total_ticks_in_path > 1 else 0.5 # Avoid div by zero if path is 1 tick
                sim_exit_time_delta_seconds = int(60 * fraction)
                sim_exit_time = one_m_candle_time_utc + timedelta(seconds=min(59, sim_exit_time_delta_seconds)) # Cap at 59s
                
                exit_reason_tsl = "trailing_stop_simulated" 
                if position_data_sim.get("breakeven_activated", False):
                    be_level = position_data_sim.get("breakeven_level")
                    if be_level is not None:
                        if (pos_type == "long" and exit_price_tsl >= be_level) or \
                           (pos_type == "short" and exit_price_tsl <= be_level):
                            exit_reason_tsl = "breakeven_stop_simulated"
                elif not position_data_sim["trailing_activated"] and current_tsl_to_check == position_data_sim.get("initial_stop"):
                    exit_reason_tsl = "initial_stop_simulated"

                logging.info(f"Stufe 3.1 Sim: {exit_reason_tsl.upper()} HIT for {position_data_sim['symbol']} at tick {i+1}. Price={tick_price:.5f}, StopLevel={exit_price_tsl:.5f}. SimExitTime: {sim_exit_time}")
                exit_details = self._finalize_simulated_exit(position_data_sim, exit_price_tsl, sim_exit_time, exit_reason_tsl)
                return exit_details, position_data_sim
                
        logger.debug(f"Stufe 3.1 Sim: No TSL hit for {position_data_sim['symbol']}. Final TSL: {position_data_sim['current_stop']:.5f}, BestPrice: {position_data_sim['best_price']:.5f}")
        return None, position_data_sim

    def _finalize_simulated_exit(self, sim_pos_data, exit_price, exit_time, exit_reason):
        exit_qty = sim_pos_data["remaining_quantity"]
        sim_pos_data["is_active"] = False
        sim_pos_data["exit_price"] = exit_price
        sim_pos_data["exit_time"] = exit_time
        sim_pos_data["exit_reason"] = exit_reason
        sim_pos_data["remaining_quantity"] = 0
        
        return {
            "symbol": sim_pos_data["symbol"], "position_id": sim_pos_data.get("position_id"),
            "exit_price": exit_price, "exit_time": exit_time, "exit_reason": exit_reason,
            "exit_quantity": exit_qty, "remaining_quantity": 0, "full_exit": True,
            "breakeven_activated": sim_pos_data.get("breakeven_activated", False),
            "exit_level": sim_pos_data.get("current_tp_level", 0)
        }

    def _simulate_intra_candle_advanced_exits(self, symbol, original_main_candle_series):
        if symbol not in self.positions or not self.positions[symbol].get("is_active", False):
            return [], self.positions.get(symbol) 
        if self.use_standard_sl_tp:
            return [], self.positions[symbol]

        simulated_position = copy.deepcopy(self.positions[symbol])

        original_main_candle_time = original_main_candle_series.name
        if not isinstance(original_main_candle_time, pd.Timestamp): original_main_candle_time = pd.Timestamp(original_main_candle_time)
        if original_main_candle_time.tzinfo is None: original_main_candle_time = original_main_candle_time.tz_localize('UTC')
        else: original_main_candle_time = original_main_candle_time.tz_convert('UTC')
        
        logger.info(f"--- Stufe 2: Intra-Main-Candle Sim Start ({symbol}) for main candle ending {original_main_candle_time} ---")
        log_initial_state = (f"Initial Sim State: Qty={simulated_position['remaining_quantity']:.8f}, Stop={simulated_position['current_stop']:.5f}, BestPrice={simulated_position['best_price']:.5f}, TSLActive={simulated_position['trailing_activated']}")
        logger.debug(log_initial_state)

        main_interval_str = getattr(Z_config, 'interval', '5m') 
        main_interval_minutes = Backtest.parse_interval_to_minutes(main_interval_str)
        if main_interval_minutes is None or main_interval_minutes <=0:
            logger.error(f"Stufe 2: Invalid main interval '{main_interval_str}'. Aborting simulation for this candle.")
            return [], simulated_position

        main_candle_logical_start_time = original_main_candle_time - timedelta(minutes=main_interval_minutes)
        
        try:
            one_min_df = Backtest.fetch_data(
                symbol=symbol, interval="1m",
                end_time=original_main_candle_time, 
                start_time_force=main_candle_logical_start_time
            )
            if one_min_df is None or one_min_df.empty:
                logger.warning(f"Stufe 2: No 1m data for {symbol} in main candle range {main_candle_logical_start_time} to {original_main_candle_time}. No Stufe 2/3 sim possible.")
                return [], simulated_position
            one_min_df = one_min_df[(one_min_df.index >= main_candle_logical_start_time) & (one_min_df.index < original_main_candle_time)]
            if one_min_df.empty:
                logger.warning(f"Stufe 2: No 1m data after strict time filtering for {symbol}. No Stufe 2/3 sim possible.")
                return [], simulated_position
            logger.info(f"Stufe 2: Found {len(one_min_df)} 1m candles for main candle {original_main_candle_time}.")
        except Exception as e:
            logger.error(f"Stufe 2: Error fetching/filtering 1m data for {symbol}: {e}", exc_info=True)
            return [], simulated_position

        exits_found_in_main_candle = []
        pos_type = simulated_position["position_type"]
        entry_price_for_be = simulated_position["entry_price"]

        for one_m_candle_time_idx, one_m_candle_series in one_min_df.iterrows():
            if not simulated_position["is_active"]: break

            one_m_candle_time_utc = one_m_candle_time_idx.tz_convert('UTC') if one_m_candle_time_idx.tzinfo else one_m_candle_time_idx.tz_localize('UTC')
            one_m_ohlc = {
                'open': float(one_m_candle_series['open']), 'high': float(one_m_candle_series['high']),
                'low': float(one_m_candle_series['low']), 'close': float(one_m_candle_series['close'])
            }
            logger.debug(f"--- Stufe 2: Processing 1m candle @ {one_m_candle_time_utc} O={one_m_ohlc['open']:.5f} H={one_m_ohlc['high']:.5f} L={one_m_ohlc['low']:.5f} C={one_m_ohlc['close']:.5f} ---")
            
            # STUFE 2.b: Trailing Stop Logic
            tsl_exit_details_from_1m_processing = None
            needs_tsl_seconds_sim = False
            if simulated_position["trailing_activated"] or (not simulated_position.get("breakeven_activated") and simulated_position["current_stop"] == simulated_position.get("initial_stop")):
                current_stop_to_check = simulated_position["current_stop"]
                if pos_type == "long":
                    if one_m_ohlc['low'] <= current_stop_to_check: needs_tsl_seconds_sim = True
                    if one_m_ohlc['high'] > simulated_position["best_price"]: needs_tsl_seconds_sim = True
                else: 
                    if one_m_ohlc['high'] >= current_stop_to_check: needs_tsl_seconds_sim = True
                    if one_m_ohlc['low'] < simulated_position["best_price"]: needs_tsl_seconds_sim = True
            
            if needs_tsl_seconds_sim:
                logger.debug(f"Stufe 2.b.ii: TSL Seconds Simulation (Stufe 3.1) triggered for {symbol} at {one_m_candle_time_utc}.")
                tsl_exit_details_from_1m_processing, simulated_position = self._simulate_seconds_for_trailing_stop(
                    simulated_position, one_m_ohlc, one_m_candle_time_utc
                )
                if tsl_exit_details_from_1m_processing:
                    exits_found_in_main_candle.append(tsl_exit_details_from_1m_processing)
                    break 
            else: 
                made_tsl_related_update_coarse = False
                if pos_type == "long":
                    if one_m_ohlc['high'] > simulated_position["best_price"]:
                        simulated_position["best_price"] = one_m_ohlc['high']; made_tsl_related_update_coarse = True
                else: 
                    if one_m_ohlc['low'] < simulated_position["best_price"]:
                        simulated_position["best_price"] = one_m_ohlc['low']; made_tsl_related_update_coarse = True
                
                if not simulated_position["trailing_activated"] and made_tsl_related_update_coarse :
                    price_mv_pct = 0
                    if simulated_position["entry_price"] > 0:
                        if pos_type == "long": price_mv_pct = ((simulated_position["best_price"] - simulated_position["entry_price"]) / simulated_position["entry_price"]) * 100
                        else: price_mv_pct = ((simulated_position["entry_price"] - simulated_position["best_price"]) / simulated_position["entry_price"]) * 100
                    if price_mv_pct >= self.trailing_activation_threshold:
                        simulated_position["trailing_activated"] = True
                        logging.debug(f"Stufe 2.b.iii: TSL ACTIVATED (coarse) for {symbol} via 1m OHLC.")

                if simulated_position["trailing_activated"] and made_tsl_related_update_coarse: # Update TSL if best_price changed
                    new_tsl_level = self._calculate_new_tsl_level(simulated_position, simulated_position["best_price"])
                    self._update_tsl_stop_if_better(simulated_position, new_tsl_level)
                
                current_stop_coarse_check = simulated_position["current_stop"]
                tsl_hit_coarse = False; exit_price_coarse_tsl = None
                is_tsl_or_initial_relevant_coarse = simulated_position["trailing_activated"] or \
                   (not simulated_position.get("breakeven_activated") and current_stop_coarse_check == simulated_position.get("initial_stop"))

                if is_tsl_or_initial_relevant_coarse:
                    if pos_type == "long" and one_m_ohlc['low'] <= current_stop_coarse_check:
                        tsl_hit_coarse = True; exit_price_coarse_tsl = current_stop_coarse_check
                    elif pos_type == "short" and one_m_ohlc['high'] >= current_stop_coarse_check:
                        tsl_hit_coarse = True; exit_price_coarse_tsl = current_stop_coarse_check
                
                if tsl_hit_coarse:
                    reason_coarse = "trailing_stop_1m_ohlc"
                    if not simulated_position["trailing_activated"] and current_stop_coarse_check == simulated_position.get("initial_stop"):
                        reason_coarse = "initial_stop_1m_ohlc"
                    elif simulated_position.get("breakeven_activated", False):
                        be_level = simulated_position.get("breakeven_level")
                        if be_level is not None and ((pos_type == "long" and exit_price_coarse_tsl >= be_level) or (pos_type == "short" and exit_price_coarse_tsl <= be_level)):
                            reason_coarse = "breakeven_stop_1m_ohlc"
                    
                    logging.info(f"Stufe 2.b.iii: TSL/Initial HIT (coarse) for {symbol} on 1m OHLC {one_m_candle_time_utc}. Price={exit_price_coarse_tsl:.5f}, Stop={current_stop_coarse_check:.5f}")
                    exit_event = self._finalize_simulated_exit(simulated_position, exit_price_coarse_tsl, one_m_candle_time_utc, reason_coarse)
                    exits_found_in_main_candle.append(exit_event)
                    break 

            if not simulated_position["is_active"]: continue

            # STUFE 2.c: Feste SL/TP-Konflikte
            current_sl_for_conflict, next_tp_for_conflict = self._get_current_fixed_sl_and_next_tp(simulated_position)
            sl_touched_by_1m = (current_sl_for_conflict is not None) and \
                               ((pos_type == "long" and one_m_ohlc['low'] <= current_sl_for_conflict) or \
                                (pos_type == "short" and one_m_ohlc['high'] >= current_sl_for_conflict))
            tp_touched_by_1m = (next_tp_for_conflict is not None) and \
                               ((pos_type == "long" and one_m_ohlc['high'] >= next_tp_for_conflict) or \
                                (pos_type == "short" and one_m_ohlc['low'] <= next_tp_for_conflict))

            if sl_touched_by_1m and tp_touched_by_1m:
                logger.info(f"Stufe 2.c.i: Conflict on 1m candle {one_m_candle_time_utc} for {symbol}. Escalating to Stufe 3.2.")
                hit_order = self._resolve_1m_candle_sl_tp_ambiguity(one_m_ohlc, current_sl_for_conflict, next_tp_for_conflict, pos_type)
                logger.info(f"Stufe 3.2 Result for {symbol} (1m conflict): '{hit_order}' hit first.")
                if hit_order == "sl":
                    sl_conflict_reason = "stop_loss_conflict_res" # Generic, could be BE or TSL if they became the fixed stop
                    exit_event = self._finalize_simulated_exit(simulated_position, current_sl_for_conflict, one_m_candle_time_utc, sl_conflict_reason)
                    exits_found_in_main_candle.append(exit_event)
                    break 
                elif hit_order == "tp":
                    price_hit_tp = one_m_ohlc['high'] if pos_type == "long" else one_m_ohlc['low']
                    tp_conflict_exits = self._update_take_profit_from_sim_state(simulated_position, price_hit_tp, one_m_candle_time_utc, "take_profit_conflict_res")
                    if tp_conflict_exits: exits_found_in_main_candle.extend(tp_conflict_exits)
                    if not simulated_position["is_active"]: break 
                if not simulated_position["is_active"]: continue

            # STUFE 2.d: Standard SL/TP Checks (wenn kein Konflikt oben zur Schließung führte)
            if not simulated_position["is_active"]: continue
            
            # 1. Check SL (current_stop könnte initial, TSL oder BE sein)
            # Nur prüfen, wenn kein Konflikt mit TP vorlag oder dieser TP bevorzugte
            if not (sl_touched_by_1m and tp_touched_by_1m and hit_order == "tp"): # Don't re-check SL if TP conflict was resolved to TP
                final_sl_to_check_d = simulated_position["current_stop"] 
                sl_hit_in_2d = False; sl_hit_price_2d = None
                if (pos_type == "long" and one_m_ohlc['low'] <= final_sl_to_check_d): sl_hit_in_2d = True; sl_hit_price_2d = final_sl_to_check_d
                elif (pos_type == "short" and one_m_ohlc['high'] >= final_sl_to_check_d): sl_hit_in_2d = True; sl_hit_price_2d = final_sl_to_check_d
                
                if sl_hit_in_2d:
                    reason_2d_sl = "stop_loss_1m_ohlc" # Default reason
                    if simulated_position.get("breakeven_activated", False): reason_2d_sl = "breakeven_stop_1m_ohlc"
                    elif simulated_position.get("trailing_activated", False) and final_sl_to_check_d != simulated_position.get("initial_stop"):
                        reason_2d_sl = "trailing_stop_1m_ohlc"
                    elif not simulated_position.get("trailing_activated", False) and final_sl_to_check_d == simulated_position.get("initial_stop"):
                         reason_2d_sl = "initial_stop_1m_ohlc"
                    logging.info(f"Stufe 2.d: SL HIT for {symbol} on 1m OHLC {one_m_candle_time_utc}. Price={sl_hit_price_2d:.5f}, Stop={final_sl_to_check_d:.5f}")
                    exit_event = self._finalize_simulated_exit(simulated_position, sl_hit_price_2d, one_m_candle_time_utc, reason_2d_sl)
                    exits_found_in_main_candle.append(exit_event)
                    break 
            
            if not simulated_position["is_active"]: continue

            # 2. Check Take Profit
            if not (sl_touched_by_1m and tp_touched_by_1m and hit_order == "sl"): # Don't re-check TP if SL conflict was resolved to SL
                price_for_tp_check_2d = one_m_ohlc['high'] if pos_type == "long" else one_m_ohlc['low']
                tp_exits_2d = self._update_take_profit_from_sim_state(simulated_position, price_for_tp_check_2d, one_m_candle_time_utc, "take_profit_1m_ohlc")
                if tp_exits_2d: exits_found_in_main_candle.extend(tp_exits_2d)
            
            if not simulated_position["is_active"]: break 
        
        logger.info(f"--- Stufe 2: Intra-Main-Candle Sim End ({symbol}). Found {len(exits_found_in_main_candle)} exit events. ---")
        log_final_state = (f"Final Sim State for {symbol} (after 1m loop): Active={simulated_position['is_active']}, Qty={simulated_position['remaining_quantity']:.8f}, Stop={simulated_position['current_stop']:.5f}, ExitReason={simulated_position.get('exit_reason', 'N/A')}")
        logger.debug(log_final_state)
        return exits_found_in_main_candle, simulated_position

    def _resolve_main_candle_conflict_via_1m(self, symbol, main_candle_series, initial_sl, initial_tp):
        position = self.positions.get(symbol) 
        if not position or not position["is_active"]:
            logger.warning(f"Resolve Main Candle Conflict: Called for inactive/non-existent position {symbol}")
            return "tp", main_candle_series.name 

        main_candle_time = main_candle_series.name
        if not isinstance(main_candle_time, pd.Timestamp): main_candle_time = pd.Timestamp(main_candle_time)
        if main_candle_time.tzinfo is None: main_candle_time = main_candle_time.tz_localize('UTC')
        else: main_candle_time = main_candle_time.tz_convert('UTC')

        logger.info(f"Stufe 1.B: Resolving Main Entry Candle Conflict for {symbol} @ {main_candle_time}")
        logger.info(f"  Initial SL: {initial_sl:.5f}, Initial TP: {initial_tp:.5f}, Type: {position['position_type']}")

        main_interval_str = getattr(Z_config, 'interval', '5m')
        main_interval_minutes = Backtest.parse_interval_to_minutes(main_interval_str)
        if main_interval_minutes is None or main_interval_minutes <= 0 :
            logger.warning(f"Could not parse main interval for entry conflict. Defaulting to TP priority at main candle time.")
            return "tp", main_candle_time

        mc_start_time = main_candle_time - timedelta(minutes=main_interval_minutes)
        mc_end_time = main_candle_time
        
        try:
            one_min_df_entry_conflict = Backtest.fetch_data(
                symbol=symbol, interval="1m",
                end_time=mc_end_time, start_time_force=mc_start_time
            )
            if one_min_df_entry_conflict is None or one_min_df_entry_conflict.empty:
                logger.warning(f"EntryConflict: No 1m data for {symbol}. Defaulting to TP at main candle time.")
                return "tp", main_candle_time
            
            one_min_df_entry_conflict = one_min_df_entry_conflict[
                (one_min_df_entry_conflict.index >= mc_start_time) & 
                (one_min_df_entry_conflict.index < mc_end_time)
            ]
            if one_min_df_entry_conflict.empty:
                logger.warning(f"EntryConflict: No 1m data after strict filtering for {symbol}. Defaulting to TP at main candle time.")
                return "tp", main_candle_time

            for one_m_candle_time_idx, one_m_candle in one_min_df_entry_conflict.iterrows():
                l, h = float(one_m_candle['low']), float(one_m_candle['high'])
                o, c = float(one_m_candle['open']), float(one_m_candle['close'])
                one_m_ohlc_dict = {'open':o, 'high':h, 'low':l, 'close':c}
                sl_hit = (position["position_type"] == "long" and l <= initial_sl) or \
                           (position["position_type"] == "short" and h >= initial_sl)
                tp_hit = (position["position_type"] == "long" and h >= initial_tp) or \
                           (position["position_type"] == "short" and l <= initial_tp)

                if sl_hit and tp_hit:
                    logger.info(f"EntryConflict: Ambiguity in 1m sub-candle @ {one_m_candle_time_idx}. Using Stufe 3.2 path simulation.")
                    hit_order_str = self._resolve_1m_candle_sl_tp_ambiguity(one_m_ohlc_dict, initial_sl, initial_tp, position["position_type"])
                    return hit_order_str, one_m_candle_time_idx
                if sl_hit: return "sl", one_m_candle_time_idx
                if tp_hit: return "tp", one_m_candle_time_idx
            
            logger.warning(f"EntryConflict: 1m data did not show clear SL or TP hit for initial levels. Defaulting TP at main candle time.")
            return "tp", main_candle_time
        except Exception as e:
            logger.error(f"Error in _resolve_main_candle_conflict_via_1m: {e}", exc_info=True)
            return "tp", main_candle_time

    def update_position_standard_mode(self, symbol, main_candle_h, main_candle_l, main_candle_time):
        if symbol not in self.positions or not self.positions[symbol]["is_active"] or not self.use_standard_sl_tp:
            return None 
        
        position = self.positions[symbol]
        pos_type = position["position_type"]
        sl_price = position.get("standard_sl")
        tp_price = position.get("standard_tp")

        if sl_price is None or tp_price is None: return None 
        if main_candle_time.tzinfo is None: main_candle_time = main_candle_time.replace(tzinfo=timezone.utc)
        else: main_candle_time = main_candle_time.astimezone(timezone.utc)

        sl_triggered = False; tp_triggered = False
        if pos_type == "long":
            if main_candle_l <= sl_price: sl_triggered = True
            if main_candle_h >= tp_price: tp_triggered = True
        else: 
            if main_candle_h >= sl_price: sl_triggered = True
            if main_candle_l <= tp_price: tp_triggered = True
        
        if sl_triggered and tp_triggered:
            logger.warning(f"Standard Mode: {symbol} SL and TP triggered on same main candle {main_candle_time}. Using 1m resolution for order.")
            main_candle_series_dummy = pd.Series(name=main_candle_time) 
            resolution, resolved_time = self._resolve_main_candle_conflict_via_1m(symbol, main_candle_series_dummy, sl_price, tp_price)
            
            if resolution == "sl": return self._close_position(symbol, sl_price, resolved_time, "standard_stop_loss_resolved")
            else: return self._close_position(symbol, tp_price, resolved_time, "standard_take_profit_resolved")
        elif sl_triggered: return self._close_position(symbol, sl_price, main_candle_time, "standard_stop_loss")
        elif tp_triggered: return self._close_position(symbol, tp_price, main_candle_time, "standard_take_profit")
        return None

    def _close_position(self, symbol, exit_price, exit_time_dt, exit_reason):
        if symbol not in self.positions or not self.positions[symbol]["is_active"]:
            logging.warning(f"Attempted to close already inactive or non-existent position for {symbol}")
            return None

        position_to_close = self.positions[symbol] 
        
        if not isinstance(exit_time_dt, datetime): exit_time_dt = pd.to_datetime(exit_time_dt)
        if exit_time_dt.tzinfo is None: exit_time_dt = exit_time_dt.replace(tzinfo=timezone.utc)
        else: exit_time_dt = exit_time_dt.astimezone(timezone.utc)

        exit_quantity_val = position_to_close["remaining_quantity"]
        position_to_close["is_active"] = False
        position_to_close["exit_price"] = float(exit_price)
        position_to_close["exit_time"] = exit_time_dt
        position_to_close["exit_reason"] = exit_reason
        position_to_close["remaining_quantity"] = 0.0 
        
        logging.info(f"POSITION CLOSED (Tracker Store): {symbol} (ID: {position_to_close.get('position_id')}). Reason: {exit_reason}, Price: {exit_price:.5f}, Time: {exit_time_dt}")

        return { 
            "symbol": symbol, "position_id": position_to_close.get("position_id"),
            "exit_price": float(exit_price), "exit_time": exit_time_dt, "exit_reason": exit_reason,
            "exit_quantity": exit_quantity_val, "remaining_quantity": 0.0, "full_exit": True,
            "breakeven_activated": position_to_close.get("breakeven_activated", False),
            "exit_level": position_to_close.get("current_tp_level", 0) 
        }

    def get_position(self, symbol):
        if symbol in self.positions:
            return copy.deepcopy(self.positions.get(symbol))
        return None

    def get_all_active_positions(self):
        active = {}
        for symbol, pos_data in self.positions.items():
            if pos_data.get("is_active", False):
                active[symbol] = copy.deepcopy(pos_data)
        return active

    def _update_take_profit_from_sim_state(self, sim_pos_data, current_price_for_tp, current_time_utc, base_reason="take_profit"):
        """Manages multi-TP logic within a simulation. Modifies sim_pos_data. Returns list of exit events."""
        if self.use_standard_sl_tp: 
            std_tp = sim_pos_data.get("standard_tp")
            if std_tp is not None:
                tp_hit = False
                if sim_pos_data["position_type"] == "long" and current_price_for_tp >= std_tp: tp_hit = True
                elif sim_pos_data["position_type"] == "short" and current_price_for_tp <= std_tp: tp_hit = True

                if tp_hit:
                    logging.info(f"Standard TP HIT for {sim_pos_data['symbol']} at {std_tp:.5f} within 1m candle {current_time_utc}")
                    # _finalize_simulated_exit modifies sim_pos_data to inactive
                    return [self._finalize_simulated_exit(sim_pos_data, std_tp, current_time_utc, "standard_take_profit")]
            return []

        triggered_tp_exits = []
        pos_type = sim_pos_data["position_type"]
        entry_price = sim_pos_data["entry_price"]

        for level_idx in range(len(sim_pos_data.get("take_profit_levels",[]))):
            if not sim_pos_data["is_active"]: break 
            
            level_flag = f"reached_{['first', 'second', 'third'][level_idx]}" if level_idx < 3 else f"reached_{level_idx+1}"
            if sim_pos_data.get(level_flag, False): continue

            # Ensure TP levels and quantities lists are long enough
            if level_idx >= len(sim_pos_data.get("take_profit_levels",[])) or level_idx >= len(sim_pos_data.get("take_profit_quantities",[])):
                logging.warning(f"TP level/quantity index out of bounds for {sim_pos_data['symbol']}. Level idx: {level_idx}")
                break 

            tp_price_level = sim_pos_data["take_profit_levels"][level_idx]
            qty_to_exit_at_level = sim_pos_data["take_profit_quantities"][level_idx]
            if qty_to_exit_at_level <= 1e-9 : continue

            tp_hit = False
            if pos_type == "long" and current_price_for_tp >= tp_price_level: tp_hit = True
            elif pos_type == "short" and current_price_for_tp <= tp_price_level: tp_hit = True

            if tp_hit:
                actual_exit_qty = min(qty_to_exit_at_level, sim_pos_data["remaining_quantity"])
                if actual_exit_qty <= 1e-9: continue

                reason = f"{base_reason}_{level_idx+1}"
                logging.info(f"TP Level {level_idx+1} HIT for {sim_pos_data['symbol']} at {tp_price_level:.5f} (1m candle {current_time_utc}). Exiting {actual_exit_qty:.8f}")

                sim_pos_data[level_flag] = True
                sim_pos_data["remaining_quantity"] -= actual_exit_qty
                sim_pos_data["current_tp_level"] = level_idx + 1
                
                is_full_exit_this_tp = sim_pos_data["remaining_quantity"] <= 1e-9
                
                partial_exit_detail = {
                    "symbol": sim_pos_data["symbol"], "position_id": sim_pos_data.get("position_id"),
                    "exit_price": tp_price_level, "exit_time": current_time_utc, "exit_reason": reason,
                    "exit_quantity": actual_exit_qty, "remaining_quantity": sim_pos_data["remaining_quantity"],
                    "full_exit": is_full_exit_this_tp, "exit_level": level_idx + 1,
                    "breakeven_activated": sim_pos_data.get("breakeven_activated", False)
                }
                triggered_tp_exits.append(partial_exit_detail)
                sim_pos_data.get("partial_exits", []).append(copy.deepcopy(partial_exit_detail)) # Add to internal list as well

                if is_full_exit_this_tp:
                    sim_pos_data["is_active"] = False
                    sim_pos_data["exit_price"] = tp_price_level
                    sim_pos_data["exit_time"] = current_time_utc
                    sim_pos_data["exit_reason"] = reason
                    break 

                if level_idx == 1 and self.enable_breakeven and not sim_pos_data.get("breakeven_activated", False):
                    be_buffer_pct = getattr(Z_config, 'breakeven_buffer_pct', 0.05) / 100 
                    be_level = entry_price * (1 + be_buffer_pct) if pos_type == "long" else entry_price * (1 - be_buffer_pct)
                    if (pos_type == "long" and be_level > sim_pos_data["current_stop"]) or \
                       (pos_type == "short" and be_level < sim_pos_data["current_stop"]):
                        logger.info(f"Breakeven for {sim_pos_data['symbol']} activated at {be_level:.5f} (stop moved from {sim_pos_data['current_stop']:.5f}) after TP{level_idx+1}")
                        sim_pos_data["current_stop"] = be_level
                        sim_pos_data["breakeven_activated"] = True
                        sim_pos_data["breakeven_level"] = be_level
                
                is_final_configured_tp_level = (level_idx == len(sim_pos_data.get("take_profit_levels",[])) - 1)
                if level_idx == 1 and self.enable_trailing_take_profit and not is_final_configured_tp_level: # After TP2, if not the last fixed TP
                    if not sim_pos_data.get("tp_trailing_activated", False):
                        sim_pos_data["tp_trailing_activated"] = True
                        sim_pos_data["best_tp_price"] = current_price_for_tp 
                        logger.info(f"Trailing Take Profit (for subsequent levels) for {sim_pos_data['symbol']} activated. Start best_tp_price: {sim_pos_data['best_tp_price']:.5f}")
        
        # Trailing Take Profit logic for the very last portion, if activated and position still active
        if sim_pos_data.get("tp_trailing_activated", False) and sim_pos_data["is_active"]:
            # This usually applies if all fixed TPs are hit OR if the last TP itself is a trailing one.
            # Let's assume it's for the remainder after all fixed TPs defined by take_profit_levels are processed.
            if sim_pos_data["current_tp_level"] >= len(sim_pos_data.get("take_profit_levels",[])):
                best_tp_price_updated_final_trail = False
                if pos_type == "long":
                    if current_price_for_tp > sim_pos_data["best_tp_price"]:
                        sim_pos_data["best_tp_price"] = current_price_for_tp; best_tp_price_updated_final_trail = True
                else: 
                    if current_price_for_tp < sim_pos_data["best_tp_price"]:
                        sim_pos_data["best_tp_price"] = current_price_for_tp; best_tp_price_updated_final_trail = True
                
                # Check if trailing TP stop is hit (always check once activated for this stage)
                trailing_tp_stop_level = 0.0
                trailing_tp_hit_final = False
                if pos_type == "long":
                    trailing_tp_stop_level = sim_pos_data["best_tp_price"] * (1 - self.third_level_trailing_distance / 100)
                    if current_price_for_tp <= trailing_tp_stop_level: trailing_tp_hit_final = True
                else: 
                    trailing_tp_stop_level = sim_pos_data["best_tp_price"] * (1 + self.third_level_trailing_distance / 100)
                    if current_price_for_tp >= trailing_tp_stop_level: trailing_tp_hit_final = True

                if trailing_tp_hit_final:
                    logger.info(f"Trailing TP (final portion) STOP HIT for {sim_pos_data['symbol']} at {trailing_tp_stop_level:.5f}")
                    exit_event = self._finalize_simulated_exit(sim_pos_data, trailing_tp_stop_level, current_time_utc, f"{base_reason}_trailing_final")
                    triggered_tp_exits.append(exit_event)
        return triggered_tp_exits
    
    def _resolve_sl_tp_priority(self, symbol, sl_price, tp_price, current_time, position_type):
        """
        Resolves SL/TP priority using finer time intervals if available.

        Args:
            symbol (str): Trading symbol.
            sl_price (float): Stop-Loss price.
            tp_price (float): Take-Profit price.
            current_time (datetime): Timestamp of the candle where conflict occurred (must be timezone-aware).
            position_type (str): "long" or "short".

        Returns:
            Tuple[str, pd.Timestamp]: ("sl" or "tp", timestamp_of_hit) or ("tp", original_current_time) on failure/default.
                                       The timestamp is the start time of the 1m candle where the hit occurred.
        """
        # Ensure current_time is timezone-aware UTC
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)
        else:
            current_time = current_time.astimezone(timezone.utc)

        original_current_time = current_time # Keep original for fallback

        logger.info(f"Resolving SL/TP conflict for {symbol} within candle ending at {current_time} (UTC)")
        logger.info(f"  SL: {sl_price}, TP: {tp_price}, Type: {position_type}")

        # Determine the original candle interval from Z_config
        original_interval_str = getattr(Z_config, 'interval', '5m') # Default to 5m
        original_interval_minutes = Backtest.parse_interval_to_minutes(original_interval_str)

        if original_interval_minutes is None:
            logger.warning(f"Could not parse original interval '{original_interval_str}'. Defaulting resolution to TP.")
            return "tp", original_current_time # Return original time on failure

        # Define hierarchy of finer intervals to try
        finer_intervals = ["1m"] # Start with 1m
        resolution_interval = None

        # Find the finest interval that is smaller than the original
        for interval in finer_intervals:
            interval_minutes = Backtest.parse_interval_to_minutes(interval)
            if interval_minutes is not None and interval_minutes < original_interval_minutes:
                resolution_interval = interval
                break

        if resolution_interval is None:
            logger.warning(f"No finer interval found than {original_interval_str}. Defaulting resolution to TP.")
            return "tp", original_current_time # Return original time on failure

        logger.info(f"Attempting resolution using {resolution_interval} interval.")

        # Calculate the time range for fetching finer data (the duration of the original candle)
        candle_start_time = current_time - timedelta(minutes=original_interval_minutes)
        candle_end_time = current_time

        logger.info(f"Fetching {resolution_interval} data for {symbol} from {candle_start_time} to {candle_end_time}")

        try:
            # Forcing lookback_hours to cover just the candle duration + buffer
            required_hours = (candle_end_time - candle_start_time).total_seconds() / 3600 + 0.1

            # Fetch data using the Backtest module's function
            detailed_data = Backtest.fetch_data(
                symbol=symbol,
                interval=resolution_interval,
                lookback_hours=required_hours,
                end_time=candle_end_time,
                start_time_force=candle_start_time # Force the start time
            )

            if detailed_data is None or detailed_data.empty:
                logger.warning(f"No detailed ({resolution_interval}) data found for {symbol} between {candle_start_time} and {candle_end_time}. Defaulting resolution to TP.")
                return "tp", original_current_time # Return original time on failure

            if detailed_data.empty:
                logger.warning(f"Detailed ({resolution_interval}) data found, but none within the exact candle range {candle_start_time} to {candle_end_time}. Defaulting resolution to TP.")
                return "tp", original_current_time # Return original time on failure

            logger.info(f"Received {len(detailed_data)} detailed candles for resolution.")

            # Iterate through the finer candles to see what was hit first
            for idx, candle in detailed_data.iterrows(): # idx is the Timestamp of the 1m candle
                candle_low = float(candle['low'])
                candle_high = float(candle['high'])
                sl_hit = False
                tp_hit = False

                if position_type == "long":
                    if candle_low <= sl_price: sl_hit = True
                    if candle_high >= tp_price: tp_hit = True
                else: # short
                    if candle_high >= sl_price: sl_hit = True
                    if candle_low <= tp_price: tp_hit = True

                # Check priority within the finer candle
                if sl_hit and tp_hit:
                    logger.info(f"SL and TP hit within the same {resolution_interval} candle at {idx}. Prioritizing SL.")
                    return "sl", idx # Return the 1m candle timestamp (idx)
                elif sl_hit:
                    logger.info(f"SL ({sl_price}) hit first at {idx} (Low: {candle_low}, High: {candle_high})")
                    return "sl", idx # Return the 1m candle timestamp (idx)
                elif tp_hit:
                    logger.info(f"TP ({tp_price}) hit first at {idx} (Low: {candle_low}, High: {candle_high})")
                    return "tp", idx # Return the 1m candle timestamp (idx)

            logger.warning(f"Resolution logic completed without finding SL or TP hit in detailed data for {symbol}. Defaulting to TP.")
            return "tp", original_current_time # Return original time on failure

        except Exception as e:
            logger.error(f"Error during SL/TP priority resolution fetch/processing for {symbol}: {e}", exc_info=True)
            return "tp", original_current_time # Return original time on error

# --- Global Helper Function ---
def _calculate_pnl(position_type, entry_price, exit_price, quantity, commission_rate=None):
    """Calculate Profit/Loss including commission."""
    if commission_rate is None:
        # Safely get commission rate from config with a default
        commission_rate = getattr(Z_config, 'taker_fee_parameter', 0.00045) # Default 0.045%

    # Ensure inputs are numeric and handle potential errors
    try:
        entry_price = float(entry_price)
        exit_price = float(exit_price)
        quantity = float(quantity)
        # Ensure quantity is positive for calculations
        if quantity <= 0:
             logging.error(f"Invalid quantity for PnL calculation: qty={quantity}. Must be positive.")
             return 0.0
    except (ValueError, TypeError) as e:
        logging.error(f"Invalid input for PnL calculation: entry={entry_price}, exit={exit_price}, qty={quantity}. Error: {e}")
        return 0.0 # Return 0 PnL on error

    # Calculate raw PnL based on position type
    if position_type == "long":
        raw_pnl = quantity * (exit_price - entry_price)
    elif position_type == "short":
        raw_pnl = quantity * (entry_price - exit_price)
    else:
        logging.error(f"Invalid position_type '{position_type}' for PnL calculation.")
        return 0.0 # Return 0 PnL for invalid type

    # Calculate commission fees for entry and exit
    entry_commission = quantity * entry_price * commission_rate
    exit_commission = quantity * exit_price * commission_rate
    total_commission = entry_commission + exit_commission

    # Calculate net PnL after deducting commissions
    net_pnl = raw_pnl - total_commission

    # Handle potential NaN result (e.g., if prices were somehow invalid despite float conversion)
    if np.isnan(net_pnl):
        logging.warning(f"PnL calculation resulted in NaN (raw_pnl={raw_pnl}, commission={total_commission}). Inputs: entry={entry_price}, exit={exit_price}, qty={quantity}. Returning 0.0.")
        return 0.0

    return net_pnl


