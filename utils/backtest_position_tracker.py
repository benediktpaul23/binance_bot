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

class BacktestPositionTracker:
    """
    Tracks positions during backtesting, handles standard and advanced position management.
    """
    def __init__(self):
        # Configuration parameters (fetch safely using getattr)
        self.use_standard_sl_tp = getattr(Z_config, 'use_standard_sl_tp', False)
        self.commission_rate = getattr(Z_config, 'taker_fee_parameter', 0.00045)

        # Standard SL/TP parameters (only if use_standard_sl_tp is True)
        self.take_profit_parameter = 0.03 # Default TP if not set
        self.stop_loss_parameter = 0.02 # Default SL if not set
        if self.use_standard_sl_tp:
            self.take_profit_parameter = getattr(Z_config, 'take_profit_parameter', self.take_profit_parameter)
            self.stop_loss_parameter = getattr(Z_config, 'stop_loss_parameter', self.stop_loss_parameter)
            logging.info(f"Tracker Init: Using Standard SL ({self.stop_loss_parameter*100:.2f}%) / TP ({self.take_profit_parameter*100:.2f}%)")
        else:
             # Advanced parameters (only if use_standard_sl_tp is False)
             self.trailing_activation_threshold = getattr(Z_config, 'activation_threshold', 0.5)
             self.trailing_distance = getattr(Z_config, 'trailing_distance', 1.9)
             self.take_profit_levels = getattr(Z_config, 'take_profit_levels', [1.8, 3.7, 4.0])
             self.take_profit_size_percentages = getattr(Z_config, 'take_profit_size_percentages', [35, 35, 30])
             self.third_level_trailing_distance = getattr(Z_config, 'third_level_trailing_distance', 2.5)
             self.enable_breakeven = getattr(Z_config, 'enable_breakeven', False)
             self.enable_trailing_take_profit = getattr(Z_config, 'enable_trailing_take_profit', True)
             logging.info("Tracker Init: Using Advanced Position Management (Trailing/Multi-TP)")


        # Dictionary to store position data for each symbol
        self.positions = {}

    def open_position(self, symbol, entry_price, position_type, quantity, entry_time):
        """Open a new position and initialize its state."""
        if symbol in self.positions and self.positions[symbol].get("is_active", False):
            logging.warning(f"Attempted to open a new position for {symbol} while one is already active. Ignoring.")
            return None # Return None or the existing position? Returning None is safer.

        # Ensure entry_time is timezone-aware (UTC)
        if entry_time.tzinfo is None:
            entry_time = entry_time.replace(tzinfo=timezone.utc)
        else:
            entry_time = entry_time.astimezone(timezone.utc)

        position_id = f"{symbol}_{entry_time.strftime('%Y%m%d%H%M%S')}_{position_type}"

        position_data = {
            "symbol": symbol,
            "position_id": position_id,
            "entry_price": entry_price,
            "position_type": position_type,
            "quantity": quantity, # Initial total quantity
            "entry_time": entry_time,
            "remaining_quantity": quantity, # Initially, remaining is total
            "is_active": True,
            "partial_exits": [], # List to store partial exit details
            "exit_price": None,
            "exit_time": None,
            "exit_reason": None,
            "slippage": 0.0, # Will be updated externally after opening

            # --- Initialize based on Standard vs Advanced ---
            "standard_sl": None,
            "standard_tp": None,
            "initial_stop": None,
            "current_stop": None,
            "activation_price": None,
            "trailing_activated": False,
            "best_price": entry_price,
            "take_profit_levels": [],
            "take_profit_quantities": [],
            "current_tp_level": 0,
            "reached_first": False,
            "reached_second": False,
            "reached_third": False,
            "tp_trailing_activated": False,
            "breakeven_activated": False,
            "breakeven_level": None, # Store the level when activated
            "best_tp_price": entry_price,
        }

        if self.use_standard_sl_tp:
            # Calculate and store standard SL/TP
            if position_type == "long":
                position_data["standard_sl"] = entry_price * (1 - self.stop_loss_parameter)
                position_data["standard_tp"] = entry_price * (1 + self.take_profit_parameter)
            else: # short
                position_data["standard_sl"] = entry_price * (1 + self.stop_loss_parameter)
                position_data["standard_tp"] = entry_price * (1 - self.take_profit_parameter)
            # Set current_stop for consistency, although standard logic might bypass it
            position_data["current_stop"] = position_data["standard_sl"]
            position_data["take_profit_levels"] = [position_data["standard_tp"]] # Store TP in levels format

        else:
            # Calculate advanced/trailing initial state
            if position_type == "long":
                position_data["initial_stop"] = entry_price * (1 - self.trailing_distance / 100)
                position_data["activation_price"] = entry_price * (1 + self.trailing_activation_threshold / 100)
            else: # short
                position_data["initial_stop"] = entry_price * (1 + self.trailing_distance / 100)
                position_data["activation_price"] = entry_price * (1 - self.trailing_activation_threshold / 100)
            position_data["current_stop"] = position_data["initial_stop"]
            position_data["take_profit_levels"] = self._calculate_tp_levels(entry_price, position_type)
            position_data["take_profit_quantities"] = self._calculate_tp_quantities(quantity)

        self.positions[symbol] = position_data
        logging.info(f"Position opened in tracker for {symbol} (ID: {position_id})")
        return position_data # Return the created position data


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



    def update_position(self, symbol, current_price, current_time):
         """
         Update position state based on current price, handling SL, TP (standard or advanced).
         *** NOTE: For Advanced mode, this function is largely superseded by _simulate_intra_candle_advanced_exits ***
         *** It remains primarily for STANDARD SL/TP checks based on the overall candle High/Low. ***

         current_price here represents the price point relevant for the check (e.g., low for long SL, high for long TP).
         Returns exit details dictionary if an exit occurred, otherwise None.
         """
         if symbol not in self.positions or not self.positions[symbol].get("is_active", False):
             return None # No active position

         position = self.positions[symbol]
         position_type = position["position_type"]

         # Ensure current_time is timezone-aware UTC
         if current_time.tzinfo is None:
             current_time = current_time.replace(tzinfo=timezone.utc)
         else:
             current_time = current_time.astimezone(timezone.utc)


         # --- Standard SL/TP Logic ---
         if self.use_standard_sl_tp:
             sl_price = position.get("standard_sl")
             tp_price = position.get("standard_tp")

             if sl_price is None or tp_price is None:
                 logging.error(f"Standard SL/TP prices not found for active position {symbol}. Cannot update.")
                 return None # Cannot proceed without prices

             sl_triggered = False
             tp_triggered = False

             # IMPORTANT: current_price passed here is typically the High or Low of the WHOLE candle
             if position_type == "long":
                 # Check SL against the LOW price passed in
                 if current_price <= sl_price: sl_triggered = True
                 # Check TP against the HIGH price passed in (requires separate call)
                 # This basic check doesn't handle simultaneous hits well, relies on _resolve_sl_tp_priority
                 # Let's assume the caller passes the relevant price (low for SL check, high for TP check)
                 if current_price >= tp_price: tp_triggered = True # This check might be inaccurate depending on what price is passed
             else: # short
                 # Check SL against the HIGH price passed in
                 if current_price >= sl_price: sl_triggered = True
                 # Check TP against the LOW price passed in
                 if current_price <= tp_price: tp_triggered = True # This check might be inaccurate

             # The standard logic handles one trigger at a time based on the single price point passed.
             # Conflict resolution happens externally (Phase 1) or by sequence of calls.
             if sl_triggered:
                 # Use _close_position for consistency
                 logging.debug(f"Standard SL triggered for {symbol} via update_position (Price: {current_price}, SL: {sl_price})")
                 return self._close_position(symbol, sl_price, current_time, "standard_stop_loss")
             elif tp_triggered:
                 logging.debug(f"Standard TP triggered for {symbol} via update_position (Price: {current_price}, TP: {tp_price})")
                 return self._close_position(symbol, tp_price, current_time, "standard_take_profit")
             else:
                 return None # No standard exit triggered by this specific price check

         # --- Advanced Trailing/Multi-TP Logic ---
         else:
             # --- THIS PATH SHOULD NOT BE NORMALLY REACHED IF SIMULATION IS USED ---
             # If this is called in advanced mode, it implies simulation wasn't used or failed.
             # Log a warning and potentially fall back to the old (less accurate) method,
             # or simply return None as simulation should handle it.
             logger.warning(f"update_position called in Advanced Mode for {symbol}. This should be handled by simulation. Skipping.")
             return None


    # --- Keep other methods like _calculate_tp_levels, _calculate_tp_quantities, ---
    # --- _update_trailing_stop, _update_take_profit, _close_position, get_position etc. ---
    # --- Ensure they are consistent with the logic above. ---

    def _calculate_tp_levels(self, entry_price, position_type):
        """Calculate take profit price levels based on config percentages."""
        tp_levels = []
        for level_pct in self.take_profit_levels:
            if position_type == "long":
                tp_price = entry_price * (1 + level_pct / 100)
            else: # short
                tp_price = entry_price * (1 - level_pct / 100)
            tp_levels.append(tp_price)
        return tp_levels

    def _calculate_tp_quantities(self, total_quantity):
        """Calculate quantities for each take profit level based on config percentages."""
        tp_quantities = []
        remaining_qty = total_quantity
        total_pct = sum(self.take_profit_size_percentages)

        # Normalize percentages if they don't sum to 100
        normalized_percentages = self.take_profit_size_percentages
        if not np.isclose(total_pct, 100):
            logging.warning(f"TP size percentages ({self.take_profit_size_percentages}) do not sum to 100. Normalizing.")
            if total_pct > 0:
                 normalized_percentages = [(p / total_pct) * 100 for p in self.take_profit_size_percentages]
            else:
                 # Avoid division by zero if sum is 0
                 equal_pct = 100 / len(self.take_profit_size_percentages) if len(self.take_profit_size_percentages) > 0 else 0
                 normalized_percentages = [equal_pct] * len(self.take_profit_size_percentages)


        for i, pct in enumerate(normalized_percentages):
             # Avoid floating point errors by calculating precisely
             if i == len(normalized_percentages) - 1:
                 # Last level gets the exact remaining quantity
                 level_qty = remaining_qty
             else:
                 level_qty = total_quantity * (pct / 100)

             # Ensure quantity does not exceed remaining
             level_qty = min(level_qty, remaining_qty)
             tp_quantities.append(level_qty)
             remaining_qty -= level_qty
             # Break if remaining quantity is zero or negligible
             if remaining_qty < 1e-9: # Use a small threshold for floating point comparison
                 # Assign 0 to remaining levels if any
                 tp_quantities.extend([0.0] * (len(normalized_percentages) - (i + 1)))
                 break

        # Final check if quantities sum up correctly
        if not np.isclose(sum(tp_quantities), total_quantity):
             logging.warning(f"Calculated TP quantities {sum(tp_quantities)} do not sum to total {total_quantity}. Last element adjusted.")
             # Adjust last element slightly if needed due to floating point math
             if tp_quantities:
                 tp_quantities[-1] = total_quantity - sum(tp_quantities[:-1])


        return tp_quantities


    def _update_trailing_stop(self, symbol, current_price):
        """Update trailing stop logic. current_price is the relevant high/low."""
        position = self.positions[symbol]
        position_type = position["position_type"]
        entry_price = position["entry_price"]

        # Update best price observed
        if position_type == "long":
            if current_price > position["best_price"]:
                position["best_price"] = current_price
        else: # short
            if current_price < position["best_price"]:
                position["best_price"] = current_price

        # Check if trailing activation threshold is met
        if not position["trailing_activated"]:
            price_movement_pct = 0
            if position_type == "long":
                if entry_price > 0: price_movement_pct = ((position["best_price"] - entry_price) / entry_price) * 100
            else: # short
                if entry_price > 0: price_movement_pct = ((entry_price - position["best_price"]) / entry_price) * 100

            if price_movement_pct >= self.trailing_activation_threshold:
                position["trailing_activated"] = True
                # logging.info(f"Trailing stop ACTIVATED for {symbol} at price {current_price}")

        # Adjust stop loss if trailing is active
        if position["trailing_activated"]:
            new_stop = position["current_stop"] # Start with current stop
            if position_type == "long":
                proposed_stop = position["best_price"] * (1 - self.trailing_distance / 100)
                # Only move stop up
                new_stop = max(position["current_stop"], proposed_stop)
            else: # short
                proposed_stop = position["best_price"] * (1 + self.trailing_distance / 100)
                # Only move stop down
                new_stop = min(position["current_stop"], proposed_stop)

            if new_stop != position["current_stop"]:
                 # logging.info(f"Trailing stop UPDATED for {symbol} from {position['current_stop']:.8f} to {new_stop:.8f}")
                 position["current_stop"] = new_stop


        # Check if current stop loss is triggered
        stop_triggered = False
        if position_type == "long":
            if current_price <= position["current_stop"]: stop_triggered = True
        else: # short
            if current_price >= position["current_stop"]: stop_triggered = True

        if stop_triggered:
            reason = "breakeven_stop" if position.get("breakeven_activated", False) else "trailing_stop"
            # logging.info(f"STOP TRIGGERED for {symbol} ({reason}) at price {current_price} (Stop Level: {position['current_stop']})")
            return True, reason

        return False, None


    def _update_take_profit(self, symbol, current_price, current_time):
        """Update multi-level take profit logic. current_price is the relevant high/low."""
        position = self.positions[symbol]
        position_type = position["position_type"]
        entry_price = position["entry_price"]
        tp_levels = position["take_profit_levels"]
        position_id = position.get("position_id", f"{symbol}_{position['entry_time']}")

        # Check TP levels sequentially
        for level_index in range(len(tp_levels)):
             level_reached_flag = f"reached_{['first', 'second', 'third'][level_index]}" # e.g., "reached_first"
             tp_price = tp_levels[level_index]
             exit_qty = position["take_profit_quantities"][level_index]
             exit_reason = f"take_profit_{level_index + 1}"

             # Skip if this level was already reached or quantity for this level is zero
             if position.get(level_reached_flag, False) or exit_qty <= 1e-9:
                 continue

             # Check if TP level is hit
             tp_hit = False
             if position_type == "long":
                 if current_price >= tp_price: tp_hit = True
             else: # short
                 if current_price <= tp_price: tp_hit = True

             if tp_hit:
                 # Ensure we don't exit more than remaining quantity (important for last level)
                 actual_exit_qty = min(exit_qty, position["remaining_quantity"])
                 if actual_exit_qty <= 1e-9: # Skip if remaining is negligible
                     continue

                 # Record partial/final exit
                 exit_details = {
                     "exit_time": current_time,
                     "exit_price": tp_price, # Exit at the TP level price
                     "exit_quantity": actual_exit_qty,
                     "exit_level": level_index + 1,
                     "exit_reason": exit_reason,
                     "position_id": position_id
                 }
                 position["partial_exits"].append(exit_details)

                 # Update position state
                 position[level_reached_flag] = True
                 position["remaining_quantity"] -= actual_exit_qty
                 position["current_tp_level"] = level_index + 1

                 # Check if this is the final exit for the position
                 is_final_tp_level = (level_index == len(tp_levels) - 1)
                 is_full_exit = (position["remaining_quantity"] <= 1e-9) or is_final_tp_level # Close if negligible qty left or last TP

                 # logger.info(f"TAKE PROFIT {level_index + 1}/{len(tp_levels)} HIT for {symbol} (ID: {position_id}) at {tp_price}")
                 # logger.info(f"  Exited Quantity: {actual_exit_qty:.8f}, Remaining: {position['remaining_quantity']:.8f}")


                 # Handle post-TP actions (Breakeven, Trailing TP activation)
                 if level_index == 1: # After TP2 is hit
                     # Activate Breakeven Stop
                     if self.enable_breakeven and not position.get("breakeven_activated", False):
                         breakeven_level = entry_price # Simple breakeven at entry
                         # Add small buffer to cover fees if desired (e.g., 0.05%)
                         buffer = 0.0005
                         if position_type == "long":
                             breakeven_level = entry_price * (1 + buffer)
                             if breakeven_level > position["current_stop"]: # Only move stop up
                                 position["current_stop"] = breakeven_level
                                 position["breakeven_activated"] = True
                                 position["breakeven_level"] = breakeven_level
                                 # logger.info(f"Breakeven stop activated and moved to {breakeven_level:.8f} for {symbol}")
                         else: # short
                             breakeven_level = entry_price * (1 - buffer)
                             if breakeven_level < position["current_stop"]: # Only move stop down
                                 position["current_stop"] = breakeven_level
                                 position["breakeven_activated"] = True
                                 position["breakeven_level"] = breakeven_level
                                 # logger.info(f"Breakeven stop activated and moved to {breakeven_level:.8f} for {symbol}")


                     # Activate Trailing Take Profit for the last portion
                     if self.enable_trailing_take_profit and not is_final_tp_level: # Check if there's a next level
                          if not position.get("tp_trailing_activated", False):
                               position["tp_trailing_activated"] = True
                               position["best_tp_price"] = current_price # Start trailing from current price
                               # logger.info(f"Trailing Take Profit activated for {symbol} (level 3 onwards) starting from {current_price}")


                 # Handle Trailing Take Profit for the last level
                 if is_final_tp_level and position.get("tp_trailing_activated", False):
                     # Update best TP price
                     if position_type == "long":
                         if current_price > position["best_tp_price"]:
                             position["best_tp_price"] = current_price
                         # Calculate trailing TP stop
                         trailing_tp_stop = position["best_tp_price"] * (1 - self.third_level_trailing_distance / 100)
                         if current_price <= trailing_tp_stop:
                             # Trailing TP stop hit, close remaining position
                             final_exit_qty = position["remaining_quantity"]
                             if final_exit_qty > 1e-9:
                                  exit_details_trailing = {
                                      "exit_time": current_time,
                                      "exit_price": trailing_tp_stop, # Exit at the stop level
                                      "exit_quantity": final_exit_qty,
                                      "exit_level": 3,
                                      "exit_reason": "take_profit_3_trailing",
                                      "position_id": position_id
                                  }
                                  position["partial_exits"].append(exit_details_trailing)
                                  position["remaining_quantity"] = 0
                                  is_full_exit = True
                                  exit_reason = "take_profit_3_trailing"
                                  # logger.info(f"Trailing TP (L3) STOP HIT for {symbol} at {trailing_tp_stop:.8f}. Closed remaining {final_exit_qty:.8f}.")

                     else: # short
                         if current_price < position["best_tp_price"]:
                              position["best_tp_price"] = current_price
                         # Calculate trailing TP stop
                         trailing_tp_stop = position["best_tp_price"] * (1 + self.third_level_trailing_distance / 100)
                         if current_price >= trailing_tp_stop:
                             # Trailing TP stop hit
                             final_exit_qty = position["remaining_quantity"]
                             if final_exit_qty > 1e-9:
                                 exit_details_trailing = {
                                     "exit_time": current_time,
                                     "exit_price": trailing_tp_stop, # Exit at the stop level
                                     "exit_quantity": final_exit_qty,
                                     "exit_level": 3,
                                     "exit_reason": "take_profit_3_trailing",
                                     "position_id": position_id
                                 }
                                 position["partial_exits"].append(exit_details_trailing)
                                 position["remaining_quantity"] = 0
                                 is_full_exit = True
                                 exit_reason = "take_profit_3_trailing"
                                 # logger.info(f"Trailing TP (L3) STOP HIT for {symbol} at {trailing_tp_stop:.8f}. Closed remaining {final_exit_qty:.8f}.")


                 # Finalize position if it's fully closed
                 if is_full_exit:
                     position["is_active"] = False
                     position["exit_price"] = exit_details["exit_price"] # Use the last exit price
                     position["exit_time"] = exit_details["exit_time"]
                     position["exit_reason"] = exit_reason # Use the reason from the last exit step
                     # logger.info(f"POSITION CLOSED for {symbol} (ID: {position_id}) after final TP ({exit_reason}).")


                 # Return the details of the exit that just occurred
                 return {
                     "symbol": symbol,
                     "position_id": position_id,
                     "exit_price": exit_details["exit_price"],
                     "exit_time": exit_details["exit_time"],
                     "exit_reason": exit_details["exit_reason"],
                     "exit_quantity": exit_details["exit_quantity"],
                     "exit_level": exit_details["exit_level"],
                     "remaining_quantity": position["remaining_quantity"],
                     "full_exit": is_full_exit,
                     "breakeven_activated": position.get("breakeven_activated", False)
                 }

             # If TP not hit for this level, continue to check next level (or exit loop)


        # If Trailing TP is active but the fixed TP3 level wasn't hit, still check the trailing stop
        if position.get("tp_trailing_activated", False) and not position.get("reached_third", False) and position.get("remaining_quantity", 0) > 0:
             if position_type == "long":
                 if current_price > position["best_tp_price"]: position["best_tp_price"] = current_price
                 trailing_tp_stop = position["best_tp_price"] * (1 - self.third_level_trailing_distance / 100)
                 if current_price <= trailing_tp_stop:
                      final_exit_qty = position["remaining_quantity"]
                      if final_exit_qty > 1e-9:
                          exit_details_trailing = {
                              "exit_time": current_time, "exit_price": trailing_tp_stop, "exit_quantity": final_exit_qty,
                              "exit_level": 3, "exit_reason": "take_profit_3_trailing", "position_id": position_id
                          }
                          position["partial_exits"].append(exit_details_trailing)
                          position["remaining_quantity"] = 0
                          position["is_active"] = False
                          position["exit_price"] = trailing_tp_stop
                          position["exit_time"] = current_time
                          position["exit_reason"] = "take_profit_3_trailing"
                          # logger.info(f"Trailing TP (L3) STOP HIT (standalone check) for {symbol} at {trailing_tp_stop:.8f}.")
                          return { # Return the full exit details
                               "symbol": symbol, "position_id": position_id, "exit_price": trailing_tp_stop, "exit_time": current_time,
                               "exit_reason": "take_profit_3_trailing", "exit_quantity": final_exit_qty, "exit_level": 3,
                               "remaining_quantity": 0, "full_exit": True, "breakeven_activated": position.get("breakeven_activated", False)
                           }
             else: # short
                 if current_price < position["best_tp_price"]: position["best_tp_price"] = current_price
                 trailing_tp_stop = position["best_tp_price"] * (1 + self.third_level_trailing_distance / 100)
                 if current_price >= trailing_tp_stop:
                     final_exit_qty = position["remaining_quantity"]
                     if final_exit_qty > 1e-9:
                         exit_details_trailing = {
                             "exit_time": current_time, "exit_price": trailing_tp_stop, "exit_quantity": final_exit_qty,
                             "exit_level": 3, "exit_reason": "take_profit_3_trailing", "position_id": position_id
                         }
                         position["partial_exits"].append(exit_details_trailing)
                         position["remaining_quantity"] = 0
                         position["is_active"] = False
                         position["exit_price"] = trailing_tp_stop
                         position["exit_time"] = current_time
                         position["exit_reason"] = "take_profit_3_trailing"
                         # logger.info(f"Trailing TP (L3) STOP HIT (standalone check) for {symbol} at {trailing_tp_stop:.8f}.")
                         return { # Return the full exit details
                               "symbol": symbol, "position_id": position_id, "exit_price": trailing_tp_stop, "exit_time": current_time,
                               "exit_reason": "take_profit_3_trailing", "exit_quantity": final_exit_qty, "exit_level": 3,
                               "remaining_quantity": 0, "full_exit": True, "breakeven_activated": position.get("breakeven_activated", False)
                           }

        return None # No TP hit in this update


    def _close_position(self, symbol, exit_price, exit_time, exit_reason):
        """Internal helper to mark a position as closed and return exit details."""
        if symbol not in self.positions or not self.positions[symbol]["is_active"]:
             logging.warning(f"Attempted to close already inactive or non-existent position for {symbol}")
             return None

        position = self.positions[symbol]

        # Ensure exit_time is timezone-aware UTC
        if exit_time.tzinfo is None:
             exit_time = exit_time.replace(tzinfo=timezone.utc)
        else:
             exit_time = exit_time.astimezone(timezone.utc)


        exit_quantity = position["remaining_quantity"] # Close the remaining quantity

        # Mark as inactive and record exit details
        position["is_active"] = False
        position["exit_price"] = exit_price
        position["exit_time"] = exit_time
        position["exit_reason"] = exit_reason
        position["remaining_quantity"] = 0 # Set remaining to zero

        # Add to partial exits list for consistency in reporting if needed? Or just return final state.
        # Let's just return the final state clearly marked as full_exit=True

        exit_result = {
            "symbol": symbol,
            "position_id": position.get("position_id"),
            "exit_price": exit_price,
            "exit_time": exit_time,
            "exit_reason": exit_reason,
            "exit_quantity": exit_quantity, # The quantity that was exited now
            "remaining_quantity": 0,
            "full_exit": True, # Mark this as closing the entire remaining position
            "breakeven_activated": position.get("breakeven_activated", False),
            "exit_level": position.get("current_tp_level", 0) # Record which TP level was active if relevant
        }

        # logging.info(f"Closing position {position.get('position_id')} for {symbol} via _close_position. Reason: {exit_reason}")
        return exit_result
    
    def _simulate_intra_candle_advanced_exits(self, symbol, original_candle_data):
        """
        Simulates exit logic within a single original interval candle using 1-minute data
        for advanced position management (multi-TP, trailing SL).
        MODIFIED to detect and flag 1m SL/TP ambiguity.

        Args:
            symbol (str): Trading symbol.
            original_candle_data (pd.Series): Data for the original interval candle (e.g., 15m)
                                            Requires 'open', 'high', 'low', 'close', and the timestamp index.

        Returns:
            Tuple[List[Dict], Dict]:
                - List of exit detail dictionaries triggered within this candle.
                - The final state of the position *after* simulating this candle.
                Returns the *original* position state if no 1m data is found or simulation fails.
        """
        if symbol not in self.positions or not self.positions[symbol].get("is_active", False):
            return [], self.positions.get(symbol) # No active position or not found

        if self.use_standard_sl_tp:
            # This simulation is only for advanced mode
            return [], self.positions[symbol]

        original_position_state = self.positions[symbol]
        # --- Deep Copy Needed ---
        simulated_position = copy.deepcopy(original_position_state)

        original_candle_time = original_candle_data.name # This is the *end* time of the original candle

        # Ensure original_candle_time is timezone-aware UTC
        if not isinstance(original_candle_time, pd.Timestamp):
            logger.error(f"Invalid original_candle_time type: {type(original_candle_time)}. Expected Timestamp.")
            return [], original_position_state
        if original_candle_time.tzinfo is None:
            original_candle_time = original_candle_time.tz_localize('UTC')
        else:
            original_candle_time = original_candle_time.tz_convert('UTC')


        logger.info(f"--- Intra-Candle Simulation Start ({symbol}) for candle ending {original_candle_time} ---")
        logger.debug(f"Initial Sim State: Qty={simulated_position['remaining_quantity']:.8f}, Stop={simulated_position['current_stop']:.8f}, TPs Reached={simulated_position.get('current_tp_level',0)}")

        # --- 1. Fetch 1-Minute Data for the Original Candle's Duration ---
        original_interval_str = getattr(Z_config, 'interval', '15m') # Get original interval
        original_interval_minutes = Backtest.parse_interval_to_minutes(original_interval_str)
        if original_interval_minutes is None:
            logger.warning(f"Could not parse original interval '{original_interval_str}' for simulation. Aborting simulation.")
            return [], original_position_state # Return original state on error

        candle_start_time = original_candle_time - timedelta(minutes=original_interval_minutes)
        candle_end_time = original_candle_time
        resolution_interval = "1m" # Hardcode to 1m for simulation

        logger.info(f"Fetching {resolution_interval} data for {symbol} from {candle_start_time} to {candle_end_time}")

        try:
            detailed_data = Backtest.fetch_data(
                symbol=symbol,
                interval=resolution_interval,
                end_time=candle_end_time,
                start_time_force=candle_start_time
            )

            if detailed_data is None or detailed_data.empty:
                logger.warning(f"No detailed ({resolution_interval}) data found for {symbol} during {candle_start_time} to {candle_end_time}. Skipping simulation for this candle.")
                return [], original_position_state

            detailed_data = detailed_data[(detailed_data.index >= candle_start_time) & (detailed_data.index < candle_end_time)]
            if detailed_data.empty:
                logger.warning(f"No detailed ({resolution_interval}) data remains for {symbol} after strict time filtering {candle_start_time} to {candle_end_time}. Skipping simulation.")
                return [], original_position_state

            logger.info(f"Found {len(detailed_data)} {resolution_interval} candles for simulation.")

        except Exception as e:
            logger.error(f"Error fetching {resolution_interval} data for simulation: {e}", exc_info=True)
            return [], original_position_state

        # --- 2. Simulate Step-by-Step through 1-Minute Candles ---
        exits_found_this_candle = []
        position_type = simulated_position["position_type"]
        entry_price = simulated_position["entry_price"]
        position_id = simulated_position.get("position_id", f"{symbol}_{simulated_position['entry_time']}")

        # --- Simulation Loop ---
        for candle_1m_time, candle_1m in detailed_data.iterrows():
            if not simulated_position["is_active"]:
                logger.debug(f"Sim: Position became inactive at {candle_1m_time}. Stopping simulation for this candle.")
                break

            if candle_1m_time.tzinfo is None: candle_1m_time_utc = candle_1m_time.tz_localize('UTC')
            else: candle_1m_time_utc = candle_1m_time.tz_convert('UTC')

            try:
                candle_low = float(candle_1m['low'])
                candle_high = float(candle_1m['high'])
                candle_open = float(candle_1m['open'])
                candle_close = float(candle_1m['close'])
            except (ValueError, TypeError) as e:
                logger.error(f"Sim: Invalid price data in 1m candle at {candle_1m_time_utc}: {e}. Skipping.")
                continue

            logger.debug(f"--- Simulating 1m candle: {candle_1m_time_utc} O:{candle_open:.5f} H:{candle_high:.5f} L:{candle_low:.5f} C:{candle_close:.5f} ---")
            logger.debug(f"Sim State Before 1m: Qty={simulated_position['remaining_quantity']:.8f}, Stop={simulated_position['current_stop']:.8f}, TPs={simulated_position.get('current_tp_level',0)}, BEActive={simulated_position.get('breakeven_activated', False)}, TPTrailActive={simulated_position.get('tp_trailing_activated', False)}")

            # --- A. Update Trailing State ---
            price_for_best_update = candle_high if position_type == "long" else candle_low
            stop_moved_in_this_step = False
            best_price_updated = False
            if position_type == "long":
                if price_for_best_update > simulated_position["best_price"]:
                    simulated_position["best_price"] = price_for_best_update
                    best_price_updated = True
            else:
                if price_for_best_update < simulated_position["best_price"]:
                    simulated_position["best_price"] = price_for_best_update
                    best_price_updated = True

            if not simulated_position["trailing_activated"]:
                price_movement_pct = 0
                if entry_price > 0:
                    if position_type == "long": price_movement_pct = ((simulated_position["best_price"] - entry_price) / entry_price) * 100
                    else: price_movement_pct = ((entry_price - simulated_position["best_price"]) / entry_price) * 100
                if price_movement_pct >= self.trailing_activation_threshold:
                    simulated_position["trailing_activated"] = True
                    logger.debug(f"Sim: Trailing activated at 1m candle {candle_1m_time_utc}")

            new_stop = simulated_position["current_stop"]
            stop_updated_reason = ""
            if simulated_position.get("breakeven_activated", False):
                be_level = simulated_position.get("breakeven_level")
                if be_level is not None:
                    if position_type == "long" and be_level > new_stop:
                        new_stop = be_level; stop_updated_reason = "breakeven"
                    elif position_type == "short" and be_level < new_stop:
                        new_stop = be_level; stop_updated_reason = "breakeven"
            if simulated_position["trailing_activated"]:
                proposed_trailing_stop = 0.0
                if position_type == "long":
                    proposed_trailing_stop = simulated_position["best_price"] * (1 - self.trailing_distance / 100)
                    if proposed_trailing_stop > new_stop:
                        new_stop = proposed_trailing_stop; stop_updated_reason = "trailing"
                else:
                    proposed_trailing_stop = simulated_position["best_price"] * (1 + self.trailing_distance / 100)
                    if proposed_trailing_stop < new_stop:
                        new_stop = proposed_trailing_stop; stop_updated_reason = "trailing"
            if new_stop != simulated_position["current_stop"]:
                logger.debug(f"Sim: Stop moved from {simulated_position['current_stop']:.8f} to {new_stop:.8f} due to {stop_updated_reason} at {candle_1m_time_utc}")
                simulated_position["current_stop"] = new_stop
                stop_moved_in_this_step = True
            # --- END OF SECTION A ---

            # ***** AMBIGUITY DETECTION START *****
            sl_condition_met = False
            current_sim_stop = simulated_position["current_stop"]
            if position_type == "long":
                if candle_low <= current_sim_stop: sl_condition_met = True
            else:
                if candle_high >= current_sim_stop: sl_condition_met = True

            tp_condition_met = False
            for level_index in range(len(simulated_position["take_profit_levels"])):
                level_reached_flag = f"reached_{['first', 'second', 'third'][level_index]}"
                if not simulated_position.get(level_reached_flag, False) and simulated_position["take_profit_quantities"][level_index] > 1e-9:
                    tp_price = simulated_position["take_profit_levels"][level_index]
                    if position_type == "long":
                        if candle_high >= tp_price: tp_condition_met = True; break
                    else:
                        if candle_low <= tp_price: tp_condition_met = True; break
            if not tp_condition_met and simulated_position.get("tp_trailing_activated", False) and simulated_position["remaining_quantity"] > 1e-9:
                trailing_tp_stop_level = 0.0
                if position_type == "long":
                    trailing_tp_stop_level = simulated_position["best_tp_price"] * (1 - self.third_level_trailing_distance / 100)
                    if candle_low <= trailing_tp_stop_level: tp_condition_met = True
                else:
                    trailing_tp_stop_level = simulated_position["best_tp_price"] * (1 + self.third_level_trailing_distance / 100)
                    if candle_high >= trailing_tp_stop_level: tp_condition_met = True

            ambiguity_detected = sl_condition_met and tp_condition_met
            if ambiguity_detected:
                logging.warning(f"Sim: Ambiguity detected at {candle_1m_time_utc}! SL ({current_sim_stop:.8f}) AND TP conditions met within 1m candle (L:{candle_low:.8f} H:{candle_high:.8f}). Prioritizing SL.")
            # ***** AMBIGUITY DETECTION END *****

            # --- B. Check Stop Loss Hit ---
            sl_hit_price = None
            if position_type == "long":
                if candle_low <= current_sim_stop: sl_hit_price = current_sim_stop
            else:
                if candle_high >= current_sim_stop: sl_hit_price = current_sim_stop

            if sl_hit_price is not None:
                exit_time = candle_1m_time_utc
                # ***** EXIT REASON MODIFICATION START *****
                exit_reason_base = "initial_stop"
                if simulated_position.get("breakeven_activated", False): exit_reason_base = "breakeven_stop"
                elif simulated_position.get("trailing_activated", False): exit_reason_base = "trailing_stop"
                final_exit_reason = f"{exit_reason_base}_unsure" if ambiguity_detected else exit_reason_base
                # ***** EXIT REASON MODIFICATION END *****
                exit_quantity = simulated_position["remaining_quantity"]
                logger.info(f"Sim: STOP HIT ({final_exit_reason}) within candle at {exit_time}. SL Level: {current_sim_stop:.8f}, Trigger Price Approx: {sl_hit_price:.8f} (Low: {candle_low:.8f}, High: {candle_high:.8f})")
                exit_details = {
                    "exit_time": exit_time, "exit_price": sl_hit_price, "exit_quantity": exit_quantity,
                    "exit_level": simulated_position.get("current_tp_level", 0),
                    "exit_reason": final_exit_reason, # Use final reason
                    "position_id": position_id, "full_exit": True,
                    "breakeven_activated": simulated_position.get("breakeven_activated", False),
                    "remaining_quantity": 0
                }
                exits_found_this_candle.append(exit_details)
                simulated_position["is_active"] = False
                simulated_position["remaining_quantity"] = 0
                simulated_position["exit_price"] = sl_hit_price
                simulated_position["exit_time"] = exit_time
                simulated_position["exit_reason"] = final_exit_reason # Update state
                logger.debug(f"Sim: Position closed by SL. Exiting simulation for original candle.")
                break

            # --- C. Check Take Profit Levels (Only if SL not hit) ---
            tp_levels = simulated_position["take_profit_levels"]
            tp_hit_in_this_step = False
            for level_index in range(len(tp_levels)):
                if not simulated_position["is_active"]: break
                level_reached_flag = f"reached_{['first', 'second', 'third'][level_index]}"
                tp_price = tp_levels[level_index]
                exit_qty_config = simulated_position["take_profit_quantities"][level_index]
                if simulated_position.get(level_reached_flag, False) or exit_qty_config <= 1e-9: continue
                tp_hit = False; tp_hit_price = None
                if position_type == "long":
                    if candle_high >= tp_price: tp_hit = True; tp_hit_price = tp_price
                else:
                    if candle_low <= tp_price: tp_hit = True; tp_hit_price = tp_price
                if tp_hit and tp_hit_price is not None:
                    actual_exit_qty = min(exit_qty_config, simulated_position["remaining_quantity"])
                    if actual_exit_qty <= 1e-9: continue
                    exit_time = candle_1m_time_utc
                    exit_reason = f"take_profit_{level_index + 1}"
                    is_final_tp_level_config = (level_index == len(tp_levels) - 1)
                    tp_hit_in_this_step = True
                    logger.info(f"Sim: TP{level_index + 1} HIT at {exit_time}. TP Level: {tp_price:.8f}, Trigger Price Approx: {tp_hit_price:.8f} (Low: {candle_low:.8f}, High: {candle_high:.8f})")
                    logger.info(f"Sim: Exiting Qty: {actual_exit_qty:.8f} / Remaining Before: {simulated_position['remaining_quantity']:.8f}")
                    simulated_position[level_reached_flag] = True
                    simulated_position["remaining_quantity"] -= actual_exit_qty
                    simulated_position["current_tp_level"] = level_index + 1
                    is_full_exit = (simulated_position["remaining_quantity"] <= 1e-9)
                    logger.info(f"Sim: Remaining Qty After TP{level_index + 1}: {simulated_position['remaining_quantity']:.8f}")
                    exit_details = {
                        "exit_time": exit_time, "exit_price": tp_hit_price, "exit_quantity": actual_exit_qty,
                        "exit_level": level_index + 1, "exit_reason": exit_reason, "position_id": position_id,
                        "full_exit": is_full_exit, "breakeven_activated": simulated_position.get("breakeven_activated", False),
                        "remaining_quantity": simulated_position["remaining_quantity"]
                    }
                    exits_found_this_candle.append(exit_details)
                    simulated_position['partial_exits'].append(copy.deepcopy(exit_details))
                    if level_index == 1 and self.enable_breakeven and not simulated_position.get("breakeven_activated", False):
                        breakeven_level_calc = entry_price
                        buffer = getattr(Z_config, 'breakeven_buffer_pct', 0.05) / 100
                        if position_type == "long":
                            breakeven_level_calc = entry_price * (1 + buffer)
                            if breakeven_level_calc > simulated_position["current_stop"]:
                                logger.debug(f"Sim: Activating Breakeven. Moving Stop from {simulated_position['current_stop']:.8f} to {breakeven_level_calc:.8f}")
                                simulated_position["current_stop"] = breakeven_level_calc; simulated_position["breakeven_activated"] = True; simulated_position["breakeven_level"] = breakeven_level_calc; stop_moved_in_this_step = True
                        else:
                            breakeven_level_calc = entry_price * (1 - buffer)
                            if breakeven_level_calc < simulated_position["current_stop"]:
                                logger.debug(f"Sim: Activating Breakeven. Moving Stop from {simulated_position['current_stop']:.8f} to {breakeven_level_calc:.8f}")
                                simulated_position["current_stop"] = breakeven_level_calc; simulated_position["breakeven_activated"] = True; simulated_position["breakeven_level"] = breakeven_level_calc; stop_moved_in_this_step = True
                    if level_index == 1 and self.enable_trailing_take_profit and not is_final_tp_level_config:
                        if not simulated_position.get("tp_trailing_activated", False):
                            simulated_position["tp_trailing_activated"] = True
                            simulated_position["best_tp_price"] = candle_high if position_type == "long" else candle_low
                            logger.debug(f"Sim: Trailing TP activated at {candle_1m_time_utc}, starting best TP price {simulated_position['best_tp_price']:.8f}")
                    if is_full_exit:
                        simulated_position["is_active"] = False; simulated_position["exit_price"] = tp_hit_price
                        simulated_position["exit_time"] = exit_time; simulated_position["exit_reason"] = exit_reason
                        logger.debug(f"Sim: Position closed by final TP{level_index + 1}.")

            # --- D. Check Trailing Take Profit ---
            if simulated_position.get("tp_trailing_activated", False) and simulated_position["is_active"]:
                tp_best_price_updated = False
                if position_type == "long":
                    if candle_high > simulated_position["best_tp_price"]: simulated_position["best_tp_price"] = candle_high; tp_best_price_updated = True
                else:
                    if candle_low < simulated_position["best_tp_price"]: simulated_position["best_tp_price"] = candle_low; tp_best_price_updated = True
                trailing_tp_stop_price = None; trailing_tp_stop_hit = False; trailing_tp_stop_level = 0.0
                if position_type == "long":
                    trailing_tp_stop_level = simulated_position["best_tp_price"] * (1 - self.third_level_trailing_distance / 100)
                    if candle_low <= trailing_tp_stop_level: trailing_tp_stop_hit = True; trailing_tp_stop_price = trailing_tp_stop_level
                else:
                    trailing_tp_stop_level = simulated_position["best_tp_price"] * (1 + self.third_level_trailing_distance / 100)
                    if candle_high >= trailing_tp_stop_level: trailing_tp_stop_hit = True; trailing_tp_stop_price = trailing_tp_stop_level
                if trailing_tp_stop_hit and trailing_tp_stop_price is not None:
                    exit_time = candle_1m_time_utc; exit_reason = "take_profit_3_trailing"; exit_quantity = simulated_position["remaining_quantity"]
                    logger.info(f"Sim: Trailing TP Stop HIT at {exit_time}. Stop Level: {trailing_tp_stop_level:.8f}, Trigger Price Approx: {trailing_tp_stop_price:.8f} (Low: {candle_low:.8f}, High: {candle_high:.8f})")
                    exit_details = {
                        "exit_time": exit_time, "exit_price": trailing_tp_stop_price, "exit_quantity": exit_quantity,
                        "exit_level": 3, "exit_reason": exit_reason, "position_id": position_id, "full_exit": True,
                        "breakeven_activated": simulated_position.get("breakeven_activated", False), "remaining_quantity": 0
                    }
                    exits_found_this_candle.append(exit_details)
                    simulated_position["is_active"] = False; simulated_position["remaining_quantity"] = 0
                    simulated_position["exit_price"] = trailing_tp_stop_price; simulated_position["exit_time"] = exit_time
                    simulated_position["exit_reason"] = exit_reason
                    logger.debug(f"Sim: Position closed by Trailing TP Stop. Exiting simulation for original candle.")
                    break
            # --- End of 1m candle simulation ---

        # --- 3. Return Results ---
        if not exits_found_this_candle:
            logger.info(f"--- Intra-Candle Simulation End ({symbol}) for candle ending {original_candle_time}: No SL/TP hits found in 1m data. ---")
        else:
            logger.info(f"--- Intra-Candle Simulation End ({symbol}) for candle ending {original_candle_time}: Found {len(exits_found_this_candle)} exit events. ---")
            logger.debug(f"Final Sim State: Active={simulated_position['is_active']}, Qty={simulated_position['remaining_quantity']:.8f}, Stop={simulated_position['current_stop']:.8f}, Reason={simulated_position.get('exit_reason', 'N/A')}")

        return exits_found_this_candle, simulated_position


    def get_position(self, symbol):
        """Get current position data for a symbol."""
        return self.positions.get(symbol)

    def get_all_positions(self):
        """Get all currently active positions."""
        return {k: v for k, v in self.positions.items() if v.get("is_active", False)}

    # get_position_summary remains the same


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


