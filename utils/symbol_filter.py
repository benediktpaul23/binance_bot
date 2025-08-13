#symbol_filter.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import logging
import Z_config # Angenommen, Z_config ist korrekt konfiguriert
import pytz # Import für Zeitzonen
# from datetime import time # time wird in is_within_trading_time benötigt und ist schon oben importiert

# Globale Dictionaries entfernt:
# symbols_under_observation = {}
# active_observation_periods = {}
# observation_period_tracking = {} # War ohnehin ungenutzt

def update_symbol_position_status(
    symbol: str,
    has_position: bool,
    active_observation_periods_dict: dict, # NEU
    observed_symbols_dict: dict,           # NEU
    position_type: str = None
):
    """
    Aktualisiert den Positionsstatus eines Symbols in den übergebenen Tracking-Systemen.
    """
    # Aktualisiere in observed_symbols_dict
    if symbol in observed_symbols_dict:
        observed_symbols_dict[symbol]['has_position'] = has_position
        if position_type:
            observed_symbols_dict[symbol]['position_type'] = position_type
    else:
        # Optional: Symbol hinzufügen, falls es nicht existiert, aber einen Positionsstatus erhält
        # Dies hängt von der Gesamtlogik ab, ob ein Symbol erst beobachtet werden muss,
        # bevor es eine Position haben kann.
        # observed_symbols_dict[symbol] = {
        #     'has_position': has_position,
        #     'position_type': position_type,
        #     # Ggf. Standardwerte für 'start_time', 'end_time' etc. setzen oder Fehler loggen
        # }
        logging.warning(f"Symbol {symbol} nicht in observed_symbols_dict gefunden bei update_symbol_position_status.")


    # Aktualisiere in active_observation_periods_dict
    if symbol in active_observation_periods_dict:
        active_observation_periods_dict[symbol]['has_position'] = has_position
        if not has_position:
            current_time = datetime.now(timezone.utc)
            if active_observation_periods_dict[symbol].get('active_period'):
                period_end = active_observation_periods_dict[symbol]['active_period']['end_time']
                if current_time > period_end:
                    active_observation_periods_dict[symbol]['active_period'] = None
                    logging.info(f"Beobachtungszeitraum für {symbol} in active_observation_periods_dict beendet (keine Position mehr).")
    else:
        # active_observation_periods_dict[symbol] = {'has_position': has_position, 'active_period': None}
        logging.warning(f"Symbol {symbol} nicht in active_observation_periods_dict gefunden bei update_symbol_position_status.")

    logging.info(f"Positionsstatus für {symbol} aktualisiert: {'Aktiv' if has_position else 'Keine Position'}")

# check_price_change_threshold bleibt unverändert, da es keinen globalen State verwendet.
def check_price_change_threshold(
    df: pd.DataFrame,
    min_pct: float,
    max_pct: float,
    lookback_minutes: int,
    direction: str = 'both'
) -> tuple:
    if df is None or df.empty:
        logging.warning("Keine Daten für Preisänderungsprüfung verfügbar")
        return False, 0, 0, 0
    complete_candles = df[df['is_complete'] == True].copy()
    if len(complete_candles) < 2:
        logging.warning("Nicht genügend vollständige Candles für Preisänderungsprüfung")
        return False, 0, 0, 0
    if len(complete_candles) >= 2:
        interval_minutes = (complete_candles.index[1] - complete_candles.index[0]).total_seconds() / 60
        if interval_minutes <= 0:
            interval_minutes = 5
    else:
        interval_minutes = 5
    lookback_intervals = int(lookback_minutes / interval_minutes)
    if len(complete_candles) <= lookback_intervals:
        logging.warning(f"Nicht genügend Candles für {lookback_minutes}min Lookback (benötigt: {lookback_intervals+1}, vorhanden: {len(complete_candles)})")
        return False, 0, 0, 0
    current_index = len(complete_candles) - 1
    end_price = complete_candles.iloc[current_index]['close']
    start_index = max(0, current_index - lookback_intervals)
    start_price = complete_candles.iloc[start_index]['close']
    price_change_pct = ((end_price - start_price) / start_price) * 100
    threshold_reached = False
    if direction == 'both':
        abs_price_change_pct = abs(price_change_pct)
        threshold_reached = (abs_price_change_pct >= min_pct) and (abs_price_change_pct <= max_pct)
    elif direction == 'plus':
        threshold_reached = (price_change_pct >= min_pct) and (price_change_pct <= max_pct)
    elif direction == 'minus':
        threshold_reached = (price_change_pct <= -min_pct) and (price_change_pct >= -max_pct)
    return threshold_reached, price_change_pct, start_price, end_price

# is_within_trading_time bleibt unverändert, da es keinen globalen State verwendet.
def is_within_trading_time(current_time: datetime) -> bool:
    time_filter_active = getattr(Z_config, 'time_filter_active', True)
    if not time_filter_active:
        return True
    trading_days = getattr(Z_config, 'trading_days', [0, 1, 2, 3, 4])
    start_hour = getattr(Z_config, 'trading_start_hour', 0)
    start_minute = getattr(Z_config, 'trading_start_minute', 0)
    end_hour = getattr(Z_config, 'trading_end_hour', 23)
    end_minute = getattr(Z_config, 'trading_end_minute', 59)
    timezone_str = getattr(Z_config, 'trading_timezone', 'UTC')
    try:
        tz = pytz.timezone(timezone_str) # 'tz' statt 'timezone' um Namenskonflikt zu vermeiden
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=pytz.UTC)
        localized_time = current_time.astimezone(tz)
    except Exception as e:
        logging.warning(f"Fehler bei der Zeitzonenkonvertierung: {e}. Verwende Originalzeit.")
        localized_time = current_time
    weekday = localized_time.weekday()
    if weekday not in trading_days:
        logging.info(f"Tag {weekday} (wobei 0=Montag) ist kein Handelstag. Konfigurierte Handelstage: {trading_days}")
        return False
    trading_start = pd.Timestamp(f"{start_hour:02d}:{start_minute:02d}").time() # pd.Timestamp für time
    trading_end = pd.Timestamp(f"{end_hour:02d}:{end_minute:02d}").time()
    current_time_only = localized_time.time()
    within_time_window = trading_start <= current_time_only <= trading_end
    if not within_time_window:
        logging.info(f"Zeit {current_time_only} liegt außerhalb des Handelszeitfensters ({trading_start}-{trading_end})")
    return within_time_window


def should_observe_symbol(
    symbol: str,
    df: pd.DataFrame,
    current_time: datetime,
    observed_symbols_dict: dict, # NEU
    min_pct: float,
    max_pct: float,
    lookback_minutes: int,
    observation_hours: int,
    direction: str = 'both',
    check_trading_time: bool = True
) -> bool:
    """
    Bestimmt, ob ein Symbol beobachtet werden sollte. Verwendet übergebenes observed_symbols_dict.
    """
    print(f"\n--- SYMBOL OBSERVATION CHECK FOR {symbol} AT {current_time} ---")
    filtering_active = getattr(Z_config, 'filtering_active', True)
    if not filtering_active:
        print(f"Filterung ist deaktiviert. Symbol {symbol} wird immer beobachtet.")
        return True # Früher Ausstieg, keine Modifikation von observed_symbols_dict hier

    beobachten_active = getattr(Z_config, 'beobachten_active', True)
    if check_trading_time and not is_within_trading_time(current_time):
        print(f"Zeitfilter: Symbol {symbol} wird übersprungen, da aktuelle Zeit außerhalb des Handelszeitfensters")
        return False

    if symbol in observed_symbols_dict:
        entry = observed_symbols_dict[symbol]
        # Prüfe, ob 'end_time' im Eintrag vorhanden ist
        if 'end_time' not in entry:
             logging.error(f"Symbol {symbol} in observed_symbols_dict, aber 'end_time' fehlt. Eintrag: {entry}")
             # Handhabung für fehlendes 'end_time', z.B. Symbol entfernen oder Standard annehmen
             del observed_symbols_dict[symbol] # Vorsichtige Annahme: ungültiger Eintrag
        elif current_time <= entry['end_time']:
            remaining = entry['end_time'] - current_time
            print(f"Symbol {symbol} wird bereits beobachtet (bis {entry['end_time'].strftime('%Y-%m-%d %H:%M')})")
            print(f"Verbleibende Beobachtungszeit: {remaining.total_seconds()/3600:.1f} Stunden")
            return True # Wird bereits beobachtet
        else:
            print(f"Beobachtungszeit für {symbol} ist abgelaufen.")
            # Nur entfernen, wenn keine Position mehr besteht
            if not entry.get('has_position', False):
                 del observed_symbols_dict[symbol]
            else:
                 print(f"Beobachtungszeit für {symbol} abgelaufen, aber Position ist noch aktiv. Bleibt in observed_symbols_dict.")


    observation_end_time = current_time + timedelta(hours=observation_hours)
    new_entry_data = {
        'start_time': current_time,
        'end_time': observation_end_time,
        'has_position': False, # Standardmäßig keine Position
        'position_id': None, # Standard
        'price_change_pct': 0.0 # Standard
    }

    if not beobachten_active:
        observed_symbols_dict[symbol] = new_entry_data
        logging.info(f"✓ {symbol} wird beobachtet ohne Preisänderungsprüfung (Preisänderungsprüfung deaktiviert)")
        logging.info(f"  Beobachtung bis: {observation_end_time.strftime('%Y-%m-%d %H:%M')}")
        return True

    threshold_reached, price_change_pct_val, start_price, end_price = check_price_change_threshold(
        df, min_pct, max_pct, lookback_minutes, direction
    )

    if threshold_reached:
        new_entry_data['price_change_pct'] = price_change_pct_val
        observed_symbols_dict[symbol] = new_entry_data
        direction_text = "positiv" if price_change_pct_val > 0 else "negativ"
        logging.info(f"✓ {symbol} erfüllt Preisänderungskriterien: {price_change_pct_val:.2f}% ({direction_text}) in den letzten {lookback_minutes} Minuten")
        logging.info(f"  Start-Preis: {start_price}, End-Preis: {end_price}")
        logging.info(f"  Beobachtung bis: {observation_end_time.strftime('%Y-%m-%d %H:%M')}")
        return True

    return False


def check_position_expiry(
    symbol: str,
    tracker, # BacktestPositionTracker-Instanz
    current_time: datetime,
    close_price: float,
    observed_symbols_dict: dict,           # NEU
    active_observation_periods_dict: dict, # NEU
    close_positions: bool = False
) -> dict:
    """
    Überprüft Positionsablauf. Verwendet übergebene Dictionaries.
    """
    active_period = None
    if symbol in active_observation_periods_dict:
        active_period = active_observation_periods_dict[symbol].get('active_period')

    if symbol in observed_symbols_dict and observed_symbols_dict[symbol].get('has_position', False):
        symbol_data = observed_symbols_dict[symbol]
        position = tracker.get_position(symbol) # get_position ist Teil des BacktestPositionTracker

        # Log-Details
        print(f"\n--- POSITION EXPIRY CHECK FOR {symbol} ---")
        print(f"Observed since: {symbol_data.get('start_time', 'N/A')}") # .get für Sicherheit
        print(f"Observation ends: {symbol_data.get('end_time', 'N/A')}")
        print(f"Current time: {current_time}")
        if 'end_time' in symbol_data:
            remaining_time = symbol_data['end_time'] - current_time
            hours_remaining = remaining_time.total_seconds() / 3600
            print(f"Remaining observation time: {hours_remaining:.2f} hours")
        else:
            print("Remaining observation time: N/A (end_time fehlt)")
        print(f"close_positions setting: {close_positions}")

        if position and position.get("is_active"):
            if 'end_time' in symbol_data and current_time > symbol_data['end_time']:
                print(f"⏱️ Beobachtungszeit für {symbol} abgelaufen")
                if close_positions:
                    exit_result = tracker._close_position( # _close_position ist Teil des Trackers
                        symbol=symbol,
                        exit_price=close_price,
                        exit_time=current_time,
                        exit_reason="observation_timeout"
                    )
                    # Wichtig: Dictionaries direkt aktualisieren
                    observed_symbols_dict[symbol]['has_position'] = False
                    if symbol in active_observation_periods_dict:
                        active_observation_periods_dict[symbol]['has_position'] = False
                        # Ggf. active_period zurücksetzen
                        if active_observation_periods_dict[symbol].get('active_period') and \
                           current_time > active_observation_periods_dict[symbol]['active_period']['end_time']:
                            active_observation_periods_dict[symbol]['active_period'] = None
                    print(f"Position für {symbol} geschlossen zum Preis {close_price}")
                    return exit_result
                else:
                    print(f"Position für {symbol} bleibt offen trotz abgelaufener Beobachtungszeit (close_positions=False)")
            else:
                print(f"Position für {symbol} ist noch innerhalb der Beobachtungszeit oder end_time fehlt.")
        else:
            print(f"Keine aktive Position für {symbol} im Tracker gefunden.")
            # Korrigiere Status, falls Inkonsistenz
            if observed_symbols_dict[symbol].get('has_position', False):
                 observed_symbols_dict[symbol]['has_position'] = False
                 logging.warning(f"Inkonsistenz: observed_symbols_dict sagte Position für {symbol}, aber Tracker nicht. Korrigiert.")
            if symbol in active_observation_periods_dict and active_observation_periods_dict[symbol].get('has_position', False):
                 active_observation_periods_dict[symbol]['has_position'] = False
                 logging.warning(f"Inkonsistenz: active_observation_periods_dict sagte Position für {symbol}, aber Tracker nicht. Korrigiert.")

    return None


def get_observation_period(
    symbol: str,
    current_time: datetime,
    observed_symbols_dict: dict # NEU
) -> dict:
    """
    Gibt den Beobachtungszeitraum für ein Symbol zurück. Verwendet übergebenes observed_symbols_dict.
    """
    if symbol in observed_symbols_dict:
        # 'start_time' und 'end_time' sollten vorhanden sein, wenn das Symbol beobachtet wird
        # Füge eine Sicherheitsprüfung hinzu, falls die Struktur unerwartet ist
        start_t = observed_symbols_dict[symbol].get('start_time')
        end_t = observed_symbols_dict[symbol].get('end_time')
        if start_t and end_t:
            return {'start_time': start_t, 'end_time': end_t}
        else:
            logging.error(f"Symbol {symbol} in observed_symbols_dict, aber Start/Endzeit fehlen. {observed_symbols_dict[symbol]}")
            # Fallback oder Fehlerbehandlung
            # Hier könnte man es entfernen oder mit Defaults neu erstellen, je nach Anforderung
            del observed_symbols_dict[symbol] # Beispiel: Entferne ungültigen Eintrag

    # Symbol ist neu oder wurde entfernt - berechne Beobachtungszeitraum
    observation_hours = getattr(Z_config, 'symbol_observation_hours', 4) # Default aus Z_config
    observation_end_time = current_time + timedelta(hours=observation_hours)

    # In Beobachtungsliste eintragen (observed_symbols_dict wird hier modifiziert)
    observed_symbols_dict[symbol] = {
        'start_time': current_time,
        'end_time': observation_end_time,
        'has_position': False,
        'position_id': None,
        'price_change_pct': 0.0
    }
    return {'start_time': current_time, 'end_time': observation_end_time}

# verify_observation_indicators bleibt unverändert, da es keinen globalen State verwendet.
def verify_observation_indicators(df, observation_start, symbol):
    observation_mask = (df.index >= observation_start)
    first_candles = df[observation_mask].head(3)
    if first_candles.empty:
        print(f"⚠️ Keine Candles im Beobachtungszeitraum für {symbol} gefunden!")
        return False
    all_indicators_valid = True
    # Kommentarblock für Indikatorprüfung bleibt bestehen
    return all_indicators_valid


def clean_expired_observations(
    current_time: datetime,
    observed_symbols_dict: dict # NEU
):
    """
    Entfernt abgelaufene Symbole aus der Beobachtungsliste. Verwendet übergebenes observed_symbols_dict.
    """
    to_remove = []
    for symbol, data in observed_symbols_dict.items():
        # Prüfe ob 'end_time' existiert, bevor darauf zugegriffen wird
        if 'end_time' in data:
            if current_time > data['end_time'] and not data.get('has_position', False):
                to_remove.append(symbol)
        else:
            logging.warning(f"Symbol {symbol} in observed_symbols_dict ohne 'end_time'. Kann nicht auf Ablauf geprüft werden.")

    for symbol in to_remove:
        del observed_symbols_dict[symbol]
        logging.info(f"Symbol {symbol} aus Beobachtungsliste entfernt (Beobachtungszeit abgelaufen, keine Position)")

def print_observation_status(observed_symbols_dict: dict): # NEU
    """Gibt den aktuellen Status der Beobachtungsliste aus. Verwendet übergebenes observed_symbols_dict."""
    if not observed_symbols_dict:
        print("Keine Symbole in der Beobachtungsliste.")
        return

    current_time = datetime.now(timezone.utc)
    print(f"\nAktuelle Beobachtungsliste ({len(observed_symbols_dict)} Symbole):")
    print("-" * 80)
    print(f"{'Symbol':<10} {'Start':<20} {'Ende':<20} {'Position':<10} {'Restzeit':<15} {'Preisänderung (%)':<20}")
    print("-" * 80)

    for symbol, data in sorted(observed_symbols_dict.items()):
        start_str = data.get('start_time', pd.NaT).strftime('%Y-%m-%d %H:%M') if pd.notna(data.get('start_time')) else "N/A"
        end_str = data.get('end_time', pd.NaT).strftime('%Y-%m-%d %H:%M') if pd.notna(data.get('end_time')) else "N/A"
        remaining_str = "N/A"
        if pd.notna(data.get('end_time')):
            remaining = data['end_time'] - current_time
            remaining_str = f"{remaining.total_seconds() / 3600:.1f}h" if remaining.total_seconds() > 0 else "abgelaufen"
        
        price_change_str = f"{data.get('price_change_pct', 0.0):.2f}"

        print(f"{symbol:<10} {start_str:<20} "
              f"{end_str:<20} "
              f"{'Ja' if data.get('has_position', False) else 'Nein':<10} {remaining_str:<15} "
              f"{price_change_str:<20}")
    print("-" * 80)


def manage_observation_periods(
    symbol: str,
    current_time: datetime,
    active_observation_periods_dict: dict, # NEU
    observed_symbols_dict: dict,           # NEU
    has_position: bool = None
) -> bool:
    """
    Verwaltet Beobachtungszeiträume. Verwendet übergebene Dictionaries.
    """
    # Initialisiere für dieses Symbol, falls noch nicht vorhanden
    if symbol not in active_observation_periods_dict:
        active_observation_periods_dict[symbol] = {
            'active_period': None,
            'has_position': False # Standardwert
        }
    # Wenn has_position übergeben wird, aktualisiere es direkt.
    # Diese Logik wird oft von update_symbol_position_status übernommen.
    # Hier kann es redundant sein oder für direkte Manipulation dienen.
    if has_position is not None:
        active_observation_periods_dict[symbol]['has_position'] = has_position
        # Wenn keine Position mehr aktiv ist und der Zeitraum abgelaufen ist, kann er entfernt werden
        if not has_position and active_observation_periods_dict[symbol].get('active_period'):
            period_end = active_observation_periods_dict[symbol]['active_period']['end_time']
            if current_time > period_end:
                active_observation_periods_dict[symbol]['active_period'] = None
                logging.info(f"Beobachtungszeitraum für {symbol} in active_observation_periods_dict entfernt (keine Position mehr, Zeitraum abgelaufen).")


    # Prüfe, ob ein aktiver Beobachtungszeitraum existiert und abgelaufen ist
    current_active_period = active_observation_periods_dict[symbol].get('active_period')
    if current_active_period:
        period_end_time = current_active_period['end_time']
        # Wenn Zeitraum abgelaufen UND keine Position mehr aktiv ist, setze Periode zurück
        if current_time > period_end_time and not active_observation_periods_dict[symbol].get('has_position', False):
            active_observation_periods_dict[symbol]['active_period'] = None
            logging.info(f"Beobachtungszeitraum für {symbol} abgelaufen und keine Position mehr. Zurückgesetzt.")

    # Wenn kein aktiver Zeitraum oder der alte abgelaufen ist, versuche einen neuen zu aktivieren
    if not active_observation_periods_dict[symbol].get('active_period'):
        if symbol in observed_symbols_dict:
            symbol_observation_data = observed_symbols_dict[symbol]
            # Stelle sicher, dass es eine einzelne Periode ist, nicht eine Liste von Perioden.
            # Die Struktur von observed_symbols_dict ist {symbol: {period_data}}
            # und nicht {symbol: [list_of_periods]} wie in einer früheren Version des Codes.
            if isinstance(symbol_observation_data, dict) and \
               'start_time' in symbol_observation_data and \
               'end_time' in symbol_observation_data:
                
                obs_start_time = symbol_observation_data['start_time']
                obs_end_time = symbol_observation_data['end_time']

                # Prüfe, ob diese Periode für die Aktivierung relevant ist
                if obs_start_time <= current_time <= obs_end_time:
                    active_observation_periods_dict[symbol]['active_period'] = {
                        'start_time': obs_start_time,
                        'end_time': obs_end_time,
                        'price_change_pct': symbol_observation_data.get('price_change_pct', 0.0)
                        # Weitere Felder aus symbol_observation_data könnten hierher kopiert werden, falls nötig
                    }
                    logging.info(f"Neuer Beobachtungszeitraum für {symbol} aktiviert: {obs_start_time} bis {obs_end_time}")
                # else:
                #    logging.debug(f"Verfügbare Periode für {symbol} ({obs_start_time}-{obs_end_time}) nicht im aktuellen Zeitfenster {current_time}.")
            # else:
            #    logging.debug(f"Keine gültige einzelne Beobachtungsperiode in observed_symbols_dict für {symbol} gefunden oder Struktur unerwartet.")
        # else:
        #    logging.debug(f"Symbol {symbol} nicht in observed_symbols_dict, keine Periode zu aktivieren.")


    # Finale Prüfung, ob eine Position geöffnet werden kann
    final_active_period = active_observation_periods_dict[symbol].get('active_period')
    if final_active_period:
        within_period = final_active_period['start_time'] <= current_time <= final_active_period['end_time']
        
        # Die Logik für 'close_position' ist komplex und hängt davon ab, ob Seitenwechsel erlaubt sind.
        # Hier wird die Logik aus deinem Snippet beibehalten.
        close_position_config = getattr(Z_config, 'close_position', False) # Default False, wenn nicht in Config
        
        # Position kann geöffnet werden, wenn:
        # 1. Wir uns im aktiven Zeitraum befinden UND
        # 2. Entweder keine Position besteht ODER Z_config.close_position False ist (erlaubt Seitenwechsel)
        can_open_position = within_period and \
                            (not active_observation_periods_dict[symbol].get('has_position', False) or not close_position_config)
        
        # Logging für Klarheit
        # if within_period:
        #    logging.debug(f"Für {symbol}: Innerhalb Periode. Has_pos: {active_observation_periods_dict[symbol].get('has_position', False)}, close_pos_cfg: {close_position_config}, CanOpen: {can_open_position}")
        return can_open_position

    return False