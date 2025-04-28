#symbol_filter.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import logging
import Z_config
import pytz
from datetime import datetime, time

# Globales Dictionary zur Verfolgung der beobachteten Symbole
symbols_under_observation = {}

def update_symbol_position_status(symbol, has_position, position_type=None):
    """
    Aktualisiert den Positionsstatus eines Symbols im globalen Tracking-System.
    
    Args:
        symbol: Trading-Symbol
        has_position: True wenn eine Position aktiv ist, False sonst
        position_type: Optional, Typ der Position ('long' oder 'short')
    """
    global symbols_under_observation, active_observation_periods
    
    # Aktualisiere in symbols_under_observation
    if symbol in symbols_under_observation:
        symbols_under_observation[symbol]['has_position'] = has_position
        if position_type:
            symbols_under_observation[symbol]['position_type'] = position_type
    
    # Aktualisiere in active_observation_periods
    if symbol in active_observation_periods:
        active_observation_periods[symbol]['has_position'] = has_position
        # Wenn keine Position mehr aktiv ist und der Beobachtungszeitraum abgelaufen ist,
        # kann ein neuer Zeitraum aktiviert werden
        if not has_position:
            current_time = datetime.now(timezone.utc)
            if active_observation_periods[symbol]['active_period']:
                period_end = active_observation_periods[symbol]['active_period']['end_time']
                if current_time > period_end:
                    active_observation_periods[symbol]['active_period'] = None
                    logging.info(f"Beobachtungszeitraum für {symbol} beendet (keine Position mehr)")
    
    logging.info(f"Positionsstatus für {symbol} aktualisiert: {'Aktiv' if has_position else 'Keine Position'}")

def check_price_change_threshold(
    df: pd.DataFrame,
    min_pct: float,
    max_pct: float,
    lookback_minutes: int,
    direction: str = 'both'  # 'plus', 'minus', oder 'both'
) -> tuple:
    """
    Überprüft, ob ein Symbol eine Preisänderung innerhalb eines bestimmten Bereichs
    und in der gewünschten Richtung im angegebenen Zeitfenster erreicht hat.
    
    Args:
        df: DataFrame mit OHLCV-Daten und 'is_complete' Flag
        min_pct: Mindest-Preisänderung in Prozent (absoluter Wert)
        max_pct: Maximale Preisänderung in Prozent (absoluter Wert)
        lookback_minutes: Zeitfenster in Minuten, in dem die Preisänderung überprüft wird
        direction: 'plus' für positive, 'minus' für negative, 'both' für beide Richtungen
    
    Returns:
        Tuple mit (erreicht_schwellenwert, änderung_prozent, start_preis, end_preis)
    """
    if df is None or df.empty:
        logging.warning("Keine Daten für Preisänderungsprüfung verfügbar")
        return False, 0, 0, 0
    
    # Nur vollständige Candles verwenden - wichtig: Kopie erstellen, um Warnungen zu vermeiden
    complete_candles = df[df['is_complete'] == True].copy()
    if len(complete_candles) < 2:
        logging.warning("Nicht genügend vollständige Candles für Preisänderungsprüfung")
        return False, 0, 0, 0
    
    # Bestimme, wie viele Candles dem lookback_minutes entsprechen
    # Prüfe das Zeitintervall der Daten
    if len(complete_candles) >= 2:
        # Berechne das typische Intervall aus den ersten beiden Candles
        interval_minutes = (complete_candles.index[1] - complete_candles.index[0]).total_seconds() / 60
        if interval_minutes <= 0:
            # Fallback, wenn die Berechnung ungültig ist
            interval_minutes = 5  # Standard-Annahme
    else:
        interval_minutes = 5  # Standard-Annahme
    
    lookback_intervals = int(lookback_minutes / interval_minutes)
    
    # Überprüfe, ob genügend Candles für die Berechnung vorhanden sind
    if len(complete_candles) <= lookback_intervals:
        logging.warning(f"Nicht genügend Candles für {lookback_minutes}min Lookback (benötigt: {lookback_intervals+1}, vorhanden: {len(complete_candles)})")
        return False, 0, 0, 0
    
    # Aktueller Preis ist der letzte vollständige Candle
    current_index = len(complete_candles) - 1
    end_price = complete_candles.iloc[current_index]['close']
    
    # Der Startpreis ist der Schlusskurs "lookback_intervals" Candles vor dem aktuellen Candle
    start_index = max(0, current_index - lookback_intervals)
    start_price = complete_candles.iloc[start_index]['close']
    
    # Berechne die prozentuale Preisänderung
    price_change_pct = ((end_price - start_price) / start_price) * 100
    
    # Prüfe Richtung der Preisänderung basierend auf dem direction Parameter
    threshold_reached = False
    
    if direction == 'both':
        # Prüfe auf Magnitude unabhängig von der Richtung
        abs_price_change_pct = abs(price_change_pct)
        threshold_reached = (abs_price_change_pct >= min_pct) and (abs_price_change_pct <= max_pct)
    elif direction == 'plus':
        # Nur positive Änderungen (Preisanstieg)
        threshold_reached = (price_change_pct >= min_pct) and (price_change_pct <= max_pct)
    elif direction == 'minus':
        # Nur negative Änderungen (Preisrückgang)
        # Beachte: Für negative Änderungen müssen wir die Zeichen umkehren, da min_pct und max_pct als positive Zahlen angegeben werden
        threshold_reached = (price_change_pct <= -min_pct) and (price_change_pct >= -max_pct)
    
    return threshold_reached, price_change_pct, start_price, end_price



def is_within_trading_time(current_time: datetime) -> bool:
    """
    Prüft, ob die aktuelle Zeit innerhalb des konfigurierten Handelszeitfensters liegt.
    Berücksichtigt den time_filter_active Schalter.
    
    Args:
        current_time: Zu überprüfende DateTime (wird in die trading_timezone konvertiert)
    
    Returns:
        Boolean: True, wenn Zeit im Handelszeitfenster liegt oder der Filter deaktiviert ist
    """
    # Prüfe, ob der Zeitfilter aktiv sein soll
    time_filter_active = Z_config.time_filter_active if hasattr(Z_config, 'time_filter_active') else True
    
    # Wenn der Zeitfilter nicht aktiv ist, immer True zurückgeben
    if not time_filter_active:
        return True
    
    # Hole die Konfiguration aus Z_config
    trading_days = Z_config.trading_days if hasattr(Z_config, 'trading_days') else [0, 1, 2, 3, 4]
    
    start_hour = Z_config.trading_start_hour if hasattr(Z_config, 'trading_start_hour') else 0
    start_minute = Z_config.trading_start_minute if hasattr(Z_config, 'trading_start_minute') else 0
    end_hour = Z_config.trading_end_hour if hasattr(Z_config, 'trading_end_hour') else 23
    end_minute = Z_config.trading_end_minute if hasattr(Z_config, 'trading_end_minute') else 59
    
    timezone_str = Z_config.trading_timezone if hasattr(Z_config, 'trading_timezone') else 'UTC'
    
    # Konvertiere die aktuelle Zeit in die konfigurierte Zeitzone
    try:
        timezone = pytz.timezone(timezone_str)
        if current_time.tzinfo is None:
            # Wenn keine Zeitzone angegeben ist, nehmen wir UTC an
            current_time = current_time.replace(tzinfo=pytz.UTC)
        
        localized_time = current_time.astimezone(timezone)
    except Exception as e:
        logging.warning(f"Fehler bei der Zeitzonenkonvertierung: {e}. Verwende Originalzeit.")
        localized_time = current_time
    
    # Prüfe den Wochentag (0 = Montag, 1 = Dienstag, ..., 6 = Sonntag)
    weekday = localized_time.weekday()
    if weekday not in trading_days:
        logging.info(f"Tag {weekday} (wobei 0=Montag) ist kein Handelstag. Konfigurierte Handelstage: {trading_days}")
        return False
    
    # Erstelle Zeitobjekte für Start und Ende des Handelszeitfensters
    trading_start = time(hour=start_hour, minute=start_minute)
    trading_end = time(hour=end_hour, minute=end_minute)
    current_time_only = localized_time.time()
    
    # Prüfe, ob die aktuelle Zeit im Handelszeitfenster liegt
    within_time_window = trading_start <= current_time_only <= trading_end
    
    if not within_time_window:
        logging.info(f"Zeit {current_time_only} liegt außerhalb des Handelszeitfensters ({trading_start}-{trading_end})")
    
    return within_time_window


def should_observe_symbol(
    symbol: str,
    df: pd.DataFrame,
    current_time: datetime,
    min_pct: float,
    max_pct: float,
    lookback_minutes: int,
    observation_hours: int,
    direction: str = 'both',  # Neuer Parameter für Richtung
    check_trading_time: bool = True
) -> bool:
    """
    Bestimmt, ob ein Symbol basierend auf seiner Preisänderung und Zeitkriterien
    beobachtet werden sollte.
    
    Args:
        symbol: Trading-Symbol
        df: DataFrame mit OHLCV-Daten
        current_time: Aktuelle Zeit für die Beobachtungsprüfung
        min_pct: Mindest-Preisänderung in Prozent
        max_pct: Maximale Preisänderung in Prozent
        lookback_minutes: Anzahl der Minuten für den Rückblick
        observation_hours: Beobachtungsdauer in Stunden
        direction: Richtung der Preisänderung ('plus', 'minus', 'both')
        check_trading_time: Wenn True, wird auch geprüft, ob die aktuelle Zeit im Handelszeitfenster liegt
    
    Returns:
        Boolean: True, wenn das Symbol beobachtet werden sollte, sonst False
    """
    global symbols_under_observation
    
    print(f"\n--- SYMBOL OBSERVATION CHECK FOR {symbol} AT {current_time} ---")
    
    # Hauptschalter für alle Filter
    filtering_active = Z_config.filtering_active if hasattr(Z_config, 'filtering_active') else True
    
    # Wenn die Filterung komplett deaktiviert ist, immer True zurückgeben
    if not filtering_active:
        print(f"Filterung ist deaktiviert. Symbol {symbol} wird immer beobachtet.")
        return True
    
    # Prüfe, ob die Preisänderungsprüfung aktiv ist
    beobachten_active = Z_config.beobachten_active if hasattr(Z_config, 'beobachten_active') else True
    
    # Prüfe, ob die aktuelle Zeit im Handelszeitfenster liegt (nur wenn check_trading_time=True)
    if check_trading_time and not is_within_trading_time(current_time):
        print(f"Zeitfilter: Symbol {symbol} wird übersprungen, da aktuelle Zeit außerhalb des Handelszeitfensters")
        return False
    
    # Prüfe, ob Symbol bereits beobachtet wird (unabhängig von beobachten_active)
    if symbol in symbols_under_observation:
        end_time = symbols_under_observation[symbol]['end_time']
        # Beobachtungszeit noch nicht abgelaufen
        if current_time <= end_time:
            remaining = end_time - current_time
            print(f"Symbol {symbol} wird bereits beobachtet (bis {end_time.strftime('%Y-%m-%d %H:%M')})")
            print(f"Verbleibende Beobachtungszeit: {remaining.total_seconds()/3600:.1f} Stunden")
            return True
        else:
            # Beobachtungszeit abgelaufen
            print(f"Beobachtungszeit für {symbol} ist abgelaufen.")
            # Symbol aus der Beobachtungsliste entfernen
            del symbols_under_observation[symbol]
    
    # Wenn beobachten_active = False ist, füge das Symbol sofort zur Beobachtungsliste hinzu
    if not beobachten_active:
        # Berechne das Ende der Beobachtungszeit
        observation_end_time = current_time + timedelta(hours=observation_hours)
        
        # Füge Symbol zur Beobachtungsliste hinzu
        symbols_under_observation[symbol] = {
            'end_time': observation_end_time,
            'start_time': current_time,
            'has_position': False,
            'position_id': None,
            'price_change_pct': 0.0  # Kein Preischeck durchgeführt
        }
        
        logging.info(f"✓ {symbol} wird beobachtet ohne Preisänderungsprüfung (Preisänderungsprüfung deaktiviert)")
        logging.info(f"  Beobachtung bis: {observation_end_time.strftime('%Y-%m-%d %H:%M')}")
        
        return True
    
    # Wenn beobachten_active = True, prüfe auf Preisänderung im definierten Bereich und mit der gewünschten Richtung
    threshold_reached, price_change_pct, start_price, end_price = check_price_change_threshold(
        df, min_pct, max_pct, lookback_minutes, direction
    )
    
    if threshold_reached:
        # Berechne das Ende der Beobachtungszeit
        observation_end_time = current_time + timedelta(hours=observation_hours)
        
        # Füge Symbol zur Beobachtungsliste hinzu mit zusätzlichen Informationen
        symbols_under_observation[symbol] = {
            'end_time': observation_end_time,
            'start_time': current_time,
            'has_position': False,
            'position_id': None,
            'price_change_pct': price_change_pct
        }
        
        direction_text = "positiv" if price_change_pct > 0 else "negativ"
        logging.info(f"✓ {symbol} erfüllt Preisänderungskriterien: {price_change_pct:.2f}% ({direction_text}) in den letzten {lookback_minutes} Minuten")
        logging.info(f"  Start-Preis: {start_price}, End-Preis: {end_price}")
        logging.info(f"  Beobachtung bis: {observation_end_time.strftime('%Y-%m-%d %H:%M')}")
        
        return True
    
    return False



def check_position_expiry(
    symbol: str,
    tracker,  # BacktestPositionTracker-Instanz
    current_time: datetime,
    close_price: float,
    close_positions: bool = False
) -> dict:
    """
    Überprüft, ob die Beobachtungszeit für eine Position abgelaufen ist
    und schließt sie automatisch, wenn close_positions=True ist.
    
    Args:
        symbol: Trading-Symbol
        tracker: BacktestPositionTracker-Instanz
        current_time: Aktuelle Zeit
        close_price: Aktueller Schlusskurs für das Schließen der Position
        close_positions: Wenn True, werden Positionen nach Ablauf der Beobachtungszeit geschlossen
    
    Returns:
        Dict mit dem Ergebnis oder None, wenn keine Position geschlossen wurde
    """
    global symbols_under_observation, active_observation_periods
    
    # Prüfe, ob der Beobachtungszeitraum aktiv ist
    active_period = None
    if symbol in active_observation_periods:
        active_period = active_observation_periods[symbol]['active_period']
    
    # Prüfe, ob das Symbol unter Beobachtung steht und eine aktive Position hat
    if symbol in symbols_under_observation and symbols_under_observation[symbol].get('has_position', False):
        symbol_data = symbols_under_observation[symbol]
        position = tracker.get_position(symbol)
        
        # Berechne die verbleibende Zeit
        remaining_time = symbol_data['end_time'] - current_time
        hours_remaining = remaining_time.total_seconds() / 3600
        
        # Detaillierteres Logging hinzufügen
        print(f"\n--- POSITION EXPIRY CHECK FOR {symbol} ---")
        print(f"Observed since: {symbol_data['start_time']}")
        print(f"Observation ends: {symbol_data['end_time']}")
        print(f"Current time: {current_time}")
        print(f"Remaining observation time: {hours_remaining:.2f} hours")
        print(f"close_positions setting: {close_positions}")
        
        # Prüfe, ob die Position noch aktiv ist
        if position and position["is_active"]:
            # Prüfe, ob die Beobachtungszeit abgelaufen ist
            if current_time > symbol_data['end_time']:
                print(f"⏱️ Beobachtungszeit für {symbol} abgelaufen")
                
                if close_positions:
                    # Schließe die Position
                    exit_result = tracker._close_position(
                        symbol=symbol,
                        exit_price=close_price,
                        exit_time=current_time,
                        exit_reason="observation_timeout"
                    )
                    
                    # Aktualisiere den Status im Beobachtungswörterbuch
                    symbols_under_observation[symbol]['has_position'] = False
                    if symbol in active_observation_periods:
                        active_observation_periods[symbol]['has_position'] = False
                    
                    print(f"Position für {symbol} geschlossen zum Preis {close_price}")
                    print(f"Exit Grund: observation_timeout")
                    
                    return exit_result
                else:
                    print(f"Position für {symbol} bleibt offen trotz abgelaufener Beobachtungszeit (close_positions=False)")
                    return None
            else:
                print(f"Position für {symbol} ist noch innerhalb der Beobachtungszeit")
        else:
            print(f"Keine aktive Position für {symbol} gefunden")
    
    return None


    
def get_observation_period(symbol, current_time):
    """
    Gibt den Beobachtungszeitraum für ein Symbol zurück.
    
    Args:
        symbol: Trading-Symbol
        current_time: Aktuelle Zeit für die Beobachtungsprüfung
    
    Returns:
        Dict mit start_time und end_time für den Beobachtungszeitraum
    """
    global symbols_under_observation
    
    if symbol in symbols_under_observation:
        # Symbol ist bereits in der Beobachtungsliste
        return {
            'start_time': symbols_under_observation[symbol]['start_time'],
            'end_time': symbols_under_observation[symbol]['end_time']
        }
    else:
        # Symbol ist neu - berechne Beobachtungszeitraum
        observation_hours = Z_config.symbol_observation_hours if hasattr(Z_config, 'symbol_observation_hours') else 4
        observation_end_time = current_time + timedelta(hours=observation_hours)
        
        # In Beobachtungsliste eintragen
        symbols_under_observation[symbol] = {
            'start_time': current_time,
            'end_time': observation_end_time,
            'has_position': False,
            'position_id': None,
            'price_change_pct': 0.0
        }
        
        return {
            'start_time': current_time,
            'end_time': observation_end_time
        }

def verify_observation_indicators(df, observation_start, symbol):
    """
    Überprüft, ob die Indikatoren für die ersten Candles im Beobachtungszeitraum korrekt berechnet wurden.
    
    Args:
        df: DataFrame mit berechneten Indikatoren
        observation_start: Startzeit des Beobachtungszeitraums
        symbol: Symbol für die Protokollierung
        
    Returns:
        Boolean: True wenn alle Indikatoren korrekt berechnet wurden
    """
    # Extrahiere die ersten 3 Candles im Beobachtungszeitraum
    observation_mask = (df.index >= observation_start)
    first_candles = df[observation_mask].head(3)
    
    if first_candles.empty:
        print(f"⚠️ Keine Candles im Beobachtungszeitraum für {symbol} gefunden!")
        return False
    
    # Überprüfe, ob die wichtigsten Indikatoren berechnet wurden
    #print(f"\nÜberprüfung der ersten Candles im Beobachtungszeitraum für {symbol}:")
    
    all_indicators_valid = True
    
    """for idx, (candle_time, candle) in enumerate(first_candles.iterrows()):
        print(f"Candle {idx+1} bei {candle_time}:")
        
        # Überprüfe kritische Indikatoren
        indicators_to_check = ['rsi', 'ema_fast', 'ema_slow', 'ema_baseline', 'trend', 'trend_strength']
        
        for indicator in indicators_to_check:
            if indicator not in candle or pd.isna(candle[indicator]):
                print(f"  ✗ {indicator}: nicht berechnet oder NaN")
                all_indicators_valid = False
            else:
                print(f"  ✓ {indicator}: {candle[indicator]}")
    
    if all_indicators_valid:
        print(f"✅ Alle Indikatoren für die ersten Candles im Beobachtungszeitraum wurden korrekt berechnet")
    else:
        print(f"❌ Einige Indikatoren für die ersten Candles im Beobachtungszeitraum wurden nicht korrekt berechnet")"""
    
    return all_indicators_valid



def clean_expired_observations(current_time):
    """
    Entfernt abgelaufene Symbole aus der Beobachtungsliste.
    
    Args:
        current_time: Aktuelle Zeit für die Prüfung
    """
    global symbols_under_observation
    
    # Liste der zu entfernenden Symbole erstellen
    to_remove = []
    
    for symbol, data in symbols_under_observation.items():
        if current_time > data['end_time'] and not data['has_position']:
            to_remove.append(symbol)
    
    # Symbole aus der Liste entfernen
    for symbol in to_remove:
        del symbols_under_observation[symbol]
        logging.info(f"Symbol {symbol} aus Beobachtungsliste entfernt (Beobachtungszeit abgelaufen)")

def print_observation_status():
    """Gibt den aktuellen Status der Beobachtungsliste aus."""
    global symbols_under_observation
    
    if not symbols_under_observation:
        print("Keine Symbole in der Beobachtungsliste.")
        return
    
    current_time = datetime.now(timezone.utc)
    print(f"\nAktuelle Beobachtungsliste ({len(symbols_under_observation)} Symbole):")
    print("-" * 80)
    print(f"{'Symbol':<10} {'Start':<20} {'Ende':<20} {'Position':<10} {'Restzeit':<15}")
    print("-" * 80)
    
    for symbol, data in sorted(symbols_under_observation.items()):
        remaining = data['end_time'] - current_time
        remaining_str = f"{remaining.total_seconds() / 3600:.1f}h" if remaining.total_seconds() > 0 else "abgelaufen"
        print(f"{symbol:<10} {data['start_time'].strftime('%Y-%m-%d %H:%M'):<20} "
              f"{data['end_time'].strftime('%Y-%m-%d %H:%M'):<20} "
              f"{'Ja' if data['has_position'] else 'Nein':<10} {remaining_str:<15}")
    
    print("-" * 80)


observation_period_tracking = {}



# Globales Tracking-Dictionary für Beobachtungszeiträume
active_observation_periods = {}  # Format: {symbol: {'active_period': period_data, 'has_position': bool}}

def manage_observation_periods(symbol, current_time, has_position=None):
    """
    Verwaltet die Beobachtungszeiträume und ihren Status.
    
    Args:
        symbol: Trading-Symbol
        current_time: Aktuelle Zeit für die Prüfung
        has_position: Wenn angegeben, aktualisiert den Positionsstatus
    
    Returns:
        bool: True wenn der Beobachtungszeitraum aktiv ist und eine Position geöffnet werden kann
    """
    global active_observation_periods, symbols_under_observation
    
    # Initialisiere für dieses Symbol, falls noch nicht vorhanden
    if symbol not in active_observation_periods:
        active_observation_periods[symbol] = {
            'active_period': None,
            'has_position': False
        }
    
    # Aktualisiere den Positionsstatus, wenn angegeben
    if has_position is not None:
        active_observation_periods[symbol]['has_position'] = has_position
        
        # Wenn keine Position mehr aktiv ist und der Beobachtungszeitraum abgelaufen ist,
        # kann dieser Zeitraum aus dem Tracking entfernt werden
        if not has_position and active_observation_periods[symbol]['active_period']:
            period_end = active_observation_periods[symbol]['active_period']['end_time']
            if current_time > period_end:
                active_observation_periods[symbol]['active_period'] = None
                logging.info(f"Beobachtungszeitraum für {symbol} entfernt (keine Position mehr und Zeitraum abgelaufen)")
    
    # Prüfe, ob ein aktiver Beobachtungszeitraum existiert
    if active_observation_periods[symbol]['active_period']:
        # Wenn der Zeitraum abgelaufen ist und keine Position mehr aktiv ist,
        # kann ein neuer Zeitraum aktiviert werden
        period_end = active_observation_periods[symbol]['active_period']['end_time']
        if current_time > period_end and not active_observation_periods[symbol]['has_position']:
            active_observation_periods[symbol]['active_period'] = None
            logging.info(f"Beobachtungszeitraum für {symbol} abgelaufen und keine Position mehr aktiv")
    
    # Wenn kein aktiver Zeitraum existiert, prüfe, ob ein neuer aktiviert werden kann
    if not active_observation_periods[symbol]['active_period']:
        # Prüfe, ob ein Zeitraum in symbols_under_observation verfügbar ist
        if symbol in symbols_under_observation:
            # Sortiere nach Startzeit, um den nächsten zu finden
            available_periods = [
                period for period in symbols_under_observation[symbol]
                if period['start_time'] <= current_time <= period['end_time']
            ]
            
            if available_periods:
                # Aktiviere den ersten verfügbaren Zeitraum
                active_observation_periods[symbol]['active_period'] = available_periods[0]
                logging.info(f"Neuer Beobachtungszeitraum für {symbol} aktiviert: {available_periods[0]['start_time']} bis {available_periods[0]['end_time']}")
    
    # Prüfe, ob ein Beobachtungszeitraum aktiv ist und eine Position geöffnet werden kann
    if active_observation_periods[symbol]['active_period']:
        period = active_observation_periods[symbol]['active_period']
        within_period = period['start_time'] <= current_time <= period['end_time']
        
        # Wenn close_position=False, kann eine Position auch dann geöffnet werden,
        # wenn bereits eine Position aktiv ist (für Seitenwechsel)
        close_position = Z_config.close_position if hasattr(Z_config, 'close_position') else False
        can_open_position = within_period and (
            not active_observation_periods[symbol]['has_position'] or not close_position
        )
        
        return can_open_position
    
    return False

