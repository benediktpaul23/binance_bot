from utils.binance import fetch_data
import pandas as pd
from datetime import datetime, timezone, timedelta
import time
import logging
import os

# Angenommen, deine fetch_data-Funktion und alle erforderlichen Abhängigkeiten sind bereits definiert

def create_custom_daily_files(symbols, output_dir='symbol_changes', days=30):
    """
    Erstellt für jedes Symbol eine Datei mit täglichen Daten (5:00 bis 5:00) für einen Monat
    
    Args:
        symbols: Liste der zu überwachenden Symbole (z.B. ['BTCUSDT', 'ETHUSDT'])
        output_dir: Verzeichnis für die Ausgabedateien
        days: Anzahl der Tage zurück (standardmäßig 30 Tage)
    """
    # Stelle sicher, dass das Ausgabeverzeichnis existiert
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Verzeichnis {output_dir} erstellt")
    
    # Berechne End- und Startdatum
    end_time = datetime.now(timezone.utc)
    
    # Runde auf 5:00 Uhr des aktuellen Tages
    if end_time.hour < 5:
        # Wenn es vor 5:00 Uhr ist, nehmen wir 5:00 Uhr des Vortages
        end_time = end_time.replace(hour=5, minute=0, second=0, microsecond=0) - timedelta(days=1)
    else:
        # Sonst nehmen wir 5:00 Uhr des aktuellen Tages
        end_time = end_time.replace(hour=5, minute=0, second=0, microsecond=0)
    
    # Für jeden 5:00-5:00 Tag benötigen wir 24 Stunden Daten, also 24 * days Stunden insgesamt
    lookback_hours = 24 * days
    
    for symbol in symbols:
        print(f"\nVerarbeite {symbol}...")
        
        try:
            # Hole stündliche Daten für den gesamten Zeitraum
            # (wir verwenden 1h-Intervall für bessere Genauigkeit an den 5:00-Grenzen)
            df = fetch_data.fetch_data(
                symbol=symbol, 
                interval='1h',
                lookback_hours=lookback_hours,
                end_time=end_time
            )
            
            if df is None or df.empty:
                print(f"Keine Daten für {symbol} gefunden")
                continue
            
            # Wir erstellen eine neue DataFrame für die 5:00-5:00 Tage
            custom_days = []
            
            # Startdatum für die Schleife (30 Tage zurück von end_time)
            current_day_start = end_time - timedelta(days=days-1)
            
            for day in range(days):
                # Definiere Start (5:00) und Ende (4:59 des nächsten Tages) für diesen Tag
                day_start = current_day_start.replace(hour=5, minute=0, second=0, microsecond=0)
                day_end = (day_start + timedelta(days=1)).replace(hour=4, minute=59, second=59, microsecond=999999)
                
                # Filtere Daten für diesen Zeitraum
                day_data = df[(df.index >= day_start) & (df.index <= day_end)]
                
                if not day_data.empty:
                    # Berechne OHLCV für diesen benutzerdefinierten Tag
                    day_open = day_data.iloc[0]['open']
                    day_high = day_data['high'].max()
                    day_low = day_data['low'].min()
                    day_close = day_data.iloc[-1]['close']
                    day_volume = day_data['volume'].sum()
                    
                    # Berechne Veränderung und Prozent
                    day_change = day_close - day_open
                    day_change_percent = (day_change / day_open) * 100
                    
                    # Füge die Daten zur Liste hinzu
                    custom_days.append({
                        'date': day_start.strftime('%Y-%m-%d'),
                        'time_start': day_start.strftime('%H:%M'),
                        'time_end': day_end.strftime('%H:%M'),
                        'open': day_open,
                        'high': day_high,
                        'low': day_low,
                        'close': day_close,
                        'volume': day_volume,
                        'change': day_change,
                        'change_percent': day_change_percent
                    })
                    
                    print(f"Tag {day_start.strftime('%Y-%m-%d')} verarbeitet: Open={day_open}, Close={day_close}, Change={day_change:.2f} ({day_change_percent:.2f}%)")
                else:
                    print(f"Keine Daten für Tag {day_start.strftime('%Y-%m-%d')} gefunden")
                
                # Gehe zum nächsten Tag
                current_day_start += timedelta(days=1)
            
            # Erstelle DataFrame aus den gesammelten Daten
            if custom_days:
                custom_df = pd.DataFrame(custom_days)
                
                # Speichere als CSV im angegebenen Verzeichnis
                output_file = os.path.join(output_dir, f"{symbol}_daily_5to5.csv")
                custom_df.to_csv(output_file, index=False)
                print(f"Daten für {symbol} in {output_file} gespeichert ({len(custom_days)} Tage)")
            else:
                print(f"Keine Tage für {symbol} verarbeitet")
                
        except Exception as e:
            print(f"Fehler bei der Verarbeitung von {symbol}: {e}")
    
    print("\nVerarbeitung abgeschlossen. Alle Dateien wurden im Verzeichnis '{output_dir}' gespeichert.")

# Beispielaufruf:
if __name__ == "__main__":
    # Liste der zu überwachenden Symbole
    symbols_to_track = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'SOLUSDT']
    
    # Erstelle die Dateien
    create_custom_daily_files(symbols_to_track, output_dir='symbol_changes', days=30*12*15)