import pandas as pd
import os

# --- Konfiguration ---
input_filename = 'all_runs_optimization_results_ml.csv'
# Neuer Name für die Ausgabedatei, um die Sortierung anzuzeigen
output_filename = 'profitable_runs_sorted.csv'
column_to_filter_and_sort = 'total_profit'
# -------------------

# Sicherstellen, dass die Pfade relativ zum Skriptverzeichnis sind
script_dir = os.path.dirname(os.path.abspath(__file__))
input_filepath = os.path.join(script_dir, input_filename)
output_filepath = os.path.join(script_dir, output_filename)

print(f"Lese CSV-Datei: {input_filepath}")

try:
    # Lese die CSV-Datei in ein pandas DataFrame
    df = pd.read_csv(input_filepath)
    print(f"Erfolgreich {len(df)} Zeilen gelesen.")

    # Überprüfe, ob die Spalte zum Filtern und Sortieren existiert
    if column_to_filter_and_sort not in df.columns:
        print(f"Fehler: Spalte '{column_to_filter_and_sort}' nicht in der Datei gefunden.")
        print("Verfügbare Spalten:", list(df.columns))
        exit() # Beendet das Skript

    print(f"Filtere nach '{column_to_filter_and_sort}' >= 0...")

    # Konvertiere die Spalte in einen numerischen Typ (Fehler -> NaN)
    df[column_to_filter_and_sort] = pd.to_numeric(df[column_to_filter_and_sort], errors='coerce')

    # Filtere das DataFrame (behalte nur Zeilen >= 0)
    # .copy() wird verwendet, um eine explizite Kopie zu erstellen und Warnungen zu vermeiden
    df_filtered = df[df[column_to_filter_and_sort] >= 0].copy()

    # Überprüfen, ob nach dem Filtern noch Daten vorhanden sind
    if df_filtered.empty:
        print(f"Nach dem Filtern wurden keine Zeilen mit nicht-negativem '{column_to_filter_and_sort}' gefunden.")
        # Optional: Eine leere Datei speichern oder nichts tun
        # print(f"Speichere leere Datei: {output_filepath}")
        # df_filtered.to_csv(output_filepath, index=False) # Erstellt eine Datei nur mit Headern
        print("Es wird keine Ausgabedatei erstellt, da keine profitablen Durchläufe gefunden wurden.")
    else:
        print(f"{len(df_filtered)} Zeilen mit nicht-negativem '{column_to_filter_and_sort}' gefunden.")

        # *** NEU: Sortiere das gefilterte DataFrame ***
        # Sortiere nach der Zielspalte, 'ascending=False' für absteigende Reihenfolge (höchster Wert zuerst)
        print(f"Sortiere Ergebnisse nach '{column_to_filter_and_sort}' (absteigend)...")
        df_sorted = df_filtered.sort_values(by=column_to_filter_and_sort, ascending=False)

        # Speichere das gefilterte UND SORTIERTE DataFrame
        print(f"Speichere gefilterte und sortierte Daten in: {output_filepath}")
        df_sorted.to_csv(output_filepath, index=False)

        print("Skript erfolgreich abgeschlossen.")

except FileNotFoundError:
    print(f"Fehler: Eingabedatei '{input_filepath}' nicht gefunden.")
    print("Stelle sicher, dass die CSV-Datei im selben Ordner wie das Skript liegt und der Name korrekt ist.")
except Exception as e:
    print(f"Ein unerwarteter Fehler ist aufgetreten: {e}")