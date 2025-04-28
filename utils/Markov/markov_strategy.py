import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

# Pfad zum Ordner mit den CSV-Dateien
folder_path = 'symbol_changes'

# Alle CSV-Dateien mit dem Muster *_daily_5to5.csv finden
file_pattern = os.path.join(folder_path, '*_daily_5to5.csv')
csv_files = glob.glob(file_pattern)

# Ergebnisse für alle Sequenzlängen speichern
all_sequence_results = {}

for file_path in csv_files:
    try:
        # Symbolname aus dem Dateinamen extrahieren
        symbol = os.path.basename(file_path).split('_')[0]
        
        print(f"\n{'='*50}")
        print(f"Analyse für {symbol}:")
        print(f"{'='*50}")
        
        # CSV mit Header laden
        df = pd.read_csv(file_path)
        
        # Datum als Index setzen und nach Datum sortieren
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        df.set_index('date', inplace=True)
        
        # Tage als Plus oder Minus klassifizieren basierend auf change_percent
        df['Zustand'] = np.where(df['change_percent'] >= 0, 'Plus', 'Minus')
        
        # Einfache Zustandsverteilung
        zustaende = df['Zustand'].value_counts()
        minus_tage = zustaende.get('Minus', 0)
        plus_tage = zustaende.get('Plus', 0)
        
        print(f"Verteilung der Tage: Plus: {plus_tage} ({plus_tage/len(df):.2%}), Minus: {minus_tage} ({minus_tage/len(df):.2%})")
        
        # Analyse aller möglichen Minus-Sequenzen
        # Zählen, wie viele aufeinanderfolgende Minus-Tage es maximal gibt
        current_streak = 0
        max_streak = 0
        all_streaks = []
        
        for i in range(len(df)):
            if df['Zustand'].iloc[i] == 'Minus':
                current_streak += 1
            else:
                if current_streak > 0:
                    all_streaks.append(current_streak)
                    max_streak = max(max_streak, current_streak)
                current_streak = 0
        
        # Letzten Streak hinzufügen, falls die Daten mit Minus enden
        if current_streak > 0:
            all_streaks.append(current_streak)
            max_streak = max(max_streak, current_streak)
        
        print(f"\nMaximale Anzahl aufeinanderfolgender Minus-Tage: {max_streak}")
        
        # Histogramm der Streak-Längen
        streak_counts = pd.Series(all_streaks).value_counts().sort_index()
        print("\nVerteilung der Minus-Streak-Längen:")
        for length, count in streak_counts.items():
            print(f"{length} Minus-Tage in Folge: {count} mal")
        
        # Analyse für jede mögliche Sequenzlänge
        sequence_results = {}
        
        for seq_length in range(1, max_streak + 1):
            # Finde alle Positionen, an denen genau seq_length Minus-Tage aufeinander folgen
            positions = []
            current_streak = 0
            
            for i in range(len(df)):
                if df['Zustand'].iloc[i] == 'Minus':
                    current_streak += 1
                    if current_streak == seq_length and i+1 < len(df):
                        positions.append(i+1)  # Position des Tages nach der Sequenz
                else:
                    current_streak = 0
            
            # Wenn Positionen gefunden wurden, analysiere den folgenden Tag
            if positions:
                next_day_states = [df['Zustand'].iloc[pos] for pos in positions]
                plus_count = next_day_states.count('Plus')
                minus_count = next_day_states.count('Minus')
                total = plus_count + minus_count
                
                plus_prob = plus_count / total if total > 0 else 0
                minus_prob = minus_count / total if total > 0 else 0
                
                sequence_results[seq_length] = {
                    'Anzahl': total,
                    'Plus_danach': plus_count,
                    'Minus_danach': minus_count,
                    'Plus_Wahrscheinlichkeit': plus_prob,
                    'Minus_Wahrscheinlichkeit': minus_prob
                }
        
        # Ausgabe der Ergebnisse für jede Sequenzlänge
        print("\nWahrscheinlichkeit für Plus-Tag nach X aufeinanderfolgenden Minus-Tagen:")
        for seq_length, results in sequence_results.items():
            print(f"Nach {seq_length} Minus-Tagen: Plus: {results['Plus_Wahrscheinlichkeit']:.2%}, Anzahl Fälle: {results['Anzahl']}")
        
        # Speichern der Ergebnisse für dieses Symbol
        all_sequence_results[symbol] = sequence_results
        
        # Visualisierung
        seq_lengths = list(sequence_results.keys())
        plus_probs = [results['Plus_Wahrscheinlichkeit'] for results in sequence_results.values()]
        sample_sizes = [results['Anzahl'] for results in sequence_results.values()]
        
        # Erstelle eine Figur mit zwei Subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # Plot der Plus-Wahrscheinlichkeiten
        bars = ax1.bar(seq_lengths, plus_probs, color='green')
        ax1.axhline(y=0.5, color='red', linestyle='--')
        ax1.set_xlabel('Anzahl aufeinanderfolgender Minus-Tage')
        ax1.set_ylabel('Wahrscheinlichkeit für Plus-Tag danach')
        ax1.set_title(f'{symbol}: Wahrscheinlichkeit für Plus-Tag nach X Minus-Tagen')
        ax1.set_xticks(seq_lengths)
        ax1.set_ylim([0, 1])
        
        # Beschriftung der Balken mit Prozenten
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1%}', ha='center', va='bottom')
        
        # Plot der Stichprobengrößen
        bars2 = ax2.bar(seq_lengths, sample_sizes, color='blue')
        ax2.set_xlabel('Anzahl aufeinanderfolgender Minus-Tage')
        ax2.set_ylabel('Anzahl der Fälle')
        ax2.set_title(f'{symbol}: Stichprobengröße für jede Sequenzlänge')
        ax2.set_xticks(seq_lengths)
        
        # Beschriftung der Balken mit Anzahl
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{symbol}_minus_sequence_analysis.png')
        plt.close()
        
    except Exception as e:
        print(f"Fehler bei der Analyse von {file_path}: {str(e)}")
        import traceback
        traceback.print_exc()

# Zusammenfassung aller Symbole
print("\n\n" + "="*80)
print("ZUSAMMENFASSUNG ALLER SYMBOLE:")
print("="*80)

# Tabelle mit allen Ergebnissen erstellen
summary_data = []
for symbol, seq_results in all_sequence_results.items():
    for seq_length, results in seq_results.items():
        if results['Anzahl'] >= 10:  # Nur Sequenzen mit ausreichender Stichprobengröße
            summary_data.append({
                'Symbol': symbol,
                'Minus_Sequenz_Länge': seq_length,
                'Plus_Wahrscheinlichkeit': results['Plus_Wahrscheinlichkeit'],
                'Anzahl_Fälle': results['Anzahl']
            })

summary_df = pd.DataFrame(summary_data)
summary_df = summary_df.sort_values(['Plus_Wahrscheinlichkeit', 'Anzahl_Fälle'], ascending=[False, False])

print("\nTop Handelsmöglichkeiten (höchste Wahrscheinlichkeit für Plus-Tag, mindestens 10 Fälle):")
print(summary_df.head(10))

# Speichern der Zusammenfassung
summary_df.to_csv('minus_sequence_analysis_summary.csv', index=False)
print("\nZusammenfassung wurde in 'minus_sequence_analysis_summary.csv' gespeichert.")

print("\nAnalyse abgeschlossen.")