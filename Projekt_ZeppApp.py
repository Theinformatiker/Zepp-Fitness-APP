# -*- coding: utf-8 -*-
"""
Projektarbeit
Darstellung und Auswertung des Datensatzes
"Zepp Fitness App"

@author: Volkan korunan
Projekt fertig

"""

import xmltodict
import json
import pandas as pd
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import folium
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from PIL import Image
import io

# Definieren des Pfads zur GPX-Datei
file_path = r'C:\Users\vkkor\.spyder-py3\Woche 4\Projekt\datein\zepp/datei2.gpx'

# XML-Datei lesen und in ein Dictionary umwandeln
try:
    with open(file_path, 'r', encoding='utf-8') as xml_file:
        doc = xmltodict.parse(xml_file.read())
        
    # Dictionary in JSON umwandeln und speichern
    json_data = json.dumps(doc, indent=4)
    with open('testJsonConvert.json', 'w') as json_file:
        json_file.write(json_data)

except FileNotFoundError:
    print(f"Die Datei '{file_path}' wurde nicht gefunden.")
    exit()
except Exception as e:
    print(f"Ein Fehler ist aufgetreten: {e}")
    exit()

# JSON in ein Python-Dictionary laden
try:
    data = json.loads(json_data)  # Verwende die Variable direkt statt die Datei erneut zu lesen
except json.JSONDecodeError as e:
    print(f"Fehler beim Lesen der JSON-Daten: {e}")
    exit()

# Liste zum Speichern extrahierter Informationen
track_points = []

# Iteration über JSON-Daten und Extraktion von Informationen
try:
    for trkpt in data['gpx']['trk']['trkseg']['trkpt']:
        # Verwenden Sie Standardwerte, wenn ein Schlüssel fehlt
        extensions = trkpt.get('extensions', {}).get('ns3:TrackPointExtension', {})
        speed = extensions.get('ns3:speed', None)
        heart_rate = extensions.get('ns3:hr', None)
        cad = extensions.get('ns3:cad', None) 

        point = {
            'latitude': float(trkpt['@lat']),
            'longitude': float(trkpt['@lon']),
            'elevation': float(trkpt['ele']),
            'time': pd.to_datetime(trkpt['time']),
            'speed': float(speed) if speed is not None else None,
            'heart_rate': float(heart_rate) if heart_rate is not None else None,
            'cadence': float(cad) if cad is not None else None
        }
        track_points.append(point)
except KeyError as e:
    print(f"Key error: {e}. Bitte überprüfen Sie die JSON-Struktur.")
    exit()
except ValueError as e:
    print(f"Value error: {e}. Überprüfen Sie die Datenwerte.")
    exit()

# Erstellen des DataFrames
df = pd.DataFrame(track_points)

# DataFrame ausgeben
print(f"df: \n {df}")

print("DataFrame info Ausgabe: \n ")
df.info()

print("DataFrame Describe Ausgabe: \n")
df.describe()


# Um rechnung von Meter pro Sekunde in KM pro Stunde 
df['Speed'] = df['speed'] *3.6


# Behandlung von fehlenden Herzfrequenzdaten durch Interpolation
df['Heart_rate'] = df['heart_rate'].interpolate(method='linear')
df['Heart_rate'] = df['Heart_rate'].round()

# um die ersten nan Werte die nicht interpoliert werden konnten,
# auf den ersten gemessenen wert zu setzten
df['Heart_rate'] = df['Heart_rate'].backfill() 

# NaN-Werte in der Spalte 'Cadence' durch 0 ersetzen
df['Cadence'] = df['cadence'].where(df['cadence'].notna(), 0) 
df['Cadence'] = df['Cadence'].interpolate(method='linear')


# Median der Cadence-Spalte berechnen, unter Ausschluss von Nullen
median_cadence = df['Cadence'][df['Cadence'] != 0].median()

# Nullen durch den Median ersetzen
df['Cadence'] = df['Cadence'].replace(0, median_cadence)



# Extrahieren der Minuten und Sekunden in separate Spalten
time1 =datetime.datetime.strptime('10:09:2024:15:26:42',"%d:%m:%Y:%H:%M:%S")
timeaware =time1.replace(tzinfo=datetime.timezone.utc)

df['time_delta'] = df['time'] - timeaware
df['seconds'] = df['time_delta'].dt.components['seconds']
df['minutes'] = df['time_delta'].dt.components['minutes']
df['stunde'] = df['time_delta'].dt.components['hours']
df['Fullminutes'] = df['stunde'] *60 + df['minutes']


print(f" cdenszSum = {df['Cadence'].sum()}")

# Scatterplot  von der Herzfreuenz  va Geschwindigkeit
sns.scatterplot(data=df, x='Speed', y='Heart_rate')

# Polynomiale Regressionslinie dritten Grades hinzufügen
#sns.regplot(data=df, x='Speed', y='Heart_rate', scatter=False, color='black', order=3, line_kws={"label":"Regressionslinie (5. Grad)"})


# Achsenbeschriftungen und Titel hinzufügen
plt.xlabel('Geschwindigkeit (km/h)')
plt.ylabel('Herzfrequenz (bpm)')
plt.title('Herzfrequenz vs. Geschwindigkeit')

# Herzfrequenzlinien hinzufügen
hr_max = df['Heart_rate'].max()
hr_mean = df['Heart_rate'].mean()
hr_min = df['Heart_rate'].min()
plt.axhline(y=hr_max, color='red', linestyle='--', label='Max Herzfrequenz')
plt.axhline(y=hr_mean, color='green', linestyle='--', label='Durchschnitt Herzfrequenz')
plt.axhline(y=hr_min, color='violet', linestyle='--', label='Min Herzfrequenz')

# Legende hinzufügen und anpassen
plt.legend(loc='lower right', fontsize='8')  # Position und Größe anpassen

# Plot anzeigen
plt.show()



# Erstellen des ersten Scatterplots für die erste y-Achse (Herzfrequenz)
fig, ax1 = plt.subplots()

sns.lineplot(data=df, x='Fullminutes', y='Heart_rate', ax=ax1, color='b', label='Herzfrequenz')
ax1.set_xlabel('Minutes')
ax1.set_ylabel('Herzfrequenz (bpm)', color='b')
ax1.tick_params(axis='y', labelcolor='b')

# Herzfrequenzlinien hinzufügen
hr_max = df['Heart_rate'].max()
hr_mean = df['Heart_rate'].mean()
hr_min = df['Heart_rate'].min()
ax1.axhline(y=hr_max, color='red', linestyle='--', label='Max Herzfrequenz')
ax1.axhline(y=hr_mean, color='green', linestyle='--', label='Durchschnitt Herzfrequenz')
ax1.axhline(y=hr_min, color='violet', linestyle='--', label='Min Herzfrequenz')

# Erstellen der sekundären y-Achse
ax2 = ax1.twinx()
sns.lineplot(data=df, x='Fullminutes', y='Speed', ax=ax2, color='orange', label='Geschwindigkeit')
ax2.set_ylabel('Speed (km/h)', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

# Legenden zusammenführen und anzeigen
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='lower right', fontsize='6')

plt.title('Time in Minutes vs Herzfrequenz & Speed')
plt.show()



# Scatterplot erstellen
sns.scatterplot(data=df, x='minutes', y='Heart_rate')

# Achsenbeschriftungen und Titel hinzufügen
plt.xlabel('minutes')
plt.ylabel('Herzfrequenz (bpm)')
plt.title('Time in minutes vs Herzfrequenz')
# Herzfrequenzlinien hinzufügen
hr_max = df['Heart_rate'].max()
hr_mean = df['Heart_rate'].mean()
hr_min = df['Heart_rate'].min()
plt.axhline(y=hr_max, color='red', linestyle='--', label='Max Herzfrequenz')
plt.axhline(y=hr_mean, color='green', linestyle='--', label='Durchschnitt Herzfrequenz')
plt.axhline(y=hr_min, color='violet', linestyle='--', label='Min Herzfrequenz')


# Legende hinzufügen und anpassen
plt.legend(loc='lower right', fontsize='8',)  # Position und Größe anpassen
# Plot anzeigen
plt.show()


#Histo Plot

# Histogramm mit Seaborn erstellen
sns.histplot(df['Heart_rate'], bins=20, color='skyblue', kde=False)

# Achsenbeschriftungen und Titel hinzufügen
plt.xlabel('Herzfrequenz (bpm)')
plt.ylabel('Anzahl der Vorkommen')
plt.title('Verteilung der Herzfrequenz')

# Y-Achse in 60er-Schritten aufteilen bis 600
plt.yticks(range(0, 601, 60))

# Plot anzeigen
plt.show()

#Histo Plot

# Histogramm mit Seaborn erstellen
sns.histplot(df['Speed'], bins=20, color='skyblue', kde=False)

# Achsenbeschriftungen und Titel hinzufügen
plt.xlabel('Geschwindigkeit km/h')
plt.ylabel('Anzahl der Vorkommen')
plt.title('Verteilung der Geschwindigkeit')


# Plot anzeigen
plt.show()


# Plot erstellen
plt.figure(figsize=(12, 6))
sns.lineplot(x='minutes', y='elevation', data=df, marker='o')
plt.title('Elevation über die Zeit')
plt.xlabel('Fullminutes')
plt.ylabel('Höhe (Elevation in Metern)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

def plot_histogram(df, x_col, title='Histogramm', xlabel='X-Achse', ylabel='Häufigkeit', bins=20):
    """
    Plottet ein Histogramm basierend auf den angegebenen Parametern.

    :param df: DataFrame, der die Daten enthält
    :param x_col: Spaltenname für die x-Achse
    :param title: Titel des Plots
    :param xlabel: Beschriftung der x-Achse
    :param ylabel: Beschriftung der y-Achse
    :param bins: Anzahl der Bins im Histogramm
    """
    plt.figure(figsize=(12, 6))
    sns.histplot(df[x_col], bins=bins,color='skyblue', kde=False)  # KDE (Kernel Density Estimate) hinzufügen, wenn gewünscht
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()



# Funktion zur Erstellung von Boxplots
def create_boxplot(data, y_var, title):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data, y=y_var)
    plt.title(title)
    plt.show()

# Boxplots für verschiedene Variablen erstellen
create_boxplot(df, 'elevation', 'Verteilung der Höhe')
create_boxplot(df, 'Speed', 'Verteilung der Geschwindigkeit')
create_boxplot(df, 'heart_rate', 'Verteilung der Herzfrequenz')
create_boxplot(df, 'cadence', 'Verteilung der Kadenz')


# Funktion aufrufen für verschiedene Histogramme
plot_histogram(df, x_col='minutes', title='Histogramm der Minuten', xlabel='Minuten', ylabel='Häufigkeit', bins=5)
plot_histogram(df, x_col='elevation', title='Histogramm der Höhe', xlabel='Höhe (in Metern)', ylabel='Häufigkeit', bins=6)




breitengrade = df.latitude  # Breitengrade
laengengrade = df.longitude  # Längengrade
# Konstante: Radius der Erde in Kilometern
EARTH_RADIUS_KM = 6371.0
def haversine_distance(lat1, lon1, lat2, lon2):
    # Umwandlung der Koordinaten von Grad in Radian
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Berechnung der Unterschiede der Koordinaten
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine-Formel
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    # Berechnung der Entfernung
    distance = EARTH_RADIUS_KM * c
    return distance

# Berechnung der gesamten zurückgelegten Strecke
total_distance = 0.0

for i in range(1, len(breitengrade)):
    # Berechnung der Entfernung zwischen aufeinanderfolgenden Punkten
    distance = haversine_distance(breitengrade[i-1], laengengrade[i-1], breitengrade[i], laengengrade[i])
    total_distance += distance

print(f"Die zurückgelegte Strecke beträgt etwa {total_distance:.2f} km.")


elvMax= df['elevation'].min()
print(f"eveo max {elvMax} ")

# Laufstrecke Darstellen auf dem Browser und abspeichern als Bild

# Beispiel: Listen von Längen- und Breitengraden
breitengrade = df.latitude  # Breitengrade
laengengrade = df.longitude  # Längengrade

# Startpunkt für die Karte setzen (Mittelpunkt der ersten Koordinate)
karte = folium.Map(location=[breitengrade[0], laengengrade[0]], zoom_start=14.45)

# Eine Liste der Koordinatenpaare erstellen
koordinaten = list(zip(breitengrade, laengengrade))

# Strecke (Linie) auf der Karte einzeichnen
folium.PolyLine(locations=koordinaten, color='blue', weight=2, opacity=1).add_to(karte)

# Speichern der Karte als HTML
karte.save("karte4.html")

# Pfad zum geckodriver angeben
gecko_path = r"C:\WebDriver\geckodriver.exe"  # Ersetze durch den tatsächlichen Pfad zu geckodriver.exe

# Firefox Options und Service konfigurieren
options = Options()
options.headless = False  # Setze auf True, wenn du den Browser im Hintergrund ausführen möchtest

# Service mit dem geckodriver-Pfad erstellen
service = Service(executable_path=gecko_path)

# Selenium WebDriver für Firefox initialisieren
driver = webdriver.Firefox(service=service, options=options)

# Screenshot als PNG im Arbeitsspeicher speichern
screenshot = driver.get_screenshot_as_png()

# Bild mit Pillow öffnen und speichern
image = Image.open(io.BytesIO(screenshot))
image.save('karte_screenshot4.png')  # Speichert das Bild als PNG

# WebDriver beenden
driver.quit()

img_data = karte._to_png(5)
img = Image.open(io.BytesIO(img_data))
img.save('Karte4.png')
