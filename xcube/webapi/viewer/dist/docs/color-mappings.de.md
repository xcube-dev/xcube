Eine benutzerdefinierte Farbzuordnung ordnet Datenwerte oder Bereiche von 
Datenwerten Farbwerten zu. Die Zeilen im Textfeld haben die allgemeine 
Syntax `<value>`: `<color>`, wobei `<color>` sein kann:

* eine Liste von RGB-Werten, mit Werten im Bereich von 0 bis 255, zum Beispiel 
  `255,165,0` für die Farbe Orange;
* ein hexadezimaler RGB-Wert, z.B. `#FFA500`;
* oder ein gültiger [HTML-Farbname](https://www.w3schools.com/colors/colors_names.asp)
  wie `Orange`, `BlanchedAlmond` oder `MediumSeaGreen`.

Der Farbwert kann durch einen Deckungswert (Alpha-Wert) im Bereich von 0 bis 1 
ergänzt werden, zum Beispiel `110,220,230,0.5` oder `#FFA500,0.8` oder `Blue,0`. 
Hexadezimale Werte können auch mit einem Alpha-Wert geschrieben werden, wie `#FFA500CD`.

Die Interpretation des `<value>` hängt vom ausgewählten Farbzuordnungstyp ab

* Typ **Node**: Die Farbkodierung ist ein linearer Verlauf zwischen den Farben 
  mit relativen Abständen, die durch `<value>` angegeben werden, welche 
  die Kontrollwerte oder _Knoten_ darstellen.
* Typ **Bound**: Angrenzende Werte bilden Wertgrenzen (bounds), die der `<color>` 
  zugeordnet sind, die mit dem ersten `<value>` der Grenze verbunden ist. Der 
  letzte Farbwert wird daher ignoriert.
* Typ **Key**: Die Werte sind ganze Zahlen, die direkt die zugehörige Farbe 
  identifizieren. Sollte für Daten des Typs Integer verwendet werden. Zusammen 
  mit einer kategorischen Normalisierung **CAT** ermöglicht dieser Farbzuordnungstyp 
  auch eine kategorische Kartenlegende. 
