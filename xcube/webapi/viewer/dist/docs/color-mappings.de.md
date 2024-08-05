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

* **Kontinuierlich:** Kontinuierliche Farbzuordnung, bei der jeder 
  `<value>` eine Stützstelle eines Farbverlaufs darstellt.
* **Schrittweise:** Schrittweise Farbzuordnung, bei der die Werte 
  Bereichsgrenzen darstellen, die einer einzelnen Farbe zugeordnet werden.
  Eine `<color>` wird dem ersten `<value>` eines Grenzbereiches zugeordnet. 
  Der letzte Farbwert wird ignoriert.
* **Kategorisch:** Werte stellen eindeutige Kategorien oder Indizes dar, 
  die einer Farbe zugeordnet sind. Der Inhalt des Datensatzes sowie der 
  `<value>` muss dem Typ Integer entsprechen. Wenn eine Kategorie keinen 
  `<value>` in der Farbzuordnung hat, wird diese transparent dargestellt. 
  Geeignet für kategorische Datensätze. 
