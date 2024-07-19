Eine _Nutzer-Variable_ ist eine Variable, die durch einen _Namen_, einen 
_Titel_, _Einheiten_ und einen algebraischen _Ausdruck_ definiert wird, der 
verwendet wird, um die Arraydaten der Variable zu berechnen. Nutzer-Variablen 
werden dem aktuell ausgewählten Datensatz hinzugefügt und deren Ausdrücke 
werden im Kontext des ausgewählten Datensatzes ausgewertet.

**Name**: Ein Name, der innerhalb der Variablen des ausgewählten Datensatzes 
eindeutig ist. Der Name muss mit einem Buchstaben beginnen, gefolgt von 
Buchstaben oder Ziffern.

**Titel**: Optionaler Anzeigename der Variablen in der Benutzeroberfläche.

**Einheiten**: Optionale physikalische Einheiten der berechneten Datenwerte. 
Zum Beispiel werden Einheiten verwendet, um Zeitreihen zu gruppieren.

**Ausdruck**: Ein algebraischer Ausdruck, der verwendet wird, um die Datenwerte 
der Variablen zu berechnen. Die Syntax entspricht der von [Python-Ausdrücken](https://docs.python.org/3/reference/expressions.html). 
Der Ausdruck kann die folgenden Namen referenzieren:
- die Datensätze des aktuellen Datensatzes;
- die numpy-Konstanten `e`, `pi`, `nan`, `inf`;
- alle [numpy ufunc](https://numpy.org/doc/stable/reference/ufuncs.html) Funktionen;
- die [`where`](https://docs.xarray.dev/en/stable/generated/xarray.where.html) Funktion.

Die Mehrheit der numerischen und logischen Operatoren von Python wird 
unterstützt. Jedoch können die logischen Operatoren `and`, `or` und `not` nicht 
mit Array-Variablen verwendet werden, da sie boolesche Werte als Operanden 
erfordern. Stattdessen können die bitweisen Operatoren `&`, `|`, `~` oder 
die entsprechenden Funktionen `logical_and()`, `logical_or()` und `logical_not()` 
verwendet werden. Auch eingebaute Python-Funktionen wie `min()` und `max()` werden 
nicht unterstützt. Alternativen sind hier `fmin()` und `fmax()`.

Ausdrucksbeispiele:

- Maskieren, wo eine Variable `chl` kleiner als null ist: `where(chl >= 0, chl, nan)`
- Sentinel-2 Vegetationsindex oder NDVI: `(B08 - B04) / (B08 + B04)`
- Sentinel-2 Feuchtigkeitsindex: `(B8A - B11) / (B8A + B11)`

Invalide Ausdrücke geben eine Fehlermeldung zurück.

STRG+LEER: aktiviert die Autovervollständigung, welche verfügbare 
Python-Funktionen und Konstanten auflistet