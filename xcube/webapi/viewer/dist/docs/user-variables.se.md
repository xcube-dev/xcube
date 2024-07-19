En _användarvariabel_ är en variabel som definieras av ett _namn_, en _titel_, 
_enheter_ och ett algebraisk _uttryck_ som används för att beräkna variabelns 
arraydata. Användarvariabler läggs till den för närvarande valda datasetet 
och deras uttryck utvärderas i kontexten av det valda datasetet.

**Namn**: Ett namn som är unikt inom det valda datasetets variabler. Namnet 
måste börja med en bokstav, följt av bokstäver eller siffror.

**Titel**: Valfritt visningsnamn för variabeln i användargränssnittet.

**Enheter**: Valfria fysiska enheter för de beräknade datavärdena. Till exempel 
används enheter för att gruppera tidsserier.

**Uttryck**: Ett algebraisk uttryck som används för att beräkna variabelns 
datavärden. Syntaxen är densamma som för [Python-uttryck](https://docs.python.org/3/reference/expressions.html). 
Uttrycket kan referera till följande namn:
- de aktuella datasetets datavariabler;
- numpy-konstanterna `e`, `pi`, `nan`, `inf`;
- alla [numpy ufunc](https://numpy.org/doc/stable/reference/ufuncs.html) funktioner;
- funktionen [`where`](https://docs.xarray.dev/en/stable/generated/xarray.where.html).

De flesta av Pythons numeriska och logiska operatorer stöds, men de logiska 
operatorerna `and`, `or` och `not` kan inte användas med arrayvariabler eftersom 
de kräver booleska värden som operander. Använd istället bitvisa operatorer `&`, `|`, `~` 
eller motsvarande funktioner `logical_and()`, `logical_or()` och `logical_not()`. 
Python inbyggda funktioner som `min()` och `max()` stöds inte, använd istället `fmin()` 
och `fmax()`.

Exempel på uttryck:

- Maskera där en variabel `chl` är lägre än noll: `where(chl >= 0, chl, nan)`
- Sentinel-2 vegetationsindex eller NDVI: `(B08 - B04) / (B08 + B04)`
- Sentinel-2 fuktindex: `(B8A - B11) / (B8A + B11)`

Ogiltiga uttryck returnerar ett felmeddelande.

CTRL+MELLANSLAG: aktiveras autofullständningsfunktionen som 
listar tillgängliga Python funktioner och konstanter