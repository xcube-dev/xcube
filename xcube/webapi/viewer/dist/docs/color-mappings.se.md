En användardefinierad färgkarta associerar datavärden eller intervall 
av datavärden med färgvärden. Raderna i textrutan har den allmänna 
syntaxen `<value>`: `<color>`, där `<color>` kan vara

* en lista med RGB-värden, med värden i intervallet 0 till 255, 
  till exempel, `255,165,0` för färgen Orange;
* ett hexadecimalt RGB-värde, t.ex. `#FFA500`;
* eller ett giltigt [HTML-färgnamn](https://www.w3schools.com/colors/colors_names.asp) 
  som `Orange`, `BlanchedAlmond` eller `MediumSeaGreen`.

Färgvärdet kan kompletteras med ett opacitetsvärde (alpha-värde) i 
intervallet 0 till 1, till exempel `110,220,230,0.5` eller `#FFA500,0.8` 
eller `Blue,0`. Hexadecimala värden kan också skrivas med ett alpha-värde, 
såsom `#FFA500CD`.

Tolkningen av `<value>` beror på den valda färgkartläggningstypen:

* **Kontinuerlig:** Kontinuerlig färgtilldelning där varje `<value>` 
  representerar en punkt i en färggradient.
* **Stegvis:** Stegvis färgmappning där värden är gränser för intervall. 
  En `<color>` associeras med det första `<value>` i gränsintervall, medan 
  den sista färgen ignoreras.
* **Kategorisk:** Värden representerar unika kategorier eller index som 
  är mappade till en färg. Innehållet i datasetet samt `<value>` måste 
  vara av typ Integer. Om en kategori inte har något `<value>` i 
  färgkartan, kommer den att visas som genomskinlig. Lämplig för 
  kategoriska dataset.
