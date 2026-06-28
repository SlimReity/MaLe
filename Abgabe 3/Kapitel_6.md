# Abgabe 3 – Kapitel 6: Entscheidungsbäume



---

### 1. Was ist die ungefähre Tiefe eines mit 1 Million Datenpunkten (ohne Restriktionen) trainierten Entscheidungsbaums?

Ein unbeschränkter, ausbalancierter Binärbaum hat ungefähr die Tiefe log₂(m). Für m = 1.000.000 ergibt das

$$\log_2(10^6) \approx 20.$$

Also etwa Tiefe 20. In der Praxis wird es ein bisschen mehr, weil die Bäume selten perfekt ausbalanciert sind.

---

### 2. Ist die Gini-Unreinheit eines Knotens im Allgemeinen geringer oder grösser als die seines Elternteils? Ist sie im Allgemeinen kleiner/grösser oder immer kleiner/grösser?

Im Allgemeinen ist sie geringer als die des Elternknotens. Der CART-Algorithmus teilt einen Knoten so, dass die gewichtete Summe der Gini-Unreinheiten beider Kindknoten minimal wird, und diese Summe liegt unter der des Elternteils.

Für jeden einzelnen Kindknoten gilt das aber nicht. Durch die Gewichtung mit der Knotengrösse kann ein einzelnes Kind durchaus eine höhere Unreinheit haben als das Elternteil, solange das andere Kind das überkompensiert. Garantiert kleiner ist nur die gewichtete Summe der Kinder.

---

### 3. Sollte man versuchen, max_depth zu senken, wenn ein Entscheidungsbaum einen Trainingsdatensatz overfittet?

Ja. Overfitting heisst, das Modell ist zu komplex. Senkt man `max_depth`, wird der Baum flacher und einfacher. Das regularisiert ihn und verringert das Overfitting.

---

### 4. Sollte man versuchen, die Eingabemerkmale zu skalieren, wenn ein Entscheidungsbaum die Trainingsdaten underfittet?

Nein. Entscheidungsbäume reagieren nicht auf die Skalierung der Merkmale, denn pro Merkmal betrachten sie nur Schwellenwerte. Skalieren ändert am Modell nichts und hilft deshalb auch nicht gegen Underfitting. Was hilft, ist die Regularisierung zu lockern, etwa `max_depth` zu erhöhen oder `min_samples_*` zu senken.

---

### 5. Wenn es eine Stunde dauert, einen Entscheidungsbaum mit 1 Million Datenpunkten zu trainieren, wie lange etwa wird das Trainieren eines weiteren Baums mit 10 Millionen Datenpunkten dauern?

Die Trainingskomplexität von CART liegt bei etwa O(n · m · log m). Geht man von 1 auf 10 Millionen Datenpunkte, wächst die Zeit um den Faktor

$$\frac{10 \cdot m \cdot \log_2(10\,m)}{m \cdot \log_2(m)} = 10 \cdot \frac{\log_2(10^7)}{\log_2(10^6)} \approx 10 \cdot \frac{23{,}25}{19{,}93} \approx 11{,}7.$$

Das Training dauert also rund 11,7 Stunden, grob 11 bis 12 Stunden.

---

### 6. Wenn es eine Stunde dauert, einen Entscheidungsbaum mit einem gegebenen Trainingsdatensatz zu trainieren, wie lange dauert es dann ungefähr, wenn sich die Anzahl der Merkmale verdoppelt?

Die Komplexität ist linear in der Anzahl der Merkmale n. Verdoppelt sich n, verdoppelt sich grob auch die Trainingszeit. Man landet also bei etwa 2 Stunden.
