# Abgabe 2 – Kapitel 4: Training von Modellen

> Theoriefragen aus *Hands-On Machine Learning* (Aurélien Géron), Kapitel 4.
> Die Programmieraufgabe (Aufgabe 12) befindet sich im Notebook `Kapitel_4.ipynb`.

---

### 1. Welchen Trainingsalgorithmus für die lineare Regression können Sie verwenden, wenn Sie einen Trainingsdatensatz mit Millionen Merkmalen haben?

Ein Gradientenverfahren. Am besten das stochastische oder das Mini-Batch-Gradientenverfahren, eventuell auch das Batch-Gradientenverfahren. Diese Verfahren skalieren gut mit der Anzahl der Merkmale.

Die Normalengleichung und die SVD scheiden hier aus. Ihr Rechenaufwand wächst sehr stark mit der Anzahl der Merkmale, und bei Millionen Merkmalen werden sie viel zu langsam.

---

### 2. Nehmen wir an, dass die Merkmale in Ihrem Trainingsdatensatz unterschiedlich skaliert sind. Welche Algorithmen würden dadurch in Mitleidenschaft gezogen und in welcher Weise? Was können Sie dagegen tun?

Alle Gradientenverfahren. Sind die Merkmale unterschiedlich skaliert, wird die Kostenfunktion zu einer langgestreckten Schüssel. Das Gradientenverfahren braucht dann viel länger, bis es das Minimum erreicht.

Dagegen hilft der `StandardScaler`. Wendet man ihn vorher an, konvergiert das Gradientenverfahren deutlich schneller.


---

### 3. Kann das Gradientenverfahren bei einem logistischen Regressionsmodell in einem lokalen Minimum stecken bleiben?

Nein. Die Kostenfunktion der logistischen Regression ist konvex. Eine konvexe Funktion hat keine lokalen Minima, sondern nur ein einziges globales Minimum.

---

### 4. Führen alle Algorithmen für das Gradientenverfahren zum gleichen Modell, vorausgesetzt, sie laufen lange genug?

Nein. Ist die Kostenfunktion konvex und die Lernrate nicht zu gross, landen alle Verfahren in der Nähe des globalen Minimums und liefern sehr ähnliche Modelle.

Aber das stochastische und das Mini-Batch-Gradientenverfahren kommen nie ganz zur Ruhe. Sie schwanken um das Minimum herum, und so bleiben die Modelle leicht unterschiedlich.

---

### 5. Nehmen wir an, Sie verwenden das Batch-Gradientenverfahren und plotten den Validierungsfehler in jeder Epoche. Was passiert vermutlich, wenn der Validierungsfehler ständig steigt? Wie können Sie dies beheben?

Steigt auch der Trainingsfehler, ist die Lernrate vermutlich zu hoch und der Algorithmus divergiert. Abhilfe: die Lernrate verkleinern.

Steigt dagegen nur der Validierungsfehler, während der Trainingsfehler weiter sinkt, dann overfittet das Modell. In diesem Fall sollte man das Training early stoppen.

---

### 6. Ist es eine gute Idee, das Mini-Batch-Gradientenverfahren sofort zu unterbrechen, sobald der Validierungsfehler steigt?

Nein. Das Mini-Batch-Verfahren (und ebenso das stochastische) ist verrauscht, der Validierungsfehler schwankt von Schritt zu Schritt. Ein kurzzeitiger Anstieg bedeutet nicht, dass man das Optimum überschritten hat.

Besser setzt man das Training fort, speichert regelmässig das beste Modell und bricht erst dann ab, wenn der Validierungsfehler über längere Zeit nicht mehr besser geworden ist. Am Ende verwendet man das gespeicherte beste Modell.

---

### 7. Welcher der besprochenen Algorithmen für das Gradientenverfahren erreicht die Umgebung der optimalen Lösung am schnellsten? Welcher konvergiert? Wie können Sie auch die übrigen konvergieren lassen?

- Am schnellsten in die Nähe des Optimums kommt das stochastische Gradientenverfahren. Es betrachtet
  jeweils nur eine Instanz und macht daher die meisten Schritte pro Zeit. Das Mini-Batch-Verfahren
  ist ebenfalls schnell.
- Tatsächlich konvergiert nur das Batch-Gradientenverfahren.
- Damit auch die übrigen konvergieren, verkleinert man die Lernrate allmählich (Learning Schedule). Dann
  hören die Schwankungen um das Minimum nach und nach auf.

---

### 8. Sie verwenden eine polynomielle Regression, plotten die Lernkurven und bemerken, dass es zwischen dem Trainingsfehler und dem Validierungsfehler einen grossen Unterschied gibt. Was passiert? Nennen Sie drei Möglichkeiten, dies zu beheben.

Ein grosser Abstand, also niedriger Trainingsfehler bei deutlich höherem Validierungsfehler, bedeutet Overfitting. Das Modell hat eine hohe Varianz.

Drei Möglichkeiten:
1. Das Modell vereinfachen, z. B. einen niedrigeren Polynomgrad wählen.
2. Das Modell regularisieren (z. B. Ridge, Lasso).
3. Mehr Trainingsdaten sammeln.

---

### 9. Bei der Ridge-Regression bemerken Sie, dass Trainingsfehler und Validierungsfehler beinahe gleich und recht hoch sind. Krankt dieses Modell an einem hohen Bias oder an einer hohen Varianz? Sollten Sie den Regularisierungsparameter α erhöhen oder senken?

Beide Fehler sind hoch und nahezu gleich. Das Modell hat also einen hohen Bias (Underfitting), keine hohe Varianz.

Man sollte den Regularisierungsparameter α verringern, damit das Modell flexibler wird und die Daten besser anpassen kann.

---

### 10. Welche Gründe sprechen für folgende Verfahren?

a) Ridge-Regression anstelle einer einfachen linearen Regression (ohne Regularisierung)?
Etwas Regularisierung ist fast immer besser als gar keine. Ridge verringert die Varianz und damit das Overfitting, und das Modell verallgemeinert in der Regel besser auf neue Daten.

b) Lasso anstelle einer Ridge-Regression?
Lasso (L1-Regularisierung) setzt die Gewichte unwichtiger Merkmale auf exakt null und erzeugt so ein dünn besetztes Modell (automatische Merkmalsauswahl). Das ist sinnvoll, wenn man vermutet, dass nur wenige Merkmale wirklich relevant sind. Andernfalls ist Ridge meist vorzuziehen.

c) Elastic Net anstelle von Lasso-Regression?
Lasso kann sich unberechenbar verhalten, wenn es mehr Merkmale als Instanzen gibt oder wenn Merkmale stark korreliert sind. Es wählt dann willkürlich eines aus. Elastic Net (eine Mischung aus L1 und L2) ist in diesen Fällen stabiler. Daher zieht man Elastic Net dem reinen Lasso meist vor.

---

### 11. Angenommen, Sie möchten Bilder als innen/aussen und Tag/Nacht klassifizieren. Sollten Sie zwei Klassifikatoren mit logistischer Regression oder einen Klassifikator mit Softmax-Regression erstellen?

Zwei logistische Regressionen. Die beiden Eigenschaften „innen/aussen“ und „Tag/Nacht“ schliessen sich nicht gegenseitig aus, ein Bild kann z. B. aussen und nachts sein. Es handelt sich also um zwei voneinander unabhängige binäre Klassifikationen.

Die Softmax-Regression passt nur für sich gegenseitig ausschliessende Klassen, wenn eine Instanz zu genau einer Klasse gehört. Hier ist das nicht der Fall.

---

### 12. Implementieren Sie das Batch-Gradientenverfahren mit Early Stopping für die Softmax-Regression ohne Scikit-Learn, sondern nur mit NumPy.

Siehe das Notebook `Kapitel_4.ipynb`.
