# Abgabe 3 – Kapitel 7: Ensemble Learning und Random Forests



---

### 1. Fünf Modelle mit je 95 % Relevanz auf denselben Trainingsdaten – lassen sie sich kombinieren? Begründung.

Ja, das kann sich lohnen. Man fasst die fünf Modelle zu einem Voting-Klassifikator zusammen,
der oft besser abschneidet als jedes einzelne Modell. Besonders gut klappt das, wenn die Modelle
möglichst unterschiedlich sind und verschiedene Fehler machen, etwa weil sie auf unterschiedlichen
Algorithmen beruhen. Beim Mehrheitsvotum gleichen sich ihre Fehler dann gegenseitig aus.

Eine Einschränkung gibt es aber. Wenn man alle Modelle auf exakt denselben Daten und mit demselben
Verfahren trainiert hat, sind ihre Fehler vermutlich stark korreliert, und der Gewinn fällt klein
aus. Am meisten bringt die Kombination, wenn die Modelle unabhängige Fehler machen.

---

### 2. Worin unterscheiden sich Klassifikatoren mit Hard und Soft Voting?

- Hard Voting: Jeder Klassifikator gibt eine Klasse aus; gewählt wird die Klasse mit den meisten
  Stimmen (Mehrheitsentscheid).
- Soft Voting: Jeder Klassifikator gibt für jede Klasse eine Wahrscheinlichkeit aus; diese werden
  gemittelt, und die Klasse mit der höchsten mittleren Wahrscheinlichkeit gewinnt.

Soft Voting liefert oft die besseren Ergebnisse, weil es sichere Stimmen stärker gewichtet. Dafür
müssen aber alle Klassifikatoren überhaupt Wahrscheinlichkeiten ausgeben können (`predict_proba`).

---

### 3. Lässt sich Bagging über mehrere Server verteilen? Und Pasting, Boosting, Random Forests, Stacking?

- Bagging: Ja. Die einzelnen Prädiktoren sind voneinander unabhängig und lassen sich parallel auf
  mehreren Servern trainieren.
- Pasting: Ja, aus demselben Grund (unabhängige Prädiktoren).
- Random Forests: Ja, sie sind eine Form von Bagging mit Entscheidungsbäumen.
- Boosting: Nein, oder nur eingeschränkt. Boosting läuft sequenziell; jeder Prädiktor baut auf
  dem vorherigen auf, deshalb lässt sich das Training nicht einfach parallelisieren.
- Stacking: Teilweise. Die Prädiktoren einer Schicht sind voneinander unabhängig und lassen sich
  parallel trainieren. Die Schichten bauen aber aufeinander auf und müssen deshalb nacheinander
  trainiert werden.

---

### 4. Welchen Vorteil bietet die Out-of-Bag-Evaluation?

Beim Bagging sieht jeder Prädiktor nur eine zufällige Teilmenge der Trainingsdaten. Im Schnitt
bleiben etwa 37 % der Daten („out-of-bag") für diesen Prädiktor ungenutzt. Genau auf diesen nicht
gesehenen Daten kann man den Prädiktor bewerten. Der Vorteil: Man bekommt eine Schätzung der
Generalisierungsfähigkeit, ohne dafür einen separaten Validierungsdatensatz abzuzweigen. So bleibt
mehr Datenmaterial fürs Training übrig.

---

### 5. Wodurch werden Extra-Trees zufälliger als Random Forests? Wobei hilft das? Sind sie langsamer oder schneller?

- Mehr Zufall: Ein gewöhnlicher Random Forest sucht für jedes Merkmal den besten Schwellenwert.
  Extra-Trees (Extremely Randomized Trees) verwenden stattdessen zufällige Schwellenwerte und
  wählen davon den besten aus.
- Nutzen: Dieser zusätzliche Zufall erhöht die Verzerrung (Bias) etwas, senkt aber die Varianz.
  Das wirkt Overfitting entgegen und kann die Generalisierung verbessern.
- Geschwindigkeit: Extra-Trees trainieren schneller, weil die Suche nach dem optimalen
  Schwellenwert wegfällt. Genau das ist der rechenintensivste Teil.

---

### 6. AdaBoost underfittet – welche Hyperparameter wie ändern?

Bei Underfitting muss das Ensemble stärker werden:

- `n_estimators` erhöhen (mehr Prädiktoren).
- `learning_rate` erhöhen (jeder Prädiktor bekommt mehr Gewicht).
- Den Basis-Schätzer weniger einschränken (z. B. `max_depth` erhöhen), falls er zu schwach ist.

---

### 7. Gradient-Boosting overfittet – Lernrate erhöhen oder verringern?

Die Lernrate verringern. Eine kleinere `learning_rate`, kombiniert mit Early Stopping oder einer
passenden Zahl an Bäumen, regularisiert das Ensemble und verringert das Overfitting. Alternativ
kann man auch die Anzahl der Bäume reduzieren (Early Stopping).
