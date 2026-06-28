# Abgabe 3 – Kapitel 5: Support Vector Machines



---

### 1. Was ist die den Support Vector Machines zugrunde liegende Idee?

Die Grundidee ist, zwischen den Klassen nicht irgendeine Trennlinie zu ziehen, sondern die
breitestmögliche „Strasse" (Large-Margin-Klassifikation). Die SVM sucht dafür die
Entscheidungsgrenze mit dem grössten Abstand (Margin) zu den nächstgelegenen Trainingsdatenpunkten
beider Klassen. Je breiter dieser Rand ausfällt, desto besser verallgemeinert das Modell in der
Regel auf neue Daten.

---

### 2. Was ist ein Stützvektor?

Ein Stützvektor (Support Vector) ist ein Trainingsdatenpunkt, der direkt auf dem Rand der Strasse
liegt, innerhalb davon oder auf der falschen Seite. Genau diese Punkte „stützen" die
Entscheidungsgrenze, denn allein sie bestimmen ihre Lage. Verschiebt oder entfernt man einen
Datenpunkt weit ausserhalb der Strasse, bleibt das Modell unverändert. Entfernt man dagegen einen
Stützvektor, verschiebt sich die Grenze.

---

### 3. Warum ist es wichtig, beim Verwenden von SVMs die Eingabedaten zu skalieren?

SVMs reagieren empfindlich auf die Skalen der Merkmale. Weil der Margin über Abstände definiert
ist, dominiert ein Merkmal mit grossem Wertebereich alle anderen, und die SVM ignoriert die
kleinskaligen Merkmale weitgehend. Nach einer Skalierung (z. B. `StandardScaler`) sind die
Merkmale vergleichbar, die Strasse wird breiter und die Entscheidungsgrenze deutlich besser.

---

### 4. Kann ein SVM-Klassifikator einen Konfidenzwert ausgeben, wenn er einen Datenpunkt klassifiziert? Wie sieht es mit einer Wahrscheinlichkeit aus?

Konfidenzwert: Ja. Die SVM liefert über `decision_function()` den vorzeichenbehafteten Abstand
des Datenpunkts zur Entscheidungsgrenze. Diesen Wert kann man als Konfidenzmass nutzen: Je weiter
weg, desto sicherer.

Wahrscheinlichkeit: Nicht direkt. Eine echte Wahrscheinlichkeit liefert die SVM von Haus aus
nicht. Schätzen lässt sie sich nachträglich über `SVC(probability=True)`, was per
Logistic/Platt-Kalibrierung mit Kreuzvalidierung geschieht. Das verlangsamt allerdings das
Training und kann den Vorhersagen leicht widersprechen.

---

### 5. Wie können Sie zwischen LinearSVC, SVC und SGDClassifier entscheiden?

- `LinearSVC`: für lineare Probleme. Skaliert sehr gut mit der Anzahl der Datenpunkte
  (≈ O(m·n)), unterstützt aber keinen Kerneltrick. Erste Wahl bei grossen, linear separierbaren
  Datensätzen.
- `SVC`: unterstützt den Kerneltrick (z. B. RBF) und damit nichtlineare Probleme. Skaliert
  jedoch schlecht (≈ O(m²·n) bis O(m³·n)) und wird bei vielen Datenpunkten langsam. Ideal also
  für kleine bis mittelgrosse Datensätze.
- `SGDClassifier`: trainiert per stochastischem Gradientenabstieg und eignet sich für sehr grosse
  Datensätze und Online-Lernen. Standardmässig linear (kein Kerneltrick), dafür speichereffizient.

Faustregel: linear + gross → `LinearSVC`/`SGDClassifier`; nichtlinear + klein/mittel → `SVC` mit Kernel.

---

### 6. Underfitting mit RBF-Kernel: γ (gamma) und C erhöhen oder senken?

Bei Underfitting ist das Modell zu stark eingeschränkt. Man muss es flexibler machen und deshalb
beide Hyperparameter erhöhen:

- γ (gamma) erhöhen: gamma steuert die Reichweite eines Trainingspunkts. Ein grösseres gamma
  macht die Entscheidungsgrenze enger und wendiger.
- C erhöhen: C steuert den Trade-off zwischen breitem Rand und Klassifikationsfehlern. Ein
  grösseres C erlaubt weniger Randverletzungen und passt das Modell stärker an die Daten an.

Bei Overfitting wäre es genau umgekehrt: beide senken.

---

### 7. Was bedeutet es für ein Modell, ε-insensitiv zu sein?

Der Begriff stammt aus der SVM-Regression. ε-insensitiv heisst: Fehler innerhalb eines Schlauchs
der Breite ε um die Vorhersage werden gar nicht bestraft, sie zählen nicht zur Kostenfunktion.
Nur Punkte ausserhalb dieses ε-Schlauchs tragen zum Fehler bei. Das macht das Modell unempfindlich
gegenüber kleinen Abweichungen, denn mehr Trainingspunkte innerhalb des Schlauchs ändern die
Vorhersage nicht.

---

### 8. Wofür wird der Kerneltrick eingesetzt?

Der Kerneltrick erlaubt es, eine SVM so zu trainieren, als hätte man die Merkmale in einen sehr
hochdimensionalen Raum transformiert (in dem die Daten linear separierbar werden), ohne diese
Transformation tatsächlich zu berechnen. Es genügt, eine Kernelfunktion (z. B. polynomial oder
RBF) auszuwerten, die dem Skalarprodukt im transformierten Raum entspricht. So werden
nichtlineare Probleme lösbar, ohne dass man Millionen neuer Merkmale explizit erzeugen und
speichern müsste.
