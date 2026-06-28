# Abgabe 5 – Kapitel 10: Künstliche neuronale Netze mit Keras

> Theoriefragen und Übungen aus *Hands-On Machine Learning* (Aurélien Géron), Kapitel 10.
> Die abschließende Programmieraufgabe (MLP auf MNIST mit >98 % Genauigkeit) befindet sich im
> Notebook `Kapitel_10.ipynb`.

---

## 1. TensorFlow Playground

> Übungen mit dem Simulator unter <https://playground.tensorflow.org/>. Hier die
> Beobachtungen zu den einzelnen Experimenten.

### a) Die erlernten Muster
Das Standardnetz (zwei verborgene Schichten, `tanh`) findet sehr schnell eine gute
Lösung. In der ersten verborgenen Schicht lernen die Neuronen einfache Muster,
im Wesentlichen orientierte Geraden bzw. Halbebenen. Die zweite verborgene Schicht setzt
diese einfachen Muster zu komplexeren Formen zusammen. Je mehr Schichten, desto
komplexer werden die darstellbaren Muster (hierarchische Merkmalsbildung).

### b) Aktivierungsfunktionen (tanh → ReLU)
Mit ReLU statt `tanh` findet das Netz die Lösung noch schneller. Die
Entscheidungsgrenzen sind dann aber stückweise linear, also kantig statt rund. Das liegt an der Form
von ReLU: Sie besteht aus geraden Abschnitten, deshalb trennt das Netz die Daten mit einem Polygonzug
aus Geraden statt mit glatten Kurven.

### c) Risiko lokaler Minima (1 Schicht, 3 Neuronen)
Reduziert man auf eine verborgene Schicht mit drei Neuronen und trainiert mehrfach (mit
Reset dazwischen), schwankt die Trainingsdauer stark. Manchmal bleibt das Training in einem
lokalen Minimum hängen und findet keine gute Lösung. Schuld ist die zufällige
Initialisierung der Gewichte. Bei einem kleinen Netz ist die Verlustlandschaft so beschaffen,
dass je nach Startpunkt unterschiedliche Minima erreicht werden, auch schlechte.

### d) Zu kleine Netze (2 Neuronen)
Mit nur zwei Neuronen findet das Netz selbst nach vielen Versuchen keine gute Lösung
mehr. Das Modell hat zu wenige Parameter und underfittet die Daten systematisch. Die
Kapazität reicht nicht aus, um die Klassenstruktur abzubilden.

### e) Ausreichend große Netze (8 Neuronen)
Mit acht Neuronen trainiert das Netz durchgängig schnell und bleibt nie stecken. Das
illustriert eine wichtige Erkenntnis: Große Netze bleiben nur selten in lokalen Minima
hängen, und falls doch, sind diese lokalen Optima oft fast so gut wie das globale Optimum.
Sie können allerdings lange auf Plateaus verweilen.

### f) Verschwindende Gradienten in tiefen Netzen (Spirale, 4 × 8)
Beim Spiral-Datensatz mit vier verborgenen Schichten zu je acht Neuronen dauert das
Training deutlich länger und verharrt oft lange auf Plateaus. Die Neuronen der hinteren
Schichten (rechts) entwickeln sich schneller als die der vorderen (links). Das ist ein Zeichen für
verschwindende Gradienten: Der Fehler wird beim Rückwärtsdurchlauf immer kleiner, sodass die
vorderen Schichten kaum noch lernen. Abhilfe schaffen bessere Gewichtsinitialisierung, andere
Aktivierungsfunktionen, bessere Optimierer (AdaGrad, Adam) oder Batch Normalization (Kapitel 11).

### g) Eigene Experimente
Beim freien Experimentieren zeigt sich unter anderem: Eine zu hohe Lernrate lässt den Verlust
divergieren, eine zu niedrige macht das Training extrem langsam. Mehr Merkmale (z. B.
$x_1 x_2$, $\sin x_1$) erleichtern manche Aufgaben enorm. Regularisierung (L1/L2) glättet die
Entscheidungsgrenze und verhindert Overfitting, sichtbar am Abstand zwischen Trainings- und
Testverlust. Mehr Rauschen im Datensatz erhöht die Overfitting-Gefahr.

---

## 2. Ein ANN für A ⊕ B (XOR) mit klassischen Neuronen

Mit der Identität $A \oplus B = (A \land \lnot B) \lor (\lnot A \land B)$ lässt sich XOR mit
zwei verborgenen Neuronen und einem Ausgabeneuron bauen. Dafür verwende ich
Threshold Logic Units (Stufenfunktion: Ausgabe 1, falls die gewichtete Summe $\ge 0$, sonst 0).

```
        Eingaben          verborgene Schicht        Ausgabe

          A ──+1────────────►( h1 )──+1──┐
           \                  A∧¬B        \
            \  -1                          +1
             \                              ►( y )   y = h1 ∨ h2 = A ⊕ B
            -1\                            /
              \                          +1
          B ───┴──+1────────►( h2 )──────┘
                              ¬A∧B
```

Gewichte und Schwellen (Bias):

| Neuron | Funktion        | $w_A$ | $w_B$ | Bias | feuert bei |
|--------|-----------------|:-----:|:-----:|:----:|------------|
| $h_1$  | $A \land \lnot B$ | $+1$  | $-1$  | $-1$ | $(A,B)=(1,0)$ |
| $h_2$  | $\lnot A \land B$ | $-1$  | $+1$  | $-1$ | $(A,B)=(0,1)$ |

| Neuron | $w_{h_1}$ | $w_{h_2}$ | Bias | feuert bei |
|--------|:---------:|:---------:|:----:|------------|
| $y$    | $+1$      | $+1$      | $-1$ | $h_1=1 \lor h_2=1$ |

Damit gilt $y = \text{step}(h_1 + h_2 - 1)$, also $y = h_1 \lor h_2 = (A \land \lnot B) \lor
(\lnot A \land B) = A \oplus B$.

Wahrheitstabelle zur Kontrolle:

| A | B | $h_1$ | $h_2$ | $y$ |
|:-:|:-:|:-----:|:-----:|:---:|
| 0 | 0 | 0 | 0 | 0 |
| 0 | 1 | 0 | 1 | 1 |
| 1 | 0 | 1 | 0 | 1 |
| 1 | 1 | 0 | 0 | 0 |

---

## 3. Logistische Regression vs. klassisches Perzeptron

Das klassische Perzeptron verwendet die Stufenfunktion und gibt nur eine harte
Klasse aus (0 oder 1), also keine Wahrscheinlichkeiten. Trainiert wird es mit der Perzeptron-Lernregel,
und es konvergiert nur bei linear separierbaren Daten. Sonst pendelt es ohne
Garantie. Ein Maß für die Sicherheit einer Vorhersage liefert es ebenfalls nicht.

Die logistische Regression gibt dagegen für jede Instanz eine Klassenwahrscheinlichkeit
aus. Sie minimiert eine konvexe Kostenfunktion (Log-Loss / Kreuzentropie) per Gradientenabstieg
und ist daher zuverlässiger und informativer.

Umbau des Perzeptrons zu einer logistischen Regression: Man ersetzt die Stufenfunktion
durch die Sigmoid-(Logistik-)Funktion als Aktivierung und trainiert das Neuron, indem man
die Kreuzentropie per Gradientenabstieg minimiert, statt der Perzeptron-Lernregel. Eine
einzelne TLU mit logistischer Aktivierung entspricht dann genau einer logistischen Regression.

---

## 4. Warum war die Sigmoid-Funktion eine Schlüsselkomponente?

Das Trainieren von MLPs beruht auf Backpropagation, also auf dem Berechnen von Gradienten.
Die Stufenfunktion des Perzeptrons hat aber überall einen Gradienten von 0, und an der
Sprungstelle ist sie nicht differenzierbar. Damit kann der Gradientenabstieg nicht arbeiten,
es gibt keine nutzbare Steigung.

Die Sigmoid-Funktion ist dagegen überall glatt und differenzierbar und hat eine
wohldefinierte, von Null verschiedene Ableitung. So können die Gradienten durch das Netz
zurückfließen und die Gewichte sinnvoll angepasst werden. Sie ist außerdem nichtlinear,
sodass mehrere Schichten überhaupt erst zusätzliche Ausdruckskraft gewinnen.

---

## 5. Drei verbreitete Aktivierungsfunktionen

| Funktion | Formel | Wertebereich | Form |
|----------|--------|--------------|------|
| Sigmoid (Logistik) | $\sigma(z)=\dfrac{1}{1+e^{-z}}$ | $(0,\,1)$ | S-förmig, sättigt für große $\lvert z\rvert$ |
| Tanh | $\tanh(z)=\dfrac{e^{z}-e^{-z}}{e^{z}+e^{-z}}$ | $(-1,\,1)$ | S-förmig, um 0 zentriert |
| ReLU | $\operatorname{ReLU}(z)=\max(0,\,z)$ | $[0,\,\infty)$ | 0 für $z<0$, danach Gerade mit Steigung 1 |

Ein Diagramm dieser drei Funktionen entsteht im Notebook `Kapitel_10.ipynb` (Abschnitt
„Aktivierungsfunktionen"). Skizzenhaft:

```
 Sigmoid              Tanh                 ReLU
 1 ┤      ____        1 ┤      ____         ┤        /
   │    /              │    /               │      /
0.5┤   /             0 ┤───/────            │    /
   │  /                │  /                 │   /
 0 ┤_/              -1 ┤_/                 0 ┤__/______
   └───────────        └───────────         └───────────
```

---

## 6. Abmessungen eines MLP (10 → 50 → 3, ReLU)

Sei $m$ die Anzahl der Instanzen in einem Batch.

| Größe | Abmessung | Erläuterung |
|-------|-----------|-------------|
| Eingabematrix $X$ | $m \times 10$ | eine Zeile pro Instanz, 10 Merkmale |
| Gewichte $W_h$ (verborgen) | $10 \times 50$ | von 10 Eingaben zu 50 Neuronen |
| Bias $b_h$ | Länge $50$ | ein Wert pro verborgenem Neuron |
| Gewichte $W_o$ (Ausgabe) | $50 \times 3$ | von 50 Neuronen zu 3 Ausgaben |
| Bias $b_o$ | Länge $3$ | ein Wert pro Ausgabeneuron |
| Ausgabematrix $Y$ | $m \times 3$ | eine Zeile pro Instanz, 3 Ausgaben |

Gleichung für die Ausgabe (alle Neuronen mit ReLU):

$$
Y = \operatorname{ReLU}\!\Big( \operatorname{ReLU}\big( X\,W_h + b_h \big)\, W_o + b_o \Big)
$$

Die Bias-Vektoren werden dabei zeilenweise auf jede Instanz addiert (Broadcasting).

---

## 7. Anzahl der Ausgabeneuronen und Aktivierungsfunktion

| Aufgabe | Ausgabeneuronen | Aktivierung der Ausgabeschicht |
|---------|:---------------:|--------------------------------|
| Spam / Ham (binär) | 1 | Sigmoid (Wahrscheinlichkeit für „Spam") |
| MNIST (10 Ziffern, exklusiv) | 10 | Softmax (Wahrscheinlichkeitsverteilung über die Klassen) |
| Immobilienpreise (Regression, Kap. 2) | 1 | keine (lineare/identische Aktivierung) |

Bei der Spam-Klassifikation genügt ein Neuron, denn es gibt nur zwei sich gegenseitig ausschließende
Klassen. MNIST hat 10 exklusive Klassen, daher Softmax. Die Vorhersage eines einzelnen
Preises ist eine Regression mit einem reellwertigen Ausgang, also ein Neuron ohne
Aktivierungsfunktion.

---

## 8. Backpropagation und Reverse-Modus-Autodiff

Backpropagation ist das Standardverfahren zum Trainieren neuronaler Netze. Pro Mini-Batch
laufen folgende Schritte ab:

1. Vorwärtsdurchlauf: Die Daten werden durch das Netz geschickt, alle Zwischenausgaben
   werden gespeichert, und am Ende wird der Verlust berechnet.
2. Rückwärtsdurchlauf: Mit der Kettenregel wird – von der Ausgabe zurück zur Eingabe –
   der Gradient des Verlusts bezüglich jedes Parameters bestimmt.
3. Gradientenschritt: Alle Gewichte und Biases werden in Richtung des negativen Gradienten
   angepasst (Gradientenabstieg).

Diese drei Schritte wiederholen sich über viele Epochen.

Unterschied zu Reverse-Modus-Autodiff: Reverse-Modus-Autodiff ist lediglich die
Technik, mit der die Gradienten im Rückwärtsdurchlauf effizient berechnet werden.
Backpropagation ist das vollständige Trainingsverfahren = Reverse-Modus-Autodiff zur
Gradientenberechnung plus der anschließende Gradientenabstiegsschritt zur
Parameteraktualisierung. Autodiff ist also der allgemeinere Baustein, Backpropagation seine
konkrete Anwendung beim Training von Netzen.

---

## 9. Hyperparameter eines einfachen MLP und Gegenmaßnahmen bei Overfitting

Veränderbare Hyperparameter:
- Anzahl der verborgenen Schichten
- Anzahl der Neuronen pro Schicht
- Aktivierungsfunktion (verborgene Schichten und Ausgabeschicht)
- Gewichtsinitialisierung
- Optimierer und dessen Parameter (z. B. Momentum)
- Lernrate
- Batch-Größe
- Anzahl der Epochen
- Regularisierung: L1/L2, Dropout-Rate, Early Stopping

Bei Overfitting verringert man die Kapazität oder verstärkt die Regularisierung,
etwa:
- weniger verborgene Schichten und/oder Neuronen,
- Dropout hinzufügen bzw. die Dropout-Rate erhöhen,
- L1/L2-Regularisierung hinzufügen bzw. verstärken,
- Early Stopping verwenden,
- ggf. die Lernrate anpassen und die Anzahl der Epochen begrenzen.

---

## 10. Programmieraufgabe – MLP auf MNIST (> 98 %)

Die praktische Aufgabe ist im Notebook `Kapitel_10.ipynb` umgesetzt:
- Aufbau eines tiefen MLP auf MNIST,
- Bestimmung der optimalen Lernrate über exponentiell wachsende Lernrate und Verlustkurve,
- Hyperparameter-Optimierung mit Keras Tuner,
- Checkpoints, Early Stopping und Lernkurven mit TensorBoard.
