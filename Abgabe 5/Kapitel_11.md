# Abgabe 5 – Kapitel 11: Trainieren tiefer neuronaler Netze

> Theoriefragen aus *Hands-On Machine Learning* (Aurélien Géron), Kapitel 11.
> Die Programmieraufgabe (DNN auf CIFAR-10) befindet sich im Notebook `Kapitel_11.ipynb`.

---

## 1. Welches Problem lösen Glorot- und He-Initialisierung?

Sie sollen instabile Gradienten beheben, also die verschwindenden und die explodierenden Gradienten. Sind die Gewichte ungünstig initialisiert, etwa mit zu großer Streuung, dann wächst oder schrumpft das Signal von Schicht zu Schicht. Im Rückwärtsdurchlauf gehen die Gradienten dann entweder gegen null oder sie explodieren.

Glorot und He wählen die Streuung der Anfangsgewichte so, dass die Varianz der Ausgaben jeder Schicht etwa der Varianz ihrer Eingaben entspricht und die Gradienten beim Rückwärtsdurchlauf ihre Größe ungefähr behalten. So fließt das Signal stabil durch das Netz, und das Training konvergiert schneller und zuverlässiger. (Faustregel: Glorot für `tanh`/Sigmoid/Softmax, He für ReLU und Varianten.)

---

## 2. Dürfen alle Gewichte gleich initialisiert werden, wenn der Wert per He zufällig gewählt wird?

Nein. Setzt man alle Gewichte auf denselben Wert, selbst wenn dieser zufällig per He-Initialisierung bestimmt wurde, dann sind alle Neuronen einer Schicht identisch. Sie erhalten dieselben Eingaben, berechnen dasselbe und bekommen beim Backpropagation dieselben Gradienten, also dieselbe Aktualisierung. Damit bleiben sie für immer gleich. Diese Symmetrie lässt sich nie brechen, und die Schicht verhält sich effektiv wie ein einziges Neuron.

Entscheidend ist, dass die Gewichte unabhängig und zufällig gezogen werden, damit die Neuronen unterschiedliche Muster lernen. He legt nur die Streuung der Zufallsverteilung fest, nicht einen einzelnen festen Wert.

---

## 3. Dürfen die Bias-Terme mit 0 initialisiert werden?

Ja, das ist völlig in Ordnung und sogar üblich. Die Symmetrie zwischen den Neuronen bricht schon die zufällige Initialisierung der Gewichte; die Biases müssen dazu nichts beitragen. Biases mit 0 zu starten ist deshalb unproblematisch und gängige Praxis.

---

## 4. Wann verwendet man die in diesem Kapitel besprochenen Aktivierungsfunktionen?

- ReLU: solider Standard für die verborgenen Schichten, einfach, schnell zu berechnen und in der Praxis robust. Gute erste Wahl.
- Leaky ReLU (und Varianten wie RReLU, PReLU): wenn man das Problem der sterbenden Neuronen (dying ReLUs) vermeiden möchte; etwas mehr Aufwand, oft etwas besser als ReLU.
- ELU / SELU: wenn es auf bessere Genauigkeit ankommt und die Rechenzeit weniger zählt.
  SELU eignet sich besonders für selbstnormalisierende Netze (nur dichte Schichten, LeCun-Init,
  standardisierte Eingaben); dann braucht man oft keine Batch Normalization.
- Swish: gute Wahl für sehr tiefe Netze, oft etwas besser als ReLU.
- tanh: in den verborgenen Schichten kaum noch gebräuchlich, manchmal in der Ausgabeschicht,
  wenn ein Wert in $(-1, 1)$ ausgegeben werden soll.
- Sigmoid (Logistik): in der Ausgabeschicht bei binärer Klassifikation oder
  Mehrfachlabel-Klassifikation (Wahrscheinlichkeiten).
- Softmax: in der Ausgabeschicht bei Mehrklassen-Klassifikation (exklusive Klassen).

Wenn die Laufzeit kritisch ist, bleibt ReLU die pragmatische Wahl.

---

## 5. Momentum zu nah an 1 (z. B. 0,99999) beim SGD-Optimierer

Bei sehr hohem Momentum klingen die vergangenen Gradienten kaum noch ab, und ihre Beiträge summieren sich über viele Schritte auf. Der Optimierer gewinnt dadurch enorm an „Schwung". In flachen Bereichen beschleunigt er stark, schießt aber über das Minimum hinaus, beginnt zu oszillieren und braucht lange, bis er sich wieder einpendelt. Das Training wird instabil und konvergiert langsamer oder gar nicht. Das Modell überschwingt das Optimum immer wieder, bevor es zur Ruhe kommt.

---

## 6. Drei Möglichkeiten, ein dünn besetztes Modell zu erzeugen

1. Das Modell normal trainieren und anschließend die winzigen Gewichte auf 0 setzen
   (Pruning / Schwellenwert).
2. Beim Training ℓ₁-Regularisierung anwenden, die viele Gewichte aktiv gegen 0 drückt.
3. TensorFlow Model Optimization Toolkit (bzw. ein Verfahren wie das duale Mittelwert-
   /Dual-Averaging-Verfahren, FTRL) einsetzen, das gezielt dünn besetzte Modelle erzeugt.

---

## 7. Verlangsamt Dropout das Training, die Inferenz, und was ist mit MC-Dropout?

- Training: Ja, Dropout verlangsamt das Training meist etwas (grob um den Faktor
  zwei pro Epoche bzw. man braucht mehr Epochen), weil in jedem Schritt zufällig Neuronen
  deaktiviert werden und das Netz dadurch langsamer konvergiert.
- Inferenz: Nein. Zur Vorhersagezeit ist Dropout abgeschaltet; die Vorhersagen
  laufen wie bei einem normalen Netz, ohne zusätzlichen Aufwand.
- MC-Dropout: Hier wird die Inferenz langsamer, denn man führt für jede Eingabe mehrere
  Vorhersagen mit aktivem Dropout durch (z. B. 100 Durchläufe) und mittelt sie. Die Inferenz
  dauert also etwa so viele Male länger, wie Durchläufe gemacht werden. Am Training ändert
  sich gegenüber normalem Dropout nichts.

---

## 8. Programmieraufgabe – DNN auf CIFAR-10

Die praktische Aufgabe (Teile a–f) ist im Notebook `Kapitel_11.ipynb` umgesetzt:
- a) DNN mit 20 verborgenen Schichten zu je 100 Neuronen, He-Initialisierung und
  Swish-Aktivierung.
- b) Training mit Nadam und Early Stopping auf CIFAR-10 (Softmax-Ausgabe, 10 Klassen),
  jeweils mit Lernratensuche.
- c) Batch Normalization ergänzen und die Lernkurven vergleichen.
- d) Batch Normalization durch SELU ersetzen (selbstnormalisierendes Netz: standardisierte
  Eingaben, LeCun-Initialisierung, nur dichte Schichten).
- e) Regularisierung mit Alpha-Dropout und anschließend MC-Dropout ohne erneutes
  Training.
- f) Erneutes Training mit 1cycle-Scheduling.
