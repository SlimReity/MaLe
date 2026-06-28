# Abgabe 4 – Kapitel 8: Dimensionsreduktion

Theoriefragen aus *Hands-On Machine Learning* (Aurélien Géron), Kapitel 8. Die Programmieraufgaben (Aufgaben 9 und 10) befinden sich im Notebook `Kapitel_8.ipynb`.

---

### 1. Welche sind die wichtigsten Gründe, die Dimensionen eines Datensatzes zu verringern? Was sind die wichtigsten Nachteile?

Gründe:
- Schnelleres Training: Weniger Merkmale heißt in der Regel deutlich kürzere Trainingszeiten.
- Geringerer Speicherbedarf.
- Visualisierung: Auf 2 oder 3 Dimensionen reduziert kann man Muster, Cluster und Strukturen mit dem Auge erkennen.
- Rauschen und überflüssige Merkmale fliegen raus, was die Modellleistung manchmal verbessert (aber eben nicht immer).
- Bekämpfung des Fluchs der Dimensionalität (siehe Frage 2).

Nachteile:
- Informationsverlust: Ein Teil der Information geht unwiederbringlich verloren, und das kann die Leistung der nachgelagerten Modelle verschlechtern.
- Rechenaufwand: Manche Verfahren wie t-SNE sind selbst sehr rechenintensiv.
- Zusätzliche Komplexität in der Verarbeitungspipeline. Man handelt sich einen weiteren Schritt mit eigenen Hyperparametern ein.
- Schwer interpretierbare Merkmale: Die transformierten Dimensionen, etwa die Hauptkomponenten, haben meist keine anschauliche Bedeutung mehr.

---

### 2. Was ist der Fluch der Dimensionalität?

Der Fluch der Dimensionalität beschreibt, dass sich Daten in hochdimensionalen Räumen völlig anders verhalten als in unserer gewohnten 2- oder 3-dimensionalen Welt.

Mit steigender Dimensionszahl wird der Raum enorm gross, und die Trainingsdaten liegen darin extrem dünn verteilt (spärlich). Zwei zufällig gewählte Punkte sind im Mittel sehr weit voneinander entfernt. Eine neue Instanz liegt deshalb in der Regel weit von allen Trainingsbeispielen entfernt, und Vorhersagen beruhen eher auf Extrapolation als auf Interpolation. Sie werden damit viel unzuverlässiger, und die Modelle neigen stärker zum Overfitting.

Um trotz vieler Dimensionen eine vergleichbare Datendichte zu erreichen, bräuchte man eine mit der Dimensionszahl exponentiell wachsende Menge an Trainingsdaten.

---

### 3. Ist die Dimensionalität eines Datensatzes erst einmal reduziert, ist es möglich, die Operation umzukehren? Falls ja, wie? Falls nein, warum nicht?

In der Regel nein, zumindest nicht exakt. Bei der Reduktion geht zwangsläufig Information verloren (die verworfenen Dimensionen), und die lässt sich nicht vollständig zurückgewinnen.

Einige Verfahren bieten allerdings eine inverse Transformation, die eine Rekonstruktion liefert, die dem Original möglichst nahekommt. Bei der PCA ist das `inverse_transform`. Diese Rekonstruktion ist aber nicht mit dem Original identisch. Die Differenz dazwischen ist der Rekonstruktionsfehler.

Andere Verfahren wie t-SNE besitzen gar keine inverse Transformation. Hier ist eine Umkehrung grundsätzlich nicht möglich.

---

### 4. Lässt sich die PCA einsetzen, um die Dimensionen eines hochgradig nichtlinearen Datensatzes zu verringern?

Das kommt darauf an. Die PCA kann auch bei nichtlinearen Daten überflüssige Dimensionen entfernen, sofern es solche gibt.

Gibt es aber keine überflüssigen Dimensionen, klassisches Beispiel ist die „Swiss Roll“, dann zerstört eine lineare Projektion mittels normaler PCA die innere Struktur der Daten und verliert zu viel Information.

Für stark nichtlineare Daten ist daher die Kernel-PCA besser geeignet, oder ein anderes nichtlineares Verfahren wie LLE.

---

### 5. Sie führen eine PCA auf einem 1.000-dimensionalen Datensatz durch und legen den Anteil der erklärten Streuung auf 95 % fest. Wie viele Dimensionen hat der sich daraus ergebende Datensatz?

Das lässt sich nicht allgemein sagen, denn es hängt vollständig vom Datensatz ab.

- Sind die Merkmale stark korreliert, genügen oft schon wenige Dutzend Dimensionen für 95 % der Streuung.
- Sind die Datenpunkte dagegen nahezu zufällig verteilt (kaum Korrelation), braucht man fast alle Dimensionen, bei 1.000 Merkmalen also etwa 950.

Die Antwort liegt somit irgendwo zwischen 1 und 1.000 und steht erst fest, nachdem man die PCA auf die konkreten Daten angewendet hat.

---

### 6. In welchen Fällen würden Sie eine normale Hauptkomponentenzerlegung durchführen, wann eine inkrementelle PCA, eine randomisierte PCA oder eine Zufallsprojektion?

- Normale (vollständige) PCA: der Standardfall, wenn der Datensatz in den Arbeitsspeicher passt.
- Inkrementelle PCA (IPCA): wenn der Datensatz zu groß für den Speicher ist; sie verarbeitet die Daten in Mini-Batches. Auch für Online-Aufgaben mit laufend eintreffenden Daten geeignet. Sie ist langsamer, deshalb bei ausreichend Speicher lieber die normale PCA nehmen.
- Randomisierte PCA: wenn der Datensatz in den Speicher passt und man die Dimensionszahl deutlich reduzieren möchte. Sie ist erheblich schneller als die vollständige PCA, sofern die Zieldimension viel kleiner als die ursprüngliche ist.
- Zufallsprojektion (Random Projection): bei sehr hochdimensionalen Daten; ein schnelles, einfaches Verfahren, das die paarweisen Abstände zwischen den Instanzen näherungsweise erhält.

---

### 7. Wie können Sie die Qualität eines Algorithmus zur Dimensionsreduktion auf Ihrem Datensatz bestimmen?

- Besitzt das Verfahren eine inverse Transformation, transformiert man die reduzierten Daten zurück und misst den Rekonstruktionsfehler (mittlerer quadratischer Abstand zwischen Original und Rekonstruktion). Ein kleiner Fehler spricht für eine gute Reduktion.
- Andernfalls (oder zusätzlich) bewertet man die Reduktion indirekt: Man setzt sie als Vorverarbeitungsschritt vor eine eigentliche ML-Aufgabe wie eine Klassifikation und misst deren Leistung. Bleibt die Leistung trotz deutlich weniger Dimensionen nahezu unverändert, war die Reduktion gut.

---

### 8. Ist es sinnvoll, zwei unterschiedliche Algorithmen zur Dimensionsreduktion hintereinanderzuschalten?

Ja, das kann durchaus sinnvoll sein. Ein gängiges Vorgehen: Zunächst entfernt eine schnelle PCA den Großteil der überflüssigen Dimensionen, danach folgt ein langsameres, aber leistungsfähigeres Verfahren wie LLE oder t-SNE.

Das Ergebnis ist meist sehr ähnlich, als hätte man nur das langsame Verfahren verwendet, aber bei deutlich kürzerer Laufzeit, weil dieses nun auf viel weniger Dimensionen arbeitet.

---

### 9. & 10. Programmieraufgaben

Die beiden praktischen Aufgaben sind im Notebook `Kapitel_8.ipynb` umgesetzt:

- Aufgabe 9: Random Forest auf MNIST, Dimensionsreduktion per PCA (95 % erklärte Streuung) und Vergleich der Laufzeiten und Genauigkeiten, zusätzlich mit einem `SGDClassifier`.
- Aufgabe 10: Visualisierung der ersten 5.000 MNIST-Bilder mit t-SNE sowie Vergleich mit PCA, LLE und MDS.
