# Abgabe 1 – Kapitel 1: Die Machine-Learning-Landschaft



---

### 1. Wie würden Sie Machine Learning definieren?

Machine Learning ist ein Subset von Künstlicher Intelligenz. Es ist die Kunst, Computer so vorzubereiten, dass sie aus Daten lernen, ohne dass jeder einzelne Schritt explizit programmiert wird.

---

### 2. Können Sie vier Arten von Anwendungen nennen, für die Machine Learning gut geeignet ist?

ML ist besonders gut bei:

1. Komplexen Problemen, wo ein einfacher Algorithmus nicht reicht, wie bei der Spracherkennung
2. Aufgaben mit vielen verschachtelten Regeln, denn das Modell kann diese alle auf einmal anwenden
3. Umgebungen, die sich stets verändern, denn das Modell passt sich an
4. Wenn man viele Daten hat und daraus Erkenntnisse gewinnen will, wie beim Data Mining

---

### 3. Was ist ein gelabelter Trainingsdatensatz?

Ein Trainingsdatensatz, bei dem das gewünschte Ergebnis komplett mitgeliefert wird.

---

### 4. Was sind die zwei verbreitetsten Aufgaben beim überwachten Lernen?

- Klassifikation: Vorhersage einer Kategorie/Klasse.
- Regression: Vorhersage eines numerischen Werts.

---

### 5. Können Sie vier häufig anzutreffende Aufgaben für unüberwachtes Lernen nennen?

1. Clustering: In einer Datenmenge Gruppierungen finden.
2. Dimensionsreduktion: Zu viele Dimensionen machen es nahezu unmöglich, die Daten schlau zu analysieren. Die Dimensionsreduktion wirkt dem entgegen.
3. Anomalie-/Ausreissererkennung.
4. Assoziationsregel-Lernen: Zusammenhänge zwischen Merkmalen aufdecken.


---

### 6. Was für einen Algorithmus würden Sie verwenden, um einen Roboter über verschiedene unbekannte Oberflächen laufen zu lassen?

Reinforcement Learning. Der Agent probiert Aktionen aus, erhält Belohnungen oder wird bestraft und lernt nach und nach eine optimale Strategie.

---

### 7. Welche Art Algorithmus würden Sie verwenden, um Ihre Kunden in unterschiedliche Gruppen einzuteilen?

- Clustering, wenn man nicht vorgibt, wie die Gruppen aussehen sollen. Der Algorithmus findet die Gruppen selbst.
- Klassifikation, wenn die Gruppen bereits definiert sind und man gelabelte Beispiele hat.

---

### 8. Würden Sie die Aufgabe, Spam zu erkennen, als überwachte oder unüberwachte Lernaufgabe einstufen?

Überwachtes Lernen. Das Modell wird mit vielen Beispielen trainiert, die bereits als „Spam“ bzw. „kein Spam“ gelabelt sind.

---

### 9. Was ist ein Onlinelernsystem?

Ein System, das inkrementell lernt: Die Daten werden sequenziell zugeführt, einzeln oder in kleinen Mini-Batches, und das Modell wird laufend aktualisiert.


---

### 10. Was ist Out-of-Core-Lernen?

Eine Technik für Datensätze, die nicht in den Arbeitsspeicher (RAM) passen. Der Algorithmus lädt jeweils nur einen Teil der Daten, trainiert auf diesem Mini-Batch und wiederholt das, bis alle Daten verarbeitet sind. Es nutzt oft Online-Learning-Techniken.

---

### 11. Welche Art Algorithmus beruht auf einem Ähnlichkeitsmass, um Vorhersagen zu treffen?

Instanzbasiertes Lernen. Das System merkt sich die Trainingsbeispiele und vergleicht neue Instanzen über ein Ähnlichkeitsmass mit ihnen, wie bei k-Nearest-Neighbors.

---

### 12. Worin besteht der Unterschied zwischen einem Modellparameter und einem Modellhyperparameter?

Modellparameter werden vom Modell aus den Daten gelernt und bestimmen die Vorhersage (z. B. die Gewichte und der Achsenabschnitt einer linearen Funktion).
- Hyperparameter: gehören zum Lernalgorithmus, nicht zum Modell. Sie werden vor dem Training festgelegt und nicht aus den Daten gelernt. Sie steuern, wie gelernt wird.

---

### 13. Wonach suchen modellbasierte Algorithmen? Welche Strategie führt am häufigsten zum Erfolg? Wie treffen sie Vorhersagen?

Sie suchen nach den optimalen Werten der Modellparameter, um das bestmögliche Ergebnis zu liefern. Die Strategie, die am häufigsten zum Erfolg führt, ist die Minimierung der Kostenfunktion.

---

### 14. Können Sie vier der wichtigsten Herausforderungen beim Machine Learning benennen?

1. Zu wenige Trainingsdaten.
2. Under- bzw. Overfitting.
3. Daten mit schlechter Qualität (Rauschen, Fehler, Ausreisser).
4. Nutzlose Features.


---

### 15. Welches Problem liegt vor, wenn Ihr Modell auf den Trainingsdaten eine sehr gute Qualität erbringt, aber schlecht auf neue Daten verallgemeinert? Nennen Sie drei Lösungsansätze.

Das ist Overfitting. Das Modell hat das Rauschen der Trainingsdaten auswendig gelernt. Lösungsansätze:

1. Einfacheres Modell wählen oder das Modell regularisieren.
2. Mehr Trainingsdaten sammeln.
3. Rauschen reduzieren, Fehler korrigieren, Ausreisser entfernen.

---

### 16. Was ist ein Testdatensatz, und warum sollte man einen verwenden?

Ein zurückgehaltener Teil der Daten, der zum Schätzen der Generalisierungsfaehigkeit auf neuen, ungesehenen Daten verwendet wird, bevor das Modell in Produktion geht. Man verwendet ihn, um einzuschätzen, wie gut das Modell auf neue Daten verallgemeinert.

---

### 17. Was ist der Zweck eines Validierungsdatensatzes?

Er dient dazu, Modelle zu vergleichen und Hyperparameter auszuwählen bzw. zu tunen, ohne den Testdatensatz zu verschwenden. So bleibt der Test eine unberührte Schätzung des Generalisierungsfehlers.

---

### 18. Was ist das Train-Dev-Set, wann brauchen Sie es, und wie verwenden Sie es?

Das Train-Dev-Set braucht man, wenn ein Mismatch droht zwischen den Trainingsdaten und den Daten, die in Produktion auftreten (z. B. Trainingsbilder aus dem Web, aber in Produktion dann Bilder aus einer App).

Man hält einen Teil der Trainingsdaten zurück (das Train-Dev-Set) und trainiert auf dem Rest.

---

### 19. Was kann schiefgehen, wenn Sie Hyperparameter mithilfe der Testdaten einstellen?

Das Modell wird zu sehr an den Testdatensatz angepasst (Data-Leakage). Der gemessene Test-Score wird dann zu optimistisch und ist keine verlässliche Schätzung. Deshalb wird das Validierungsset verwendet, um Hyperparameter zu tunen.
