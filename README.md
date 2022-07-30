# EM-Algorithmus für Hidden Markov Modelle

Der Code ist die Implementierung meiner Bachelorarbeit.

### EM-Algorithmus
Die Methode EM nimmt Beobachtungen *y*, Anzahl der Zustände *n* und Anzahl der Ausgabesymbole *m* entgegen und gibt die Übergangsmatrix *P*, die Startwahrscheinlichkeiten *pi*, die Emissionsmatrix *b* und die Likelihood zurück.

### Generieren von Beobachtungen
Die Methode sampleHMM nimmt die Übergangsmatrix *P*, die Emissionswahrscheinlichkeiten *b*, die Startsahrscheinlichkeiten *pi* und die Anzahl der zu generierenden Beobachtungen *T* entgegen und gibt T Beobachtungen *y* und die zugehörigen Zustände *z* zurück.

### Beispiele
Im Code sind Beispiele zur Anwendung der Methoden in Form von Test 1 bis Test 5 gegeben, wobei jedes Mal auch ein Plot erstellt wird. Dabei wird der EM-Algorithmus jeweils von 10 zufälligen Startposition gestartet.
