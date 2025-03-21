# Interpolazione Polinomiale - Spline Cubica

Progetto python che realizza una libreria che realizza l'interpolazione di un insieme di punti con algoritmo spline. 
I coefficienti delle funzioni interpolanti sono calcolati risolvendo un sistema tridiagonale utilizzando un algoritmo di Decomposione LU ed un algoritmo di sostituzione indietro e avanti, in combinazione. 

## Struttura
Ciascuno dei tre algoritmi è implementato nel proprio modulo python all'interno della cartella `code`.  Il file `lu.py` contiene l'algoritmo per la Decomposizione LU, partendo da tre `numpy.array` che rappresentano le tre diagonali, il file `tls.py` contiene l'implementazione delle funzioni `backward` e `foreward`, contenenti il codice necessario per la sostituzione indietro e avanti, rispettivamente; inoltre contiene una funzione `solver`, che fa da involucro alle due precedenti. L'invocazione della funzione `solver` richiama nell'ordine opportuno le altre due funzioni per risolvere il sistema. Infine il file `spline.py` contiene l'implementazione della classe `CubicSpline`; nel costruttore della classe vengono calcolati i coefficienti delle funzioni interpolanti, mentre il metodo `eval` permette di calcolare il valore della spline un qualsiasi altro punto compreso tra il nodo iniziale e finale. 
Nella cartella `test` sono contenuti i test di verifica del codice, divisi per modulo. Se un modulo contiene più elementi da testare, il file contiene una sottoclasse della classe `unittest.TestCase` per ogni oggetto.

## Download e Utilizzo

Scarica la repository con il comando
```
git clone https://github.com/GianMarcoCoppari/cubicspline.git
```

Per utilizzare la libreria in un file python è necessario che questo sia nella stessa cartella `code`, o che i tre file siano disponibili in una delle cartelle in cui python cerca i moduli.


### Python Scripts
Per includere la libreria nello script `file.py` assicurati di inserire il file nella stessa cartella del file `spline.py` o di includere la cartella `code` nella lista di cartelle da cui Python importa i moduli. Il modulo `CubicSpline` sarà quindi disponibile importandolo con la seguente linea di codice
```
from spline import CubicSpline
```

## Features
L'obiettivo principarle del pacchetto è costruire una spline cubica, dato un insieme di punti sul piano e le appropriate condizioni al contorno sulle derivate prime della funzione. Tuttavia, il programma è anche in grado di effettuare la Decomposizione LU di matrici tridiagonali e risolvere sistemi lineari tridiagonali, ricevendo in input le tre diagonali principali che descrivono la matrice ed il sistema, rispettivamente.
Gli algoritmi sono implementati nei moduli `lu.py` e `tls.py`, rispettivamente.
