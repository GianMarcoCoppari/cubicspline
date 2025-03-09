# Cubic Spline Interpolation

Realizzazione di una libreria python contenente una classe `CubicSpline` che effettua l'interpolazione con polinomio cubico di un insieme di punti forniti come `numpy.array`. La determinazione dei coefficienti è eseguita risolvendo un sistema di equazioni lineari tridiagonale.

## Usage
### Run tests

Ogni cartella del progett deve contenere un file `__init__.py` in modo da considerare ogni file come modulo e ogni cartella package. Per include il file `lu.py`, presente nella cartella `code`, all'interno del corrispondente file di test `lu_test.py`, presente nella cartella `test`, basterà utilizzare la linea di codice `from ..code.lu import lu` per importare direttamente la funzione `lu()`.

I test sono stati eseguiti con il comando `pytest --cov=spline --cov-report html .\test\lu_test.py` ed il risultato è visionabile con il comando `.\htmlcov\index.html`.

Per eseguire i test è necessario utilizzare il relative import (dot notation) nel modulo `spline.py` presente nella cartella `.\code`.

```
from .lu import lu
from .tls import tls
```

### Importing Module in Python Shell or Script 

Per poter utilizzare i moduli nella shell python o per essere importati come moduli in altri file è necessario modificare le due import line del file `spline.py` in

```
from lu import lu
from tls import tls
```

rimuovendo così i relative imports.
Il file deve trovarsi nella stessa cartella del file `spline.py`, oppure occorre lanciare la shell python dalla stessa cartella del file `spline.py`.