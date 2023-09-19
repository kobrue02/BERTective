# BERTective
author profiling software for BND summer of code  
pip install -r requirements.txt  
Das Projekt befindet sich noch in der tiefsten Entwicklungsphase, daher gibt es noch kein zentrales ausführbares Programm.  
  
![architecture](https://github.com/kobrue02/BERTective/blob/main/architecture.drawio.svg)
[UPDATE] main.py lässt sich ausführen, Dokumentation folgt.    
`python main.py -dd' um Daten herunterzuladen, 'python main.py -z' um Vektoren zu generieren.
(mit crawl_all_datasets.py lassen sich die Daten crawlen, daher werden die .json-Dateien nach und nach aus dem Repository entfernt.) 
Nicht optimiert für Mac OSX.  

downloading all data and setting up BERTective:  
`pip install -r requirements.txt`   
`python main.py -t -dd -dw`   
`python main.py -t -b -s`  
`python main.py -t -bo -bw -bz -bs`  
`python main.py -t -tr -gender -f ortho -m multiclass -src GUTENBERG ACHGUT`

