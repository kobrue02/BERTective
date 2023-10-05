# BERTective
BERTective ist ein auf Deep Learning mit linguistischen Features basierendes Tool, welches für das Author Profiling anhand einer Texteingabe entwickelt wurde.

## Anwendung
### Installation
`git clone https://github.com/kobrue02/BERTective.git`  
`pip install -r requirements.txt`  
### Download der Daten und Vorbereitung der Features
#### Trainingsdaten herunterladen
`python main.py -dd -dw`  
Mit `-dd` werden die Trainingsdaten heruntergeladen und mit `-dw` die Wiktionary-Wortlisten.
#### Korpus aufbauen
`python main.py -b`  
`-b` generiert aus den verfügbaren Trainingsdaten eine DataCorpus-Instanz, welche für die weitere Nutzung verwendet wird und alle Texte samt Autoren-Annotationen enthält. Das DataCorpus wird in einer AVRO-Datei gespeichert.
#### Features berechnen
`python main.py -bw`    
Mit `-bw` werden mithilfe der Wiktionary-Wortlisten Vektorrepräsentationen für jeden Text im DataCorpus generiert. Diese werden in einer JSON-Datei gespeichert.  
`python main.py -bz`  
`-bz` generiert mit dem ZDL-Regionalkorpus für jeden Text ein zweidimensionales Array, welches die PPM-Werte jedes Wortes im jeweiligen Text enthält. Die erzeugten Vektoren werden in pickle-Dateien gespeichert.  
`python main.py -bo`    
`-bo` generiert Vektoren, welche anhand von öffentlichen Wortlisten auf korrekturen.de bestimmte Schreibweisen einiger Begriffe repräsentieren. Die Ergebnisse werden in einer JSON-Datei gespeichert.  
`python main.py -bs`  
`-bs` greift lokal auf eine Python-Klasse zu, mithilfe welcher einige statistische Eigenschaften der Texte berechnet werden, die ebenfalls für das Training genutzt werden können.
### Training und Evaluation
`python main.py -tr -edu -f ortho`  
`python main.py -tr -regio -f zdl -m rnn -n 20000`  
`python main.py -tr -age -f all`  
`python main.py -tr -gender -f all` 
Mit `-tr` wird ein neues Modell trainiert. Die Flags `-edu`, `-regio`, `-age`und `-gender` bestimmen, für welche Autor-Eigenschaft das Modell trainiert werden soll, und mit `-f` können die Features gewählt werden, die verwendet werden sollen: `ortho`, `zdl`, `wikt`, `stat` und `all`. Mit `-m` kann zwischen einem statistischen multiclass-Classifier und einem RNN-Modell gewählt werden. RNN ist aktuell nur für Regiolektvorhersage verfügbar.
### Inference auf eigenen Texten
`python main.py -pr "Dies ist ein Text, dessen Autor leider unbek…"`  
  
oder mit einer txt-Datei:  
`python main.py -pr text_sample.txt`  
(Es kann ein beliebiger Dateiname gewählt werden.)
## Aufbau des Programms
![architecture](https://github.com/kobrue02/BERTective/blob/main/architecture.drawio.svg)
