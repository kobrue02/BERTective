# BERTective
BERTective ist ein auf Deep Learning mit linguistischen Features basierendes Tool, welches für die Profiling des Urhebers eines Textes entwickelt wurde.

## Anwendung
### Installation
git clone https://github.com/kobrue02/BERTective.git  
pip install -r requirements.txt  
### Download der Daten und Vorbereitung der Features
#### Trainingsdaten herunterladen
python main.py -dd -dw  
#### Korpus aufbauen
python main.py -b  
#### Features berechnen
python main.py -bw  
python main.py -bz  
python main.py -bo  
python main.py -bs  
### Training und Evaluation
python main.py -tr -edu -f ortho  
python main.py -tr -regio -f zdl -m rnn -n 20000
python main.py -tr -age -f all
python main.py -tr -gender -f all  
### Inference auf eigenen Texten
python main.py -pr """Er hörte leise Schritte hinter sich. Das bedeutete nichts Gutes. Wer würde ihm schon folgen, spät in der Nacht und dazu noch in dieser engen Gasse mitten im übel beleumundeten Hafenviertel? Gerade jetzt, wo er das Ding seines Lebens gedreht hatte und mit der Beute verschwinden wollte! Hatte einer seiner zahllosen Kollegen dieselbe Idee gehabt, ihn beobachtet und abgewartet, um ihn nun um die Früchte seiner Arbeit zu erleichtern? Oder gehörten die Schritte hinter ihm zu einem der unzähligen Gesetzeshüter dieser Stadt, und die stählerne Acht um seine Handgelenke würde gleich zuschnappen?""  
  
oder mit einer txt-Datei:  
python main.py -pr text_sample.txt   #der Dateiname ist egal
## Aufbau des Programms
![architecture](https://github.com/kobrue02/BERTective/blob/main/architecture.drawio.svg)
