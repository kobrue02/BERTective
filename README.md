# BERTective
BERTective ist ein auf Deep Learning mit linguistischen Features basierendes Tool, welches für die Profiling des Urhebers eines Textes entwickelt wurde.

## Installation
git clone https://github.com/kobrue02/BERTective.git  
pip install -r requirements.txt  
#### Trainingsdaten herunterladen
python main.py -dd -dw  
#### Korpus aufbauen
python main.py -b  
#### Features berechnen
python main.py -bw  
python main.py -bz  
python main.py -bo  
python main.py -bs  

# Aufbau des Programms
![architecture](https://github.com/kobrue02/BERTective/blob/main/architecture.drawio.svg)
