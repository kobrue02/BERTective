# BERTective
BERTective ist ein auf Deep Learning mit linguistischen Features basierendes Tool, welches für das Author Profiling anhand einer Texteingabe entwickelt wurde.

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
python main.py -pr """Weit hinten, hinter den Wortbergen, fern der Länder Vokalien und Konsonantien leben die Blindtexte. Abgeschieden wohnen sie in Buchstabhausen an der Küste des Semantik, eines großen Sprachozeans. Ein kleines Bächlein namens Duden fließt durch ihren Ort und versorgt sie mit den nötigen Regelialien. Es ist ein paradiesmatisches Land, in dem einem gebratene Satzteile in den Mund fliegen. Nicht einmal von der allmächtigen Interpunktion werden die Blindtexte beherrscht – ein geradezu unorthographisches Leben. Eines Tages aber beschloß eine kleine Zeile Blindtext, ihr Name war Lorem Ipsum, hinaus zu gehen in die weite Grammatik. Der große Oxmox riet ihr davon ab, da es dort wimmele von bösen Kommata, wilden Fragezeichen und hinterhältigen Semikoli, doch das Blindtextchen ließ sich nicht beirren. Es packte seine sieben Versalien, schob sich sein Initial in den Gürtel und machte sich auf den Weg. Als es die ersten Hügel des Kursivgebirges erklommen hatte, warf es einen letzten Blick zurück auf die Skyline seiner Heimatstadt Buchstabhausen, die Headline von Alphabetdorf und die Subline seiner eigenen Straße, der Zeilengasse. Wehmütig lief ihm eine rhetorische Frage über die Wange, dann setzte es seinen Weg fort. Unterwegs traf es eine Copy. Die Copy warnte das Blindtextchen, da, wo sie herkäme wäre sie zigmal umgeschrieben worden und alles, was von ihrem Ursprung noch übrig wäre, sei das Wort "und" und das Blindtextchen solle umkehren und wieder in sein eigenes, sicheres Land zurückkehren. Doch alles Gutzureden konnte es nicht überzeugen und so dauerte es nicht lange, bis ihm ein paar heimtückische Werbetexter auflauerten, es mit Longe und Parole betrunken machten und es dann in ihre Agentur schleppten, wo sie es für ihre Projekte wieder und wieder mißbrauchten. Und wenn es nicht umgeschrieben wurde, dann benutzen Sie es immernoch."""  
  
oder mit einer txt-Datei:  
python main.py -pr text_sample.txt   #der Dateiname ist egal
## Aufbau des Programms
![architecture](https://github.com/kobrue02/BERTective/blob/main/architecture.drawio.svg)
