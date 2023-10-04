# BERTective
BERTective ist ein auf Deep Learning mit linguistischen Features basierendes Tool, welches für das Author Profiling anhand einer Texteingabe entwickelt wurde.

## Anwendung
### Installation
git clone https://github.com/kobrue02/BERTective.git  
pip install -r requirements.txt  
### Download der Daten und Vorbereitung der Features
#### Trainingsdaten herunterladen
`python main.py -dd -dw`  
Mit `-dd` werden die Trainingsdaten heruntergeladen und mit `-dw` die Wiktionary-Wortlisten.
#### Korpus aufbauen
`python main.py -b`  
`-b` generiert aus den verfügbaren Trainingsdaten eine DataCorpus-Instanz, welche für den weiteren Verlauf verwendet wird und alle Texte und Autor-Annotationen enthält. Der DataCorpus wird in einer AVRO-Datei gespeichert.
#### Features berechnen
`python main.py -bw`    
Mit `-bw` werden mithilfe der Wiktionary-Wortlisten Vektorrepräsentationen für jeden Text im DataCorpus generiert. Diese werden in einer JSON-Datei gespeichert.  
`python main.py -bz`  
`-bz` generiert mit dem ZDL-Regionalkorpus für jeden Text ein zweidimensionales Array, welches die PPM-Werte jedes Wortes im Text enthält. Die erzeugten Vektoren werden in pickle-Dateien gespeichert.  
`python main.py -bo`    
Mit dieser Flag werden Vektoren generiert, welche die sprachliche Qualität des Textes repräsentieren. Für die Generierung der Vektoren wird korrekturen.de verwendet und die Ergebnisse werden in einer JSON-Datei gespeichert.  
`python main.py -bs`  
`-bs` ruft die Klasse Statistext auf, mithilfe welcher eine Vielzahl statistischer Eigenschaften der Texte berechnet werden, die ebenfalls fürs Training genutzt werden können.
### Training und Evaluation
python main.py -tr -edu -f ortho  
python main.py -tr -regio -f zdl -m rnn -n 20000  
python main.py -tr -age -f all  
python main.py -tr -gender -f all 
Mit `-tr` wird ein neues Modell trainiert. Die Flags `-edu`, `-regio`, `-age`und `-gender` bestimmen, für welche Autor-Eigenschaft das Modell trainiert werden soll und mit `-f` können die Features gewählt werden, die verwendet werden sollen: ortho, zdl, wikt, stat und all. Mit `-m` kann zwischen einem statistischen multiclass-Classifier und einem RNN-Modell gewählt werden. RNN ist aktuell nur für Regiolektvorhersage verfügbar.
### Inference auf eigenen Texten
python main.py -pr """Weit hinten, hinter den Wortbergen, fern der Länder Vokalien und Konsonantien leben die Blindtexte. Abgeschieden wohnen sie in Buchstabhausen an der Küste des Semantik, eines großen Sprachozeans. Ein kleines Bächlein namens Duden fließt durch ihren Ort und versorgt sie mit den nötigen Regelialien. Es ist ein paradiesmatisches Land, in dem einem gebratene Satzteile in den Mund fliegen. Nicht einmal von der allmächtigen Interpunktion werden die Blindtexte beherrscht – ein geradezu unorthographisches Leben. Eines Tages aber beschloß eine kleine Zeile Blindtext, ihr Name war Lorem Ipsum, hinaus zu gehen in die weite Grammatik. Der große Oxmox riet ihr davon ab, da es dort wimmele von bösen Kommata, wilden Fragezeichen und hinterhältigen Semikoli, doch das Blindtextchen ließ sich nicht beirren. Es packte seine sieben Versalien, schob sich sein Initial in den Gürtel und machte sich auf den Weg. Als es die ersten Hügel des Kursivgebirges erklommen hatte, warf es einen letzten Blick zurück auf die Skyline seiner Heimatstadt Buchstabhausen, die Headline von Alphabetdorf und die Subline seiner eigenen Straße, der Zeilengasse. Wehmütig lief ihm eine rhetorische Frage über die Wange, dann setzte es seinen Weg fort. Unterwegs traf es eine Copy. Die Copy warnte das Blindtextchen, da, wo sie herkäme wäre sie zigmal umgeschrieben worden und alles, was von ihrem Ursprung noch übrig wäre, sei das Wort "und" und das Blindtextchen solle umkehren und wieder in sein eigenes, sicheres Land zurückkehren. Doch alles Gutzureden konnte es nicht überzeugen und so dauerte es nicht lange, bis ihm ein paar heimtückische Werbetexter auflauerten, es mit Longe und Parole betrunken machten und es dann in ihre Agentur schleppten, wo sie es für ihre Projekte wieder und wieder mißbrauchten. Und wenn es nicht umgeschrieben wurde, dann benutzen Sie es immernoch."""  
  
oder mit einer txt-Datei:  
`python main.py -pr text_sample.txt`  
Es kann ein beliebiger Dateiname gewählt werden.
## Aufbau des Programms
![architecture](https://github.com/kobrue02/BERTective/blob/main/architecture.drawio.svg)
