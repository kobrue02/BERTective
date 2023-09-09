import spacy
import pandas
import numpy
nlp = spacy.load("de_core_news_sm")


lorem = """
Auch gibt es niemanden, der den Schmerz an sich liebt, sucht oder wünscht, nur, weil er Schmerz ist, es sei denn, es kommt zu zufälligen Umständen, in denen Mühen und Schmerz ihm große Freude bereiten können. Um ein triviales Beispiel zu nehmen, wer von uns unterzieht sich je anstrengender körperlicher Betätigung, außer um Vorteile daraus zu ziehen? Aber wer hat irgend ein Recht, einen Menschen zu tadeln, der die Entscheidung trifft, eine Freude zu genießen, die keine unangenehmen Folgen hat, oder einen, der Schmerz vermeidet, welcher keine daraus resultierende Freude nach sich zieht? Auch gibt es niemanden, der den Schmerz an sich liebt, sucht oder wünscht, nur, weil er Schmerz ist, es sei denn, es kommt zu zufälligen Umständen, in denen Mühen und Schmerz ihm große Freude bereiten können. Um ein triviales Beispiel zu nehmen, wer von uns unterzieht sich je anstrengender körperlicher Betätigung, außer um Vorteile daraus zu ziehen? Aber wer hat irgend ein Recht, einen Menschen zu tadeln, der die Entscheidung trifft, eine Freude zu genießen, die keine unangenehmen Folgen hat, oder einen, der Schmerz vermeidet, welcher keine daraus resultierende Freude nach sich zieht? Auch gibt es niemanden, der den Schmerz an sich liebt, sucht oder wünscht, nur, weil er Schmerz ist, es sei denn, es kommt zu zufälligen Umständen, in denen Mühen und Schmerz ihm große Freude bereiten können. Um ein triviales Beispiel zu nehmen, wer von uns unterzieht sich je anstrengender körperlicher Betätigung,.
"""
goethe = """
Eine wunderbare Heiterkeit hat meine ganze Seele eingenommen, gleich den süßen Frühlingsmorgen, die ich mit ganzem Herzen genieße. Ich bin allein und freue mich meines Lebens in dieser Gegend, die für solche Seelen geschaffen ist wie die meine. Ich bin so glücklich, mein Bester, so ganz in dem Gefühle von ruhigem Dasein versunken, daß meine Kunst darunter leidet. Ich könnte jetzt nicht zeichnen, nicht einen Strich, und bin nie ein größerer Maler gewesen als in diesen Augenblicken. Wenn das liebe Tal um mich dampft, und die hohe Sonne an der Oberfläche der undurchdringlichen Finsternis meines Waldes ruht, und nur einzelne Strahlen sich in das innere Heiligtum stehlen, ich dann im hohen Grase am fallenden Bache liege, und näher an der Erde tausend mannigfaltige Gräschen mir merkwürdig werden; wenn ich das Wimmeln der kleinen Welt zwischen Halmen, die unzähligen, unergründlichen Gestalten der Würmchen, der Mückchen näher an meinem Herzen fühle, und fühle die Gegenwart des Allmächtigen, der uns nach seinem Bilde schuf, das Wehen des Alliebenden, der uns in ewiger Wonne schwebend trägt und erhält; mein Freund! Wenn's dann um meine Augen dämmert, und die Welt um mich her und der Himmel ganz in meiner Seele ruhn wie die Gestalt einer Geliebten - dann sehne ich mich oft und denke : ach könntest du das wieder ausdrücken, könntest du dem Papiere das einhauchen, was so voll, so warm in dir lebt, daß es würde der Spiegel deiner Seele, wie deine Seele ist der Spiegel des unendlichen Gotte
"""
wortberge = """
Weit hinten, hinter den Wortbergen, fern der Länder Vokalien und Konsonantien leben die Blindtexte. Abgeschieden wohnen sie in Buchstabhausen an der Küste des Semantik, eines großen Sprachozeans. Ein kleines Bächlein namens Duden fließt durch ihren Ort und versorgt sie mit den nötigen Regelialien. Es ist ein paradiesmatisches Land, in dem einem gebratene Satzteile in den Mund fliegen. Nicht einmal von der allmächtigen Interpunktion werden die Blindtexte beherrscht – ein geradezu unorthographisches Leben. Eines Tages aber beschloß eine kleine Zeile Blindtext, ihr Name war Lorem Ipsum, hinaus zu gehen in die weite Grammatik. Der große Oxmox riet ihr davon ab, da es dort wimmele von bösen Kommata, wilden Fragezeichen und hinterhältigen Semikoli, doch das Blindtextchen ließ sich nicht beirren. Es packte seine sieben Versalien, schob sich sein Initial in den Gürtel und machte sich auf den Weg. Als es die ersten Hügel des Kursivgebirges erklommen hatte, warf es einen letzten Blick zurück auf die Skyline seiner Heimatstadt Buchstabhausen, die Headline von Alphabetdorf und die Subline seiner eigenen Straße, der Zeilengasse. Wehmütig lief ihm eine rhetorische Frage über die Wange, dann setzte es seinen Weg fort. Unterwegs traf es eine Copy. Die Copy warnte das Blindtextchen, da, wo sie herkäme wäre sie zigmal umgeschrieben worden und alles, was von ihrem Ursprung noch übrig wäre, sei das Wort "und" und das Blindtextchen solle umkehren und wieder in sein eigenes, sicheres Land zurückke
"""
kafka = """
Jemand musste Josef K. verleumdet haben, denn ohne dass er etwas Böses getan hätte, wurde er eines Morgens verhaftet. »Wie ein Hund!« sagte er, es war, als sollte die Scham ihn überleben. Als Gregor Samsa eines Morgens aus unruhigen Träumen erwachte, fand er sich in seinem Bett zu einem ungeheueren Ungeziefer verwandelt. Und es war ihnen wie eine Bestätigung ihrer neuen Träume und guten Absichten, als am Ziele ihrer Fahrt die Tochter als erste sich erhob und ihren jungen Körper dehnte. »Es ist ein eigentümlicher Apparat«, sagte der Offizier zu dem Forschungsreisenden und überblickte mit einem gewissermaßen bewundernden Blick den ihm doch wohlbekannten Apparat. Sie hätten noch ins Boot springen können, aber der Reisende hob ein schweres, geknotetes Tau vom Boden, drohte ihnen damit und hielt sie dadurch von dem Sprunge ab. In den letzten Jahrzehnten ist das Interesse an Hungerkünstlern sehr zurückgegangen. Aber sie überwanden sich, umdrängten den Käfig und wollten sich gar nicht fortrühren. Jemand musste Josef K. verleumdet haben, denn ohne dass er etwas Böses getan hätte, wurde er eines Morgens verhaftet. »Wie ein Hund!« sagte er, es war, als sollte die Scham ihn überleben. Als Gregor Samsa eines Morgens aus unruhigen Träumen erwachte, fand er sich in seinem Bett zu einem ungeheueren Ungeziefer verwandelt. Und es war ihnen wie eine Bestätigung ihrer neuen Träume und guten Absichten, als am Ziele ihrer Fahrt die Tochter als erste sich erhob und ihren jungen Körper dehnte. »Es.
"""
fontane = """
Im Norden der Grafschaft Ruppin, hart an der mecklenburgischen Grenze, zieht sich von dem Städtchen Gransee bis nach Rheinsberg hin (und noch darüber hinaus) eine mehrere Meilen lange Seeenkette durch eine menschenarme, nur hie und da mit ein paar alten Dörfern, sonst aber ausschließlich mit Förstereien, Glas- und Teeröfen besetzte Waldung. Einer der Seeen, die diese Seeenkette bilden, heißt „der Stechlin“. Zwischen flachen, nur an einer einzigen Stelle steil und quaiartig ansteigenden Ufern liegt er da, rundum von alten Buchen eingefaßt, deren Zweige, von ihrer eignen Schwere nach unten gezogen, den See mit ihrer Spitze berühren. Hie und da wächst ein weniges von Schilf und Binsen auf, aber kein Kahn zieht seine Furchen, kein Vogel singt, und nur selten, daß ein Habicht drüber hinfliegt und seinen Schatten auf die Spiegelfläche wirft. Alles still hier. Und doch, von Zeit zu Zeit wird es an eben dieser Stelle lebendig. Das ist, wenn es weit draußen in der Welt, sei’s auf Island, sei’s auf Java, zu rollen und zu grollen beginnt oder gar der Aschenregen der hawaiischen Vulkane bis weit auf die Südsee hinausgetrieben wird. Dann regt sich’s auch hier, und ein Wasserstrahl springt auf und sinkt wieder in die Tiefe. Das wissen alle, die den Stechlin umwohnen, und wenn sie davon sprechen, so setzen sie wohl auch hinzu: „Das mit dem Wasserstrahl, das ist nur das Kleine, das beinah Alltägliche; wenn’s aber draußen was Großes gibt, wie vor hundert Jahren in Lissabon, dann brodelt’s hier nicht bloß und sprudelt und strudelt, dann steigt statt des Wasserstrahls ein roter Hahn auf und kräht laut in die Lande hinein.“
Das ist der Stechlin, der See Stechlin.
Aber nicht nur der See führt diesen Namen, auch der Wald, der ihn umschließt. Und Stechlin heißt ebenso das langgestreckte Dorf, das sich, den Windungen des Sees folgend, um seine Südspitze herumzieht. Etwa hundert Häuser und Hütten bilden hier eine lange, schmale Gasse, die sich nur da, wo eine von Kloster Wutz her heranführende Kastanienallee die Gasse durchschneidet, platzartig erweitert. An eben dieser Stelle findet sich denn auch die ganze Herrlichkeit von Dorf Stechlin zusammen; das Pfarrhaus, die Schule, das Schulzenamt, der Krug, dieser letztere zugleich ein Eck- und Kramladen mit einem kleinen Mohren und einer Guirlande von Schwefelfäden in seinem Schaufenster. Dieser Ecke schräg gegenüber, unmittelbar hinter dem Pfarrhause, steigt der Kirchhof lehnan, auf ihm, so ziemlich in seiner Mitte, die frühmittelalterliche Feldsteinkirche mit einem aus dem vorigen Jahrhundert stammenden Dachreiter und einem zur Seite des alten Rundbogenportals angebrachten Holzarm, dran eine Glocke hängt. Neben diesem Kirchhof samt Kirche setzt sich dann die von Kloster Wutz her heranführende Kastanienallee noch eine kleine Strecke weiter fort, bis sie vor einer über einen sumpfigen Graben sich hinziehenden und von zwei riesigen Findlingsblöcken flankierten Bohlenbrücke Halt macht. Diese Brücke ist sehr primitiv. Jenseits derselben aber steigt das Herrenhaus auf, ein gelbgetünchter Bau mit hohem Dach und zwei Blitzableitern.
Auch dieses Herrenhaus heißt Stechlin, Schloß Stechlin.
"""
trapattoni = """
Es gibt im Moment in diese Mannschaft, oh, einige Spieler vergessen ihnen Profi was sie sind. Ich lese nicht sehr viele Zeitungen, aber ich habe gehört viele Situationen. Erstens: wir haben nicht offensiv gespielt. Es gibt keine deutsche Mannschaft spielt offensiv und die Name offensiv wie Bayern. Letzte Spiel hatten wir in Platz drei Spitzen: Elber, Jancka und dann Zickler. Wir müssen nicht vergessen Zickler. Zickler ist eine Spitzen mehr, Mehmet eh mehr Basler. Ist klar diese Wörter, ist möglich verstehen, was ich hab gesagt? Danke. Offensiv, offensiv ist wie machen wir in Platz. Zweitens: ich habe erklärt mit diese zwei Spieler: nach Dortmund brauchen vielleicht Halbzeit Pause. Ich habe auch andere Mannschaften gesehen in Europa nach diese Mittwoch. Ich habe gesehen auch zwei Tage die Training. Ein Trainer ist nicht ein Idiot! Ein Trainer sei sehen was passieren in Platz. In diese Spiel es waren zwei, drei diese Spieler waren schwach wie eine Flasche leer! Haben Sie gesehen Mittwoch, welche Mannschaft hat gespielt Mittwoch? Hat gespielt Mehmet oder gespielt Basler oder hat gespielt Trapattoni? Diese Spieler beklagen mehr als sie spielen! Wissen Sie, warum die Italienmannschaften kaufen nicht diese Spieler? Weil wir haben gesehen viele Male solche Spiel! Haben gesagt sind nicht Spieler für die italienisch Meisters! Strunz! Strunz ist zwei Jahre hier, hat gespielt 10 Spiele, ist immer verletzt! Was erlauben Strunz? Letzte Jahre Meister Geworden mit Hamann, eh, Nerlinger. Die
"""
schiller = """
Man nehme dieses Schauspiel für nichts anders, als eine dramatische Geschichte, die die Vortheile der dramatischen Methode, die Seele gleichsam bei ihren geheimsten Operationen zu ertappen, benuzt, ohne sich übrigens in die Schranken eines Theaterstücks einzuzäunen, oder nach dem so zweifelhaften Gewinn bei theatralischer Verkörperung zu geizen. Man wird mir einräumen, daß es eine widersinnige Zumuthung ist, binnen drei Stunden drei ausserordentliche Menschen zu erschöpfen, deren Thätigkeit von vielleicht tausend Räderchen abhänget, so wie es in der Natur der Dinge unmöglich kann gegründet seyn, daß sich drei ausserordentliche Menschen auch dem durchdringendsten Geisterkenner innerhalb vier und zwanzig Stunden entblössen. Hier war Fülle ineinandergedrungener Realitäten vorhanden, die ich unmöglich in die allzuengen Pallisaden des Aristoteles und Batteux einkeilen konnte.
Nun ist es aber nicht sowohl die Masse meines Schauspiels, als vielmehr sein Innhalt, der es von der Bühne verbannet. Die Oekonomie desselben machte es nothwendig, daß mancher Karakter auftreten mußte, der das feinere Gefühl der Tugend beleidigt, und die Zärtlichkeit unserer Sitten empört. Jeder Menschenmaler ist in diese Nothwendigkeit gesezt, wenn er anders eine Kopie der wirklichen Welt, und keine idealischen Affektationen, keine Kompendienmenschen will geliefert haben. Es ist einmal so die Mode in der Welt, daß die Guten durch die Bösen schattiert werden, und die Tugend im Kontrast mit dem Laster das lebendigste Kolorit erhält. Wer sich den Zweck vorgezeichnet hat, das Laster zu stürzen, und Religion, Moral und bürgerliche Geseze an ihren Feinden zu rächen, ein solcher muß das Laster in seiner nakten Abscheulichkeit enthüllen, und in seiner kolossalischen Grösse vor das Auge der Menschheit stellen – er selbst muß augenbliklich seine nächtlichen Labyrinthe durchwandern, – er muß sich in Empfindungen hineinzuzwingen wissen, unter deren Widernatürlichkeit sich seine Seele sträubt.
Das Laster wird hier mit samt seinem ganzen innern Räderwerk entfaltet. Es lößt in Franzen all die verworrenen Schauer des Gewissens in ohnmächtige Abstraktionen auf, skeletisirt die richtende Empfindung, und scherzt die ernsthafte Stimme der Religion hinweg. Wer es einmal so weit gebracht hat, (ein Ruhm, den wir ihm nicht beneiden) seinen Verstand auf Unkosten seines Herzens zu verfeinern, dem ist das Heiligste nicht heilig mehr – dem ist die Menschheit, die Gottheit nichts – Beide Welten sind nichts in seinen Augen. Ich habe versucht, von einem Mißmenschen dieser Art ein treffendes lebendiges Konterfey hinzuwerffen, die vollständige Mechanik seines Lastersystems auseinander zu gliedern – und ihre Kraft an der Wahrheit zu prüfen. Man unterrichte sich demnach im Verfolg dieser Geschichte, wie weit ihr’s gelungen hat – Ich denke, ich habe die Natur getroffen.
Nächst an diesem stehet ein anderer, der vielleicht nicht wenige meiner Leser in Verlegenheit sezen möchte. Ein Geist, den das äusserste Laster nur reizet um der Grösse willen, die ihm anhänget, um der Kraft willen, die es erheischet; um der Gefahren willen, die es begleiten. Ein merkwürdiger wichtiger Mensch, ausgestattet mit aller Kraft, nach der Richtung, die diese bekömmt, nothwendig entweder ein Brutus oder ein Katilina zu werden. Unglükliche Konjunkturen entscheiden für das zweyte und erst am Ende einer ungeheuren Verirrung gelangt er zu dem ersten. Falsche Begriffe von Thätigkeit und Einfluß, Fülle von Kraft, die alle Geseze übersprudelt, mußten sich natürlicher Weise an bürgerlichen Verhältnissen zerschlagen, und zu diesen enthousiastischen Träumen von Grösse und Wirksamkeit durfte sich nur eine Bitterkeit gegen die unidealische Welt gesellen, so war der seltsame Donquixote fertig, den wir im Räuber Moor verabscheuen und lieben, bewundern und bedauern. Ich werde es hoffentlich nicht erst anmerken dörfen, daß ich dieses Gemählde so wenig nur allein Räubern vorhalte, als die Satyre des Spaniers nur allein Ritter geisselt.
Auch ist izo der grosse Geschmak, seinen Wiz auf Kosten der Religion spielen zu lassen, daß man beinahe für kein Genie mehr paßirt, wenn man nicht seinen gottlosen Satyr auf ihren heiligsten Wahrheiten sich herumtummeln läßt. Die edle Einfalt der Schrift muß sich in alltäglichen Assembleen von den sogenannten wizigen Köpfen mißhandeln, und ins Lächerliche verzerren lassen; denn was ist so heilig und ernsthaft, das, wenn man es falsch verdreht, nicht belacht werden kann? – Ich kann hoffen, daß ich der Religion und der wahren Moral keine gemeine Rache verschafft habe, wenn ich diese muthwilligen Schriftverächter in der Person meiner schändlichsten Räuber dem Abscheu der Welt überliefere.
Aber noch mehr. Diese unmoralische Karaktere, von denen vorhin gesprochen wurde, mußten von gewissen Seiten glänzen, ja oft von Seiten des Geistes gewinnen, was sie von Seiten des Herzens verlieren. Hierinn habe ich nur die Natur gleichsam wörtlich abgeschrieben. Jedem, auch dem Lasterhaftesten ist gewissermassen der Stempel des göttlichen Ebenbilds aufgedrükt, und vielleicht hat der grosse Bösewicht keinen so weiten Weg zum grossen Rechtschaffenen, als der kleine; denn die Moralität hält gleichen Gang mit den Kräften, und je weiter die Fähigkeit, desto weiter und ungeheurer ihre Verirrung, desto imputabler ihre Verfälschung.
Klopstoks Adramelech wekt in uns eine Empfindung, worinn Bewunderung in Abscheu schmilzt. Miltons Satan folgen wir mit schauderndem Erstaunen durch das unwegsame Chaos. Die Medea der alten Dramatiker bleibt bei all ihren Greueln noch ein grosses staunenswürdiges Weib, und Shakespears Richard hat so gewiß an Leser einen Bewunderer, als er auch ihn hassen würde, wenn er ihm vor der Sonne stünde. Wenn es mir darum zu thun ist, ganze Menschen hinzustellen, so muß ich auch ihre Vollkommenheiten mitnehmen, die auch dem bösesten nie ganz fehlen. Wenn ich vor dem Tyger gewarnt haben will, so darf ich seine schöne blendende Flekenhaut nicht übergehen, damit man nicht den Tyger beym Tyger vermisse. Auch ist ein Mensch, der ganz Bosheit ist, schlechterdings kein Gegenstand der Kunst, und äussert eine zurükstossende Kraft, statt daß er die Aufmerksamkeit der Leser fesseln sollte. Man würde umblättern, wenn er redet. Eine edle Seele erträgt so wenig anhaltende moralische Dissonanzen, als das Ohr das Gekrizel eines Messers auf Glas.
Aber eben darum will ich selbst mißrathen haben, dieses mein Schauspiel auf der Bühne zu wagen. Es gehört beiderseits, beim Dichter und seinem Leser, schon ein gewisser Gehalt von Geisteskraft dazu; bei jenem, daß er das Laster nicht ziere, bei diesem, daß er sich nicht von einer schönen Seite bestechen lasse, auch den häßlichen Grund zu schäzen. Meiner Seits entscheide ein Dritter – aber von meinen Lesern bin ich es nicht ganz versichert. Der Pöbel, worunter ich keineswegs die Gassenkehrer allein will verstanden wissen, der Pöbel wurzelt, (unter uns gesagt) weit um, und gibt zum Unglük – den Ton an. Zu kurzsichtig mein Ganzes auszureichen, zu kleingeistisch mein Grosses zu begreifen, zu boshaft mein Gutes wissen zu wollen, wird er, fürcht’ ich, fast meine Absicht vereiteln, wird vielleicht eine Apologie des Lasters, das ich stürze, darinn zu finden meynen, und seine eigene Einfalt den armen Dichter entgelten lassen, dem man gemeiniglich alles, nur nicht Gerechtigkeit wiederfahren läßt.
Es ist das ewige Dacapo mit Abdera und Demokrit, und unsre gute Hippokrate müßten ganze Plantagen Nießwurz erschöpfen, wenn sie dem Unwesen durch ein heilsames Dekokt abhelfen wollten. Noch so viele Freunde der Wahrheit mögen zusammenstehen, ihren Mitbürgern auf Kanzel und Schaubühne Schule zu halten, der Pöbel hört nie auf, Pöbel zu seyn, und wenn Sonne und Mond sich wandeln, und Himmel und Erde veralten wie ein Kleid. Vielleicht hätt’ ich den schwachherzigen zu frommen der Natur minder getreu seyn sollen; aber wenn jener Käfer, den wir alle kennen, auch den Mist aus den Perlen stört, wenn man Exempel hat, daß Feuer verbrannt, und Wasser ersäuft habe, soll darum Perle – Feuer – und Wasser konfiscirt werden?
Ich darf meiner Schrift, zufolge ihrer merkwürdigen Katastrophe mit Recht einen Plaz unter den moralischen Büchern versprechen; das Laster nimmt den Ausgang, der seiner würdig ist. Der Verirrte tritt wieder in das Gelaise der Geseze. Die Tugend geht siegend davon. Wer nur so billig gegen mich handelt, mich ganz zu lesen, mich verstehen zu wollen, von dem kann ich erwarten, daß er – nicht den Dichter bewundere, aber den rechtschaffenen Mann in mir hochschäze.
"""
comedian = """
Meine Dames un Herren, ich bin heut hier un freu mich sehr, Ihne einige humorvolle Gedanken mitzuteilen. Die Komödie ist eine Kunstform, die uns zum Lachen bringt, und sie hat die erstaunliche Fähigkeit, unsern Alltag ein wenig heller und leichter zu machen.
Kenen Sie das Gefühl, wenn Sie versuchn, eine Fliege zu erwischen und sie Ihn immer wieder entwischt? Das ist wie mein Glück in der Liebe - es fliegt ständig davon! Oder wie ist es mit dem Versuch, gesund zu leben? Ich meinte, wer braucht schon Gemüse, wenn Schokolade so viel besser schmeckt? Un das Fitnessstudio? Nun, ich habe eine Mitgliedschaft, aber mein Lieblingssport ist es, die Fernbedienung zu suchen. Ich laufe schneller als jeder Marathonläufer, wenn ich sie nicht finden kann.
Schließlich möchte ich Ihnen sagen, das das Leben selbst die beste Comedy-Show ist. Wir alle machen Fehler, stolpern über unsre eigenen Füße und lachen über unsre Missgeschicke. Lassen Sie uns gemeinsam über die kleinen Dinge im Leben lachen un uns dran erinnern, das Humor eine universale Sprache ist, die uns verbindet.
Vielen Dank dafür, dass Sie zugehört haben, un ich hoffe, Sie habn heute ein Lächeln auf den Lippen. Möge Ihr Tag mit Gelächter un Freude erfüllt sein, un möge Ihr Herz vor Glück un Heiterkeit überfließen, so wie ein überladner Kühlschrank, der sich weigert, geschlossen zu bleiben. Ha, ha, ha!
"""
short = """
recht … kurzer ... Text.
"""
texts = [comedian, goethe, lorem, kafka, trapattoni, fontane, wortberge, schiller, short]


def char_per_word(document: str) -> float:
    """
    calculate the average count of characters per word contained in the given text input
    :param document: text as string
    :return: document’s average word length
    """
    doc = nlp(document)
    total_words_length = 0
    word_count = 0
    for token in doc:
        if not token.is_space and not token.is_punct:
            total_words_length += len(token.text)
            word_count += 1
    if word_count > 0:
        return total_words_length / word_count
    else:
        return word_count


def words_per_sent(document: str) -> float:
    """
    calculate the average count of words per sentence contained in the given text input
    :param document: text as string
    :return: document’s average word count per sentence
    """
    doc = nlp(document)
    total_sent_count = 0
    total_word_count = 0
    for sentence in doc.sents:
        total_sent_count += 1
        for token in sentence:
            if not token.is_space and not token.is_punct:
                total_word_count += 1
    if total_sent_count > 0:
        return total_word_count / total_sent_count
    else:
        return total_sent_count


def morph_per_word(document: str) -> float:
    """
    calculate the average count of morphemes per word contained in the given text input
    :param document: text as string
    :return: document’s average morpheme count per word
    """
    #doc = nlp(document)
    #total_morph_
    #word_count = 0
    #for token in doc:
    #    if not token.is_space and not token.is_punct:
    #    total_word_count += 1

    return 9.6


data = []
for text in texts:
    row = {
        "Text": text,
        "Function1_Result": char_per_word(text),
        "Function2_Result": words_per_sent(text),
        "Function3_Result": morph_per_word(text),
    }
    data.append(row)

df = pandas.DataFrame(data)


for text in texts:
    print(text[0:50], char_per_word(text), words_per_sent(text))
