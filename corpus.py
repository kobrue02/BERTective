"""
Enthält Korpus-Klasse in welcher alle Trainingsdaten aus
allen verschiedenen Quellen gespeichert werden können
"""

import pandas as pd
from fastavro import writer, parse_schema, reader

class DataObject:

    """
    Wrapper für annotierte Text-Samples. 
    Muss Text enthalten, annotierte Features sind optional.
    """

    def __init__(
            self, 
            text: str, 
            author_age: int = 0, 
            author_gender: str = "NONE", 
            author_regiolect: str = "NONE",
            author_education: str = "NONE",
            source: str = "NONE") -> None:

        """
        :param text: roher Text (z.B. Reddit-Post oder Blogeintrag)

        :param author_age: das Alter des Autors des Textes während des Verfassens (int)

        :param author_gender: das Geschlecht des Autors (M/F)

        :param author_regiolect: Herkunft des Autors innerhalb von Deutschland. \
        DE-NORTH-WEST, DE-NORTH-EAST, DE-MIDDLE-WEST, DE-MIDDLE-EAST, \
        DE-SOUTH-WEST oder DE-SOUTH-EAST

        :param author_education: Bildungsgrad des Autors. Akzeptiert "finished_highschool", \
        "in_university", "has_bachelor", "has_master", \
        "has_phd", "apprentice", "NONE"

        :param source: die Quelle des Texts (z.B. Reddit)
        """
        
        self.text: str = text
        self.author_age:  int = author_age
        self.author_gender: str = author_gender
        self.author_regiolect: str = author_regiolect
        self.author_education: str = author_education
        self.source: str = source

        self.content = {'text': text,
                        'author_age': author_age,
                        'author_gender': author_gender,
                        'author_regiolect': author_regiolect,
                        'author_education': author_education,
                        'source': source}


class DataCorpus:

    """
    Datenbank für annotierte Text-Samples. Kann inkrementell aufgebaut werden.
    Einzelne Sampels lassen sich per Index ansteuern.
    """

    def __init__(self) -> None:
        self.corpus: list[DataObject] = []
        
    def add_item(self, item: DataObject):

        if self.__verify(item):

            last_id = len(self.corpus) - 1
            item.content['id'] = last_id + 1
            self.corpus.append(item)

        else:
            raise ValueError("item couldn't be verified.")
    
    def __verify(self, item: DataObject) -> bool:
        """ verify whether a DataObject contains all required information in the right formats """
        features = ['text', 'author_age', 'author_gender', 'author_regiolect', 'author_education', 'source']

        if not list(item.content.keys()) == features:
            return False
        
        if not isinstance(item.content['text'], str):
            return False
        
        if not (isinstance(item.content['author_age'], int) or item.content["author_age"] is None):
            return False
        
        if not item.content['author_regiolect'] in ["DE-NORTH-WEST",
                                                    "DE-NORTH-EAST",
                                                    "DE-MIDDLE-WEST",
                                                    "DE-MIDDLE-EAST",
                                                    "DE-SOUTH-WEST",
                                                    "DE-SOUTH-EAST",
                                                    "NONE"]:
            return False
        
        if not item.content['author_education'] in ["finished_highschool",
                                                    "in_university",
                                                    "has_bachelor",
                                                    "has_master",
                                                    "has_phd",
                                                    "apprentice",
                                                    "NONE"]:
            return False

        if not isinstance(item.content['source'], str):
            return False
        
        return True
    
    def as_dataframe(self) -> pd.DataFrame:
        
        data = pd.DataFrame()
        data['text'] = [item.text for item in self.corpus]
        data['author_age'] = [item.author_age for item in self.corpus]
        data['author_gender'] = [item.author_gender for item in self.corpus]
        data['author_regiolect'] = [item.author_regiolect for item in self.corpus]
        data['author_education'] = [item.author_education for item in self.corpus]
        data['source'] = [item.source for item in self.corpus]

        return data
    
    def save_to_avro(self, path):
        # specifying the avro schema
        schema = {
                'doc': 'corpus',
                'name': 'corpus',
                'namespace': 'corpus',
                'type': 'record',
                'fields': [
                    {'name': 'text', 'type': 'string'},
                    {'name': 'author_age', 'type': 'int'},
                    {'name': 'author_gender', 'type': 'string'},
                    {'name': 'author_regiolect', 'type': 'string'},
                    {'name': 'author_education', 'type': 'string'},
                    {'name': 'source', 'type': 'string'}
                    ]
                }
        parsed_schema = parse_schema(schema)
        dataframe = self.as_dataframe()
        records = dataframe.to_dict('records')

        # writing to avro file format
        with open(path, 'wb') as out:
            writer(out, parsed_schema, records)

    def read_avro(self, path: str):
        # reading it back into pandas dataframe
        avro_records = []

        #Read the Avro file
        with open(path, 'rb') as fo:
            avro_reader = reader(fo)
            for record in avro_reader:
                avro_records.append(record)

        #Convert to pd.DataFrame
        df_avro = pd.DataFrame(avro_records)

        for index, row in df_avro.iterrows():
            obj = DataObject(row['text'], 
                             row['author_age'],
                             row['author_gender'],
                             row['author_regiolect'],
                             row['author_education'],
                             row['source'])
            
            self.add_item(obj)



    def __getitem__(self, i):

        try:
            if self.corpus[i].content['id'] == i:
                return self.corpus[i]
            else:
                raise ValueError("The data bank hasn't been indexed properly.")
        except IndexError:
            raise IndexError(f"The ID is out of range. Corpus contains {len(self)} items.")
        
    
    def __len__(self):
        return len(self.corpus)
        

if __name__ == "__main__":

    corpus = DataCorpus()
    #corpus.read_avro('data/corpus.avro')
    #print(corpus.as_dataframe().head())
    sample_1 = DataObject("sfgfghjghj?",
                        28,
                        "male",
                        "DE-NORTH-WEST",
                        "has_bachelor",
                        "reddit")
    corpus.add_item(sample_1)
    sample_2 = DataObject("YYYY",
                        265,
                        "female",
                        "DE-NORTH-WEST",
                        "has_master",
                        "reddit")
    corpus.add_item(sample_2)
    print(corpus.as_dataframe().head())
    corpus.save_to_avro('data/corpus.avro')


        

        
