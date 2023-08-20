"""
Enthält Korpus-Klasse in welcher alle Trainingsdaten aus
allen verschiedenen Quellen gespeichert werden können
"""

import pandas as pd


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
        
        self.content = {'text': text,
                        'author_age': author_age,
                        'author_gender': author_gender,
                        'author_regiolect': author_regiolect,
                        'author_education': author_education,
                        'source': source}
        
        self.text = text
        self.author_age = author_age
        self.author_gender = author_gender
        self.author_regiolect = author_regiolect
        self.author_education = author_education
        self.source = source


class DataCorpus:

    """
    Datenbank für annotierte Text-Samples. Kann inkrementell aufgebaut werden.
    Einzelne Sampels lassen sich per Index ansteuern.
    """

    def __init__(self) -> None:
        self.corpus = []
        
    def add_item(self, item: DataObject):

        if self.__verify(item):

            last_id = len(self.corpus) - 1
            item.content['id'] = last_id + 1
            self.corpus.append(item)

        else:
            raise ValueError("item couldn't be verified.")
    
    def __verify(self, item: DataObject) -> bool:
        
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
        print(self.corpus)
        data['text'] = [item.text for item in self.corpus]
        data['author_age'] = [item.author_age for item in self.corpus]
        data['author_gender'] = [item.author_gender for item in self.corpus]
        data['author_regiolect'] = [item.author_regiolect for item in self.corpus]
        data['author_education'] = [item.author_education for item in self.corpus]
        data['source'] = [item.source for item in self.corpus]

        return data


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
    sample_1 = DataObject("Hallo wie geht es euch?",
                        22,
                        "male",
                        "DE-NORTH-WEST",
                        "has_bachelor",
                        "reddit")
    corpus.add_item(sample_1)
    sample_2 = DataObject("XXXXXXXXXXX",
                        15,
                        "male",
                        "DE-NORTH-WEST",
                        "has_bachelor",
                        "reddit")
    corpus.add_item(sample_2)
    print(corpus.as_dataframe().head())


        

        
