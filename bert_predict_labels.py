from transformers import BertTokenizerFast

import pandas as pd

tokenizer = BertTokenizerFast.from_pretrained('bert-base-german-cased')