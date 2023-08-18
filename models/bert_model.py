"""
Verwendet Methoden und Klassen aus https://towardsdatascience.com/named-entity-recognition-with-bert-in-pytorch-a454405e0b6a
BertModel kann mit annotierten Daten trainiert werden und diese vorhersagen
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from transformers import BertForTokenClassification, BertTokenizerFast
import torch
from torch.utils.data import DataLoader
from torch.optim import SGD

from tqdm import tqdm

tokenizer = BertTokenizerFast.from_pretrained('bert-base-german-cased')
# bert params
bert_parameters = {
    "padding": 'max_length', 
    "max_length": 512, 
    "truncation": True, 
    "return_tensors": "pt"
    }

# label only first token or all subwords
label_all_tokens = False
labels_to_ids = {'O': 0, 'X': 1}
ids_to_labels = {0: 'O', 1: 'X'}

def align_label_example(tokenized_input, labels):
    
        word_ids = tokenized_input.word_ids()
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                try:
                  label_ids.append(labels_to_ids[labels[word_idx]])
                except:
                  label_ids.append(-100)
            else:
                label_ids.append(labels_to_ids[labels[word_idx]] if label_all_tokens else -100)
            previous_word_idx = word_idx

        return label_ids

def align_label(texts, labels):
    tokenized_inputs = tokenizer(texts, padding='max_length', max_length=512, truncation=True)
    word_ids = tokenized_inputs.word_ids()
    previous_word_idx = None
    label_ids = []

    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)
        elif word_idx != previous_word_idx:
            try:
                label_ids.append(labels_to_ids[labels[word_idx]])
            except:
                label_ids.append(-100)
        else:
            try:
                label_ids.append(labels_to_ids[labels[word_idx]] if label_all_tokens else -100)
            except:
                label_ids.append(-100)
        previous_word_idx = word_idx

    return label_ids

class DataSequence(torch.utils.data.Dataset):

    def __init__(self, df):
        lb = [i.split() for i in df.iloc[:, 1].values.tolist()]
        txt = df['text'].values.tolist()
        self.texts = [tokenizer(str(i), **bert_parameters) for i in txt]
        self.labels = [align_label(i,j) for i,j in zip(txt, lb)]

    def __len__(self):
        return len(self.labels)

    def get_batch_data(self, idx):
        return self.texts[idx]

    def get_batch_labels(self, idx):
        return torch.LongTensor(self.labels[idx])

    def __getitem__(self, idx):
        batch_data = self.get_batch_data(idx)
        batch_labels = self.get_batch_labels(idx)
        return batch_data, batch_labels


class BertModel(torch.nn.Module):

    def __init__(self):
        super(BertModel, self).__init__()
        self.bert = BertForTokenClassification.from_pretrained('bert-base-german-cased', num_labels=2)

    def forward(self, input_id, mask, label):
        output = self.bert(input_ids=input_id, attention_mask=mask, labels=label, return_dict=False)
        return output
    
def train_loop(model, df_train, df_val):

    train_dataset = DataSequence(df_train)
    val_dataset = DataSequence(df_val)

    train_dataloader = DataLoader(train_dataset, num_workers=4, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, num_workers=4, batch_size=BATCH_SIZE)

    #use_cuda = torch.cuda.is_available()
    device = torch.device("cpu")
    optimizer = SGD(model.parameters(), lr=LEARNING_RATE)

    #if use_cuda:
    #    model = model.cuda()
    best_acc = 0
    best_loss = 1000
    for epoch_num in range(EPOCHS):

        total_acc_train = 0
        total_loss_train = 0

        model.train()

        for train_data, train_label in tqdm(train_dataloader):

            train_label = train_label.to(device)
            mask = train_data['attention_mask'].squeeze(1).to(device)
            input_id = train_data['input_ids'].squeeze(1).to(device)

            optimizer.zero_grad()
            loss, logits = model(input_id, mask, train_label)

            for i in range(logits.shape[0]):

              logits_clean = logits[i][train_label[i] != -100]
              label_clean = train_label[i][train_label[i] != -100]

              predictions = logits_clean.argmax(dim=1)
              acc = (predictions == label_clean).float().mean()
              total_acc_train += acc
              total_loss_train += loss.item()

            loss.backward()
            optimizer.step()

        model.eval()

        total_acc_val = 0
        total_loss_val = 0

        for val_data, val_label in val_dataloader:

            val_label = val_label.to(device)
            mask = val_data['attention_mask'].squeeze(1).to(device)
            input_id = val_data['input_ids'].squeeze(1).to(device)

            loss, logits = model(input_id, mask, val_label)

            for i in range(logits.shape[0]):

              logits_clean = logits[i][val_label[i] != -100]
              label_clean = val_label[i][val_label[i] != -100]

              predictions = logits_clean.argmax(dim=1)
              acc = (predictions == label_clean).float().mean()
              total_acc_val += acc
              total_loss_val += loss.item()

        val_accuracy = total_acc_val / len(df_val)
        val_loss = total_loss_val / len(df_val)

        print(
            f'Epochs: {epoch_num + 1} | Loss: {total_loss_train / len(df_train): .3f} | Accuracy: {total_acc_train / len(df_train): .3f} | Val_Loss: {total_loss_val / len(df_val): .3f} | Accuracy: {total_acc_val / len(df_val): .3f}')

def evaluate(model, df_test):
    test_dataset = DataSequence(df_test)
    test_dataloader = DataLoader(test_dataset, num_workers=4, batch_size=1)
    #use_cuda = torch.cuda.is_available()
    device = torch.device("cpu")
    #if use_cuda:
    #    model = model.cuda()
    total_acc_test = 0.0
    for test_data, test_label in test_dataloader:
            test_label = test_label.to(device)
            mask = test_data['attention_mask'].squeeze(1).to(device)
            input_id = test_data['input_ids'].squeeze(1).to(device)
            loss, logits = model(input_id, mask, test_label)

            for i in range(logits.shape[0]):
              
              logits_clean = logits[i][test_label[i] != -100]
              label_clean = test_label[i][test_label[i] != -100]
              print("clean label: ")
              print(label_clean)
              predictions = logits_clean.argmax(dim=1)
              print("predictions: ")
              print(predictions)
              acc = (predictions == label_clean).float().mean()
              total_acc_test += acc
    val_accuracy = total_acc_test / len(df_test)
    print(f'Test Accuracy: {total_acc_test / len(df_test): .3f}')

def align_word_ids(texts):
    tokenized_inputs = tokenizer(texts, padding='max_length', max_length=512, truncation=True)
    word_ids = tokenized_inputs.word_ids()
    previous_word_idx = None
    label_ids = []
    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)
        elif word_idx != previous_word_idx:
            try:
                label_ids.append(1)
            except:
                label_ids.append(-100)
        else:
            try:
                label_ids.append(1 if label_all_tokens else -100)
            except:
                label_ids.append(-100)
        previous_word_idx = word_idx

    return label_ids


def evaluate_one_text(model, sentence):
    #use_cuda = torch.cuda.is_available()
    device = torch.device("cpu")
    #if use_cuda:
    #    model = model.cuda()

    text = tokenizer(sentence, padding='max_length', max_length = 512, truncation=True, return_tensors="pt")
    mask = text['attention_mask'].to(device)
    input_id = text['input_ids'].to(device)
    label_ids = torch.Tensor(align_word_ids(sentence)).unsqueeze(0).to(device)

    logits = model(input_id, mask, None)
    logits_clean = logits[0][label_ids != -100]

    predictions = logits_clean.argmax(dim=1).tolist()
    prediction_label = [ids_to_labels[i] for i in predictions]
    print(sentence)
    print(prediction_label)

LEARNING_RATE = 5e-3
EPOCHS = 1
BATCH_SIZE = 2

if __name__ == "__main__":
    example_text = """ Sie hatte immer wieder bedenken, weil ich nicht ihr "Traummann" sei aber sie Gefühle für mich hat. """
    tokens = tokenizer(example_text, **bert_parameters)
    word_ids = tokens.word_ids()
    print(tokenizer.convert_ids_to_tokens(tokens.input_ids[0]))
    print(word_ids)
