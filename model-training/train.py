import torch
import seaborn
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import CamembertForSequenceClassification, CamembertTokenizer, AdamW
from tqdm import tqdm

BERT_MODEL_NAME = "camembert-base"
BERT_TOKENIZER_NAME = "camembert-base"
DEVICE="cpu"
MAX_LEN = 512
BATCH_SIZE = 12
N_EPOCHS = 3

# Chargement du jeu de donnees
dataset = pd.read_csv("allocine.csv")
 
dataset['pos'] = dataset.polarity
dataset['neg'] = dataset.polarity.apply(lambda p: 1 if p==0 else 0)

dataset = pd.concat([
    dataset[dataset.pos == 1].sample(9592),
    dataset[dataset.neg == 1].sample(9592)
],ignore_index=True)

train_df, test_val_df = train_test_split(dataset, test_size = 0.2)
val_df, test_df = train_test_split(test_val_df, test_size = 0.7)

tokenizer = CamembertTokenizer.from_pretrained(BERT_TOKENIZER_NAME,do_lower_case=True)

reviews = dataset['review'].values.tolist()
sentiments = dataset[['pos','neg']].values.tolist()

# La fonction batch_encode_plus encode un batch de donnees
encoded_batch = tokenizer.batch_encode_plus(reviews,
  add_special_tokens=True, max_length=MAX_LEN, return_token_type_ids=False,padding="max_length",
  truncation=True, return_attention_mask = True, return_tensors = 'pt')
 
# On transforme la liste des sentiments en tenseur
sentiments = torch.FloatTensor(sentiments)

split_border = int(len(sentiments)*0.8)
 
train_dataset = TensorDataset(
    encoded_batch['input_ids'][:split_border],
    encoded_batch['attention_mask'][:split_border],
    sentiments[:split_border])
validation_dataset = TensorDataset(
    encoded_batch['input_ids'][split_border:],
    encoded_batch['attention_mask'][split_border:],
    sentiments[split_border:])

train_dataloader = DataLoader(
            train_dataset,
            sampler = RandomSampler(train_dataset),
            batch_size = BATCH_SIZE)
 
validation_dataloader = DataLoader(
            validation_dataset,
            sampler = SequentialSampler(validation_dataset),
            batch_size = BATCH_SIZE)


model = CamembertForSequenceClassification.from_pretrained('camembert-base',num_labels = 2)

optimizer = AdamW(model.parameters(),lr = 2e-5, eps = 1e-8 )

# On va stocker nos tensors sur mon cpu : je n'ai pas mieux
device = torch.device(DEVICE)
model = model.to(device)

# Pour enregistrer les stats a chaque epoque
training_stats = []
# Boucle d'entrainement
for epoch in range(N_EPOCHS):
     
    print("")
    print(f'########## Epoch {epoch+1} / {N_EPOCHS} ##########')
    print('Training...')
 
 
    # On initialise la loss pour cette epoque
    total_train_loss = 0
 
    # On met le modele en mode 'training'
    # Dans ce mode certaines couches du modele agissent differement
    model.train()
 
    # Pour chaque batch
    bar_iter = tqdm(enumerate(train_dataloader),total=len(train_dataloader) ,desc=f"Epoch {epoch}: ")
    for step, batch in bar_iter:
         
        # On recupere les donnees du batch
        input_id = batch[0].to(device)
        attention_mask = batch[1].to(device)
        sentiment = batch[2].to(device)
 
        # On met le gradient a 0
        model.zero_grad()        
 
        # On passe la donnee au model et on recupere la loss et le logits (sortie avant fonction d'activation)
        output = model(input_id, 
                             token_type_ids=None, 
                             attention_mask=attention_mask, 
                             labels=sentiment)
        loss = output.loss 
        logits = output.logits

        # On incremente la loss totale
        # .item() donne la valeur numerique de la loss
        total_train_loss += loss.item()
        mean_loss = total_train_loss /step if step != 0 else 1
        bar_iter.set_postfix(train_loss=mean_loss)

        # Backpropagtion
        loss.backward()
 
        # On actualise les parametrer grace a l'optimizer
        optimizer.step()
 
    # On calcule la  loss moyenne sur toute l'epoque
    avg_train_loss = total_train_loss / len(train_dataloader)   
 
    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))  
     
    # Enregistrement des stats de l'epoque
    training_stats.append(
        {
            'epoch': epoch + 1,
            'Training Loss': avg_train_loss,
        }
    )
 
print("Model saved!")
torch.save(model.state_dict(), "./sentiments.pt")

model.eval()
predictions = []
sentiments = []
bar_iter = tqdm(enumerate(validation_dataloader),total=len(validation_dataloader) ,desc="Evaluation : ")
for step,batch in bar_iter:
  
  # On recupere les donnees du batch
  input_id = batch[0].to(device)
  attention_mask = batch[1].to(device)
  sentiment = batch[2].to(device)      

  # On passe la donnee au model et on recupere la loss et le logits (sortie avant fonction d'activation)
  output = model(input_id, attention_mask=attention_mask)
  loss = output.loss 
  logits = output.logits
  
  

  predictions.extend(torch.softmax(logits, dim=1).tolist())
  sentiments.extend(sentiment.tolist())

sentiments_1d = list(map(np.argmax,sentiments))
predictions_1d = list(map(np.argmax,predictions))

print(metrics.f1_score(sentiments_1d, predictions_1d, average='weighted', zero_division=0))
cm = metrics.confusion_matrix(sentiments_1d, predictions_1d)
cm_bis = [[elem / sum(row) for elem in row] for row in cm]
seaborn.heatmap(cm_bis,annot=True, fmt='.2%', cmap='Blues')