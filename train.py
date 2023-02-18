import json
from nltk_utils import tokenize, stem, bagset
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import NeuralNet

with open('tagresp.json','r') as f:
    tagresp=json.load(f)

wordset= []
tags=[]
xy=[]
for ele in tagresp['tagresp']:
    tag=ele['tag']
    tags.append(tag)
    for pattern in ele['patterns']:
        w=tokenize(pattern)
        wordset.extend(w)
        xy.append((w,tag))
ignore_words=['?','!','.',',']
wordset=[stem(w) for w in wordset if w not in ignore_words]
wordset=sorted(set(wordset))
tags=sorted(set(tags))

X_train=[]
y_train=[]
for (sent, tag) in xy:
    bag=bagset(sent,wordset)
    X_train.append(bag)

    label=tags.index(tag)
    y_train.append(label)

X_train=np.array(X_train)
y_train=np.array(y_train)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_Annieples=len(X_train)
        self.x_data=X_train
        self.y_data=y_train

    def __getitem__(self,index):
        return self.x_data[index],self.y_data[index]

    def __len__(self):
        return self.n_Annieples

#Hyperparameters
batch_size=8
hidden_size=8
output_size=len(tags) #no of classes
input_size=len(X_train[0])
learning_rate=0.001
epochno=2000

dataset=ChatDataset()
train_loader= DataLoader(dataset=dataset,batch_size=batch_size, shuffle=True,num_workers=0)

device=torch.device('cpu')
model= NeuralNet(input_size,hidden_size,output_size).to(device)
 
#loss and optimizer
criterion= nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochno):
    for(words, labels) in train_loader:
        words=words.to(device)
        labels = labels.to(dtype=torch.long)
        labels=labels.to(device)

        #forward
        outputs= model(words)
        loss=criterion(outputs, labels)

        #backward and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if(epoch+1)%100==0:
        print(f'epoch {epoch+1}/{epochno},loss={loss.item():.4f}')
        
print(f'final loss, loss={loss.item():.4f}')


data= {
    "model_state": model.state_dict(),
    "input_size":input_size,
    "output_size":output_size,
    "hidden_size":hidden_size,
    "wordset":wordset,
    "tags":tags
}

FILE="data.pth"
torch.save(data,FILE)

print(f'Training complete. File saved to {FILE}')