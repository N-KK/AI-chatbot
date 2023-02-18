import random
import json
import torch
from model import NeuralNet
from nltk_utils import bagset, tokenize

device=torch.device('cpu')

with open('tagresp.json','r') as f:
    tagresp=json.load(f)

FILE="data.pth"
data=torch.load(FILE)

input_size=data["input_size"]
hidden_size=data["hidden_size"]
output_size=data["output_size"]
wordset=data["wordset"]
tags=data["tags"]
model_state=data["model_state"]

model= NeuralNet(input_size,hidden_size,output_size).to(device)
model.load_state_dict(model_state)
model.eval()

botname="Annie"


def get_response(msg):
    sentence=tokenize(msg)
    X= bagset(sentence,wordset)
    X=X.reshape(1,X.shape[0])
    X=torch.from_numpy(X)

    output=model(X)
    _, predicted=torch.max(output,dim=1)
    tag=tags[predicted.item()]

    probs=torch.softmax(output, dim=1)
    prob=probs[0][predicted.item()]

    if prob.item()>0.75:
        for ele in tagresp["tagresp"]:
            if tag==ele["tag"]:
                return random.choice(ele["responses"])
    return "I do not understand..."
 
    
