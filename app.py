from pyvi import ViTokenizer
import numpy as np
import os
import torch
from flask import Flask, request,render_template,jsonify
from flask_cors import CORS
import torch.nn as nn
import flask
from custom_model import AttentionModel
import joblib

app = Flask(__name__)
CORS(app)



def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    return model

model_path = './model'
load_model = load_checkpoint(os.path.join(model_path,'checkpoint.pth'))

# load_model = joblib.load(os.path.join(model_path,'model_joblib.pkl'))

vocab2index = torch.load(os.path.join(model_path,'vocab.pth'))
# vocab2index = joblib.load(os.path.join(model_path,'vocab_joblib.pkl'))

# CONST_THRESHOLD = 0.8

def tokenize(text):
    list_token = ViTokenizer.tokenize(text)
    return list_token.split(' ')

def encode_sentence(text, vocab2index, N=75):
    tokenized = tokenize(text)
    encoded = np.zeros(N, dtype=int)
    enc1 = np.array([vocab2index.get(word, vocab2index["UNK"]) for word in tokenized])
#     print(len(enc1))
    length = min(N, len(enc1))
    encoded[:length] = enc1[:length]
#     print(len(encoded))
    encoded_array = torch.from_numpy(encoded.astype(np.float32))
    encoded_array = torch.reshape(encoded_array,(1,N))
     ## padding
    pad_enc = torch.zeros([29,N])
    encoded_array_pad = torch.cat([encoded_array,pad_enc])
    return encoded_array_pad.long()
    # return encoded_array.long()



@app.route('/predict',methods=['POST'])
def predict():
    # text=request.get_data(as_text=True)
    input_data = request.get_json(force=True)
    text = input_data['message']
    enc_vector = encode_sentence(text,vocab2index,22)
    preds = load_model(enc_vector)
    prop_preds = nn.functional.softmax(preds,dim=1)

    pred_idx = prop_preds.argmax().item()

    label = ['type_edu','case','career']
    probability = prop_preds.tolist()[0][pred_idx]

    return jsonify({"intent": label[pred_idx],\
                    "probability":probability,\
                    "message":text})


if __name__=="__main__":
    port=int(os.environ.get('PORT',5000))
    app.run(port=port,debug=True,use_reloader=False)
