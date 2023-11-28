import numpy as np
from flask import Flask, request, jsonify
from sklearn.metrics import accuracy_score, f1_score, classification_report
import  sys, os, re, logging
from transformers import AutoTokenizer, pipeline, AutoModelForMaskedLM



tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
model = AutoModelForMaskedLM.from_pretrained("models/GenomeBERT", trust_remote_code=True)
fill = pipeline('fill-mask', model=model, tokenizer=tokenizer)

app = Flask(__name__)# Load the model

# fill = pipeline('fill-mask', model=model_name, tokenizer=tokenizer)

@app.route('/api',methods=['POST'])
def predict():
    # Get the data from the POST request.
    data = request.get_json(force=True)    # Make prediction using model loaded from disk as per the data.
    print(data)
    prediction = model.predict([[np.array(data['exp'])]])    # Take the first value of prediction
    resp = fill(test_seq)[0]['token_str'] 
    return jsonify(resp)

if __name__ == '__main__':
    app.run(port=5000, debug=True)