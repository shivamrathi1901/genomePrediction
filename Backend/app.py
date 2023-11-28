
from flask import Flask, request, jsonify
from flask_cors import CORS
# from sklearn.metrics import accuracy_score, f1_score, classification_report
import  sys
# from transformers import AutoTokenizer, pipeline, AutoModelForMaskedLM

tokenizer_name = "zhihan1996/DNABERT-2-117M"
model_name = "models/GenomeBERT"

# tokenizer_name = sys.argv[1]
# model_name = sys.argv[2]
# tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
# model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
# fill = pipeline('fill-mask', model=model, tokenizer=tokenizer)

app = Flask(__name__)# Load the model
CORS(app)
# fill = pipeline('fill-mask', model=model_name, tokenizer=tokenizer)

@app.route('/api',methods=['POST'])
def predict():
    # Get the data from the POST request.
    data = request.json['sequence']
    # data = request.get_json(force=True)    # Make prediction using model loaded from disk as per the data.
    print(data)
    
    # prediction = model.predict([[np.array(data['exp'])]])    # Take the first value of prediction
    # resp = fill(data)[0]['token_str'] 
    return jsonify("CATT")

if __name__ == '__main__':
    app.run(port=5000, debug=True)