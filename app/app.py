from flask import Flask, render_template, request, jsonify
import torch
from model import TextSimilarityModel

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TextSimilarityModel(
    model_path='models/best_model.pth',
    classifier_head_path='models/classifier_head.pth',
    tokenizer_path='models/tokenizer',
    device=device
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    premise = request.form['premise']
    hypothesis = request.form['hypothesis']

    predicted_label = model.predict_label(premise, hypothesis)

    return jsonify({'label': predicted_label})

if __name__ == '__main__':
    app.run(debug=True)
