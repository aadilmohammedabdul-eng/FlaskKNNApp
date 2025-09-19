# app.py
from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np

app = Flask(__name__)

# Load the pre-trained KNN model
model = joblib.load('knn_model.joblib')

# Simple HTML form to input features
html = """
<!DOCTYPE html>
<html>
<head><title>KNN Prediction</title></head>
<body>
  <h2>KNN Prediction Web Interface</h2>
  <form id="predictForm">
    <label>Feature 1: <input type="number" step="any" name="f1" required></label><br>
    <label>Feature 2: <input type="number" step="any" name="f2" required></label><br>
    <label>Feature 3: <input type="number" step="any" name="f3" required></label><br>
    <label>Feature 4: <input type="number" step="any" name="f4" required></label><br>
    <button type="submit">Predict</button>
  </form>
  <h3 id="result"></h3>
  <script>
    const form = document.getElementById('predictForm');
    form.onsubmit = async e => {
      e.preventDefault();
      const formData = new FormData(form);
      const data = {
        f1: parseFloat(formData.get('f1')),
        f2: parseFloat(formData.get('f2')),
        f3: parseFloat(formData.get('f3')),
        f4: parseFloat(formData.get('f4'))
      };
      const response = await fetch('/predict', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(data)
      });
      const result = await response.json();
      document.getElementById('result').innerText = "Predicted class: " + result.prediction;
    };
  </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(html)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array([[data['f1'], data['f2'], data['f3'], data['f4']]])
    pred = model.predict(features)[0]
    return jsonify({'prediction': int(pred)})

if __name__ == '__main__':
    app.run(debug=True)
