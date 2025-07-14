from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('sonar_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = [float(x) for x in request.form.values()]
        input_array = np.asarray(input_data).reshape(1, -1)
        prediction = model.predict(input_array)[0]

        result = "Rock 🪨" if prediction == 1 else "Mine 🧨"
        return render_template('index.html', prediction_text=f"🔎 Prediction: {result}")
    except Exception as e:
        return render_template('index.html', prediction_text=f"❌ Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=False)
