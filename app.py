import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__)

# Load the model, scaler, and polynomial features converter
model = pickle.load(open("poly_regmodel.pkl", "rb"))
scaler = pickle.load(open("scaling.pkl", "rb"))
polynomial_converter = pickle.load(open("polynomial_converter.pkl", "rb"))

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print("Input Data:", data)
    
    # Convert input data to numpy array
    input_array = np.array(list(data.values())).reshape(1, -1)
    print("Input Array:", input_array) 
     
    # Scale the polynomial features
    scaled_data = scaler.transform(input_array)
    print("Scaled Data:", scaled_data)
    
    # Transform the input data to polynomial features
    new_data = polynomial_converter.transform(scaled_data)
    print("Polynomial Features:", new_data)
   
    # Predict the output
    output = model.predict(new_data)
    print("Model Output:", output[0])
    
    return jsonify(output[0])

if __name__ == "__main__":
    app.run(debug=True)
