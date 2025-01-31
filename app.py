from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

# Initialize Flask app
app = Flask(__name__)#Creates a new Flask web application. __name__ is passed to specify the name of the app.

# Load the saved objects (model, label encoders, and scaler)
# Define file paths
model_path = r"C:\Users\bhumi\OneDrive\Desktop\model\rf_model (8).pkl"
label_encoder_size_path = r"C:\Users\bhumi\OneDrive\Desktop\model\label_encoder_size.pkl"
label_encoder_gender_path = r"C:\Users\bhumi\OneDrive\Desktop\Cloth_size\Cloth_size\model\label_encoder_gender.pkl"
scaler_path = r"C:\Users\bhumi\OneDrive\Desktop\model\scaler (1).pkl"


# Load the Random Forest model
with open(model_path, 'rb') as model_file:
    rf_model = pickle.load(model_file)

# Load the LabelEncoder for size
with open(label_encoder_size_path, 'rb') as le_file:
    le_size = pickle.load(le_file)

# Load the LabelEncoder for gender
with open(label_encoder_gender_path, 'rb') as le_gender_file:
    le_gender = pickle.load(le_gender_file)

# Load the StandardScaler
with open(scaler_path, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Get the input values from the form
            age = int(request.form["age"])  # Age input from form
            gender = request.form["gender"].lower()  # Gender input from form
            height = float(request.form["height"])  # Height input from form (in feet)
            weight = float(request.form["weight"])  # Weight input from form (in kg)

            # Validate inputs
            if not (1 <= age <= 8):
                raise ValueError("Age must be between 1 and 8 years.")
            if not (1 <= height <= 4):
                raise ValueError("Height must be between 1 and 4 feet.")
            if not (1 <= weight <= 25):
                raise ValueError("Weight must be between 1 and 25 kg.")
            if gender not in ['male', 'female']:
                raise ValueError("Gender must be 'male' or 'female'.")

            # Map 'male' to 'boy' and 'female' to 'girl'
            gender_mapped = 'boy' if gender == 'male' else 'girl'

            # Prepare the input data in the same format as the training data
            input_data = np.array([[age, gender_mapped, height, weight]])

            # Ensure the gender input is properly transformed using the fitted LabelEncoder for gender
            input_data[:, 1] = le_gender.transform(input_data[:, 1])  # Using transform for gender only

            # Apply StandardScaler to the input data
            input_data_scaled = scaler.transform(input_data)

            # Make the prediction for size
            predicted_size_encoded = rf_model.predict(input_data_scaled)

            # Decode the predicted size using the correct LabelEncoder for size
            predicted_size = le_size.inverse_transform(predicted_size_encoded)[0]

            # Return the result to the frontend, passing the age and gender
            return render_template("index.html", prediction=predicted_size)
        except ValueError as e:
            # Handle validation errors and return the error message
            return render_template("index.html", prediction=f"Error: {e}", age=None)

    return render_template("index.html", prediction=None, age=None)

if __name__ == "__main__":
    app.run(debug=True)
