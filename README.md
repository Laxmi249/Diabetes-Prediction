# Diabetes-Prediction

# 🩺 Diabetes Prediction using Machine Learning

This project is a simple web application built with **Flask** that uses a trained machine learning model to predict whether a person is likely to have diabetes based on medical parameters.

## 💡 Overview

This application allows users to input health-related information such as glucose level, blood pressure, insulin, BMI, and more. It then uses a pre-trained model (e.g., Logistic Regression, Random Forest, etc.) to predict if the user is diabetic or not.

## 🚀 Features

- User-friendly web interface (Flask)
- Machine learning model for prediction
- Takes 8 health-related inputs
- Gives immediate feedback on diabetes risk
- Deployed locally or can be hosted on cloud platforms like Heroku or PythonAnywhere

## 📊 Tech Stack

- Python
- Flask (Web Framework)
- Scikit-learn (Machine Learning)
- Pandas & NumPy (Data Handling)
- HTML/CSS (Frontend)
- Matplotlib (for visualizations, optional)

## 📁 Project Structure

diabetes_prediction/ ├── static/ │ └── style.css ├── templates/ │ └── index.html ├── app.py ├── model.pkl ├── requirements.txt └── README.md


## 🧠 Model Details

- Dataset: [PIMA Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- Algorithm Used: Logistic Regression (or mention your actual algorithm)
- Accuracy: ~80% (you can update with actual score)

## 🛠️ Setup Instructions

1. Clone the repository  
git clone https://github.com/Laxmi249/Diabetes-Prediction.git

2. Navigate to the project folder  
cd Diabetes-Prediction

3. Create a virtual environment (optional but recommended)  
python -m venv venv venv\Scripts\activate

4. Install dependencies  
pip install -r requirements.txt

5. Run the app  
python app.py


