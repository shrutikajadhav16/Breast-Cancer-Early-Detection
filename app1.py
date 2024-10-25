from flask import Flask, request, send_file, jsonify, render_template
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__, static_url_path='/static')

# Load patient data from CSV (change this to the path of your CSV file)
df = pd.read_csv('hospital.csv')

# Load the prediction model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('prediction.html')

@app.route('/generate-chart', methods=['POST'])
def generate_chart():
    data = request.get_json()
    visualization_type = data['visualizationType']
    graph_type = data['graphType']

    # Chart data based on the selected visualization type
    if visualization_type == 'patient_gender':
        chart_data = df['patient_gender'].value_counts()
        title = 'Patient Gender Distribution'
    elif visualization_type == 'patient_race':
        chart_data = df['patient_race'].value_counts()
        title = 'Patient Race Distribution'
    elif visualization_type == 'department_referral':
        chart_data = df['department_referral'].value_counts()
        title = 'Department Distribution'
    elif visualization_type == 'age_distribution':
        chart_data = df['Age_Group'].value_counts()
        title = 'Patient Age Distribution'

    # Create the plot
    plt.figure(figsize=(6.8, 6))
    if graph_type == 'bar':
        bars = chart_data.plot(kind='bar')
        y_label = 'Number of Patients'
        for bar in bars.patches:
            bars.annotate(f'{int(bar.get_height())}', 
                          (bar.get_x() + bar.get_width() / 2, bar.get_height()), 
                          ha='center', va='bottom')
    elif graph_type == 'pie':
        labels = [f'{index} ({count})' for index, count in zip(chart_data.index, chart_data.values)]
        chart_data.plot(kind='pie', labels=labels, startangle=90, legend=False)
        y_label = ''
        plt.axis('equal')

    plt.title(title)
    plt.ylabel(y_label)

    # Save the plot to a bytes buffer
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close()  # Close the plot to free memory

    return send_file(img, mimetype='image/png')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = np.array(features).reshape(1, -1)
    final_features = scaler.transform(final_features)    
    prediction = model.predict(final_features)
    y_probabilities_test = model.predict_proba(final_features)
    y_prob_success = y_probabilities_test[:, 1]
    
    output = round(prediction[0], 2)
    y_prob = round(y_prob_success[0], 3)

    if output == 0:
        prediction_text = 'THE PATIENT IS MORE LIKELY TO HAVE A BENIGN CANCER WITH PROBABILITY VALUE {}'.format(y_prob)
    else:
        prediction_text = 'THE PATIENT IS MORE LIKELY TO HAVE A MALIGNANT CANCER WITH PROBABILITY VALUE {}'.format(y_prob)
    
    return render_template('prediction.html', prediction_text=prediction_text)

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

if __name__ == "__main__":
    app.run(debug=True)
