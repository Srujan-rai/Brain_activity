from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load('brain_activity.pkl')  

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    eeg_label_offset_seconds = float(request.form['eeg_label_offset_seconds'])
    spectrogram_label_offset_seconds = float(request.form['spectrogram_label_offset_seconds'])
    seizure_vote = float(request.form['seizure_vote'])
    lpd_vote = float(request.form['lpd_vote'])
    gpd_vote = float(request.form['gpd_vote'])
    lrda_vote = float(request.form['lrda_vote'])
    grda_vote = float(request.form['grda_vote'])
    other_vote = float(request.form['other_vote'])

    new_data = pd.DataFrame({
        'eeg_label_offset_seconds': [eeg_label_offset_seconds],
        'spectrogram_label_offset_seconds': [spectrogram_label_offset_seconds],
        'seizure_vote': [seizure_vote],
        'lpd_vote': [lpd_vote],
        'gpd_vote': [gpd_vote],
        'lrda_vote': [lrda_vote],
        'grda_vote': [grda_vote],
        'other_vote': [other_vote]
    })

   
    predicted = model.predict(new_data)
    index = np.argmax(predicted)
    classes = ['Seizure', 'GPD', 'LRDA', 'GRDA', 'LPD', 'Other']
    predicted_class = classes[index]

    return jsonify({'predicted_class': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
