import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle


app = Flask(__name__)

dataset = pd.read_csv('diabetes_data_processed.csv')

dataset_X = dataset.iloc[:,0:-1].values

from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
dataset_X1 = ss.fit_transform(dataset_X)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    symptoms = []
    Age=0
    Polydipsia = 0
    Sudden_weight_loss = 0
    Partial_paresis = 0
    Irritability = 0
    Polyphagia = 0
    Visual_blurring = 0

    Age = int(request.form['Age'])
    Gender = request.form['Gender']
    #Polyria = request.form['Polyria']

    symptoms = request.form.getlist('symptom')

    for symptm in symptoms:
        if symptm == "Polydipsia" :
            Polydipsia = 1
        elif symptm == "Swl" :
            Sudden_weight_loss = 1
        elif symptm == "Paresis" :
            Partial_paresis = 1
        elif symptm == "Irritability" :
            Irritability = 1
        elif symptm == "Polyphagia" :
            Polyphagia = 1
        elif symptm == "Blurring" :
            Visual_blurring = 1
        else:
            Others = 1



    '''
    if Polyria is None:
        print("None")
    else:
        print(Polyria)
    '''

    print(Age)
    print(Gender)
    print(symptoms)

    print(Polydipsia)


    model = pickle.load(open('model.pkl','rb'))
    predict_features = [np.array([Polydipsia,Sudden_weight_loss,Partial_paresis,Irritability,Polyphagia,Age,Visual_blurring])]

    #predict_features = [np.array([0,1,1,0,0,46,1])]

    final_predict_features = ss.transform(predict_features)

    prediction = model.predict(ss.transform(predict_features))

    print(prediction[0])

    if prediction[0] == 0 :
        output = " You don't have Diabetes.Keep living healthy! "
    else:
        output = "There is high chance of diabetes. Consult a doctor."

    return render_template('index.html',prediction_output='{}'.format(output))


'''
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model.predict( sc.transform(final_features) )

    if prediction == 1:
        pred = "You have Diabetes, please consult a Doctor."
    elif prediction == 0:
        pred = "You don't have Diabetes."
    output = pred

    return render_template('index.html',prediction[0])
'''








if __name__ == "__main__":
    app.run(debug=True)

