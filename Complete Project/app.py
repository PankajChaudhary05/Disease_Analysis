from flask import Flask, render_template, request,redirect, url_for, session
import os, joblib, pickle
import numpy as np
from flask_mysqldb import MySQL
import re
import MySQLdb.cursors

app = Flask(__name__)


###################################################Login ###############################################
app.secret_key = 'jbfjhbef'

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'backend'
mysql = MySQL(app)

@app.route('/')
@app.route('/login', methods=['GET', 'POST'])
def login():
	msg = ''
	if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
		username = request.form['username']
		password = request.form['password']
		cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
		cursor.execute(
			'SELECT * FROM users WHERE username = % s \
			AND password = % s', (username, password, ))
		account = cursor.fetchone()
		if account:
			session['loggedin'] = True
			session['id'] = account['id']
			session['username'] = account['username']
			msg = 'Logged in successfully !'
			return render_template('index.html', msg=msg)
		else:
			msg = 'Incorrect username / password !'
	return render_template('login.html', msg=msg)
@app.route('/register', methods=['GET', 'POST'])
###############################################################register #################################################################
def register():
	msg = ''
	if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form and 'address' in request.form and 'city' in request.form and 'country' in request.form and 'postalcode' in request.form and 'organisation' in request.form:
		username = request.form['username']
		password = request.form['password']
		email = request.form['email']
		organisation = request.form['organisation']
		address = request.form['address']
		city = request.form['city']
		state = request.form['state']
		country = request.form['country']
		postalcode = request.form['postalcode']
		cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
		cursor.execute(
			'SELECT * FROM users WHERE username = % s', (username, ))
		account = cursor.fetchone()
		if account:
			msg = 'Account already exists !'
		elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
			msg = 'Invalid email address !'
		elif not re.match(r'[A-Za-z0-9]+', username):
			msg = 'name must contain only characters and numbers !'
		else:
			cursor.execute('INSERT INTO users VALUES \
			(NULL, % s, % s, % s, % s, % s, % s, % s, % s, % s)',
						(username, password, email, 
							organisation, address, city,
							state, country, postalcode, ))
			mysql.connection.commit()
			msg = 'You have successfully registered !'
	elif request.method == 'POST':
		msg = 'Please fill out the form !'
	return render_template('register.html', msg=msg)



###########################Index #############################################
@app.route('/index/')
def index():
    return render_template("index.html")
############################Contact ########################################3
@app.route('/contact/')
def contact():
    return render_template("contact.html")
########################topic_details ##################################################################################################
@app.route('/detail/')
def detail():
    return render_template("detail.html")

###################################################topic listing #############################################################################################
@app.route('/listing/')
def listing():
    return render_template("listing.html")
#######################################3#############next page #####################################################################################
@app.route('/next/')
def next():
    return render_template("next.html")
##################################################Diabetic Prediction ######################################################
Standardscaler=pickle.load(open('models/standardscaler.pkl','rb'))
model_prediction=pickle.load(open('models/modelforprediction.pkl','rb'))
@app.route('/diabetic_home/')
def diabetic_home():
    return render_template("diabetic_home.html")

@app.route("/Predict_model", methods=['POST', 'GET'])
def Predict_model():
    if request.method=="POST":
        Pregnancies=float(request.form.get("Pregnancies"))
        Glucose=float(request.form.get("Glucose"))
        BloodPressure=float(request.form.get("BloodPressure"))
        SkinThickness=float(request.form.get("SkinThickness"))
        Insulin=float(request.form.get("Insulin"))
        BMI=float(request.form.get("BMI"))
        DiabetesPedigreeFunction=float(request.form.get("DiabetesPedigreeFunction"))
        Age=float(request.form.get('Age'))
        new_data_scaled = Standardscaler.transform([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        result=model_prediction.predict(new_data_scaled)
        return render_template('diabetic_result.html',result=result)

    else:
        return render_template("diabetic_home.html")



#####################################################Stroke disease #############################################################

@app.route('/stroke_home/')
def stroke_home():
    return render_template("stroke_home.html")
@app.route("/result", methods=['POST', 'GET'])

def result():
    gender = int(request.form['gender'])
    age = int(request.form['age'])
    hypertension = int(request.form['hypertension'])
    heart_disease = int(request.form['heart_disease'])
    work_type = int(request.form['work_type'])
    Residence_type = int(request.form['Residence_type'])
    avg_glucose_level = float(request.form['avg_glucose_level'])
    bmi = float(request.form['bmi'])
    smoking_status = int(request.form['smoking_status'])

    x = np.array([gender, age, hypertension, heart_disease, work_type, Residence_type, avg_glucose_level, bmi,
                  smoking_status]).reshape(1, -1)
    scaler = pickle.load(open('notebook\scalar.pkl', 'rb'))

    x = scaler.transform(x)

    lr = pickle.load(open('notebook//finalized_model.pkl', 'rb'))
    y_pred_lr = lr.predict(x)

   
    if y_pred_lr == 0:
        return render_template("nostroke.html")
    else:
        return render_template("stroke.html")



##############################################Parkisons prediction ###################################################################

@app.route('/parkison_home/')
def parkison_home():
    return render_template("parkison_home.html")

@app.route('/predict_result',methods=['POST'])
def predict_result():
    model=joblib.load('notebook//model_1.pkl')
    scaler=joblib.load('notebook//scaler_1.pkl')
    input_data=[float(x) for x in request.form.values()]
    input_data_array=np.asarray(input_data)
    input_data_reshaped=input_data_array.reshape(1,-1)
    std=scaler.transform(input_data_reshaped)
    prediction=model.predict(std)

    if prediction[0]==0:
        result= 'does not have'
    else:
        result= 'has'

    prediction_result = int(prediction[0])


    return render_template('parkison_home.html',prediction_text=result)

#############################################Heart Disease type predection ####################################################
#model = pickle.load(open('models//heartDisease_model.pkl', 'rb'))
@app.route("/heart_home/")
def heart_home():
    return render_template('heart_home.html')

@app.route('/predict_1',methods=['POST'])
def predict_1():    
    model = pickle.load(open('models//heartDisease_model.pkl', 'rb'))
    """Grabs the input values and uses them to make prediction"""
    age = int(request.form["age"])
    sexe = int(request.form["sex"])
    cpt = int(request.form["cpt"])
    bp = int(request.form["bp"])
    chol = int(request.form["chol"])
    fbs = int(request.form["fbs"])
    ecg = int(request.form["ecg"])
    hr = int(request.form["hr"])
    exang = int(request.form["exang"])
    oldpeak = float(request.form["oldpeak"])
    slope = int(request.form["slope"])
    majVessel = int(request.form["majVessel"])
    thal = int(request.form["thal"])

   
    features = np.array([age, sexe, cpt, bp, chol, fbs, ecg, hr, exang, oldpeak, slope, majVessel,thal])
    features = np.reshape(features,(1, features.shape[0]))
    prediction = model.predict(features)
    value = prediction[0]

    if value == 0:
        return render_template('heart_home.html', prediction_result = f'No heart disease. You got an healthy heart')
    else:
        return render_template('heart_home.html', prediction_result = f'Heart failure possibility. You have to take a good care of your health.')
    

if __name__ == '__main__':
    app.run(debug=True)
