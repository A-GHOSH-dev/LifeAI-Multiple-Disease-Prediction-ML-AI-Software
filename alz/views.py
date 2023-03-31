from django.shortcuts import render
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

import joblib
#Alzheimers

from posixpath import abspath
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import os
import h5py
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
from keras.models import load_model
import keras.utils as image

#END

from alz.forms import BreastCancerForm, DiabetesForm, HeartDiseaseForm


def heart(request):

    df = pd.read_csv('static/Heart_train.csv')
    data = df.values
    X = data[:, :-1]
    Y = data[:, -1:]


    value = ''

    if request.method == 'POST':

        age = float(request.POST['age'])
        sex = float(request.POST['sex'])
        cp = float(request.POST['cp'])
        trestbps = float(request.POST['trestbps'])
        chol = float(request.POST['chol'])
        fbs = float(request.POST['fbs'])
        restecg = float(request.POST['restecg'])
        thalach = float(request.POST['thalach'])
        exang = float(request.POST['exang'])
        oldpeak = float(request.POST['oldpeak'])
        slope = float(request.POST['slope'])
        ca = float(request.POST['ca'])
        thal = float(request.POST['thal'])

        user_data = np.array(
            (age,
             sex,
             cp,
             trestbps,
             chol,
             fbs,
             restecg,
             thalach,
             exang,
             oldpeak,
             slope,
             ca,
             thal)
        ).reshape(1, 13)

        rf = RandomForestClassifier(
            n_estimators=16,
            criterion='entropy',
            max_depth=9
        )

        rf.fit(np.nan_to_num(X), Y)
        rf.score(np.nan_to_num(X), Y)
        predictions = rf.predict(user_data)

        if int(predictions[0]) == 1:
            value = 'have'
        elif int(predictions[0]) == 0:
            value = "don\'t have"

    return render(request,
                  'heart.html',
                  {
                      'context': value,
                      'title': 'Heart Disease Prediction',
                      'active': 'btn btn-success peach-gradient text-white',
                      'heart': True,
                      'form': HeartDiseaseForm(),
                  })


def diabetes(request):

    dfx = pd.read_csv('static/Diabetes_XTrain.csv')
    dfy = pd.read_csv('static/Diabetes_YTrain.csv')
    X = dfx.values
    Y = dfy.values
    Y = Y.reshape((-1,))


    value = ''
    if request.method == 'POST':

        pregnancies = float(request.POST['pregnancies'])
        glucose = float(request.POST['glucose'])
        bloodpressure = float(request.POST['bloodpressure'])
        skinthickness = float(request.POST['skinthickness'])
        bmi = float(request.POST['bmi'])
        insulin = float(request.POST['insulin'])
        pedigree = float(request.POST['pedigree'])
        age = float(request.POST['age'])

        user_data = np.array(
            (pregnancies,
             glucose,
             bloodpressure,
             skinthickness,
             bmi,
             insulin,
             pedigree,
             age)
        ).reshape(1, 8)

        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X, Y)

        predictions = knn.predict(user_data)

        if int(predictions[0]) == 1:
            value = 'have'
        elif int(predictions[0]) == 0:
            value = "don\'t have"

    return render(request,
                  'diabetes.html',
                  {
                      'result': value,
                      'title': 'Diabetes Disease Prediction',
                      'active': 'btn btn-success peach-gradient text-white',
                      'diabetes': True,
                      'form': DiabetesForm(),
                  }
                  )


def breast(request):

    df = pd.read_csv('static/Breast_train.csv')
    data = df.values
    X = data[:, :-1]
    Y = data[:, -1]
    print(X.shape, Y.shape)


    value = ''
    if request.method == 'POST':

        radius = float(request.POST['radius'])
        texture = float(request.POST['texture'])
        perimeter = float(request.POST['perimeter'])
        area = float(request.POST['area'])
        smoothness = float(request.POST['smoothness'])

        rf = RandomForestClassifier(
            n_estimators=16, criterion='entropy', max_depth=5)
        rf.fit(np.nan_to_num(X), Y)

        user_data = np.array(
            (radius,
             texture,
             perimeter,
             area,
             smoothness)
        ).reshape(1, 5)

        predictions = rf.predict(user_data)
        print(predictions)

        if int(predictions[0]) == 1:
            value = 'have'
        elif int(predictions[0]) == 0:
            value = "don\'t have"

    return render(request,
                  'breast.html',
                  {
                      'result': value,
                      'title': 'Breast Cancer Prediction',
                      'active': 'btn btn-success peach-gradient text-white',
                      'breast': True,
                      'form': BreastCancerForm(),
                  })


def home(request):

    return render(request,
                  'home.html')



def handler404(request):
    return render(request, '404.html', status=404)


# Alzheimers
def alhome(request):
    
    return render(request,
                  'alhome.html')


def predict(request):
    return render(request,'predict.html')

def res(request):
    
    fobj=request.FILES["img"]
    fs=FileSystemStorage()
    img=fs.save(fobj.name,fobj)
    path=fs.url(img)


    abpath=os.path.abspath(__file__)
    abpath=os.path.dirname(abpath)
    abpath=abpath.replace("\\","/")
    var=abpath+'/media/'+img    


    context={"imgname":img, "imgpath":path}

    model=load_model('C:\\Users\\anany\\alzheimersmine.h5')

    img=image.load_img(var,target_size=(128, 128))
    #208, 176
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = model.predict(images, batch_size=32)
    x=str(np.argmax(classes[0]))
    d={'0':'Mild Dementia','1':'Moderate Dementia','2':'No Dementia','3':'Very Mild Dementia'}
    ans=d[x]

    context['ans']=ans

    return render(request,'x.html',context)


#END



#Parkinsons

# Predict page
def ppredict(request):
    # Fetch data from the frontend
    mdvp_fo = float(request.POST["MDVP:Fo(Hz)"])
    mdvp_fhi = float(request.POST["MDVP:Fhi(Hz)"])
    mdvp_flo = float(request.POST["MDVP:Flo(Hz)"])
    mdvp_jitter = float(request.POST["MDVP:Jitter(%)"])
    mdvp_abs = float(request.POST["MDVP:Jitter(Abs)"])
    mdvp_rap = float(request.POST["MDVP:RAP"])
    mdvp_ppq = float(request.POST["MDVP:PPQ"])
    ddp = float(request.POST["Jitter:DDP"])
    shimmer = float(request.POST["MDVP:Shimmer"])
    mdvp_db = float(request.POST["MDVP:Shimmer(dB)"])
    mdvp_apq3 = float(request.POST["Shimmer:APQ3"])
    mdvp_apq5 = float(request.POST["Shimmer:APQ5"])
    mdvp_apq = float(request.POST["MDVP:APQ"])
    dda = float(request.POST["Shimmer:DDA"])
    nhr = float(request.POST["NHR"])
    hnr = float(request.POST["HNR"])
    rpde = float(request.POST["RPDE"])
    dfa = float(request.POST["DFA"])
    spread_1 = float(request.POST["spread1"])
    spread_2 = float(request.POST["spread2"])
    d2 = float(request.POST["D2"])
    ppe = float(request.POST["PPE"])

    data = [mdvp_fo, mdvp_fhi, mdvp_flo, mdvp_jitter, mdvp_abs, mdvp_rap, mdvp_ppq,
            ddp, shimmer, mdvp_db, mdvp_apq3, mdvp_apq5, mdvp_apq, dda, nhr, hnr, rpde,
            dfa, spread_1, spread_2, d2, ppe]

    # Import tools for preprocessing
    scaler = joblib.load("tools/scaler_joblib")

    # Import model
    model = joblib.load("tools/model_joblib")    


    # scale data
    data = scaler.transform([data])
    # Getting prediction 
    prediction = model.predict(np.array(data))

    context = {
        "prediction" : prediction}
    return render(request, "indexp.html", context=context)




def indexp(request):
    
    return render(request,
                  'indexp.html')





#end