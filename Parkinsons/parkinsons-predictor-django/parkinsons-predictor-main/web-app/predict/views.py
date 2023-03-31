from django.shortcuts import render
import pandas as pd
import numpy as np
import joblib

# Home page
def home(request):
    return render(request, "index.html")


# Predict page
def predict(request):
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
    return render(request, "index.html", context=context)