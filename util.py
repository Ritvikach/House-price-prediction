import json
import pickle

import numpy as np
import sklearn

__location=None
__data_columns=None
__model=None

def get_location_names():
    return __location

def load_saved_artifacts():
    print("loading saved artifacts....start")
    global __data_columns
    global __location
    global __model
    with open("./artifacts/columns.json",'r') as f:
        __data_columns=json.load(f)['data_columns']
        __location=__data_columns[3:]

    with open("./artifacts/home_price_model.pickle",'rb') as f:
        __model=pickle.load(f)
    print("loading the artifacts is done")

def get_estimated_price(location, sqft, bath, bhk):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index=-1

    x = [0] * len(__data_columns)
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return round(__model.predict([x])[0],2)

if __name__=='__main__':
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price("1st phase jp nagar",1000,3,3))
    print(get_estimated_price("1st phase jp nagar", 1000, 2, 2))
