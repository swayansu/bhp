import pickle
import json
# from flask import Flask, request, jsonify

import numpy as np

__locations = None
__data_cols = None
__model = None


def get_estimated_costs(location, sqft, bhk, bath):
    try:
        loc_index = __data_cols.index(location.lower())
    except:
        loc_index = -1
    X = np.zeros(len(__data_cols))
    X[0] = sqft
    X[1] = bath
    X[2] = bhk
    if(loc_index >= 0):
        X[loc_index] = 1
    return round(__model.predict([X])[0], 2)


def get_location_names():
    return __locations


def load_saved_artifacts():
    print('Loading Saved artifacts...')
    global __data_cols
    global __locations

    with open("./artifacts/columns.json", 'r') as f:
        __data_cols = json.load(f)["data_colums"]
        __locations = __data_cols[3:]

    with open("./artifacts/house_price_prediction_model.pickle", 'rb') as f:
        global __model
        __model = pickle.load(f)

    print('Loading the artifacts completeed...')


if __name__ == "__main__":
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_costs('1st Phase JP Nagar', 1000, 3, 3))
