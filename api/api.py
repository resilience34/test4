#to run :
#en mode local :dans le dossier du fichier api.py faire python app.py puis dans le navigateur aller à http://127.0.0.1:5000/

# imporeter les packages et librairies 
import pandas as pd
import numpy as np
from flask import Flask, render_template, url_for, request,jsonify
import pickle
import math
import base64
from zipfile import ZipFile
from lightgbm import LGBMClassifier
import uvicorn
import json
#app = FastAPI()
# Création de l'instance de l'application Flask
app = Flask(__name__)


# Définir une route pour la validation de l'id_client
@app.route('/check_id/<int:id_client>', methods=['GET'])#, methods=['POST']
def check_id(id_client):
    z = ZipFile("real_data_clean_test.zip")
    data_id = pd.read_csv(z.open('real_data_clean_test.csv'), encoding ='utf-8')
    all_id_client = list(data_id['SK_ID_CURR'].unique())
    
    # Vérifier si l'ID client est présent dans la liste des ID client du jeu de données
    id_client = int(id_client)
    if id_client not in all_id_client:

        return  jsonify()  # Renvoyer une réponse JSON vide
    else:
        # Sélectionner les données correspondantes à l'ID client
        check_id = data_id[data_id['SK_ID_CURR'] == id_client] # qui contient uniquement les colonnes à correspond à l'ID client spécifié
                
        # Transformer le dataset en dictionnaire
        # Convertir le dictionnaire en JSON
        json_check_id = json.dumps(check_id.to_dict(orient='records'), allow_nan=True)

        #json_check_id = check_id.to_json(orient='records')
        # Renvoyer les données clients
        return json_check_id


# Définir une route pour l'identifiant client
@app.route('/client_id/<int:id_client>', methods=['GET']) # GET par défaut 
def client_id(id_client):
    z = ZipFile("real_data_clean_test.zip")
    data_origin = pd.read_csv(z.open('real_data_clean_test.csv'), encoding ='utf-8')

    # id_client en int
    id_client = int(id_client)
    # Sélectionner les données correspondantes à l'id_client
    #client_id = data_origin[data_origin['SK_ID_CURR'] == id_client] # qui contient uniquement les colonnes à correspond à l'ID client spécifié
    client_id =  data_origin
    # Transformer le dataset en dictionnaire
    # Convertir le dictionnaire en JSON
    json_client_id = json.dumps(client_id.to_dict(orient='records'), allow_nan=True)

    #json_client_id = client_id.to_json(orient='records')

    # Renvoyer les données clients
    return json_client_id


# Définir une route pour la prédiction
@app.route('/predict/<int:id_client>', methods=['GET']) # GET par défaut 
def predict(id_client):
    z = ZipFile("df_test_imputed.zip") # le dataset final avec Standardisation et encodage
    data_clean = pd.read_csv(z.open('df_test_imputed.csv'), encoding='utf-8')

    # id_client en int
    id_client = int(id_client)
    # Sélectionner les données correspondantes à l'ID client
    predict = data_clean[data_clean['SK_ID_CURR'] == id_client] # qui contient uniquement les colonnes à correspond à l'ID client spécifié

    # Transformer le dataset en dictionnaire
    # Convertir le dictionnaire en JSON
    json_predict = json.dumps(predict.to_dict(orient='records'), allow_nan=True)

    #json_predict = predict.to_json(orient='records')

    # Renvoyer les données clients
    return json_predict


# Définir une route pour les information client
@app.route('/inf_client/<int:id_client>', methods=['GET']) # GET par défaut 
def inf_client(id_client):
    z = ZipFile("test_imputed_without_standardisation.zip")
    data_clean_without_standard = pd.read_csv(z.open('test_imputed_without_standardisation.csv'), encoding ='utf-8')

    # id_client en int
    id_client = int(id_client)

    # Sélectionner les données correspondantes à l'ID client
    inf_client = data_clean_without_standard[data_clean_without_standard['SK_ID_CURR'] == id_client] 

    # Transformer le dataset en dictionnaire
    # Convertir le dictionnaire en JSON
    json_inf_client = json.dumps(inf_client.to_dict(orient='records'), allow_nan=True)


    # Transformer le dataset en json
    # orient='records' réorganise les données du DataFrame 
    # pour les représenter sous forme de dictionnaires dans un tableau JSON
    #json_inf_client = inf_client.to_json(orient='records')

    # Renvoyer les données clients
    return json_inf_client

if __name__ == "__main__":
    app.run(debug=True)
     #uvicorn.run(app)

