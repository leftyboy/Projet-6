import pandas as pd
import os
import numpy as np
import urllib
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings("ignore")
import re
from nltk.corpus import stopwords, wordnet
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import mglearn
from mglearn import tools
from sklearn.externals import joblib
from sklearn import model_selection
import glob

# Chargement des modules
from modules import get_stop_words, count_tags, get_wordnet_pos, text_processing,text_processing_stop_words, text_processing_advanced,data_preprocessing, data_preprocessing_test, ml_create_target_df


## I. Chargement des données
current_path = os.getcwd()
parent_path = os.path.dirname(current_path)
import glob
allFiles = glob.glob(parent_path + "\data\Query*.csv")
list_ = []
for file in allFiles:
    print("Chargement du fichier: {}".format(file))
    try:
        df = pd.read_csv(file,index_col=None, header=0)
        list_.append(df)
    except FileNotFoundError:
        print("Merci de vérifier la présence de {} dans le répertoire 'data'".format(file))
df_raw = pd.concat(list_, axis = 0, ignore_index = True)
print("Concaténation effectuée et dataframe prête") 

# Nous sélectionnons un échantillon et uniquement les colonnes jugées utiles pour notre étude
df_raw = df_raw.loc[:, ['Title', 'Tags','Body']]
df_raw=df_raw.sample(50000, random_state=0)


# Suppression des lignes sans Tags
df_raw = df_raw.dropna(subset=['Tags'])


# Nettoyage et traitement des doublons
#Suppression des doublons
print('Entrées en double: {}'.format(df_raw.duplicated(subset=["Body"]).sum()))
df_raw.drop_duplicates(subset=["Body"],inplace = True)
print("Taille dataframe : {}".format(df_raw.shape))

## II. Traitement des TAGS

# feature engineering. On crée 5 nouvelles features pour les tags
question_tags_featured = ['Tag_1', 'Tag_2', 'Tag_3', 'Tag_4', 'Tag_5']

# On retire les Tags chevrons
df_raw['Tags'] = df_raw.Tags.apply(lambda x: x.strip('<').strip('>').replace('>', '').replace('<', '/'))

# On compte les Tags présents
df_raw['number_of_Tags'] = df_raw.Tags.apply(lambda x: len(x.split('/')))

# découpage des tags
tags_lists = df_raw.Tags.apply(lambda x: x.split('/'))

# Initialisation de la nouvelle liste de Tags
filled_tags_list = []
# boucle sur la liste de tags
for inner_list in tags_lists:
    # taille de la liste
    length = len(inner_list)
    # While length not equal to 5 append nans
    while length < 5:
        inner_list.append(np.nan)
        length = len(inner_list)
    # Add extended list to new list
    filled_tags_list.append(inner_list)

# Création de dataframe
tags_df = pd.DataFrame(filled_tags_list)
# Suppression de colonne vide
tags_df=tags_df[tags_df.columns[:-1]]
tags_df.index = df_raw.index
tags_df.columns = question_tags_featured

# concatenation des tags avec la dataframe de départ
df_raw = pd.concat((df_raw, tags_df), axis=1)
count_tags_df=count_tags(df_raw)

# On compte le nombre tags différents
temp_list = [x.split('/') for x in df_raw.Tags.values.tolist()]
tags_list = [y for x in temp_list for y in x]
unique_tags = list(set(tags_list))
# Suppression des nan
for value in unique_tags:
    try:
        if np.isnan(value):
            unique_tags.remove(value)
    except:
        pass
		
print("Nous possédons {} Tag différents".format(len(unique_tags)))


### Quelques variables à conserver...
# Récupération du nombre de tags et du tag le plus populaire
most_popular_tag = count_tags_df.Tag[0]
# On récupère tous les tags pour une utilisation ultérieure
unique_tags_serie = pd.Series(unique_tags).apply(lambda x: [x])


## III. Préparation de la Pipeline pour l'entrainement des données

# Réduction de dimension par séléction du pourcentage de tags à conserver
# On définit un pourcentage seuil
threshold_percentage = 50
threshold_count = 0

# Création d'une dataframe
df_work = df_raw.copy()
count_tags_df = count_tags(df_work)
top_tags = count_tags_df[count_tags_df["Pourcentage (%)"].cumsum(axis=0)<threshold_percentage]
list_top_tags = top_tags.Tag.tolist()

# On sélectionne les données qui possèdent plus que le seuil imposé
selected_tags = top_tags[top_tags["Count"] >= threshold_count]["Tag"]
print("Sélection de {} Tags dans notre jeu d'entrainement".format(selected_tags.shape[0]))

# On distingue nos données en jeu d'entrainement et jeu de test. On souhaite retirer les tags pour chaque question qui ne sont pas parmi le top 50.
mask = df_work[question_tags_featured].isin(list_top_tags)
df_work[question_tags_featured]=df_work[mask][question_tags_featured]
df_work = df_work[(df_work.T != 0).any()]

# On supprime les lignes qui contiennent au minimum 4 "Nan"
df_work = df_work.dropna(thresh=len(df_work.columns)-4)

# On prépare notre jeu de données et notre jeu d'entraienment grâce à la fonction test_split
data_train, data_test = model_selection.train_test_split(df_work, test_size=0.25)
data_train["New_tags"] = data_train[question_tags_featured].apply(lambda x: '/'.join(x.dropna()),axis=1)
data_test["New_tags"] = data_test[question_tags_featured].apply(lambda x: '/'.join(x.dropna()),axis=1)

# On applique le traitement sur les données d'entrainement pour obtenir le vecteur d'entrainment
X_train, vectorizer = data_preprocessing(data_train)

# On applique le vectorizer_train sur les données test pour obtenir le vecteur de test
X_test = data_preprocessing_test(data_test,vectorizer)

# Obtention des variables cibles
y_train = ml_create_target_df(selected_tags, data_train)
y_test = ml_create_target_df(selected_tags, data_test)



## IV. Sauvegardes

# Sauvegardes des pipelines au format .pkl
joblib.dump(X_train, parent_path + "\\pipeline\\X_train.pkl")
joblib.dump(X_test, parent_path + "\\pipeline\\X_test.pkl")
joblib.dump(data_train, parent_path + "\\pipeline\\data_train.pkl")
joblib.dump(data_test, parent_path + "\\pipeline\\data_test.pkl")
joblib.dump(y_train, parent_path + "\\pipeline\\y_train.pkl")
joblib.dump(y_test, parent_path + "\\pipeline\\y_test.pkl")

# # Sauvegardes des Dataframes et données au format .pkl
joblib.dump(vectorizer, parent_path + "\\pipeline\\vectorizer.pkl")
joblib.dump(count_tags_df, parent_path + "\\pipeline\\count_tags_df.pkl")
joblib.dump(most_popular_tag, parent_path + "\\pipeline\\most_popular_tag.pkl")
joblib.dump(unique_tags_serie, parent_path + "\\pipeline\\unique_tags_serie.pkl")
joblib.dump(unique_tags, parent_path + "\\pipeline\\unique_tags.pkl")
joblib.dump(selected_tags,parent_path + "\\pipeline\\selected_tags.pkl")
joblib.dump(list_top_tags, parent_path + "\\pipeline\\list_top_tags.pkl")

print("Pipeline OK")
print("Sauvegardes effectuées")
