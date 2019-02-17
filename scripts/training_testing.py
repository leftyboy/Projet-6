import pandas as pd
import os
import shutil
import glob
import numpy as np
import urllib
import warnings

warnings.filterwarnings("ignore")
import re
import seaborn as sns
import scipy
from sklearn.linear_model import LogisticRegression
from sklearn import (
    model_selection,
    naive_bayes,
    decomposition,
    multiclass,
    ensemble,
    preprocessing,
    metrics,
    linear_model,
)
import mglearn
from mglearn import tools
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

from modules import (
    get_lda_frequent_tags,
    mean_local_recall,
    other_scores,
    local_recall,
    get_top_probable_tags,
)

# I. Chargement de la Pipeline et d'autres données
current_path = os.getcwd()
parent_path = os.path.dirname(current_path)

X_train = joblib.load(parent_path + "\\pipeline\\X_train.pkl")
X_test = joblib.load(parent_path + "\\pipeline\\X_test.pkl")
y_train = joblib.load(parent_path + "\\pipeline\\y_train.pkl")
y_test = joblib.load(parent_path + "\\pipeline\\y_test.pkl")

data_train = joblib.load(parent_path + "\\pipeline\\data_train.pkl")
data_test = joblib.load(parent_path + "\\pipeline\\data_test.pkl")

# dataframes et données
vectorizer = joblib.load(parent_path + "\\pipeline\\vectorizer.pkl")
count_tags_df = joblib.load(parent_path + "\\pipeline\\count_tags_df.pkl")
most_popular_tag = joblib.load(parent_path + "\\pipeline\\most_popular_tag.pkl")
unique_tags_serie = joblib.load(parent_path + "\\pipeline\\unique_tags_serie.pkl")
unique_tags = joblib.load(parent_path + "\\pipeline\\unique_tags.pkl")
selected_tags = joblib.load(parent_path + "\\pipeline\\selected_tags.pkl")

# Suppression du dossier contenant nos sauvegardes
logreg_path = parent_path + "\\classifiers\\logistical_regression"
if os.path.exists(logreg_path):
    shutil.rmtree(logreg_path)
os.mkdir(logreg_path)

# On entraine sur chaque Tag notre jeu de données
classifiers = []
count = 1
for category in selected_tags:
    print(
        "{}/{} - Entrainement sur le Tag: '{}' ...".format(
            count, len(selected_tags), category
        )
    )
    # Sélection du classifieur
    clf = linear_model.LogisticRegression(random_state=0)
    clf = clf.fit(X_train, y_train[category])
    # On sauvegarde notre encodage pour une utilisation ultérieure
    joblib.dump(clf, logreg_path + "\\logreg_" + category + ".pkl")
    classifiers.append(clf)
    count = count + 1

print(
    "Prédiction de la probabilité pour chaque question d'appartenir à Tag donné en cours de traitement..."
)
test_predictions = []
count = 1
zero_array = np.zeros((X_test.shape[0]))
for classifier in classifiers:
    #     print("trclassifieur {} / {}".format(count, len(classifiers)))
    prediction = classifier.predict_proba(X_test)

    if prediction.shape[1] == 2:
        test_predictions.append(prediction[:, 1])
    elif prediction.shape[1] == 1:
        test_predictions.append(zero_array)
    count += 1
print("Traitement terminé !")

# On convertit en pd.dataframe
test_predictions_df = pd.DataFrame(np.array(test_predictions).T, columns=selected_tags)
# threshold
trusted_threshold = 0
# Map threshold
test_predictions_df_thresh = test_predictions_df.copy()
test_predictions_df_thresh[test_predictions_df_thresh < trusted_threshold] = 0

# Construction des tags prédites à partir de classifieurs,
# si il y en a plus de 5 nous conservons les 5 premiers avec la probabilité la plus forte
y_pred_ml = test_predictions_df_thresh.apply(
    get_top_probable_tags, args=(selected_tags,), axis=1
)

# conversion en pd.dataframe
y_pred_ml = pd.DataFrame(y_pred_ml, columns=["Predicted_Tags"])

# On récupère les nouveaux tags dans une dataframe
y_test_new_tags = pd.DataFrame(data_test["New_tags"].values, columns=["True_Tags"])

# On construit un tableau pour comparer tags prédits et nouveaux tags
multilabel_comparison_df = pd.concat([y_test_new_tags, y_pred_ml], axis=1)

# On applique la fonction qui nous donne nombre de mots-clés justes parmi N mots-clés prédits
multilabel_comparison_df["Local_Recall"] = multilabel_comparison_df.apply(
    local_recall, axis=1
)

# Affichages
print(multilabel_comparison_df.sample(40))
print(
    " Local_Recall moyen sur jeu de TEST : {}".format(
        multilabel_comparison_df.Local_Recall.mean()
    )
)
print(mean_local_recall(multilabel_comparison_df, unique_tags_serie))
print(other_scores(multilabel_comparison_df, unique_tags_serie))
