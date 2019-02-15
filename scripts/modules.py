import pandas as pd
import os
import numpy as np
import nltk
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings("ignore")
import re
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords, wordnet
import urllib
import scipy
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing, metrics
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

current_path = os.getcwd()
parent_path = os.path.dirname(current_path)

def get_stop_words(stop_file_path):
    """Chargement du fichier de stopwords """

    with open(stop_file_path, 'r', encoding="utf-8") as f:
        stopwords = f.readlines()
        stop_set = set(m.strip() for m in stopwords)
        return frozenset(stop_set)



def count_tags(data):
    question_tags_featured = ['Tag_1', 'Tag_2', 'Tag_3', 'Tag_4', 'Tag_5']
    count_df = pd.Series(data.loc[:, question_tags_featured].squeeze().values.reshape(-1)).value_counts()
    ct_df = pd.DataFrame({'Tag': count_df.index,
                                  'Count': count_df.values,
                                  'Pourcentage (%)': (100 * (count_df / count_df.sum())).values})
    return ct_df

def get_wordnet_pos(word):
    """Map POS tag """
    # On chosit la première lettre
    tag = nltk.pos_tag([word])[0][1][0].upper()
    # Creation de dictionnaire
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    # Return relevant tag otherwise Noun
    return tag_dict.get(tag, wordnet.NOUN)

def text_processing(raw_text):
    """
    Fonction qui permet de convertir un texte brut en une chaîne de caractères
    En entrée nous avons une chaîne brute et en sortie une seule chaîne.
    """
    # 1. Nettoyer HTML
    striped_text = BeautifulSoup(raw_text, "lxml").get_text()

    # 2. Nettoyer ce qui n'est pas une lettre
    letters_numbers_only = re.sub("[^a-zA-Z#+]", " ", striped_text)

    # 3. On change en minuscule les caractères
    words = letters_numbers_only.lower()

    return(' '.join(words.split()))


def text_processing_advanced(raw_body):
    """
    Fonction avancée qui permet de convertir un texte brut en une chaîne de caractères
    En entrée nous avons une chaîne brute et en sortie une seule chaîne. Nous utilisons les stopwords et lemmatisation.
    """
    # 1. Nettoyer HTML
    body_text = BeautifulSoup(raw_body, "lxml").get_text()

    # 2. Retirer ce qui n'est pas une lettre alphabétique
    letters_numbers_only = re.sub("[^a-zA-Z#+]", " ", body_text)

    # 3. Conversion en minuscule et découpage
    words = letters_numbers_only.lower().split()

    # 4. Chargement des stopwords
    stops = get_stop_words(parent_path + "\\resources\\stopwords.txt")

    # 5. Retrait des stop words
    meaningful_words = [w for w in words if not w in stops]

    # 6. Lemmatisation des mots
    lemmatizer = WordNetLemmatizer()
    lemmatize_meaningful_words = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in meaningful_words]

    return (" ".join( lemmatize_meaningful_words))

def data_preprocessing(data):
    """
    Fonction qui permet de traiter en entrée une dataframe avec nos documents texte et d'obtenir en sortie
    un vecteur de données après nettoyage des données au préalable.
    """
    # Concaténation
    data["Body_Title"] = data["Title"] + " " + data["Body"]
    
    #Chargement des stop words
    stops = get_stop_words(parent_path + "\\resources\\stopwords.txt")

    #On concatène les colonnes "Body" et "Title" qui contiennent toutes les deux des informations importantes.
    data["Body_Title"] = data["Title"] + " " + data["Body"]

    # Traitement sur Body_Title
    bodytitle_words_clean = pd.DataFrame(columns=['body_title_words'])
    
    # On effectue un text_processing à "Body_Title"
    bodytitle_words_clean['body_title_words'] = data.Body_Title.apply(text_processing)

    # Implémentation du Vectorizer pour créer nos bag of words
    vectorizer = TfidfVectorizer(min_df=20,analyzer="word", tokenizer=None, preprocessor=None, stop_words=stops, binary=False,max_features=10000)

    print("Vectorisation de Body_Title (Training) en bag of words en cours")
    bodytitle_words_features = vectorizer.fit_transform(bodytitle_words_clean.body_title_words)
    print("Vectorisation de Body_Title effectuée")
    print("Taille suite à vectorisation de Body_Title sur training {}".format(bodytitle_words_features.shape))
    print("Fin du data preprocessing")
    return bodytitle_words_features, vectorizer

def data_preprocessing_test(data,vectorizer):
    """
    Fonction qui permet de traiter en entrée une dataframe avec nos documents texte de test et d'obtenir en sortie
    un vecteur de données après nettoyage des données au préalable.
    """
    # Concaténation
    data["Body_Title"] = data["Title"] + " " + data["Body"]
    
    #Chargement des stop words
    stops = get_stop_words(parent_path + "\\resources\\stopwords.txt")

    #On concatène les colonnes "Body" et "Title" qui contiennent toutes les deux des informations importantes.
    data["Body_Title"] = data["Title"] + " " + data["Body"]

    # Traitement sur Body_Title
    bodytitle_words_clean = pd.DataFrame(columns=['body_title_words'])
    
    # On effectue un text_processing à "Body_Title"
    bodytitle_words_clean['body_title_words'] = data.Body_Title.apply(text_processing)

    print("Vectorisation de Body_title(Test) en bag of words en cours")
    bodytitle_words_features_test = vectorizer.transform(bodytitle_words_clean.body_title_words)
    print("Vectorisation de Body_title effectuée")
    print("Taille suite à vectorisation de Body_Title sur test {}".format(bodytitle_words_features_test.shape))
    print("Fin du data preprocessing")
    return bodytitle_words_features_test

def ml_create_target_df(selected_tags,data_set):
    """
    Création de notre matrice cible à partir de notre jeu de tags qui contient
    les tags propres à chaque question posée
    """
    question_tags_featured = ['Tag_1', 'Tag_2', 'Tag_3', 'Tag_4', 'Tag_5']
    print(" La dataframe cible contient {} tags...Traitement en cours".format(selected_tags.shape[0]))
    # Création de numpy array
    temp_array = np.zeros((data_set.shape[0], selected_tags.shape[0]))
    
    # Boucle sur tags
    for i, tag in zip(range(temp_array.shape[1]), selected_tags.values):
        temp_array[:, i] = np.sum(data_set.loc[:, question_tags_featured] == tag, axis=1)

    # On limite notre tableau aux valeurs 0 ou 1
    temp_array[temp_array > 1] = 1
    
    # Conversion en pandas dataframe
    ml_target_df = pd.DataFrame(temp_array, columns=selected_tags, dtype='int64')

    print("Traitement terminé")
    return ml_target_df

def get_lda_frequent_tags(row, min_frequency, most_frequent_tag):
    """
    Retourne les tags les plus fréqemment associés aux topics LDA
    """
    # Classement des lignes
    sorted_row = row.sort_values(ascending=False)

    # Test si au moins un Tag dépasse le seuil de fréquence minimum
    if sorted_row[0] >= min_frequency:
        acceptable_values = sorted_row[sorted_row >= min_frequency].index.values
        if len(acceptable_values) > 5 :
            output = acceptable_values[:5]
        else:
            output = acceptable_values
    # Sinon affichage du Tag le plus fréquent
    else:
        output = [most_frequent_tag]

    return output

def mean_local_recall(truth_pred_comparison_df, unique_tags_serie):
    """
    Fonction qui retourne la moyenne du Local_Recall
    """
    # Moyenne du local_recall
    truth_pred_comparison_df['Local_Recall'] = truth_pred_comparison_df.apply(local_recall, axis=1)
    mean_local_recall = truth_pred_comparison_df.Local_Recall.mean()
    print("Recall moyen : %.2f" % mean_local_recall)
    return mean_local_recall


def other_scores(truth_pred_comparison_df, unique_tags_serie):
    """
    Fonction qui retourne le F1 Score, recall
    """

    # Fractionnement des tags
    y_true_splitted = truth_pred_comparison_df.True_Tags.apply(lambda x: x.split("/"))

    # From dataframe to series
    y_pred_splitted = truth_pred_comparison_df.Predicted_Tags.apply(lambda x: x.split("/"))
    
    # Multilabel binarizer
    binarizer = preprocessing.MultiLabelBinarizer().fit(unique_tags_serie)

    # Métriques sur samples
    
    f1_score = metrics.f1_score(binarizer.transform(y_true_splitted),
                                binarizer.transform(y_pred_splitted),
                                average='samples')
    
    
    accuracy = metrics.precision_score(binarizer.transform(y_true_splitted),
                                binarizer.transform(y_pred_splitted),
                                average='samples')
    
    print("F1 Score : %.3f" % f1_score)
    print("Précision : %.3f" % accuracy)
    return f1_score, accuracy

def local_recall(row):
    """
    Fonction qui permet de retourner pour une prédiction le rappel ou la sensibilité qui correspond au
    rapport de nombre de documents correctement attribués à la classe i par le nombre de documents appartenant à la classe i.
    """

    # On obtient la liste des TAGS
    y_true = row.True_Tags.split("/")
    y_pred = row.Predicted_Tags.split("/")
    
    # Calcul de la difference
    diff = np.setdiff1d(y_true, y_pred)
    len_diff = len(y_true) - len(diff)
    local_recall = len_diff / len(y_true)
    
    # On retourne le Local recall
    return local_recall

def get_top_probable_tags(row, selected_tags):
    """
    Retourne à partir des probabilités d'une ligne le top 5 des tags associés
    """

    # On classe les probabilités
    probas = row.sort_values(ascending=False)

    # Quel nombre de TAGS est prédit ? On décompte les probas nulles
    n_pred_tags = len(probas[probas > 0])
    # Si on a au moins 5 TAGS
    if n_pred_tags >= 5 :
        # On récupère les tags associés
        out_tags = probas[:5].index.values
    elif n_pred_tags == 0:
        # Sinon on récupère le premier
        out_tags = np.array(selected_tags[0])
    else:
        out_tags = probas[:n_pred_tags].index.values

    # String en sortie avec conversion en liste
    if n_pred_tags > 1:
        out_string = "/".join(out_tags.tolist())
    elif n_pred_tags == 1:
        out_string = out_tags.tolist()[0]
    else:
        out_string = out_tags.tolist()

    return out_string
