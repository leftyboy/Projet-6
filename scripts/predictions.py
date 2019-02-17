import os
import sys
import scipy
import numpy as np
import pandas as pd
from sklearn import metrics, preprocessing
from modules import (
    ml_create_target_df,
    get_top_probable_tags,
    mean_local_recall,
    other_scores,
    text_processing,
    local_recall,
)
from sklearn.externals import joblib


def prediction(csv_file):
    """
        Fonction qui permet de prédire les tags associés à une question
        Args:
        csv_file : fichier plat au format des input
    """
    # I. Chargement des donnée en input
    current_path = os.getcwd()
    parent_path = os.path.dirname(current_path)
    print("Chargement des nouvelles questions")
    # input_questions = pd.read_csv(path + "/data/InputQuestions.csv", sep=',')
    input_questions = pd.read_csv(parent_path + "\\data\\" + csv_file, sep=",")

    # II. Chargement des variables utiles liées à notre jeu d'entrainement
    print("Chargement des données d'entrainement")
    vectorizer_train = joblib.load(parent_path + "\\pipeline\\vectorizer.pkl")
    count_tags_df = joblib.load(parent_path + "\\pipeline\\count_tags_df.pkl")
    most_popular_tag = joblib.load(parent_path + "\\pipeline\\most_popular_tag.pkl")
    unique_tags_serie = joblib.load(parent_path + "\\pipeline\\unique_tags_serie.pkl")
    unique_tags = joblib.load(parent_path + "\\pipeline\\unique_tags.pkl")
    train_selected_tags = joblib.load(parent_path + "\\pipeline\\selected_tags.pkl")
    list_top_tags = joblib.load(parent_path + "\\pipeline\\list_top_tags.pkl")

    # III. Chargement des classifieurs (logistical regression)
    training_classifiers = []
    for tag in train_selected_tags:
        training_classifiers.append(
            joblib.load(
                parent_path
                + "\\classifiers\\logistical_regression\\logreg_"
                + tag
                + ".pkl"
            )
        )

    # IV. Traitement des lignes

    # Suppression des lignes sans Tags
    input_questions = input_questions.dropna(subset=["Tags"])
    print("Dataframe dimensions:", input_questions.shape)

    # Nettoyage et traitement des doublons
    # Suppression des doublons
    print(
        "Entrées en double: {}".format(
            input_questions.duplicated(subset=["Body"]).sum()
        )
    )
    input_questions.drop_duplicates(subset=["Body"], inplace=True)

    # Sélection des features utiles
    input_questions = input_questions[["Body", "Title", "Tags"]]

    # V.  Traitement des TAGS

    # Feature engineering. On crée 5 nouvelles features pour les tags
    question_tags_featured = ["Tag_1", "Tag_2", "Tag_3", "Tag_4", "Tag_5"]
    # On retire les Tags chevrons
    input_questions["Tags"] = input_questions.Tags.apply(
        lambda x: x.strip("<").strip(">").replace(">", "").replace("<", "/")
    )
    # Découpage des tags
    tags_lists = input_questions.Tags.apply(lambda x: x.split("/"))

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
    tags_df = tags_df[tags_df.columns[:-1]]
    tags_df.index = input_questions.index
    tags_df.columns = question_tags_featured

    # concatenation des tags avec la dataframe de départ
    input_questions = pd.concat((input_questions, tags_df), axis=1)

    # On compte le nombre tags différents
    temp_list = [x.split("/") for x in input_questions.Tags.values.tolist()]
    tags_list = [y for x in temp_list for y in x]
    unique_tags = list(set(tags_list))

    # Suppression des nan
    for value in unique_tags:
        try:
            if np.isnan(value):
                unique_tags.remove(value)
        except:
            pass

    print(
        "Nous possédons dans les questions en entrée {} Tags différents".format(
            len(unique_tags)
        )
    )

    # On compare les tags avec ceux de notre base d'entrainement et on retire les Tags qui n'y sont pas
    mask = input_questions[question_tags_featured].isin(list_top_tags)
    input_questions[question_tags_featured] = input_questions[mask][
        question_tags_featured
    ]
    input_questions = input_questions[(input_questions.T != 0).any()]

    # On supprime les lignes qui contiennent au minimum 4 "Nan"
    input_questions = input_questions.dropna(thresh=len(input_questions.columns) - 4)
    data_input = input_questions.copy()

    # Regroupement avec "/"
    data_input["New_tags"] = input_questions[question_tags_featured].apply(
        lambda x: "/".join(x.dropna()), axis=1
    )

    # VI. Vectorization

    # On concatène les colonnes "Body" et "Title" qui contiennent toutes les deux des informations importantes.
    data_input["Body_Title"] = data_input["Title"] + " " + data_input["Body"]
    data_input["Body_Title"] = data_input["Body_Title"].apply(text_processing)

    # On utilise le vectorizer de notre training
    X_input = vectorizer_train.transform(data_input["Body_Title"])

    # Nos variables cible
    print("Création de la dataframe cible...")
    y_test = ml_create_target_df(train_selected_tags, data_input)

    # VII. Prédiction de la probabilité
    print(
        "Prédiction de la probabilité pour chaque question d'appartenir à Tag donné en cours de traitement..."
    )
    input_predictions = []
    count = 1
    zero_array = np.zeros((X_input.shape[0]))
    for clf in training_classifiers:
        prediction = clf.predict_proba(X_input)

        if prediction.shape[1] == 2:
            input_predictions.append(prediction[:, 1])
        elif prediction.shape[1] == 1:
            input_predictions.append(zero_array)
        count += 1
    print("Affichage des prédictions en cours...")

    # On convertit en pd.dataframe
    input_predictions_df = pd.DataFrame(
        np.array(input_predictions).T, columns=y_test.columns
    )
    # threshold
    trusted_threshold = 0.11
    # Map threshold
    input_predictions_df_thresh = input_predictions_df.copy()
    input_predictions_df_thresh[input_predictions_df_thresh < trusted_threshold] = 0

    # Construction des tags prédites à partir de classifieurs,
    # si il y en a plus de 5 nous conservons les 5 premiers avec la probabilité la plus forte
    y_input_pred_ml = input_predictions_df_thresh.apply(
        get_top_probable_tags, args=(train_selected_tags,), axis=1
    )

    # Conversion en pd.dataframe
    y_input_pred_ml = pd.DataFrame(y_input_pred_ml, columns=["Predicted_Tags"])

    # On récupère les nouveaux tags dans une dataframe
    y_test_new_tags = pd.DataFrame(data_input["New_tags"].values, columns=["True_Tags"])

    # On construit un tableau pour comparer tags prédits et nouveaux tags
    multilabel_comparison_df = pd.concat([y_input_pred_ml, y_test_new_tags], axis=1)

    # On applique la fonction qui nous donne nombre de mots-clés justes parmi N mots-clés prédits
    multilabel_comparison_df["Local_Recall"] = multilabel_comparison_df.apply(
        local_recall, axis=1
    )

    # VII. Sauvegarde
    multilabel_comparison_df.to_csv(
        parent_path + "\\output\\input_predicted_tags.csv", index=False
    )

    return (
        multilabel_comparison_df,
        mean_local_recall(multilabel_comparison_df, unique_tags_serie),
        other_scores(multilabel_comparison_df, unique_tags_serie),
    )


# On affiche les résultats
csv_file = sys.argv[1]
print(sys.argv)
comparison, local_recall, f1_score = prediction(csv_file)
print(comparison.sample(40))
print(local_recall)
print(f1_score)

if __name__ == "__prediction__":
    prediction(sys.argv)
