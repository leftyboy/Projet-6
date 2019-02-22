StackOverflow est un site de question-réponses liées au développement informatique.
Pour poser une question sur ce site, il faut entrer plusieurs tags de manière à retrouver
facilement la question par la suite.  
L’objectif du projet que nous devons réaliser est de prédire les tags associés
à une question posée par un utilisateur de la plateforme. 



1. Notebooks
"PROJET 6_Entrainement et Prédictions.ipynb"
"PROJET 6_Exploration et Pipeline.ipynb"

2. Scripts python
- Fichier pipeline.py
--> à exécuter pour construire pipeline et fichiers de sauvegardes nécessaires
pour le training et la prédiction

- Fichier modules.py 
--> Fichier contenant les modules qui permettent l'exécution des autres scripts

- Fichier training_testing.py 
--> à exécuter pour lancer un entrainement et un tester notre modèle

- Fichier predictions.py
--> On teste notre modèle avec de nouvelles données utilisateur 

3. Répertoires
classifiers/logistical_regression
--> Contient les classifieurs suivant modèle de régression logistique au format .pkl

classifiers/naive_bayes
--> Contient les classifieurs suivant modèle naive Bayes au format .pkl

classifiers/random_forest
--> Contient les classifieurs suivant modèle de Random Forest au format .pkl

data
--> Contient les données initiales au format .csv

pipeline
--> Contient les élément nécessaire pour entrainer le modèle

resources
--> Contient la liste des stopwords

output
--> Fichiers .csv en sortie

scripts
Scripts au format .py

notebooks
Emplacement de nos notebooks


4. Packages à installer
pandas
numpy
scikit-learn
nltk
pyLDAvis
mglearn
seaborn
matplotlib
bs4 (beautifulsoup)

5. Site (API)
Disponible à l’adresse suivante :  http://leftyboy.pythonanywhere.com

6.lien vers projet Github
https://github.com/leftyboy/Projet-6

7. Procédure d'éxecution des scripts (se trouvant dans le répertoire "scripts")
Exécuter le fichier pipeline.py tout d'abord pour créer le jeu de données/test pour l'entrainement.
 
Exécuter le fichier training_testing.py pour entrainer notre modèle
avec notre jeu d'entrainement et le tester avec notre jeu de test.

Exécuter le fichier predictions.py avec comme argument le fichier "InputQuestions.csv"
qui se trouve dans le répertoire "data". Il n'est pas nécessaire d'ajouter le lien
complet pour accéder au fichier "InputQuestions.csv"
