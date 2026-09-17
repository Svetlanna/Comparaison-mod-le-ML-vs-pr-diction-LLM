# comparaison ml et llm sur le titanic

## ce que j'ai fait

j'ai chargé le fichier titanic avec pandas  
j'ai retiré les lignes où il manque l'âge ou le port d'embarquement  
j'ai choisi les informations des passagers qui servent à faire la prédiction  
la colonne `Survived` est la réponse à prédire avec 0 pour décédé et 1 pour survivant  
j'ai transformé le sexe et le port d'embarquement en nombres pour le modèle ml  
j'ai gardé 20 passagers pour le test avec `random_state=42` pour retrouver les mêmes lignes à chaque fois  
j'ai entraîné la random forest sur les autres passagers puis j'ai prédit la survie des 20 passagers de test  
le modèle ml a obtenu 0,80 d'accuracy donc 16 bonnes réponses sur 20  
j'ai pris 8 passagers du jeu d'entraînement avec leur vraie réponse pour les montrer au llm dans le prompt  
ces 8 passagers sont des exemples pour le llm, ils ne remplacent pas les 20 passagers du test  
pour chacun des 20 passagers de test j'envoie les mêmes 8 exemples et le passager à prédire à groq  
le code vérifie que le llm répond bien par 0 ou 1 puis compare sa réponse à la vraie réponse  
je fais deux passages du llm avec les mêmes données et le même prompt pour comparer les scores  
à la fin le script affiche les trois scores dans un tableau

## scores

| méthode | accuracy |
| --- | ---: |
| ml | 0,80 |
| llm passage 1 | à compléter après l'exécution |
| llm passage 2 | à compléter après l'exécution |

## ce que je retiens

le modèle ml apprend avec tout le jeu d'entraînement alors que le llm utilise seulement les 8 exemples donnés dans le prompt et ce qu'il a déjà appris avant  
le résultat du ml reste le même avec les mêmes données et les mêmes réglages  
le llm peut donner un résultat moins stable même si la température est à 0  
les appels au llm prennent aussi du temps puisqu'il faut interroger l'api pour chaque passager  
le llm peut être utile pour expliquer les données et aider à rédiger une analyse
