# Projet NLI

## *Architecture de notre projet*

* **Dossier project**: contient les logs des sbatch (logslurms), les notebooks pour visualiser les données et faire des essais de codes, les results (checkpoints), notre config, notre dataloader et le modèle.

* **venv** notre environnement virtuel et **wandb** le dossier généré par l'utilisation de wandb.

* un notebook **play** qui propose à l'utilisateur de rentrer deu xphrases et d'appliquer à ces phrases notre modèle.

* un notebook **attention** qui illustre les méchanismes d'attention au sein de notre modèle.

* un fichier python **train** qui se lance via le fichier bash **submit_slurms** pour lancer des sbatch lorsque l'on veut réaliser un finetuning.

## *Dataset*

Voici, à titre indicatif un **leaderboard des modèles avec leur performance respective pour la tâche de NLI sur le dataset** que l'on étudie. Même s'il n'est pas extrêment récent, il pourra nous donner une idée des caractéristique de notre dataset et de ce que sera une bonne performance sur ce dataset.

[Dataset Leaderboard](https://nlp.stanford.edu/projects/snli/)

Vous pourrez trouver dans le fichier **load_visualize** du dossier notebooks quelques commandes nous ayant permis de prendre en main le dataset.

Vous trouverez aussi dans le notebook **test** du même fichier des essais afin de prendre en main le dataset, le charger et créer un dataloader.

## *Preprocessing*

Pour le preprocessing de nos données nous avons choisis d'extraire à chaque fois l'hypothesis et le premise, de les tokenizer puis de concatener les deux vecteurs afin de ne donner en entrée à notre modèle qu'un seul vecteur et son label. Les deux phrases étant délimitées par des séparateurs au sein du vecteur, on ne perd pas d'information en effectuant cette opération.
Vous pouvez trouver notre dataloader dans le fichier **data_utils** au sein du dossier **project**.


## *Nos modèles*

Pour le choix du modèle nous nous sommes inspiré de la documentation suivante, qui présente le leaderboard des modèles pour le Massive Text Embedding:

[Model Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)

* Nous avons tester plusieurs modèle (Roberta, Bert et all-MiniLM-L6-v2) pour réaliser notre tâches et avons opté pour Roberta au vu des bonnes perfoamnces qu'il obtenait en comparaison des autres. Il s'agit pour tous les trois de modèle transformers que nous avons adaptés pour réaliser la tâche de NLI.

[INSÉRER IMAGES DE COURBES WANDB]

* Nous avons choisi l'optimizer AdamW au vu des bonnes performances qu'il présentait en comparaison de SGD.

* Enfin pour la métrique, comme nous avons afaire à une  tâche de classification nous avons trouvé pertinent de prendre l'accuracy.

* Pour tout les autres hyperparamètres du modèle, nous avons choisi la taille des batch et réduit le max_length de chaque input pour respecter les limitations imposées par la mémoire des GPU.


## *Annexe: commandes utiles*

Commandes pour lancer des sbatch sur le DCE pour le finetuning: 

sbatch submit_slurms.sh
STop un sbatch: scancel 65749