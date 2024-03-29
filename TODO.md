# TODO

1) Researchs

- look for the bests models in the transformers librairy from huggingfaces

Results:

It exists models from hugging faces trained from the SNLI datasets

Transformers à utiliser

[ link to doc](https://discuss.huggingface.co/t/what-are-the-best-trained-model-of-nli-natural-language-inference/21030)

[all info, benchmarks on SNLI dataset](https://towardsdatascience.com/natural-language-inference-an-overview-57c0eecf6517)

[github inspirant](https://github.com/sarrabenyahia/NLI-SNLI)

[lien vers modèle hugging face](https://huggingface.co/models?dataset=dataset:multi_nli)

[Lien pour savoir load des checkpoints depuis un modèle hugginface entrainé avec le trainer](https://discuss.huggingface.co/t/trainer-load-best-model-at-end-doesnt-load-the-best-model/32206)

All-mini-v6: This is a sentence-transformers model: It maps sentences & paragraphs to a 384 dimensional dense vector space and can be used for tasks like clustering or semantic search.


- choose cost function for our problem --> accuracy

Les labels sont 0, 1 ou 2

2) Implementation

- dataloader, model, training, config: ✅

- wandb ✅

- script pour les sbatch ✅ 

- essayer Roberta ✅  - pas de bonnes performances

- faire finetune d'hyperparamètres ✅

- choisir un optimizer ✅ prend AdamW par défaut et ok

-essayer SGD en créant une classe qui hérite de Trainer et en mettant cette classe à la place de trainer dans model

3) Play with the model ✅

Faire un notebook où l'utilisateur rentre 2 phrases et on applique le modèle pour faire une prédiction.

Courbes Wandb et leurs modèles, avec les best performances

all-MiniLM-L6-v2✅
Bert-cased✅
Roberta

4) Mettre des insights visuels du modèle, de son utilisation de l'attention...✅


5) Expliquer les choix de tous les hyperparamètres, de notre modèle dans un beau readme ✅
