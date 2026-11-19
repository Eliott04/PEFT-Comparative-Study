# Étude comparative des méthodes de fine-tuning efficace (PEFT)

**Étude expérimentale des méthodes PEFT pour la classification de sentiments**, réalisée par Eliott Rakotondramanitra et Louis-Antoine Pham.

## Objectif et données

Comparer **LoRA, QLoRA, IA³ et Prompt Tuning** à un modèle de référence pré-entraîné, selon quatre critères : **précision de classification**, **nombre de paramètres entraînables**, **taille du modèle en mémoire** et **temps d'entraînement**.

Les expériences utilisent le jeu de données **IMDb** (classification binaire de sentiments) et les modèles **DistilBERT** et **RoBERTa** pré-entraînés.

## Technologies

Python · Jupyter Notebook · PyTorch · Hugging Face Transformers · PEFT · Datasets · bitsandbytes · scikit-learn · Matplotlib

## Expériences et principaux résultats

Le notebook explore une implémentation manuelle de **LoRA**, la **quantification INT8**, puis compare les méthodes **LoRA, QLoRA, IA³ et Prompt Tuning** sur DistilBERT et RoBERTa.

| Modèle | Méthode | Accuracy | Taille du modèle |
| --- | --- | ---: | ---: |
| DistilBERT | Référence pré-entraînée | 89,6 % | ~270 Mo |
| DistilBERT | LoRA | 90,6 % | ~272 Mo |
| DistilBERT | QLoRA | 90,0 % | **122,7 Mo** |
| RoBERTa | Référence pré-entraînée | 90,6 % | ~499 Mo |
| RoBERTa | QLoRA | 90,2 % | **206,1 Mo** |

**Conclusion :** les méthodes PEFT conservent des performances de classification compétitives tout en réduisant fortement le nombre de paramètres entraînés. Dans ces expériences, **QLoRA offre un compromis particulièrement intéressant entre précision et utilisation de la mémoire**.

*Ces résultats proviennent du rapport expérimental et n'ont pas été reproduits indépendamment. Le modèle de référence est un classifieur pré-entraîné, et non un entraînement complet (« full fine-tuning ») exécuté séparément.*

## Fichiers du projet

- [Notebook des expériences](PEFT_project.ipynb)
- [Rapport expérimental](PEFT_Report.pdf)
