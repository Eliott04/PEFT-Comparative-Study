# Comparative Study of Parameter-Efficient Fine-Tuning Methods (PEFT)

**Experimental study of PEFT methods for sentiment classification**, Eliott Rakotondramanitra and Louis-Antoine Pham.

## Objective and Data

Compare **LoRA, QLoRA, IA³ and Prompt Tuning** with a pretrained baseline model, based on four criteria: **classification accuracy**, **number of trainable parameters**, **model size in memory** and **training time**.

The experiments use the **IMDb** dataset (binary sentiment classification) and pretrained **DistilBERT** and **RoBERTa** models.

## Technologies

Python · Jupyter Notebook · PyTorch · Hugging Face Transformers · PEFT · Datasets · bitsandbytes · scikit-learn · Matplotlib

## Experiments and Main Results

The notebook explores a manual implementation of **LoRA**, **INT8 quantization**, then compares **LoRA, QLoRA, IA³ and Prompt Tuning** on DistilBERT and RoBERTa.

| Model | Method | Accuracy | Model size |
| --- | --- | ---: | ---: |
| DistilBERT | Pretrained baseline | 89.6% | ~270 MB |
| DistilBERT | LoRA | 90.6% | ~272 MB |
| DistilBERT | QLoRA | 90.0% | **122.7 MB** |
| RoBERTa | Pretrained baseline | 90.6% | ~499 MB |
| RoBERTa | QLoRA | 90.2% | **206.1 MB** |

**Conclusion:** PEFT methods maintain competitive classification performance while significantly reducing the number of trained parameters. In these experiments, **QLoRA offers a particularly interesting trade-off between accuracy and memory usage**.

## Project Files

- [Experiment notebook](PEFT_project.ipynb)
- [Experimental report](PEFT_Report.pdf)
