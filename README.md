# Parameter-Efficient Fine-Tuning — Comparative Study

**Comparative study of PEFT methods for sentiment classification**, by Eliott Rakotondramanitra and Louis-Antoine Pham.

## Objective & dataset

Compare **LoRA, QLoRA, IA³ and Prompt Tuning** with a pretrained-classifier baseline, assessing classification accuracy, trainable parameters, model size and training time. Experiments use the **IMDb** sentiment classification dataset with pretrained **DistilBERT** and **RoBERTa** models.

## Technologies

Python · Jupyter · PyTorch · Hugging Face Transformers · PEFT · Datasets · bitsandbytes · scikit-learn · Matplotlib

## Experiments & key results

The notebook studies manual LoRA, INT8 quantization and the four PEFT methods on DistilBERT and RoBERTa.

| Model | Method | Accuracy | Model size |
| --- | --- | ---: | ---: |
| DistilBERT | Pretrained baseline | 89.6% | ~270 MB |
| DistilBERT | LoRA | 90.6% | ~272 MB |
| DistilBERT | QLoRA | 90.0% | **122.7 MB** |
| RoBERTa | Pretrained baseline | 90.6% | ~499 MB |
| RoBERTa | QLoRA | 90.2% | **206.1 MB** |

**Key takeaway:** PEFT achieves competitive accuracy while reducing the number of trainable parameters. **QLoRA** offers a particularly favorable accuracy–memory trade-off in these experiments.

Results are reported from the accompanying study, not independently reproduced. The baseline is a pretrained sentiment classifier, not a separately executed full fine-tuning run.
