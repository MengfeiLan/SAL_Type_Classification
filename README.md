# Self-acknolwedge Kimitation Type Classification
This repository is the official implementation of Automatic Identification of Complete Description for Self-acknowledged Limitations in Randomised Controlled Trial Articles.

![Alt text](graphical_abstract.png)

## Annotation Guideline

The annotation guideline for limitation types could be found through this [link](https://drive.google.com/drive/folders/1HHPuwFVngJoKPAMwU_cbMpljBTTgkcaL?usp=sharing).

## Environment

Create a conda environment by: 
```
conda create --name <env> --file requirements.txt
```
## Train Sentence Classifiers

You can run the following command to train a classifier: 

```
python main.py --input_view_augmentation_file="data/promda_input_view.txt" --output_view_augmentation_file="data/promda_output_view.txt"
--bert_model="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" --train_set="data/train.csv" --test_set="limitation_identification/all_limitation_data.csv"
--dev_set="data/dev.csv" --fine_coarse="coarse" --target_number_augmentation=70 --augmentation_mode="PromDA output-view" --eda_augmentation_file="data/EDA_augmentation.txt"
--batch_size=2 --max_length=512 --num_epochs=20 --grad_acu_steps=4 --learning_rate=1e-5 --threshold=0.4 --checkpoint="checkpoint_5/coarse_promda_output_view_1.pth" --save_prediction=1

```
main.py --input_view_augmentation_file="data/promda_input_view.txt" --output_view_augmentation_file="data/promda_output_view.txt" --bert_model="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" --train_set="data/train.csv" --test_set="limitation_identification/all_limitation_data.csv" --dev_set="data/dev.csv" --fine_coarse="coarse" --target_number_augmentation=70 --augmentation_mode="PromDA output-view" --eda_augmentation_file="data/EDA_augmentation.txt" --batch_size=2 --max_length=512 --num_epochs=20 --grad_acu_steps=4 --learning_rate=1e-5 --threshold=0.4 --checkpoint="checkpoint_1/coarse_promda_output_view_1.pth" --save_prediction=1 --default_threshold=0.5 --from_pretrain=True





