# Self-acknolwedge Kimitation Type Classification
This repository is the official implementation of Automatic Identification of Complete Description for Self-acknowledged Limitations in Randomised Controlled Trial Articles.

![Alt text](graphical_abstract.png)

## Annotation Guideline

The annotation guideline for limitation types could be found through this [link](https://drive.google.com/drive/folders/1hz5wU1S3ma87MJEAeye0BKklVbZ56UPj?usp=sharing).

## Environment

Create a conda environment by: 
```
conda create --name <env> --file requirements.txt
```
## Train Sentence Classifiers

You can run the following command to train a classifier: 

```
python main.py --input_view_augmentation_file="data/nlg_model_mix_output_part1.txt" \
--output_view_augmentation_file="data/nlg_model_mix_output_part2.txt" --bert_model="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" 
--train_set="data/train.csv" --test_set="data/test.csv" --dev_set="data/dev.csv" --fine_coarse="coarse" --target_number_augmentation=70 \
--augmentation_mode="PromDA output-view" --eda_augmentation_file="data/sst2_augmented.txt" --batch_size=2 --max_length=512 --num_epochs=20 \
--grad_acu_steps=4 --learning_rate=1e-5 --threshold=0.4 --checkpoint="checkpoint/coarse_promda_output_view_1.pth" --save_prediction=1 --train=1
```





