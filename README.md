# Self-acknolwedge Kimitation Type Classification
This repository is the official implementation of Automatic Identification of Complete Description for Self-acknowledged Limitations in Randomised Controlled Trial Articles.

![Alt text](graphical_abstract.jpg)

# Annotation Guideline

Checkpoint: https://drive.google.com/drive/folders/1hz5wU1S3ma87MJEAeye0BKklVbZ56UPj?usp=sharing

Before running the code, make sure you have the following packages installed:
```
nltk 
numpy
pandas==1.3.5
scikit-learn
tokenizers 
torch==1.13.1
torchaudio==0.13.1
torchvision==0.14.1
tqdm
transformer
```

Either run promDA output-view augmentation by:

```bash script/run_1.sh``` 

or use the command:

```
python main.py --input_view_augmentation_file="data/nlg_model_mix_output_part1.txt" \
--output_view_augmentation_file="data/nlg_model_mix_output_part2.txt" --bert_model="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" 
--train_set="data/train.csv" --test_set="data/test.csv" --dev_set="data/dev.csv" --fine_coarse="coarse" --target_number_augmentation=70 \
--augmentation_mode="PromDA output-view" --eda_augmentation_file="data/sst2_augmented.txt" --batch_size=2 --max_length=512 --num_epochs=20 \
--grad_acu_steps=4 --learning_rate=1e-5 --threshold=0.4 --checkpoint="checkpoint/coarse_promda_output_view_1.pth" --save_prediction=1 --train=1
```





