# output view augmentation
#set PATH="$PATH:/c/Users/lecheng4/Anaconda3/envs/consort-tm"
#conda activate consort-tm
python main.py --input_view_augmentation_file="data/nlg_model_mix_output_part1.txt" --output_view_augmentation_file="data/nlg_model_mix_output_part2.txt" --bert_model="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" --train_set="data/train.csv" --test_set="data/test.csv" --dev_set="data/dev.csv" --fine_coarse="coarse" --target_number_augmentation=70 --augmentation_mode="PromDA output-view" --eda_augmentation_file="data/sst2_augmented.txt" --batch_size=2 --max_length=512 --num_epochs=20 --grad_acu_steps=4 --learning_rate=1e-5 --thresholds_multi_label=True --default_threshold=0.5 --checkpoint="checkpoint_3/coarse_promda_output_view_1.pth" --save_prediction=1 --train=1

python main.py --input_view_augmentation_file="data/nlg_model_mix_output_part1.txt" --output_view_augmentation_file="data/nlg_model_mix_output_part2.txt" \
--bert_model="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" --train_set="data/train.csv" --test_set="data/test.csv" --dev_set="data/dev.csv" \
 --fine_coarse="coarse" --target_number_augmentation=70 --augmentation_mode="PromDA output-view" --eda_augmentation_file="data/sst2_augmented.txt" --batch_size=2 --max_length=512 \
 --num_epochs=20 --grad_acu_steps=4 --learning_rate=1e-5 --thresholds_multi_label=True --default_threshold=0.5 --checkpoint="checkpoint_2/coarse_promda_output_view_1.pth" \
 --save_prediction=1 --train=1
python main.py --input_view_augmentation_file="data/nlg_model_mix_output_part1.txt" --output_view_augmentation_file="data/nlg_model_mix_output_part2.txt" \
--bert_model="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" --train_set="data/train.csv" --test_set="data/test.csv" --dev_set="data/dev.csv" \
 --fine_coarse="coarse" --target_number_augmentation=70 --augmentation_mode="PromDA output-view" --eda_augmentation_file="data/sst2_augmented.txt" --batch_size=2 --max_length=512 \
 --num_epochs=20 --grad_acu_steps=4 --learning_rate=1e-5 --thresholds_multi_label=True --default_threshold=0.5 --checkpoint="checkpoint_3/coarse_promda_output_view_1.pth" \
 --save_prediction=1 --train=1

python main.py --input_view_augmentation_file="data/nlg_model_mix_output_part1.txt" --output_view_augmentation_file="data/nlg_model_mix_output_part2.txt" \
--bert_model="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" --train_set="data/train.csv" --test_set="data/test.csv" --dev_set="data/dev.csv" \
 --fine_coarse="coarse" --target_number_augmentation=70 --augmentation_mode="PromDA output-view" --eda_augmentation_file="data/sst2_augmented.txt" --batch_size=2 --max_length=512 \
 --num_epochs=20 --grad_acu_steps=4 --learning_rate=1e-5 --thresholds_multi_label=True --default_threshold=0.5 --checkpoint="checkpoint_4/coarse_promda_output_view_1.pth" \
 --save_prediction=1 --train=1

python main.py --input_view_augmentation_file="data/nlg_model_mix_output_part1.txt" --output_view_augmentation_file="data/nlg_model_mix_output_part2.txt" \
--bert_model="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" --train_set="data/train.csv" --test_set="data/test.csv" --dev_set="data/dev.csv" \
 --fine_coarse="coarse" --target_number_augmentation=70 --augmentation_mode="PromDA output-view" --eda_augmentation_file="data/sst2_augmented.txt" --batch_size=2 --max_length=512 \
 --num_epochs=20 --grad_acu_steps=4 --learning_rate=1e-5 --thresholds_multi_label=True --default_threshold=0.5 --checkpoint="checkpoint_5/coarse_promda_output_view_1.pth" \
 --save_prediction=1 --train=1
