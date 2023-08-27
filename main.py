from argparse import ArgumentParser
from utils import * 
import pandas as pd
from dataloader import *
from framework import *

from torch.optim import Adam
import torch
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.nn import functional as F
from torch.utils.data._utils.collate import default_collate
from ast import literal_eval
from transformers import BertTokenizer, BertForTokenClassification
from sklearn.metrics import accuracy_score
import numpy as np
from check_funding import * 

def prepare_data(config, tokenizer):
	sentences_input_view, labels_input_view = read_sen_label(config.input_view_augmentation_file)
	sentences_output_view, labels_output_view = read_sen_label(config.output_view_augmentation_file)

	df_augmented_input_view = pd.DataFrame(zip(sentences_input_view, labels_input_view), columns=['sentences', 'label'])
	df_augmented_output_view = pd.DataFrame(zip(sentences_output_view, labels_output_view), columns=['sentences', 'label'])

	if config.augmentation_mode == "EDA":
		sentences_eda, labels_eda = read_sen_label_eda(config.eda_augmentation_file)
		df_augmented_eda = pd.DataFrame(zip(sentences_eda, labels_eda), columns=['sentences', 'label'])


	train_df = pd.read_csv(config.train_set)
	test_df = pd.read_csv(config.test_set)
	dev_df = pd.read_csv(config.dev_set)


	if config.fine_coarse == "coarse":
		train_df = train_df[train_df["category_annotation"] != "[False]"]
		train_df["label"] = train_df["coarse_grained_categories"]
		test_df = test_df[test_df["category_annotation"] != "[False]"]
		test_df["label"] = test_df["coarse_grained_categories"]
		dev_df = dev_df[dev_df["category_annotation"] != "[False]"]
		dev_df["label"] = dev_df["coarse_grained_categories"]

	train_data_df = train_df[["index", "sentences", "label"]]
	test_data_df = test_df[["index", "sentences", "label"]]
	dev_data_df = dev_df[["index", "sentences", "label"]]

	train_data_df["label"] = train_data_df["label"].apply(literal_eval)
	test_data_df["label"] = test_data_df["label"].apply(literal_eval)
	dev_data_df["label"] = dev_data_df["label"].apply(literal_eval)

	train_data_df["label"] = train_data_df["label"].apply(convert_single_to_list)
	test_data_df["label"] = test_data_df["label"].apply(convert_single_to_list)
	dev_data_df["label"] = dev_data_df["label"].apply(convert_single_to_list)

	c = describe_list_distribution(train_data_df["label"].to_list())

	augmentation_categories = [key for key, i in c.items() if config.target_number_augmentation - i > 0]

	print("augmentation_categories: ", augmentation_categories)

	input_view_categories = set(df_augmented_input_view.label.to_list())
	output_view_categories = set(df_augmented_output_view.label.to_list())
	if config.augmentation_mode == "EDA": 
		eda_categories = set(df_augmented_eda.label.to_list())


	if config.augmentation_mode == "oversampling":
		print("oversampling augmentation")
		for label in augmentation_categories:
			resampled_result = train_data_df.copy()
			print(label)
			resampled_result["anchor"] = train_data_df["label"].apply(select_specific_label, label = label)
			oversampling_categories = set(flatten_list(resampled_result[resampled_result["anchor"]==True].label.to_list()))
			if label in oversampling_categories:
				if len(resampled_result[resampled_result["anchor"] == True]) > config.target_number_augmentation - c[label]:
					resampled_result = resampled_result[resampled_result["anchor"] == True].sample(config.target_number_augmentation - c[label], replace=False)
					train_data_df = pd.concat([resampled_result, train_data_df])
				else:
					resampled_result = resampled_result[resampled_result["anchor"] == True].sample(len(resampled_result[resampled_result["anchor"] == True]), replace=False)
					train_data_df = pd.concat([resampled_result, train_data_df])
	elif config.augmentation_mode == "PromDA input-view": 
		print("promda input-view augmentation")
		for label in augmentation_categories:
			if label in input_view_categories:
				if len(df_augmented_input_view[df_augmented_input_view["label"] == label]) >= config.target_number_augmentation - c[label]:
					resampled_result = df_augmented_input_view[df_augmented_input_view["label"] == label].sample(config.target_number_augmentation - c[label],replace=False)
					train_data_df = pd.concat([resampled_result, train_data_df])
				else:
					resampled_result = df_augmented_input_view[df_augmented_input_view["label"] == label].sample(len(df_augmented_input_view[df_augmented_input_view["label"] == label]) ,replace=False)
					train_data_df = pd.concat([resampled_result, train_data_df])	
	elif config.augmentation_mode == "PromDA output-view": 
		print("promda output-view augmentation")
		for label in augmentation_categories:
			if label in output_view_categories:
				if len(df_augmented_output_view[df_augmented_output_view["label"] == label]) >= config.target_number_augmentation - c[label]:
					resampled_result = df_augmented_output_view[df_augmented_output_view["label"] == label].sample(config.target_number_augmentation - c[label],replace=False)
					train_data_df = pd.concat([resampled_result, train_data_df])
				else:
					resampled_result = df_augmented_output_view[df_augmented_output_view["label"] == label].sample(len(df_augmented_output_view[df_augmented_output_view["label"] == label]),replace=False)
					train_data_df = pd.concat([resampled_result, train_data_df])
	elif config.augmentation_mode == "EDA": 
		print("promda EDA augmentation")
		for label in augmentation_categories:
			if label in eda_categories:
				if len(df_augmented_eda[df_augmented_eda["label"] == label]) >= config.target_number_augmentation - c[label]:
					resampled_result = df_augmented_eda[df_augmented_eda["label"] == label].sample(config.target_number_augmentation - c[label],replace=False)
					train_data_df = pd.concat([resampled_result, train_data_df])
				else:
					resampled_result = df_augmented_eda[df_augmented_eda["label"] == label].sample(len(df_augmented_eda[df_augmented_eda["label"] == label]),replace=False)
					train_data_df = pd.concat([resampled_result, train_data_df])
	
	train_data_df["label"] = train_data_df["label"].apply(convert_single_to_list)
	test_data_df["label"] = test_data_df["label"].apply(convert_single_to_list)
	dev_data_df["label"] = dev_data_df["label"].apply(convert_single_to_list)

	checkpoint_name = "checkpoint/fine_coarse_" + config.fine_coarse + "_best_model_original_oversampling.pth"
	print("after augmentation: ", describe_list_distribution(train_data_df["label"].to_list()))
	print("save to: ", config.checkpoint)

	unique_labels = select_unique_labels(train_data_df.label.to_list()).union(select_unique_labels(test_data_df.label.to_list())).union(select_unique_labels(dev_data_df.label.to_list()))
	label_index =  [i for i in range(len(unique_labels))]

	if not config.from_pretrain:
		labels_to_id = dict(zip(unique_labels, label_index))
		with open(config.checkpoint.strip(".pth") + "_labels.txt", 'w') as file:
			for key, i in labels_to_id.items():
				file.write(str(key) + "\t" + str(i) + '\n')
	else:
		labels_to_id = load_label_from_pretrained(checkpoint_name.strip(".pth") + "_labels.txt")

	train_data_df["label_id"] = train_data_df["label"].map(lambda x: [labels_to_id[y] for y in x if y in labels_to_id])
	test_data_df["label_id"] = test_data_df["label"].map(lambda x: [labels_to_id[y] for y in x if y in labels_to_id])
	dev_data_df["label_id"] = dev_data_df["label"].map(lambda x: [labels_to_id[y] for y in x if y in labels_to_id])

	train_data_df['label_id'] = train_data_df['label_id'].apply(lambda d: d if isinstance(d, list) else [])
	test_data_df['label_id'] = test_data_df['label_id'].apply(lambda d: d if isinstance(d, list) else [])
	dev_data_df['label_id'] = dev_data_df['label_id'].apply(lambda d: d if isinstance(d, list) else [])

	num_label = len(labels_to_id)


	return labels_to_id, unique_labels, checkpoint_name, num_label, train_data_df, test_data_df, dev_data_df


if __name__ == '__main__':

	parser = ArgumentParser()
	parser.add_argument('--input_view_augmentation_file', type=str, help='the file contains input-view augmentation results')
	parser.add_argument('--output_view_augmentation_file', type=str, help='the file contains output-view augmentation results')
	parser.add_argument('--bert_model', type=str, help='the name of the pretrained bert model')
	parser.add_argument('--train_set', type=str, help='the name of the train set')
	parser.add_argument('--test_set', type=str, help='the name of the test set')
	parser.add_argument('--dev_set', type=str, help='the name of the dev set')
	parser.add_argument('--fine_coarse', type=str, help='indicate which level of categories should be used - choose from fine and coarse')
	parser.add_argument('--from_pretrain', type=bool, help='if or not start from the existing checkpoint')
	parser.add_argument('--target_number_augmentation', type=int, help='target number of samples after augmentation')
	parser.add_argument('--augmentation_mode', type=str, help='indicate the type of augmentation - choose from EDA, Oversampling, PromDA input-view, and PromDA output-view')
	parser.add_argument('--eda_augmentation_file', type=str, help='the file contains eda augmentation results')
	parser.add_argument('--batch_size', type=int, help='batch size')
	parser.add_argument('--max_length', type=int, help='maximum sequence length')
	parser.add_argument('--num_epochs', type=int, help='number of epochs')
	parser.add_argument('--grad_acu_steps', type=int, help='number of gradient accumulation steps')
	parser.add_argument('--learning_rate', type=float, help='learning rate')
	parser.add_argument('--threshold', type=float, help='threshold for multi-label prediction result')
	parser.add_argument('--checkpoint', type=str, help='name of the checkpoint for save/load')
	parser.add_argument('--save_prediction', type=int, help='if save the prediction results')
	parser.add_argument('--train', type=bool, help='training the model or not')
	parser.add_argument('--augment', type=bool, help='if or not augment the data')


	config = parser.parse_args()    
	print("config", config)


	tokenizer = BertTokenizer.from_pretrained(config.bert_model)
	labels_to_id, unique_labels, checkpoint_name, num_label, train_data_df, test_data_df, dev_data_df = prepare_data(config, tokenizer)
	
	train_dataset, val_dataset = Dataset(train_data_df, tokenizer, config.max_length, num_label), Dataset(dev_data_df, tokenizer, config.max_length,  num_label)

	train_dataloader = torch.utils.data.DataLoader(train_dataset, collate_fn = my_collate_fn, batch_size=2, shuffle=True)
	val_dataloader = torch.utils.data.DataLoader(val_dataset, collate_fn = my_collate_fn, batch_size=2)

	test_dataset = Dataset(test_data_df, tokenizer, config.max_length, num_label)

	test_dataloader = torch.utils.data.DataLoader(test_dataset, collate_fn = my_collate_fn, batch_size=2, shuffle=True)


	model_augmented = BertForTokenClassification.from_pretrained(config.bert_model, num_labels=num_label)


	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	if config.from_pretrain == True:
		model.load_state_dict(torch.load(checkpoint_name))

	model_augmented.to(device)
	
	loss_func = nn.BCELoss()
	loss_func = loss_func.cuda()

	rerun = True
	if config.train:
		while rerun == True:
			labels_to_id, unique_labels, checkpoint_name, num_label, train_data_df, test_data_df, dev_data_df = prepare_data(config, tokenizer)
			train_dataset, val_dataset = Dataset(train_data_df, tokenizer, config.max_length, num_label), Dataset(dev_data_df, tokenizer, config.max_length,  num_label)
			train_dataloader = torch.utils.data.DataLoader(train_dataset, collate_fn = my_collate_fn, batch_size=2, shuffle=True)

			model_augmented = BertForTokenClassification.from_pretrained(config.bert_model, num_labels=num_label)
			model_augmented.to(device)
			rerun = train(model_augmented, train_dataloader, val_dataloader, config.learning_rate, tokenizer, config.max_length, config.num_epochs, \
			config.checkpoint, config.grad_acu_steps, labels_to_id, config.threshold, loss_func)
			print("Rerun: ", rerun)
			if config.augmentation_mode in ["EDA", "PromDA output-view", "PromDA input-view", "oversampling"] and rerun: 
				print("Reselect the augmentation samples. ")


	model_augmented.load_state_dict(torch.load(config.checkpoint))

	alllabels, allpreds, allinputs = evaluate(model_augmented, test_dataloader, tokenizer, config.max_length, config.threshold, loss_func, labels_to_id)
	predicted_tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in allinputs]
	origin_sentences = go_back_to_origin(predicted_tokens)

	for i in range(len(origin_sentences)):
		if check_funding(origin_sentences[i]):
			allpreds[i].append(labels_to_id["Funding"])

	p, r, f, total = evaluation(alllabels, allpreds, labels_to_id)

	print("test precision: ", p)
	print("test recall: ", r)
	print("test f1: ", f)
	print("test total: ", total)

	encode_dict = {value: key for key, value in enumerate(set(unique_labels))}
	encode_dict_reverse = {key: value for key, value in enumerate(set(unique_labels))}

	print("encode_dict: ", encode_dict)
	print("encode_dict_reverse: ", encode_dict_reverse)

	onehot_alllabels = [convert_to_category_specific_list(i, unique_labels) for i in alllabels]
	onehot_allpreds = [convert_to_category_specific_list(i, unique_labels) for i in allpreds]
	
	accuracy_rate = accuracy_score(onehot_alllabels, onehot_allpreds)

	id_to_labels = {i:key for key, i in labels_to_id.items()}

	with open(config.checkpoint.strip(".pth") + ".txt", "w") as file:
		file.write("precision: " + str(p))
		file.write("\n")
		file.write("recall: " + str(r))
		file.write("\n")
		file.write("f1 score(average): " + str(f))
		file.write("\n")
		file.write("results by categories: " + str(total))
		file.write("\n")
		file.write("accuracy rate: " + str(accuracy_rate))

	
	alllabels = convert_id_to_label(alllabels, id_to_labels)
	allpreds = convert_id_to_label(allpreds, id_to_labels)

	for i in range(len(origin_sentences)):
		if check_funding(origin_sentences[i]):
			allpreds[i].append("Funding")

	if config.save_prediction:
		data = pd.DataFrame(zip(origin_sentences, alllabels, allpreds), columns=['sentence', 'label', 'pred'])
		data.to_csv(config.checkpoint.split("/")[1].strip(".pth") + ".csv")

