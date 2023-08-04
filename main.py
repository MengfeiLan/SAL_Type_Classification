from argparse import ArgumentParser
from dataloader import Dataset
import pandas as pd
from model import BertClassifier
from framework import train, evaluate
import torch
from transformers import BertTokenizer, BertConfig, AdamW, get_cosine_schedule_with_warmup, BertForTokenClassification
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from simcse import SimCSE
from ast import literal_eval
from utils import *


simcse_model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")


def txt_to_dict(filename):
    with open(filename, 'r') as file:
        content = file.read()
        lines = content.splitlines()
        d = {}
        for line in lines:
            key, value = line.split('	')
            d[key] = value
    return d

def select_top_similar_sentences(l1, l2, simcse_model):
	simcse_model.build_index(l1)
	results = simcse_model.search(l2)
	return results

def select_unique_labels(label_lists):
	all_labels = []
	for label_list in label_lists:
		all_labels.extend(label_list)
	return set(all_labels)


def main():
	parser = ArgumentParser()

	parser.add_argument('--num_epochs', type=int, help='number of epochs')

	parser.add_argument('--train_scope', type=str,
		help="select from whole_discussion_coarse/whole_discussion_fine/limitation_only_coarse/limitation_only_fine")

	parser.add_argument('--test_scope', type=str,
		help="select from whole_discussion_coarse/whole_discussion_fine/limitation_only_coarse/limitation_only_fine")

	parser.add_argument('--model_path', type=str, 
		help='name of the pretrained model in huggingface')

	parser.add_argument('--max_length', type=int, 
		help='sentence max length')

	parser.add_argument('--train_data_path', type=str, 
		help='train dataset file')

	parser.add_argument('--dev_data_path', type=str, 
		help='dev dataset file')

	parser.add_argument('--test_data_path', type=str,
		help='test dataset file')

	parser.add_argument('--save_directory', type=str,
		help='path to save the checkpoint')

	parser.add_argument('--batch_size', type=int,
		help='batch size') 

	parser.add_argument('--predict', type=bool, help='predict mode') 

	parser.add_argument("--learning_rate", type=float, help="learning rate")

	parser.add_argument("--threshold", type=float, help="probablity threshold under one-to-multi setting")

	parser.add_argument('--checkpoint', type=str,
		help='path to the pretrained checkpoint') 
	
	parser.add_argument("--grad_acu_steps", type=int,
		help="number of grad acu steps")
	
	parser.add_argument("--neg_percentage", type=int,
		help="percentage of the negative examples")

	parser.add_argument("--fine_coarse", type = str, 
		help="fine grained or coarse grained")
	
	parser.add_argument("--random_or_sim", type = str, 
		help="select negative sample randomly or by similarity")

	parser.add_argument('--checkpoint_name', type=str,
		help='pretrained checkpoint name') 

	parser.add_argument('--from_pretrain', type=str, 
		help='from the pretrained checkpoint or not')

	parser.add_argument('--target_size', type=int, 
		help="the target size for each category")

	opt = parser.parse_args()

	sentences_1, labels_1 = read_sen_label("data/nlg_model_mix_output_part1.txt")
	# load input-view augmentation
	sentences_2, labels_2 = read_sen_label("data/nlg_model_mix_output_part2.txt")
	# load output-view augmentation

	df_augmented_1 = pd.DataFrame(zip(sentences_1, labels_1), columns=['sentences', 'label'])
	df_augmented_2 = pd.DataFrame(zip(sentences_2, labels_2), columns=['sentences', 'label'])

	save_directory = opt.save_directory
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	print("Hyperparameters :", opt)

	train_df = pd.read_csv(opt.train_data_path)
	test_df = pd.read_csv(opt.test_data_path)
	dev_df = pd.read_csv(opt.dev_data_path)

	fine_coarse = opt.fine_coarse
	from_pretrain = opt.from_pretrain


	if fine_coarse == "coarse":
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



	train_data_df = train_df[["pmid", "sentences", "section_name", "categories", "category_annotation"]]
	test_data_df = test_df[["pmid", "sentences", "section_name", "categories", "category_annotation"]]
	dev_data_df = dev_df[["pmid", "sentences", "section_name", "categories", "category_annotation"]]

	train_data_df["categories"] = train_data_df["categories"].apply(literal_eval)
	test_data_df["categories"] = test_data_df["categories"].apply(literal_eval)
	dev_data_df["categories"] = dev_data_df["categories"].apply(literal_eval)

	train_data_df = train_data_df.reset_index()
	test_data_df = test_data_df.reset_index()
	dev_data_df = dev_data_df.reset_index()

	train_data_df.columns = ["index", "pmid", "sentence_content", "section_name", "label", "category_annotation"]
	test_data_df.columns = ["index", "pmid", "sentence_content", "section_name", "label", "category_annotation"]
	dev_data_df.columns = ["index", "pmid", "sentence_content", "section_name", "label", "category_annotation"]

	convert_to_coarse_grained = {"DiagnosticCriteria": "Population", "VerySpecificPopulation": "Population", 
								"ConvenienceSampling": "Population", "Population": "Population", 
								"Unicentric": "Unicentric", "Setting": "Setting",
								"CompositeIntervention": "Intervention", "NonstandardTreatmentCharacteristics": "Intervention",
								"Intervention": "Intervention", "NoPlaceboGroup": "Control",
								"ActivePlacebo": "Control", "CareAsUsualControlGroup": "Control",
								"Control": "Control", "RelevantOutcomeExcluded": "OutcomeMeasures", 
								"PrecisionOfMeasurement": "OutcomeMeasures", "ValidityOfMeasurement": "OutcomeMeasures",
								"ResponsivenessOfMeasurement": "OutcomeMeasures", "OutcomeMeasures": "OutcomeMeasures",
								"HighLossToFollowUp": "MissingData", "UnbalancedDropout": "MissingData", 
								"MissingData": "MissingData", "SampleSize": "UnderpoweredStudy",
								"UnderpoweredStudy": "UnderpoweredStudy", "UnbalancedGroups": "Randomization",
								"PoorRandomizationMethods": "Randomization", "Randomization": "Randomization",
								"MultipleTesting": "StatisticalAnalysis", "ConfoundingFactors": "StatisticalAnalysis",
								"StatisticalAnalysis": "StatisticalAnalysis", "ExperimentalPhaseDuration": "StudyDuration",
								"FollowUpDuration": "StudyDuration", "StudyDuration": "StudyDuration",
								"Patient": "Blinding", "StudyTeam": "Blinding",
								"Blinding": "Blinding", "StudyDesign": "StudyDesign",
								"Funding": "Funding", "Generalization": "Generalization", "OTHER": "OTHER",
								"0": "0", 0: "0"
								}

	if opt.train_scope == "limitation_only_fine":
		train_data_df = train_data_df[train_data_df["category_annotation"] != "[False]"]

	elif opt.train_scope == "limitation_only_coarse":
		train_data_df = train_data_df[train_data_df["category_annotation"] != "[False]"]
		train_data_df["label"] = train_data_df["label"].map(lambda x: [convert_to_coarse_grained[y] for y in x if y in convert_to_coarse_grained])

	elif opt.train_scope == "whole_discussion_fine":
		train_data_df = train_data_df[train_data_df['section_name'].str.contains("Discussion|DISCUSSION|discussion|limitation|Limitation|LIMITATION|weakness|WEAKNESS|Weakness|caveat")==True]

	elif opt.train_scope == "whole_discussion_coarse":
		print(train_data_df.keys())
		train_data_df = train_data_df[train_data_df['section_name'].str.contains("Discussion|DISCUSSION|discussion|limitation|Limitation|LIMITATION|weakness|WEAKNESS|Weakness|caveat")==True]
		train_data_df["label"] = train_data_df["label"].map(lambda x: [convert_to_coarse_grained[y] for y in x if y in convert_to_coarse_grained])

	if opt.neg_percentage:
		train_df = pd.DataFrame(columns = ["index", "pmid", "sentence_content", "section_name", "label", "category_annotation"])
		train_data_df = train_data_df[train_data_df['section_name'].str.contains("Discussion|DISCUSSION|discussion|limitation|Limitation|LIMITATION|weakness|WEAKNESS|Weakness|caveat")==True]
		if opt.fine_coarse == "coarse":
			train_data_df["label"] = train_data_df["label"].map(lambda x: [convert_to_coarse_grained[y] for y in x if y in convert_to_coarse_grained])
		pmids = set(train_data_df["pmid"].to_list())
		train_data_df_pos = train_data_df[train_data_df["category_annotation"] != "[False]"]
		train_data_df_neg = train_data_df[train_data_df["category_annotation"] == "[False]"]

		for pmid in pmids:
			train_data_df_pos_sample = train_data_df_pos[train_data_df_pos["pmid"] == pmid]
			if opt.random_or_sim == "sim":
				neg_sample_list = train_data_df_neg["sentence_content"].to_list()
				pos_sample_list = train_data_df_pos["sentence_content"].to_list()

				train_data_df_neg_sample = pd.DataFrame(columns = ["index", "pmid", "sentence_content", "section_name", "label", "category_annotation"])

				for pos_sample in pos_sample_list:
					sim_sentence = select_top_similar_sentences(neg_sample_list, pos_sample, simcse_model)
					try:
						train_data_df_neg_sample = pd.concat([train_data_df_neg_sample, train_data_df_neg[train_data_df_neg["sentence_content"]==sim_sentence[0][0]]])
					except:
						print(neg_sample_list, pos_sample)
			else:
				train_data_df_neg_sample = train_data_df_neg[train_data_df_neg["pmid"] == pmid].sample(n = int(len(train_data_df_pos_sample) * 0.01 * opt.neg_percentage), replace=True)

			train_df = pd.concat([train_df, train_data_df_pos_sample])
			train_df = pd.concat([train_df, train_data_df_neg_sample])

		train_data_df = train_df


	if opt.test_scope == "limitation_only_fine":
		test_data_df = test_data_df[test_data_df["category_annotation"] != "[False]"]
		dev_data_df = dev_data_df[dev_data_df["category_annotation"] != "[False]"]

	elif opt.test_scope == "limitation_only_coarse":
		test_data_df = test_data_df[test_data_df["category_annotation"] != "[False]"]
		dev_data_df = dev_data_df[dev_data_df["category_annotation"] != "[False]"]
		test_data_df["label"] = test_data_df["label"].map(lambda x: [convert_to_coarse_grained[y] for y in x if y in convert_to_coarse_grained])
		dev_data_df["label"] = dev_data_df["label"].map(lambda x: [convert_to_coarse_grained[y] for y in x if y in convert_to_coarse_grained])
		
	elif opt.test_scope == "whole_discussion_fine":
		test_data_df = test_data_df[test_data_df['section_name'].str.contains("Discussion|DISCUSSION|discussion|limitation|Limitation|LIMITATION|weakness|WEAKNESS|Weakness|caveat")==True]
		dev_data_df = dev_data_df[dev_data_df['section_name'].str.contains("Discussion|DISCUSSION|discussion|limitation|Limitation|LIMITATION|weakness|WEAKNESS|Weakness|caveat")==True]

	elif opt.test_scope == "whole_discussion_coarse":
		test_data_df = test_data_df[test_data_df['section_name'].str.contains("Discussion|DISCUSSION|discussion|limitation|Limitation|LIMITATION|weakness|WEAKNESS|Weakness|caveat")==True]
		dev_data_df = dev_data_df[dev_data_df['section_name'].str.contains("Discussion|DISCUSSION|discussion|limitation|Limitation|LIMITATION|weakness|WEAKNESS|Weakness|caveat")==True]
		test_data_df["label"] = test_data_df["label"].map(lambda x: [convert_to_coarse_grained[y] for y in x if y in convert_to_coarse_grained])
		dev_data_df["label"] = dev_data_df["label"].map(lambda x: [convert_to_coarse_grained[y] for y in x if y in convert_to_coarse_grained])


	if opt.neg_percentage:
		test_df = pd.DataFrame(columns = ["index", "pmid", "sentence_content", "section_name", "label", "category_annotation"])
		test_data_df = test_data_df[test_data_df['section_name'].str.contains("Discussion|DISCUSSION|discussion|limitation|Limitation|LIMITATION|weakness|WEAKNESS|Weakness|caveat")==True]
		if opt.fine_coarse == "coarse":
			test_data_df["label"] = test_data_df["label"].map(lambda x: [convert_to_coarse_grained[y] for y in x if y in convert_to_coarse_grained])
		pmids = set(test_data_df["pmid"].to_list())
		test_data_df_pos = test_data_df[test_data_df["category_annotation"] != "[False]"]
		test_data_df_neg = test_data_df[test_data_df["category_annotation"] == "[False]"]


		for pmid in pmids:
			test_data_df_pos_sample = test_data_df_pos[test_data_df_pos["pmid"] == pmid]

			if opt.random_or_sim == "sim":
				neg_sample_list = test_data_df_neg["sentence_content"].to_list()
				pos_sample_list = test_data_df_pos["sentence_content"].to_list()
				test_data_df_neg_sample = pd.DataFrame(columns = ["index", "pmid", "sentence_content", "section_name", "label", "category_annotation"])

				for pos_sample in pos_sample_list:	
					sim_sentence = select_top_similar_sentences(neg_sample_list, pos_sample, simcse_model)
					try:
						test_data_df_neg_sample = pd.concat([test_data_df_neg_sample, test_data_df_neg[test_data_df_neg["sentence_content"]==sim_sentence[0][0]]])
					except:
						print(neg_sample_list, pos_sample)
			else:
				test_data_df_neg_sample = test_data_df_neg[test_data_df_neg["pmid"] == pmid].sample(n = int(len(test_data_df_pos_sample) * 0.01 * opt.neg_percentage), replace=True)

			test_data_df_pos_sample = test_data_df_pos[test_data_df_pos["pmid"] == pmid]
			
			test_df = pd.concat([test_df, test_data_df_pos_sample])
			test_df = pd.concat([test_df, test_data_df_neg_sample])

		dev_df = pd.DataFrame(columns = ["index", "pmid", "sentence_content", "section_name", "label", "category_annotation"])
		dev_data_df = dev_data_df[dev_data_df['section_name'].str.contains("Discussion|DISCUSSION|discussion|limitation|Limitation|LIMITATION|weakness|WEAKNESS|Weakness|caveat")==True]
		if opt.fine_coarse == "coarse":
			dev_data_df["label"] = dev_data_df["label"].map(convert_to_coarse_grained)		
		pmids = set(dev_data_df["pmid"].to_list())
		dev_data_df_pos = dev_data_df[dev_data_df["category_annotation"] != "[False]"]
		dev_data_df_neg = dev_data_df[dev_data_df["category_annotation"] == "[False]"]


		for pmid in pmids:
			dev_data_df_pos_sample = dev_data_df_pos[dev_data_df_pos["pmid"] == pmid]
			if opt.random_or_sim == "sim":
				neg_sample_list = dev_data_df_neg["sentence_content"].to_list()
				pos_sample_list = dev_data_df_pos["sentence_content"].to_list()

				dev_data_df_neg_sample = pd.DataFrame(columns = ["index", "pmid", "sentence_content", "section_name", "label", "category_annotation"])

				for pos_sample in pos_sample_list:
					sim_sentence = select_top_similar_sentences(neg_sample_list, pos_sample, simcse_model)
					try:
						dev_data_df_neg_sample = pd.concat([dev_data_df_neg_sample, dev_data_df_neg[dev_data_df_neg["sentence_content"]==sim_sentence[0][0]]])
					except:
						print(neg_sample_list, pos_sample)
			else:
				dev_data_df_neg_sample = dev_data_df_neg[dev_data_df_neg["pmid"] == pmid].sample(n = int(len(dev_data_df_pos_sample) * 0.01 * opt.neg_percentage), replace=True)

			dev_df = pd.concat([dev_df, dev_data_df_pos_sample])
			dev_df = pd.concat([dev_df, dev_data_df_neg_sample])

		test_data_df = test_df
		dev_data_df = dev_df


	train_data_df = train_data_df[["index", "sentence_content", "label"]]
	test_data_df = test_data_df[["index", "sentence_content", "label"]]
	dev_data_df = dev_data_df[["index", "sentence_content", "label"]]

	if opt.checkpoint_name:
		checkpoint_name = opt.checkpoint_name
	else:
		checkpoint_name = str(opt.save_directory) + "/train_" + str(opt.train_scope) + \
					"_" + str(opt.neg_percentage) + "_" + str(opt.fine_coarse) + "_" + str(opt.random_or_sim) + \
					"_" + str(opt.model_path).split("/")[-1] + "_best_model.pth"

	print("save to: ", checkpoint_name)

	if not opt.predict:
		unique_labels = select_unique_labels(train_data_df.label.to_list()).union(select_unique_labels(test_data_df.label.to_list())).union(select_unique_labels(dev_data_df.label.to_list()))
		label_index =  [i for i in range(len(unique_labels))]
		labels_to_id = dict(zip(unique_labels, label_index))
	else:
		labels_to_id = txt_to_dict(checkpoint_name.strip(".pth") + ".txt")
		labels_to_id = {k:int(v) for k,v in labels_to_id.items()}
		unique_labels = labels_to_id.keys()

	print(labels_to_id)
	print(unique_labels)
	print(len(unique_labels))
	
	train_data_df["label_id"] = train_data_df["label"].map(lambda x: [labels_to_id[y] for y in x if y in labels_to_id])
	test_data_df["label_id"] = test_data_df["label"].map(lambda x: [labels_to_id[y] for y in x if y in labels_to_id])
	dev_data_df["label_id"] = dev_data_df["label"].map(lambda x: [labels_to_id[y] for y in x if y in labels_to_id])

	# train_data_df['label_id'] = train_data_df['label_id'].fillna(0)
	# test_data_df['label_id'] = test_data_df['label_id'].fillna(0)
	# dev_data_df['label_id'] = dev_data_df['label_id'].fillna(0)

	train_data_df['label_id'] = train_data_df['label_id'].apply(lambda d: d if isinstance(d, list) else [])
	test_data_df['label_id'] = test_data_df['label_id'].apply(lambda d: d if isinstance(d, list) else [])
	dev_data_df['label_id'] = dev_data_df['label_id'].apply(lambda d: d if isinstance(d, list) else [])


	model = BertClassifier(opt.model_path, len(unique_labels)).to(device)

	tokenizer = BertTokenizer.from_pretrained(opt.model_path)


	with open(checkpoint_name.strip(".pth") + ".txt", 'w') as f:
		for key, value in labels_to_id.items():
			f.write(str(key) + '\t' + str(value) + '\n')

	if not opt.predict:
		print("start to train ------ ")
		train(model, train_data_df, dev_data_df, opt.learning_rate, tokenizer, opt.max_length, opt.num_epochs, checkpoint_name, opt.grad_acu_steps, labels_to_id, opt.threshold)
		
		print("start to test ------ ")
		model.load_state_dict(torch.load(checkpoint_name))
		model.eval()

		print(checkpoint_name + " loaded. ")

		test_labels, test_preds, test_inputs = evaluate(model, test_data_df, tokenizer, opt.max_length, opt.threshold)

		id_to_labels = {int(y): x for x, y in labels_to_id.items()}
		print("id_to_labels: ", id_to_labels)
		
		bert_labels = []
		for i in test_labels:
			bert_labels.append(i.detach().cpu().numpy().tolist()[0])

		bert_preds = []
		for i in test_preds:       
			bert_preds.append(i.detach().cpu().numpy().tolist()[0])

		test_converted = [id_to_labels[int(i)] for i in bert_labels]
		pred_converted = [id_to_labels[int(i)] for i in bert_preds]

		print(classification_report(test_converted, pred_converted))
		print("F1 score is: ", f1_score(test_converted, pred_converted, average='micro'))
	
	else:

		print("start to test ------ ")
		model.load_state_dict(torch.load(checkpoint_name))
		model.eval()

		print(checkpoint_name + " loaded. ")

		test_labels, test_preds, test_inputs = evaluate(model, test_data_df, tokenizer, opt.max_length)

		id_to_labels = {int(y): x for x, y in labels_to_id.items()}
		print("id_to_labels: ", id_to_labels)
		
		bert_labels = []
		for i in test_labels:
			bert_labels.append(i.detach().cpu().numpy().tolist()[0])

		bert_preds = []
		for i in test_preds:
			bert_preds.append(i.detach().cpu().numpy().tolist()[0])

		print("bert_labels: ", bert_labels)
		print("bert_preds: ", bert_preds)

		test_converted = [id_to_labels[int(i)] for i in bert_labels]
		pred_converted = [id_to_labels[int(i)] for i in bert_preds]

		print(classification_report(test_converted, pred_converted))
		print("F1 score is: ", f1_score(test_converted, pred_converted, average='micro'))

if __name__ == "__main__":
    main()