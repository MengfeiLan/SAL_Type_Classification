from argparse import ArgumentParser
from dataloader import *
from framework import *
import pandas as pd
import torch
from torch.utils.data._utils.collate import default_collate
from transformers import BertTokenizer, BertForTokenClassification
from sklearn.metrics import accuracy_score
from check_funding import *
from main import prepare_data

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--large_scale_data_file', type=str,
                        help='the file contains large scale samples')
    parser.add_argument('--checkpoint', type=str, help="path to the checkpoint")

    config = parser.parse_args()
    print("config", config)

    thresholds_multi_label = config.thresholds_multi_label

    tokenizer = BertTokenizer.from_pretrained(config.bert_model)
    labels_to_id, unique_labels, checkpoint_name, num_label, train_data_df, test_data_df, dev_data_df = prepare_data(config)

    train_dataset, val_dataset = Dataset(train_data_df, tokenizer, config.max_length, num_label), Dataset(dev_data_df,
                                                                                                          tokenizer,
                                                                                                          config.max_length,
                                                                                                          num_label)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, collate_fn=my_collate_fn, batch_size=2, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, collate_fn=my_collate_fn, batch_size=2)

    test_dataset = Dataset(test_data_df, tokenizer, config.max_length, num_label)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, collate_fn=my_collate_fn, batch_size=2, shuffle=True)

    model_augmented = BertForTokenClassification.from_pretrained(config.bert_model, num_labels=num_label)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config.from_pretrain == True:
        model_augmented.load_state_dict(torch.load(checkpoint_name))

    model_augmented.to(device)

    model_augmented.load_state_dict(torch.load(config.checkpoint))
    thresholds = load_thresholds_from_pretrained(config.checkpoint.strip(".pth") + "_thresholds.txt")

    print("loaded_thresholds: ", thresholds)

    if thresholds_multi_label == False:
        alllabels, allpreds, allinputs = evaluate(model_augmented, test_dataloader, config.default_threshold)
    else:
        alllabels, allpreds, allinputs = evaluate_multi_thresholds(model_augmented, test_dataloader,
                                                                   config.default_threshold, thresholds, labels_to_id)

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

    id_to_labels = {i: key for key, i in labels_to_id.items()}

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
        data.to_csv(config.checkpoint.strip(".pth") + ".csv")
