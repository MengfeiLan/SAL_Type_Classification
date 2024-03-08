import ast
import pandas as pd
from argparse import ArgumentParser
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
import numpy as np
from utils import load_label_from_pretrained

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--prediction_file_location', type=str, help='location of the file that contains predictions')
    config = parser.parse_args()
    data = pd.read_csv(config.prediction_file_location)
    data["label"] = data["label"].apply(ast.literal_eval)
    data["pred"] = data["pred"].apply(ast.literal_eval)
    true = data["label"].to_list()
    pred = data["pred"].to_list()

    label_dict = load_label_from_pretrained(config.prediction_file_location[:-4] + "_labels.txt")

    print("label_dict: ", label_dict)

    categories = sorted(label_dict, key=lambda x: label_dict[x])

    true_onehot = np.zeros([len(true), len(categories)])
    for i in range(len(true)):
        for j in range(len(true[i])):
            true_onehot[i][label_dict[true[i][j]]] = 1
    pred_onehot = np.zeros([len(pred), len(categories)])
    for i in range(len(pred)):
        for j in range(len(pred[i])):
            pred_onehot[i][label_dict[pred[i][j]]] = 1

    report = classification_report(true_onehot, pred_onehot, target_names=categories, zero_division=0, digits=4)
    print(report)


# convert the BIO tags to entity spans
def bio_to_spans(sentence, tags):
    spans = []
    start = None
    current_tag = None
    for i, (token, tag) in enumerate(zip(sentence, tags)):
        # Check if the current token is part of an entity
        if tag.startswith('B-'):  # Beginning of an entity
            if start is not None:
                spans.append((start, i, current_tag[2:]))  # Add the previous span
            start = i
            current_tag = tag
        elif tag.startswith('I-'):  # Inside an entity
            if start is None:
                start = i
            elif tag[2:] != current_tag[2:]:  # Different entity type
                spans.append((start, i, current_tag[2:]))  # Add the previous span
                start = i
                current_tag = tag
        else:  # Outside an entity
            if start is not None:
                spans.append((start, i, current_tag[2:]))  # Add the previous span
                start = None
                current_tag = None
    # Add the last span if it exists
    if start is not None:
        spans.append((start, len(sentence), current_tag[2:]))
    return [(" ".join(sentence[start:end]), entity_type) for start, end, entity_type in spans]