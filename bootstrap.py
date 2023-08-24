from sklearn.utils import resample
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from ast import literal_eval
import numpy as np

n_iterations = 1000

experiment_setting = "coarse_"
accuracy = []
precision = []
recall = []
F1 = []

def save_div(a, b):
    if b != 0:
        return a / b
    else:
        return 0.0

def evaluation(gold_labels, pred_labels):
    total_pred_num, total_gold_num, total_correct_num = 0.0, 0.0, 0.0

    for i in range(len(gold_labels)):

        pred_labels_i = pred_labels[i]
        gold_labels_i = gold_labels[i]

        for idx in gold_labels_i:
            total_gold_num += 1

        for idx in pred_labels_i:
            total_pred_num += 1

            if idx in gold_labels_i:
                total_correct_num += 1

    prec = save_div(total_correct_num, total_pred_num)
    rec = save_div(total_correct_num, total_gold_num)
    f1 = save_div(2*prec*rec, prec+rec)

    return prec, rec, f1

encode_map = {"StatisticalAnalysis": 0, 
                "Population":1,
                "OTHER":2,
                "Blinding":3,
                "Intervention":4,
                "Setting":5,
                "UnderpoweredStudy":6,
                "StudyDesign":7,
                "OutcomeMeasures":8,
                "Randomization":9,
                "Generalization":10,
                "StudyDuration":11,
                "Control":12,
                "Funding":13,
                "MissingData":14}

def encode(l1):
    encoded_result = []
    for i in l1:
        new_encode = []
        for j in i:
            new_encode.append(encode_map[j])
        encoded_result.append(new_encode)
    return encoded_result


for i in range(1,6):
    df = pd.read_csv(experiment_setting + str(i) + ".csv")
    y_labels = df.label.apply(literal_eval).to_list()
    y_preds = df.pred.apply(literal_eval).to_list()

    for j in range(n_iterations):
        X_bs, y_bs = resample(y_labels, y_preds, replace=True)
        # X_bs = encode(X_bs)
        # y_bs = encode(y_bs)
        p_single, r_single, f_single = evaluation(X_bs, y_bs)
        precision.append(p_single)
        recall.append(r_single)
        F1.append(f_single)

cil_p, ciu_p = np.quantile(precision, 0.025), np.quantile(precision, 0.975)
cil_r, ciu_r = np.quantile(recall, 0.025), np.quantile(recall, 0.975)
cil_f, ciu_f = np.quantile(F1, 0.025), np.quantile(F1, 0.975)

print("precision: ", np.mean(precision))
print("recall: ", np.mean(recall))
print("f1: ", np.mean(F1))

print("precision: ", cil_p, ciu_p)
print("recall: ", cil_r, ciu_r)
print("f1: ", cil_f, ciu_f)