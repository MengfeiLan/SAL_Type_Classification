from torch.optim import Adam
from dataloader import Dataset
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

loss_func = nn.BCELoss()
loss_func = loss_func.cuda()


def predict(logits, threshold):
    # INPUT:
    ##  logits: a torch.Tensor of (batch_size, number_of_event_types) which is the output logits for each event type (none types are included).
    # OUTPUT:
    ##  a 2-dim list which has the same format with the "labels" in "loss_func" --- the predictions for all the sentences in the batch.
    ##  For example, if predictions == [[0], [2,3,4]] then the batch size is 2, and we predict no events for the first sentence and three events (2,3,and 4) for the second sentence.

    ##  INSTRUCTIONS: The most straight-forward way for prediction is to select out the indices with maximum of logits. Note that this is a multi-label classification problem, so each sentence could have multiple predicted event indices. Using what threshold for prediction is important here. You can also use the None event (index 0) as the threshold as what https://arxiv.org/pdf/2202.07615.pdf does.

    ###  YOU NEED TO WRITE YOUR CODE HERE.  ###

    probs = logits
    predictions = []
    for idx in range(probs.size(0)):
        output = []
        for l in range(0, probs.size(1)):
            if probs[idx, l] >= threshold:
                output.append(l)
        if len(output) == 0:
            # print(probs[idx])
            # print(torch.argmax(probs[idx]))
            output.append(torch.argmax(probs[idx]).item())
        else:
            output = output
        # print("preds: ", output)

        predictions.append(output)

    return predictions



def save_div(a, b):
    if b != 0:
        return a / b 
    else:
        return 0.0

def evaluation(gold_labels, pred_labels, vocab):
    inv_vocab = {v:k for k,v in vocab.items()}
    print("inv_vocab: ", inv_vocab)
    result = {}
    for label, idx in vocab.items():
        result[label] = {"prec": 0.0, "rec": 0.0, "f1": 0.0, "support": 0}

    total_pred_num, total_gold_num, total_correct_num = 0.0, 0.0, 0.0

    for i in range(len(gold_labels)):

        pred_labels_i = pred_labels[i]
        gold_labels_i = gold_labels[i]

        for idx in gold_labels_i:
            result[inv_vocab[idx]]["support"] += 1
            total_gold_num += 1
            result[inv_vocab[idx]]["rec"] += 1

        for idx in pred_labels_i:
            total_pred_num += 1
            result[inv_vocab[idx]]["prec"] += 1

            if idx in gold_labels_i:
                total_correct_num += 1
                result[inv_vocab[idx]]["f1"] += 1

    for label in result:
        counts = result[label]
        counts["prec"] = save_div(counts["f1"], counts["prec"])
        counts["rec"] = save_div(counts["f1"], counts["rec"])
        counts["f1"] = save_div(2*counts["prec"]*counts["rec"], counts["prec"]+counts["rec"])

    prec = save_div(total_correct_num, total_pred_num)
    rec = save_div(total_correct_num, total_gold_num)
    f1 = save_div(2*prec*rec, prec+rec)

    return prec, rec, f1, result

def my_collate_fn(batch):
    elem = batch[0]

    return_dict = {}
    for key in elem:
        if key == "encoded_tgt_text":
            return_dict[key] = [d[key] for d in batch]
        else:
            try:
                return_dict[key] = default_collate([d[key] for d in batch])
            except:
                return_dict[key] = [d[key] for d in batch]

    return return_dict


def train(model, train_dataloader, val_dataloader, learning_rate, tokenizer, max_len, epochs, checkpoint_name, grad_acu_steps, labels_to_id, threshold):

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    optimizer = Adam(model.parameters(), lr= learning_rate)

    if use_cuda:

            model = model.cuda()

    best_acc = 0


    for epoch_num in range(epochs):

        total_acc_train = 0
        total_loss_train = 0
        len_train_data = 0
        len_val_data = 0

        for train_input in tqdm(train_dataloader):

            train_label = train_input["labels"]
            input_id = train_input['input_ids'].squeeze(1).to(device)
            len_train_data += 1

            output = model(input_id)
            output = output.logits
            output = output[:, -1]
            output = F.softmax(output, dim=1)


            batch_loss = loss_func(output.float(), train_label.to(device).float())
            total_loss_train += batch_loss.item()

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        total_acc_val = 0
        total_loss_val = 0

        allpreds, alllabels = [], []

        with torch.no_grad():

            for val_input in val_dataloader:

                val_label = val_input["labels"]
                val_label_origin = val_input["origin_labels"]
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id)
                output = output.logits
                output = output[:, -1]
                output = F.softmax(output, dim=1)

                batch_loss = loss_func(output.float(), val_label.to(device).float())
                total_loss_val += batch_loss.item()
                len_val_data += 1

                pred_labels = predict(output, threshold)

                golden_labels = val_label_origin

                alllabels.extend(golden_labels)
                allpreds.extend(pred_labels)



            golden_cleaned = [single_label.tolist() for single_label in alllabels]
            # preds_cleaned = [single_label.tolist() for single_label in allpreds]

            alllabels = golden_cleaned
            # allpreds = preds_cleaned

            print("labels: ", alllabels)
            print("preds: ", allpreds)

        p, r, f, total = evaluation(alllabels, allpreds, labels_to_id)

        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len_train_data : .3f} \
            | Val Loss: {total_loss_val / len_val_data: .3f} \
            | Val f1: {f : .3f}', \
            "| p, r, f, total: ", p, r, f, total) \

        if f > best_acc:
            best_acc = f
            torch.save(model.state_dict(), checkpoint_name)
            print("model_saved")


            print("current acc is {:.4f}, best acc is {:.4f}".format(total_acc_val / len_val_data, best_acc))

def evaluate(model, test_dataloader, tokenizer, max_len, threshold):

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()
    alllabels = []
    allpreds = []
    allinputs = []

    with torch.no_grad():
        for test_input in tqdm(test_dataloader):
            test_label = test_input["labels"]
            test_label_origin = test_input["origin_labels"]
            input_id = test_input['input_ids'].squeeze(1).to(device)

            output = model(input_id)
            output = output.logits
            output = output[:, -1]
            output = F.softmax(output, dim=1)

            batch_loss = loss_func(output.float(), test_label.to(device).float())
            pred_labels = predict(output, threshold)

            golden_labels = test_label_origin

            alllabels.extend(golden_labels)
            allpreds.extend(pred_labels)
            allinputs.extend(input_id)

        golden_labels = [single_label.tolist() for single_label in alllabels]
        alllabels = golden_labels
        print("golden: ", alllabels)
        print("preds: ", allpreds)

        p, r, f, total = evaluation(alllabels, allpreds, labels_to_id)


    return alllabels, allpreds, allinputs

    
