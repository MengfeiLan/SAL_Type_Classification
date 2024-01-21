from torch.optim import Adam
from tqdm import tqdm
import torch
from torch.nn import functional as F
import numpy as np
from sklearn.metrics import f1_score

def predict(logits, threshold):
    probs = logits
    predictions = []
    for idx in range(probs.size(0)):
        output = []
        for l in range(0, probs.size(1)):
            if probs[idx, l] >= threshold:
                output.append(l)
        if len(output) == 0:
            output.append(torch.argmax(probs[idx]).item())
        else:
            output = output

        predictions.append(output)

    return predictions

def predict_thresholds(logits, thresholds, vocab, default_threshold):
    inv_vocab = {v:k for k,v in vocab.items()}
    probs = np.array(logits)
    predictions = []
    for idx in range(probs.shape[0]):
        output = []
        for l in range(0, probs.shape[1]):
            if inv_vocab[l] in thresholds:
                if probs[idx, l] >= thresholds[inv_vocab[l]]:
                    output.append(l)
            else:
                if probs[idx, l] >= default_threshold:
                    output.append(l)
        if len(output) == 0:
            output.append(np.argmax(probs[idx]).item())
        else:
            output = output
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

def pred_single_label(output, id, threshold):
    probs = np.array(output)
    predictions = []
    for idx in range(probs.shape[0]):
        output = []
        for l in range(0, probs.shape[1]):
            if probs[idx, l] >= threshold:
                if l == id:
                    output.append(1)
                    break
        if len(output) == 0:
            # predict = np.argmax(probs[idx])
            # if predict == id:
            #     output.append(1)
            # else:
            output.append(0)

        predictions.extend(output)

    return predictions

def evaluation_thresholds(alllabels, output, labels_to_id, default_threshold):
    thresholds = {}
    for label, id in labels_to_id.items():
        best_f1 = -1
        for threshold in [x * 0.1 for x in range(0, 10)]:
            predictions = pred_single_label(output, id, threshold)
            f1 = f1_score([1 if id in singlelabels else 0 for singlelabels in alllabels], predictions)
            if f1 > best_f1:
                best_f1 = f1
                if threshold == 0:
                    thresholds[label] = default_threshold
                else:
                    thresholds[label] = threshold

    p, r, f, total = evaluation(alllabels, predict_thresholds(output, thresholds, labels_to_id, default_threshold), labels_to_id)
    return p, r, f, thresholds, total


def train(model, train_dataloader, val_dataloader, learning_rate, tokenizer, max_len, epochs, checkpoint_name, grad_acu_steps, labels_to_id, thresholds_multi_label, default_threshold, loss_func):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    optimizer = Adam(model.parameters(), lr= learning_rate)

    if use_cuda:

            model = model.cuda()

    best_f = 0
    thresholds = {}

    if thresholds_multi_label == False:
        for epoch_num in range(epochs):
            total_loss_train = 0
            len_train_data = 0
            len_val_data = 0

            print("epoch " + str(epoch_num))

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

                    pred_labels = predict(output, default_threshold)

                    golden_labels = val_label_origin

                    alllabels.extend(golden_labels)
                    allpreds.extend(pred_labels)



                golden_cleaned = [single_label.tolist() for single_label in alllabels]

                alllabels = golden_cleaned

            p, r, f, total = evaluation(alllabels, allpreds, labels_to_id)

            print(
                f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len_train_data : .3f} \
                | Val Loss: {total_loss_val / len_val_data: .3f} \
                | Val f1: {f : .3f}', \
                "| p, r, f, total: ", p, r, f, total) \


            if f > best_f:
                best_f = f
                torch.save(model.state_dict(), checkpoint_name)
                print("model_saved")
                print("current average f1 is {:.4f}, best f1 is {:.4f}".format(f, best_f))

            if epoch_num == 4 and best_f < 0.5:
                return True, thresholds
            if epoch_num == epochs - 1:
                return False, thresholds
    else:
        for epoch_num in range(epochs):
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

            total_loss_val = 0

            allpreds, alllabels, alllabels_onehot, alloutput = [], [], [], []

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

                    golden_labels = val_label_origin

                    alllabels.extend(golden_labels)
                    alllabels_onehot.extend(val_label)
                    alloutput.extend(output.cpu().tolist())

                golden_cleaned = [single_label.tolist() for single_label in alllabels]

                alllabels = golden_cleaned

            p, r, f, thresholds, total = evaluation_thresholds(alllabels, alloutput, labels_to_id, default_threshold)

            print("thresholds: ", thresholds)

            print(
                f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len_train_data : .3f} \
                | Val Loss: {total_loss_val / len_val_data: .3f} \
                | Val f1: {f : .3f}', \
                "| p, r, f, total: ", p, r, f, total)


            if f > best_f:
                best_f = f
                best_threshold = thresholds
                torch.save(model.state_dict(), checkpoint_name)
                print("model_saved")
                print("current acc is {:.4f}, best acc is {:.4f}".format(f, best_f))
                with open(checkpoint_name.strip(".pth") + "_thresholds.txt", 'w') as file:
                    for key, i in thresholds.items():
                        file.write(str(key) + "\t" + str(i) + '\n')
            if epoch_num == 4 and best_f < 0.5:
                return True, best_threshold
            if epoch_num == epochs - 1:
                return False, best_threshold

def evaluate(model, test_dataloader, threshold):

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:

        model = model.cuda()
    alllabels = []
    allpreds = []
    allinputs = []

    with torch.no_grad():

        for test_input in test_dataloader:
            test_label = test_input["labels"]
            test_label_origin = test_input["origin_labels"]
            input_id = test_input['input_ids'].squeeze(1).to(device)

            output = model(input_id)
            output = output.logits
            output = output[:, -1]
            output = F.softmax(output, dim=1)

            pred_labels = predict(output, threshold)

            golden_labels = test_label_origin

            alllabels.extend(golden_labels)
            allpreds.extend(pred_labels)
            allinputs.extend(input_id)

        golden_labels = [single_label.tolist() for single_label in alllabels]
        alllabels = golden_labels

    return alllabels, allpreds, allinputs


def evaluate_multi_thresholds(model, test_dataloader, default_threshold, thresholds, labels_to_id):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    alllabels = []
    allinputs = []
    alloutput = []
    allpmcids = []

    with torch.no_grad():
        for test_input in tqdm(test_dataloader):
            test_label_origin = test_input["origin_labels"]
            input_id = test_input['input_ids'].squeeze(1).to(device)
            pmcid = [t for t in test_input["pmcids"]]

            output = model(input_id)
            output = output.logits
            output = output[:, -1]
            output = F.softmax(output, dim=1)

            golden_labels = test_label_origin

            alllabels.extend(golden_labels)
            allinputs.extend(input_id)
            output_logits = output.cpu().numpy()
            output_logits = np.round(output_logits, 2)
            output_logits = output_logits.tolist()
            alloutput.extend(output_logits)
            allpmcids.extend(pmcid)

        predictions = predict_thresholds(alloutput, thresholds, labels_to_id, default_threshold)
        golden_labels = [single_label.tolist() for single_label in alllabels]
        alllabels = golden_labels

    return alllabels, predictions, allinputs, allpmcids, alloutput

