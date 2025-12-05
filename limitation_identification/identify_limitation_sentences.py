import pandas as pd
import torch
from tqdm import tqdm
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
from datasets import Dataset
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers.models.bert.modeling_bert import BertForTokenClassification
from transformers import BertTokenizer, BertConfig
from ast import literal_eval as load


def combine_sentence_with_section(l1, l2):
    return_list = []
    for i in l1:
        return_list.append(i)
    for j in l2:
        return_list.append(j)

    return [return_list]

def contain_specific_section(l1):
    lower_case = [i.lower() for i in l1]
    section_all = " ".join(lower_case)
    list = ["abstract", "discussion", "limitation", "weakness", "conclusion", "discussions", "limitations", "weaknesses", "conclusions", "caveat", "shortcoming", "drawback"]
    for i in list:
        if i in section_all:
            return True
    return False

def add_list(s1):
    return [s1]

def convert_to_binary(l1):
    return [0]


maxlen = 512
pad_token_id = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
ignore_label_id = CrossEntropyLoss().ignore_index

def convert(df):
    input_ids_ls = []
    attention_mask_ls = []
    labels_ls = []
    pmcid_sids = []
    for i in range(len(df)):
        sents = (df.loc[i, 'sentence'])
#         print ("Sentence:", sents)
        labels = (df.loc[i, 'label'])
        pmcid = (df.loc[i, 'PMCID'])
        sid = (df.loc[i, 'SENTENCEID'])
        tokens = []
        label_ids = []
        for j in range(len(sents)):  # loop over each sentence
            token_tmp = []
#             print (sents[j])
            for word in sents[j]:
                word_tokens = tokenizer.tokenize(word)
                token_tmp.extend(word_tokens)
            token_tmp.extend([tokenizer.sep_token])
            label_ids.extend([ignore_label_id] * (len(token_tmp)-1)+[labels[0]])
            tokens.extend(token_tmp)


        if len(tokens) > maxlen:
            tokens = tokens[:maxlen]
            label_ids = label_ids[:maxlen]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        padding_length = maxlen - len(input_ids)
        attention_mask = [1]*len(input_ids) + [0]*padding_length
        input_ids.extend([pad_token_id] * padding_length)
        label_ids.extend([ignore_label_id] * padding_length)
        assert len(input_ids) == maxlen
        assert len(attention_mask) == maxlen
        assert len(label_ids) == maxlen
        input_ids_ls.append(input_ids)
        attention_mask_ls.append(attention_mask)
        labels_ls.append(label_ids)
        pmcid_sids.append(pmcid + "_" + sid)
    tokenized_df = pd.DataFrame(
        [input_ids_ls, attention_mask_ls, labels_ls, pmcid_sids]).transpose()
    tokenized_df.columns = ['input_ids',
                            'attention_mask', 'labels', "pmcid_sids"]
    return tokenized_df

def evaluate(model, data_loader, device=device):
    model.eval()
    val_true, val_pred = [], []
    i = 0
    for batch in tqdm(data_loader):
        i += 1
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            logits = outputs.logits
        y_pred = torch.argmax(logits, dim=-1).detach().cpu().numpy().tolist()
        y_true = batch['labels'].detach().cpu().numpy().tolist()
        real_len = torch.sum(batch['attention_mask'],
                             1).detach().cpu().numpy().tolist()
        for j in range(len(real_len)):
            pred_tmp = []
            true_tmp = []
            for k in range(real_len[j]):
                if y_true[j][k] != ignore_label_id:
                    pred_tmp.append(y_pred[j][k])
                    true_tmp.append(y_true[j][k])
            val_true.append(true_tmp)
            val_pred.append(pred_tmp)

    return val_true, val_pred


data_file = "data/11988_rct_articles.csv"

data_file = data_file[:]
all_data_with_sections_df = pd.read_csv(data_file)
all_data_with_sections_df = all_data_with_sections_df
all_data_with_sections_df["section"]= all_data_with_sections_df["section"].apply(load)
all_data_with_sections_df["token"]= all_data_with_sections_df["token"].apply(load)

exist_pmcids_limitation_dataset = ['PMC5120936','PMC5488440','PMC5152707','PMC6075465','PMC64502','PMC3502035','PMC5228538','PMC5685027','PMC4029747','PMC5834076','PMC4338564','PMC3391717','PMC6172755',
 'PMC3348565','PMC3420230','PMC5134778','PMC3661377','PMC1208877','PMC3070271','PMC3489506','PMC4407900','PMC4802874','PMC8296136',
 'PMC3853239','PMC2768730','PMC5427564','PMC5457543','PMC3943833','PMC4446825','PMC8144928','PMC2575590','PMC5508816','PMC5539623','PMC4195950','PMC3852834','PMC5987848','PMC3880176','PMC5781264','PMC5210360','PMC3386495',
 'PMC4815465','PMC3268716','PMC3128457','PMC3028347','PMC4288434','PMC3890490','PMC2874129','PMC2821389','PMC5099201',
 'PMC3623865','PMC3668094','PMC4506459','PMC3906609','PMC5762805','PMC5699022','PMC3724593','PMC3242163','PMC2876235','PMC6323792','PMC3214151','PMC3686261','PMC5434971','PMC3075549','PMC3638634','PMC5176330','PMC3919641','PMC1386726',
 'PMC3321505','PMC3623038','PMC4145444','PMC6137538','PMC5765712','PMC3298524','PMC3375195','PMC3864174','PMC5181800','PMC5644752','PMC5099314','PMC5477802',
 'PMC3756454','PMC3112406','PMC4085478','PMC5101703','PMC4349661','PMC4469977','PMC5022161','PMC4030523','PMC4770816','PMC3277464','PMC5892036','PMC5655723','PMC3109952','PMC4893154',
 'PMC3076731','PMC5348580','PMC4582716','PMC2911677','PMC3944682','PMC3213035','PMC5526285','PMC3266479','PMC4515982','PMC4019289','PMC3016167','PMC5910468','PMC4215282',
 'PMC5210423','PMC4431679','PMC6714428','PMC4572447','PMC5034014','PMC5556660','PMC5932464','PMC3835257','PMC4842267','PMC5219781','PMC8678693','PMC5442675','PMC4902320','PMC125315','PMC5390457',
 'PMC5833494','PMC4797126','PMC2924475','PMC116603','PMC3002766','PMC3208022','PMC3441849','PMC4022270','PMC3648471','PMC4703201','PMC5330033','PMC5975052','PMC4066691','PMC9380327','PMC5064025','PMC4272792','PMC9171654','PMC5718075',
 'PMC6082539','PMC3851000','PMC3103669','PMC5381088','PMC2629563','PMC4984260','PMC6371462','PMC5704386','PMC4967304','PMC5721882','PMC3551748','PMC3542201','PMC5112179','PMC1277826','PMC4553934','PMC2917572',
 'PMC3424202','PMC3882723','PMC5134798','PMC6814122','PMC4392313','PMC3217848','PMC5222895','PMC3590447','PMC4082496','PMC3036630',
 'PMC9007089','PMC3856515','PMC3682293','PMC2291568','PMC5499060','PMC4461648','PMC4145439','PMC5083764','PMC2277384','PMC5094537','PMC5662095','PMC2893119','PMC5963181',
 'PMC3641012','PMC3848019','PMC3641608','PMC5702102','PMC7939944','PMC400670','PMC3679996','PMC5544609','PMC4647309','PMC7038903','PMC2744862','PMC5845227','PMC4745832','PMC5472276','PMC5994015',
 'PMC5529963','PMC5410698','PMC4772661','PMC3018567','PMC5721594','PMC5070122','PMC5483256']

test_data_df = all_data_with_sections_df
test_data_df["CURRENT_SENTENCE"] = test_data_df.apply(lambda x: combine_sentence_with_section(x.token, x.section), axis=1)

test_data_df = test_data_df[["CURRENT_SENTENCE","pmcids", "sid"]]
test_data_df = test_data_df.reset_index()
test_data_df.columns = ["index", "sentence", "pmcids", "sid"]

test_data_df.sid = test_data_df.sid.apply(add_list)

test_data_df["label"] = test_data_df["sid"]
test_data_df["label"] = test_data_df["label"].apply(convert_to_binary)

test_data_df = test_data_df[["sentence", "label", "pmcids", "sid"]]

model_path = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
tokenizer = BertTokenizer.from_pretrained(model_path)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

MAX_LEN = 512
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 3e-5

maxlen = 512
pad_token_id = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
ignore_label_id = CrossEntropyLoss().ignore_index

tokenized_df = convert(test_data_df)
print("load the data ... ")
tokenized_test = Dataset.from_pandas(tokenized_df)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
tokenized_test.set_format("torch")
test_loader = DataLoader(tokenized_test, batch_size=4, shuffle=False)

config = BertConfig.from_pretrained(
    model_path, max_position_embeddings=maxlen, num_labels=2)  # 18
model = BertForTokenClassification.from_pretrained(
    model_path, config=config).to(device)

model.load_state_dict(torch.load("pretrained_checkpoint/pubmedbert_best_model.pth"))
model.to(device)
model.eval()
final_val_true, final_val_pred = evaluate(model, test_loader)


predictions = final_val_pred.copy()
with open("predictions.txt", "w") as f:
    for s in predictions:
        f.write(str(s) +"\n")

test_data_df["true"] = final_val_true
test_data_df["prediction"] = predictions

test_data_df.to_csv("large_scale_data_with_predictions.csv")
