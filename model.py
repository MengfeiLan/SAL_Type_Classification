from torch import nn
from transformers import BertModel
import transformers

class BertClassifier(nn.Module):

    def __init__(self, model_path, number_unique_labels, dropout=0.1):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained(model_path)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, number_unique_labels)
        self.relu = nn.ReLU(1)

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        # final_layer = self.relu(linear_output)

        return linear_output