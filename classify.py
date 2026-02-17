
# import transformers for bertclassifier
import torch
from transformers import  AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download
# import pandas as pd


# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

tokenizer = AutoTokenizer.from_pretrained('rockerritesh/maiBERT_TF')

# Creating the customized model, by adding a drop out and a dense layer on top of distil bert to get the final output for the model.

class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.roberta = AutoModel.from_pretrained('rockerritesh/maiBERT_TF', from_tf = True)
        self.l2 = torch.nn.Dropout(0.5)
        self.fc = torch.nn.Linear(768,10)

    def forward(self, ids, mask, token_type_ids):
        _, features = self.roberta(ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict=False)
        output_2 = self.l2(features)
        output = self.fc(output_2)
        return output


# Load the saved state dictionary
model_state_dict_path  = hf_hub_download(repo_id="rockerritesh/maithili_classifier", filename="maibert.bin", repo_type="model")

model_state_dict = torch.load(model_state_dict_path,map_location=torch.device(device))

# Create an instance of your model
model = BERTClass()

# Load the state dictionary into the model
model.load_state_dict(model_state_dict)

model.to(device)

target_cols = ['Politics', 'Culture', 'Sports', 'Literature', 'Entertainment',
       'Health', 'EduTech', 'Opinion', 'Interview', 'Economy']


def predict_using_maibert(df):
    df = df.copy()
    all_labels = []

    # Split the dataframe into batches
    batch_size = 32
    batches = [df[i:i+batch_size] for i in range(0, len(df), batch_size)]

    # Iterate through all batches and perform inference
    for batch_df in batches:
        texts = batch_df['translated'].tolist()

        # Tokenize and preprocess the batch of inputs
        tokenized_batch = tokenizer.batch_encode_plus(
            texts,
            truncation=True,
            add_special_tokens=True,
            max_length=256,
            padding='max_length',
            return_token_type_ids=True,
            return_tensors='pt'
        )

        input_ids = tokenized_batch['input_ids'].to(device)
        attention_mask = tokenized_batch['attention_mask'].to(device)
        token_type_ids = tokenized_batch['token_type_ids'].to(device)

        model.eval()
        with torch.no_grad():
            outputs = model(input_ids, attention_mask, token_type_ids)

        predicted_classes = torch.argmax(outputs, dim=1).tolist()
        predicted_classes = [target_cols[idx] for idx in predicted_classes]
        all_labels.extend(predicted_classes)

    df['label'] = all_labels
    return df
