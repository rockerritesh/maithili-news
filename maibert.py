# Required imports
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download

# Define the BERTClass model
class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.roberta = AutoModel.from_pretrained('google/muril-base-cased')
        self.l2 = torch.nn.Dropout(0.5)
        self.fc = torch.nn.Linear(768, 10)

    def forward(self, ids, mask, token_type_ids):
        _, features = self.roberta(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        output_2 = self.l2(features)
        output = self.fc(output_2)
        return output

# Function to predict classes
def predict_classes(df):
    # Load the pre-trained model state dictionary
    path_model = hf_hub_download(repo_id="rockerritesh/maithili_classification_model", filename="maibert.bin")
    model_state_dict = torch.load(path_model, map_location=torch.device('cpu'))

    # Load the model architecture
    model = BERTClass()
    model.load_state_dict(model_state_dict)
    model.eval()

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained('rockerritesh/maiBERT_TF')

    # List to store predicted classes
    predicted_classes = []
    
    target_cols = ['Politics', 'Culture', 'Sports', 'Literature', 'Entertainment',
       'Health', 'EduTech', 'Opinion', 'Interview', 'Economy']
    for text in df['translated']:
        # Tokenize and preprocess the input text
        tokenized_input = tokenizer.encode_plus(
            text,
            truncation=True,
            add_special_tokens=True,
            max_length=256,
            padding='max_length',
            return_token_type_ids=True
        )

        # Move tensors to the available device (GPU or CPU)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        input_ids = torch.tensor(tokenized_input['input_ids']).unsqueeze(0).to(device)
        attention_mask = torch.tensor(tokenized_input['attention_mask']).unsqueeze(0).to(device)
        token_type_ids = torch.tensor(tokenized_input['token_type_ids']).unsqueeze(0).to(device)

        # Perform inference
        with torch.no_grad():
            output = model(input_ids, attention_mask, token_type_ids)
        
        # Get the predicted class
        predicted_class = target_cols[torch.argmax(output)]
        predicted_classes.append(predicted_class)

    return predicted_classes

# Example usage
# df = pd.read_csv('your_data.csv')  # Load your CSV file containing text
# predicted_classes = predict_classes(df)
# print(predicted_classes)
