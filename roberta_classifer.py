# %% [markdown]
# To test if a general purpose sentiment analysis model can be used to predict the sentiment of political tweets.

# %%
# Define Model
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import RobertaModel
from transformers import RobertaTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import re

class RobertaClassifier(nn.Module):
    def __init__(self, freeze=False):
        super(RobertaClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H, D_out = 768, 50, 2

        # Instantiate BERT model
        self.bert = RobertaModel.from_pretrained("roberta-base")

        self.to('cuda')
        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Linear(H, D_out)
        )

        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False
    
    def forward(self, input_ids, attention_mask):

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        last_hidden_state_cls = outputs[0][:, 0, :]

        logits = self.classifier(last_hidden_state_cls)

        return logits
    
    
    def preprocess_for_roberta(data):
        # Initialize lists to store the input_ids and attention_masks
        input_ids = []
        attention_masks = []

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True)
        
        for text in tqdm(data):
            # Use tokenizer.encode_plus to tokenize and encode the text
            encoded = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                padding='max_length',
                truncation=True,
                pad_to_max_length=True,
                max_length=360, # Determined from the processed text
                return_attention_mask=True,
            )

            # Add the input_ids and attention_mask to the lists
            input_ids.append(encoded.get('input_ids'))
            attention_masks.append(encoded.get('attention_mask'))

        # Convert the lists to tensors
        input_ids = torch.tensor(input_ids)
        attention_masks = torch.tensor(attention_masks)


        return input_ids, attention_masks
    
    def create_dataloader(test_inputs, test_masks, batch_size):
        test_data = TensorDataset(test_inputs, test_masks)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
        return test_dataloader
