import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import RobertaModel
from transformers import RobertaTokenizer
from transformers import get_linear_schedule_with_warmup
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

    def get_optimizer(self, lr=1e-3):

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        return optimizer
    
    def get_scheduler(self, optimizer, data_loader, epochs):
        
        total_steps = len(data_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        return scheduler
    
    def forward(self, input_ids, attention_mask):

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        last_hidden_state_cls = outputs[0][:, 0, :]

        logits = self.classifier(last_hidden_state_cls)

        return logits
    
    
    def preprocess_for_roberta(self, data):
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
    
    def create_test_dataloader(self, inputs, masks, batch_size):
        test_data = TensorDataset(inputs, masks)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
        return test_dataloader
    
    def create_train_dataloader(self, inputs, masks, labels, batch_size):
        train_data = TensorDataset(inputs, masks, labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
        return train_dataloader
    
    
