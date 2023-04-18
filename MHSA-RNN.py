import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class MHSA_RNN(nn.Module):
    def __init__(self, num_labels):
        super(MHSA_RNN, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        self.lstm = nn.LSTM(input_size=768, hidden_size=256, num_layers=1, batch_first=True)
        self.mhsa = nn.MultiheadAttention(embed_dim=256, num_heads=4)
        self.fc = nn.Linear(in_features=256, out_features=num_labels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        h_lstm, _ = self.lstm(outputs.last_hidden_state)
        h_lstm = h_lstm.permute(1, 0, 2)
        h_mhsa, _ = self.mhsa(h_lstm, h_lstm, h_lstm)
        h_mhsa = h_mhsa.permute(1, 0, 2)
        logits = self.fc(h_mhsa)
        return logits
    
# Example usage
model = MHSA_RNN(num_labels=3)
input_ids = torch.tensor([tokenizer.encode("The bank can guarantee deposits will eventually cover future tuition costs because it invests in adjustable-rate mortgage securities.")])
attention_mask = torch.ones(input_ids.shape)
logits = model(input_ids=input_ids, attention_mask=attention_mask)
