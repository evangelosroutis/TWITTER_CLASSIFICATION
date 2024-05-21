
import torch
import torch.nn as nn

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, num_encoder_layers, dim_feedforward, num_classes,dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)  
        self.pos_encoder = PositionalEncoding(d_model) 
        self.encoder_layers = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward, dropout=dropout,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layers, num_encoder_layers)
        self.d_model = d_model
        self.output_layer = nn.Linear(d_model, num_classes)

    def forward(self, src, src_key_padding_mask):
        src = self.embedding(src) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float))#dim [batch,seq,d_model]
        src = self.pos_encoder(src) #dim [batch,seq,d_model]
        output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)#dim [batch,seq,d_model]
        #Average token representations
        valid_token_masks = ~src_key_padding_mask.unsqueeze(-1)  # add dimension for broadcasting
        sum_embeddings = torch.sum(output * valid_token_masks, dim=1)  # sum embeddings across seq_len
        num_valid_tokens = torch.sum(valid_token_masks, dim=1)  # Count non-padding tokens for each sequence in the batch
        averaged_embeddings = sum_embeddings / num_valid_tokens  # Divide sum by number of valid, ie non-padding, tokens
        output = self.output_layer(averaged_embeddings)  # pass the averaged embeddings to the output layer
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        #make sure max_len is bigger than the max of the tweet lengths
        super().__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        self.encoding[:, 0::2] = torch.sin(position * div_term)

        if d_model%2==0:
            self.encoding[:, 1::2] = torch.cos(position * div_term)
        else:
            self.encoding[:, 1::2] = torch.cos(position * div_term)[:,:-1]

        
        self.encoding = self.encoding.unsqueeze(0)  # Add batch dimension
        
    def forward(self, x):
        return x + self.encoding[:, :x.shape[1]]
