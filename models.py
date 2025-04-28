
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, n_layers, num_timesteps):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, num_timesteps, hidden_dim))
        self.transformer = nn.Transformer(d_model=hidden_dim, nhead=n_heads, num_encoder_layers=n_layers, num_decoder_layers=n_layers)
        self.fc = nn.Linear(hidden_dim * num_timesteps, output_dim)  # Flatten the output of the transformer
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, src):
        src_emb = self.embedding(src) + self.positional_encoding[:, :src.size(1), :]
        src_emb = src_emb.permute(1, 0, 2)  # (seq_len, batch, feature)
        transformer_output = self.transformer.encoder(src_emb)
        transformer_output = transformer_output.permute(1, 0, 2).contiguous().view(src.size(0), -1)  # Flatten
        transformer_output = self.dropout(transformer_output)
        predictions = self.fc(transformer_output)
        return predictions

def reset_weights(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_timesteps, num_layers, dropout_prob=0.5):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)  # Dropout layer
        self.fc = nn.Linear(hidden_dim * num_timesteps, output_dim)  # Corrected for seq_len (num_timesteps)

    def forward(self, x):
        lstm_output, _ = self.lstm(x)  # (batch_size, seq_len, hidden_dim)
        lstm_output = self.dropout(lstm_output)  # Apply dropout to LSTM output
        flattened_output = lstm_output.contiguous().view(x.size(0), -1)  # Flatten all time steps
        predictions = self.fc(flattened_output)  # Fully connected layer for classification
        return predictions
class RNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,num_timesteps, num_layers, dropout_prob=0.5):
        super(RNNClassifier, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob, nonlinearity='tanh')
        self.dropout = nn.Dropout(dropout_prob)  # Dropout layer
        self.fc = nn.Linear(hidden_dim * num_timesteps, output_dim)  # Adjust to match flattened output dimensions

    def forward(self, x):
        rnn_output, _ = self.rnn(x)  # (batch_size, seq_len, hidden_dim)
        rnn_output = self.dropout(rnn_output)  # Apply dropout to RNN output
        flattened_output = rnn_output.contiguous().view(x.size(0), -1)  # Flatten all time steps
        predictions = self.fc(flattened_output)  # Fully connected layer for classification
        return predictions
class GRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_timesteps, num_layers, dropout_prob=0.5):
        super(GRUClassifier, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)  # Dropout layer
        self.fc = nn.Linear(hidden_dim * num_timesteps, output_dim)  # Adjust to match flattened output dimensions

    def forward(self, x):
        gru_output, _ = self.gru(x)  # (batch_size, seq_len, hidden_dim)
        gru_output = self.dropout(gru_output)  # Apply dropout to GRU output
        flattened_output = gru_output.contiguous().view(x.size(0), -1)  # Flatten all time steps
        predictions = self.fc(flattened_output)  # Fully connected layer for classification
        return predictions
