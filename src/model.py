import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.2):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=0
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        last_hidden = out[:, -1, :] 
        out = self.dropout(last_hidden)
        prediction = self.head(out)
        return prediction

class CrossAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(CrossAttention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.size(1)
        hidden_expanded = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        combined = torch.cat((hidden_expanded, encoder_outputs), dim=2)
        energy = torch.tanh(self.attn(combined))
        attention = self.v(energy).squeeze(2) 
        weights = F.softmax(attention, dim=1)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs)
        return context.squeeze(1), weights

class LSTM_Seq2Seq(nn.Module):
    def __init__(self, input_dim, hidden_dim, future_forcing_dim, static_dim, output_steps=5):
        super(LSTM_Seq2Seq, self).__init__()
        self.output_steps = output_steps
        self.static_dim = static_dim
        
        # --- ENCODER ---
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        
        # --- DECODER COMPONENTS ---
        decoder_input_dim = 1 + future_forcing_dim + static_dim
        
        self.decoder_cell = nn.LSTMCell(decoder_input_dim, hidden_dim)
        self.attention = CrossAttention(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x_past, x_future_forcing, static_features, target_seq=None, teacher_forcing_ratio=0.5):
        """
        x_past structure: [Dynamic | Flow | Static]
        """
        batch_size = x_past.size(0)
        
        # 1. ENCODE
        encoder_outputs, (hidden, cell) = self.encoder(x_past)
        hidden = hidden.squeeze(0)
        cell = cell.squeeze(0)
        
        outputs = torch.zeros(batch_size, self.output_steps).to(x_past.device)
        
        # 2. IDENTIFY LAST FLOW INDEX
        # x_past includes static features at the end. 
        # The Flow variable is the last column of the Dynamic section.
        # Index = Total_Cols - Static_Cols - 1
        flow_idx = x_past.size(2) - self.static_dim - 1
        decoder_input_flow = x_past[:, -1, flow_idx].unsqueeze(1) # (Batch, 1)
        
        # 3. DECODE LOOP
        for t in range(self.output_steps):
            
            # Get forcing for this step
            current_forcing = x_future_forcing[:, t, :] 
            
            inputs_list = [decoder_input_flow, current_forcing]
            
            # Only append static features if they exist and dimension > 0
            if self.static_dim > 0 and static_features is not None:
                inputs_list.append(static_features)
            
            dec_input = torch.cat(inputs_list, dim=1)
            
            hidden, cell = self.decoder_cell(dec_input, (hidden, cell))
            context, _ = self.attention(hidden, encoder_outputs)
            
            combined = torch.cat((hidden, context), dim=1)
            prediction = self.fc_out(combined) 
            
            outputs[:, t] = prediction.squeeze(1)
            
            if target_seq is not None and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input_flow = target_seq[:, t].unsqueeze(1)
            else:
                decoder_input_flow = prediction
        
        return outputs