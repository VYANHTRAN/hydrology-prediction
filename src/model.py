import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.2):
        """
        Standard LSTM for Single-Step Prediction.
        
        Args:
            input_dim: Number of dynamic features + static features (if used).
            hidden_dim: Number of LSTM units.
        """
        super(LSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=0
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Dense layer to predict a single scalar (Flow at t+2)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Features)
        
        # 1. Run LSTM
        # out shape: (Batch, Seq_Len, Hidden)
        out, (h_n, c_n) = self.lstm(x)
        
        # 2. Take the Last Hidden State
        last_hidden = out[:, -1, :] # (Batch, Hidden)
        
        # 3. Dropout & Predict
        out = self.dropout(last_hidden)
        prediction = self.head(out) # (Batch, 1)
        
        return prediction


class CrossAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(CrossAttention, self).__init__()
        # Linear layer to calculate attention energy
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: Decoder state at current step (Batch, Hidden)
        # encoder_outputs: All encoder states (Batch, Seq_Len, Hidden)
        
        seq_len = encoder_outputs.size(1)
        
        # Repeat decoder hidden state seq_len times
        # (Batch, Seq_Len, Hidden)
        hidden_expanded = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Calculate Energy: score = v * tanh(W * [h_dec; h_enc])
        combined = torch.cat((hidden_expanded, encoder_outputs), dim=2)
        energy = torch.tanh(self.attn(combined))
        attention = self.v(energy).squeeze(2) # (Batch, Seq_Len)
        
        # Softmax to get weights
        weights = F.softmax(attention, dim=1)
        
        # Compute Context Vector (Weighted sum of encoder outputs)
        # (Batch, 1, Seq_Len) * (Batch, Seq_Len, Hidden) -> (Batch, 1, Hidden)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs)
        
        return context.squeeze(1), weights

class LSTM_Seq2Seq(nn.Module):
    def __init__(self, input_dim, hidden_dim, future_forcing_dim, static_dim, output_steps=5):
        """
        Encoder-Decoder with Cross-Attention.
        
        Args:
            input_dim: Input features for Encoder (Past Dyn + Static).
            hidden_dim: LSTM units.
            future_forcing_dim: Number of known future features (e.g., Precip, Temp).
            static_dim: Number of static attributes (to concat in Decoder).
            output_steps: Prediction horizon (5).
        """
        super(LSTM_Seq2Seq, self).__init__()
        self.output_steps = output_steps
        self.static_dim = static_dim
        
        # --- ENCODER ---
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        
        # --- DECODER COMPONENTS ---
        # Decoder Input: Previous_Flow (1) + Future_Forcing + Static
        decoder_input_dim = 1 + future_forcing_dim + static_dim
        
        self.decoder_cell = nn.LSTMCell(decoder_input_dim, hidden_dim)
        self.attention = CrossAttention(hidden_dim)
        
        # Final projection: [Decoder_Hidden + Context] -> Output
        self.fc_out = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x_past, x_future_forcing, static_features, target_seq=None, teacher_forcing_ratio=0.5):
        """
        Args:
            x_past: (Batch, Past_Seq, Input_Dim)
            x_future_forcing: (Batch, 5, Future_Forcing_Dim) - Known weather for t+1..t+5
            static_features: (Batch, Static_Dim) or None
            target_seq: (Batch, 5) - Ground truth flow (for Teacher Forcing)
            teacher_forcing_ratio: Probability of using ground truth input
            
        Returns:
            outputs: (Batch, 5) - Predicted flow sequence
        """
        batch_size = x_past.size(0)
        
        # 1. ENCODE
        # encoder_outputs: (Batch, Seq, Hidden) - Used for Attention keys/values
        # hidden, cell: (1, Batch, Hidden) - Used to init Decoder
        encoder_outputs, (hidden, cell) = self.encoder(x_past)
        
        # Remove layer dimension for LSTMCell
        hidden = hidden.squeeze(0)
        cell = cell.squeeze(0)
        
        # Initialize outputs container
        outputs = torch.zeros(batch_size, self.output_steps).to(x_past.device)
        
        # First decoder input is the Last Observed Flow (from x_past)
        # Assuming Flow is the LAST column in x_past
        decoder_input_flow = x_past[:, -1, -1].unsqueeze(1) # (Batch, 1)
        
        # 2. DECODE LOOP (t=0 to 4)
        for t in range(self.output_steps):
            
            # --- Prepare Decoder Input ---
            # Structure: [Flow(t-1), Future_Forcing(t), Static]
            
            # Get forcing for this step
            current_forcing = x_future_forcing[:, t, :] # (Batch, Forcing_Dim)
            
            inputs_list = [decoder_input_flow, current_forcing]
            
            if self.static_dim > 0 and static_features is not None:
                inputs_list.append(static_features)
            
            # Concat everything
            dec_input = torch.cat(inputs_list, dim=1)
            
            # --- Run Decoder Cell ---
            hidden, cell = self.decoder_cell(dec_input, (hidden, cell))
            
            # --- Cross Attention ---
            # Look at all encoder states to find relevant past info
            context, _ = self.attention(hidden, encoder_outputs)
            
            # --- Predict ---
            # Combine Decoder Memory (hidden) + Encoder Context (context)
            combined = torch.cat((hidden, context), dim=1)
            prediction = self.fc_out(combined) # (Batch, 1)
            
            # Store prediction
            outputs[:, t] = prediction.squeeze(1)
            
            # --- Teacher Forcing Logic ---
            # Decide input for NEXT step: Truth or Prediction?
            if target_seq is not None and torch.rand(1).item() < teacher_forcing_ratio:
                # Use Ground Truth (if training)
                decoder_input_flow = target_seq[:, t].unsqueeze(1)
            else:
                # Use Model's own prediction
                decoder_input_flow = prediction
        
        return outputs