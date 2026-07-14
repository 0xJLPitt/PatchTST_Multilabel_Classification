import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    """
    LSTM Classifier for Time-Series
    """
    def __init__(self, input_dim, num_classes, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )
        
        # bidirectional LSTM, so hidden_size * 2
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, num_classes)
        )
        
    def forward(self, x):
        # x shape: (B, T, C)
        out, _ = self.lstm(x)
        
        # Take the output from the last time step
        # out shape: (B, T, hidden_size * 2)
        last_out = out[:, -1, :]
        
        return self.classifier(last_out)
