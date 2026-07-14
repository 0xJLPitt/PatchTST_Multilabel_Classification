import torch
import torch.nn as nn

class iTransformer_Classification(nn.Module):
    def __init__(
        self, 
        seq_len: int, 
        num_variates: int, 
        d_model: int = 128, 
        nhead: int = 8, 
        num_encoder_layers: int = 3, 
        dim_feedforward: int = 512, 
        num_classes: int = 1, 
        dropout: float = 0.1, 
        pooling: str = 'mean'
    ):
        """
        iTransformer model for binary/multi-label classification.
        
        Args:
            seq_len: Length of the time series sequence.
            num_variates: Number of physical variates/features.
            d_model: Hidden dimension of the Transformer.
            nhead: Number of attention heads.
            num_encoder_layers: Number of Transformer encoder layers.
            dim_feedforward: Hidden dimension of the feedforward network in Transformer.
            num_classes: Number of output classes (1 for single binary classification).
            dropout: Dropout probability.
            pooling: Pooling method to aggregate variate features ('mean' or 'flatten').
        """
        super().__init__()
        
        self.seq_len = seq_len
        self.num_variates = num_variates
        self.d_model = d_model
        self.pooling = pooling
        
        # 1. Variate Embedding & Normalization
        # Layer Normalization for each variate's time series independently
        self.layer_norm = nn.LayerNorm(seq_len)
        
        # Linear Projection: Project entire history sequence of each variate to d_model
        self.projector = nn.Linear(seq_len, d_model)
        
        # (Note: Positional Encoding is NOT added because the tokens are "variates" and 
        # lack absolute spatial/sequential order. The temporal order is inherently captured 
        # by the Linear projector eating the entire sequence at once.)
        
        # 2. Transformer Encoder Backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True # batch_first=True means inputs are [batch_size, num_variates, d_model]
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # 3. Classification Head
        if pooling == 'flatten':
            in_dim = num_variates * d_model
        elif pooling == 'mean':
            in_dim = d_model
        else:
            raise ValueError("pooling must be 'flatten' or 'mean'")
            
        self.output_layer = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim // 2, num_classes)
        )
            
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, num_variates]
        Returns:
            logits: Tensor of shape [batch_size, num_classes] (RAW logits, NO Sigmoid)
        """
        # 1. Data Inversion: [batch_size, seq_len, num_variates] -> [batch_size, num_variates, seq_len]
        x_inverted = x.transpose(1, 2)
        
        # Normalize each variate's sequence independently
        # LayerNorm is applied over the last dimension (seq_len)
        # Shape remains: [batch_size, num_variates, seq_len]
        x_norm = self.layer_norm(x_inverted)
        
        # Project sequence to d_model
        # Shape becomes: [batch_size, num_variates, d_model]
        x_proj = self.projector(x_norm)
        
        # 2. Transformer Encoder (Cross-Variate Attention)
        # Self-Attention is calculated across the `num_variates` dimension
        # Shape remains: [batch_size, num_variates, d_model]
        x_enc = self.transformer_encoder(x_proj)
        
        # 3. Feature Aggregation (Pooling)
        if self.pooling == 'flatten':
            # Flatten to [batch_size, num_variates * d_model]
            x_pool = x_enc.reshape(x_enc.size(0), -1)
        elif self.pooling == 'mean':
            # Mean pooling across variates: [batch_size, d_model]
            x_pool = x_enc.mean(dim=1)
            
        # 4. Output Layer
        # logits shape: [batch_size, num_classes]
        # CRITICAL: NO Sigmoid activation here!
        logits = self.output_layer(x_pool)
        
        return logits

if __name__ == "__main__":
    # ==========================================
    # Test script with random tensors
    # ==========================================
    torch.manual_seed(42)
    
    # Mock dimensions
    batch_size = 32
    seq_len = 100
    num_variates = 21 # e.g. 7 joints * 3D coordinates
    num_classes = 1   # Binary classification
    
    # 1. Create random input tensor [batch_size, seq_len, num_variates]
    x = torch.randn(batch_size, seq_len, num_variates)
    
    # 2. Create random target tensor (binary classification 0 or 1)
    y_true = torch.empty(batch_size, num_classes).random_(2)
    
    # 3. Initialize the iTransformer model
    model = iTransformer_Classification(
        seq_len=seq_len, 
        num_variates=num_variates, 
        d_model=128, 
        num_classes=num_classes,
        pooling='mean'
    )
    
    # 4. Forward pass
    logits = model(x)
    
    print("--- Shapes Validation ---")
    print(f"Input shape: {x.shape} -> [batch_size, seq_len, num_variates]")
    print(f"Output (Logits) shape: {logits.shape} -> [batch_size, num_classes]")
    
    # 5. Calculate Loss with BCEWithLogitsLoss
    # Uses raw logits directly!
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(logits, y_true)
    
    print("\n--- Loss Validation ---")
    print(f"BCEWithLogitsLoss value: {loss.item():.4f}")
    
    # 6. Demonstration of the Forward Flow Explicitly
    print("\n--- Step-by-Step Architecture Demonstration ---")
    print(f"0. Original Input:    {x.shape}")
    x_inverted = x.transpose(1, 2)
    print(f"1. Inverted Data:     {x_inverted.shape} (Tokens are now Variates)")
    x_norm = model.layer_norm(x_inverted)
    print(f"2. After LayerNorm:   {x_norm.shape}")
    x_proj = model.projector(x_norm)
    print(f"3. After Projection:  {x_proj.shape} (Ready for Transformer Encoder)")
    x_enc = model.transformer_encoder(x_proj)
    print(f"4. After Transformer: {x_enc.shape} (Cross-Variate Attention Applied)")
    x_pool = x_enc.mean(dim=1)
    print(f"5. After Mean Pooling:{x_pool.shape} (Aggregated Variates)")
    final_logits = model.output_layer(x_pool)
    print(f"6. Final Logits:      {final_logits.shape} (No Sigmoid Applied)")
