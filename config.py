"""
Configuration for next-location prediction system
"""

class Config:
    # Data paths
    data_dir = "data/geolife"
    train_file = "geolife_transformer_7_train.pk"
    val_file = "geolife_transformer_7_validation.pk"
    test_file = "geolife_transformer_7_test.pk"
    
    # Model hyperparameters
    num_locations = 1187  # max_id + 1
    num_users = 46  # max_user + 1
    
    # Embedding dimensions
    loc_embed_dim = 32
    user_embed_dim = 12
    weekday_embed_dim = 8
    time_embed_dim = 12
    
    # Model architecture
    d_model = 128
    nhead = 8
    num_layers = 2
    dim_feedforward = 256
    dropout = 0.2
    
    # Training parameters
    batch_size = 128
    learning_rate = 0.0005
    weight_decay = 1e-4
    max_epochs = 150
    patience = 20
    warmup_epochs = 5
    
    # Optimization
    gradient_clip = 1.0
    label_smoothing = 0.1
    
    # Device
    device = "cuda"
    
    # Checkpointing
    checkpoint_dir = "checkpoints"
    log_dir = "logs"
    
    # Reproducibility
    seed = 42
