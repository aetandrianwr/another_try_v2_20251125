"""
Advanced configuration with optimized hyperparameters
"""

class AdvancedConfig:
    # Data paths
    data_dir = "data/geolife"
    train_file = "geolife_transformer_7_train.pk"
    val_file = "geolife_transformer_7_validation.pk"
    test_file = "geolife_transformer_7_test.pk"
    
    # Model hyperparameters
    num_locations = 1187
    num_users = 46
    
    # Embedding dimensions (optimized for <500k params)
    loc_embed_dim = 32
    user_embed_dim = 12
    weekday_embed_dim = 8
    time_embed_dim = 12
    
    # Model architecture
    d_model = 112
    nhead = 8
    num_layers = 2
    dim_feedforward = 224
    dropout = 0.15
    drop_path_rate = 0.1
    
    # Training parameters
    batch_size = 32  # Smaller batch for better generalization
    learning_rate = 0.001
    weight_decay = 1e-4
    max_epochs = 200
    patience = 35
    warmup_epochs = 15
    
    # Optimization
    gradient_clip = 1.0
    use_focal_loss = True
    focal_alpha = 0.25
    focal_gamma = 2.0
    label_smoothing = 0.05
    
    # Data augmentation
    use_mixup = True
    mixup_alpha = 0.2
    
    # Learning rate schedule
    use_cosine_schedule = True
    min_lr = 1e-6
    
    # Device
    device = "cuda"
    
    # Checkpointing
    checkpoint_dir = "checkpoints"
    log_dir = "logs"
    
    # Reproducibility
    seed = 42
    
    # Mixed precision training
    use_amp = True
