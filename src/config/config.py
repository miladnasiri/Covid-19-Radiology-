from dataclasses import dataclass
from typing import Tuple, List, Optional
from pathlib import Path

@dataclass
class DataConfig:
    data_dir: Path
    image_size: Tuple[int, int] = (224, 224)
    batch_size: int = 32
    validation_split: float = 0.1
    test_split: float = 0.1
    num_workers: int = 4
    pin_memory: bool = True
    
    # Advanced augmentation configs
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    mosaic_prob: float = 0.5
    rand_augment_n: int = 2
    rand_augment_m: int = 9
    use_auto_augment: bool = True

@dataclass
class ModelConfig:
    model_name: str = "efficientnet_v2_l"  # Using EfficientNetV2 Large
    pretrained: bool = True
    num_classes: int = 4
    dropout_rate: float = 0.3
    drop_path_rate: float = 0.2
    label_smoothing: float = 0.1
    channels: int = 3
    
    # Advanced architecture configs
    use_attention: bool = True
    attention_type: str = "cbam"  # Options: cbam, se, eca
    use_gem_pooling: bool = True
    gem_p: float = 3.0
    use_arcface: bool = True
    arcface_margin: float = 0.5
    arcface_scale: float = 30.0

@dataclass
class TrainingConfig:
    epochs: int = 100
    initial_lr: float = 1e-3
    weight_decay: float = 0.01
    optimizer: str = "adamw"  # Options: adamw, lamb, adabelief
    scheduler: str = "cosine"  # Options: cosine, onecycle, warmup_cosine
    warmup_epochs: int = 5
    grad_clip: float = 1.0
    
    # Advanced training techniques
    use_amp: bool = True  # Automatic Mixed Precision
    use_ema: bool = True  # Exponential Moving Average
    ema_decay: float = 0.9999
    use_swa: bool = True  # Stochastic Weight Averaging
    use_sam: bool = True  # Sharpness-Aware Minimization
    
    # Loss configuration
    losses: List[str] = ("cross_entropy", "focal", "dice")
    loss_weights: List[float] = (1.0, 0.5, 0.5)

@dataclass
class Config:
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    seed: int = 42
    output_dir: Path = Path("outputs")
    experiment_name: str = "covid_classification"
    
    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
