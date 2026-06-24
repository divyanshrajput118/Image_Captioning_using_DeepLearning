from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path           # ResNet50 encoder saved here
    updated_base_model_path: Path   # Full encoder+decoder model saved here

    # ResNet50 Encoder params
    params_image_size: list         # e.g. [224, 224, 3]
    params_weights: str             # "imagenet"
    params_include_top: bool        # False (we remove classification head)
    params_pooling: str

    # LSTM Decoder params
    params_vocab_size: int          # total unique words in captions
    params_max_length: int          # max caption length (e.g. 35)
    params_embedding_dim: int       # word embedding size (e.g. 256)
    params_units: int               # LSTM hidden units (e.g. 512)
    params_dropout: float           # dropout rate (e.g. 0.5)

    # Training params (used later but defined here for MLflow tracking)
    params_learning_rate: float 