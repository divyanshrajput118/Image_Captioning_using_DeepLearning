from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    images_dir: Path
    captions_file: Path
    train_img_id_path: Path
    val_img_id_path: Path
    test_img_id_path: Path
    train_imagesid_captions_path: Path
    val_imagesid_captions_path: Path
    test_imagesid_captions_path: Path

    TRAIN_SPLIT: float
    TEST_SPLIT: float
    RANDOM_STATE: int


@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path           
    updated_base_model_path: Path   

    IMAGE_SIZE: list         
    WEIGHTS: str            
    INCLUDE_TOP: bool        
    POOLING: str

    CNN_DIM: int
    VOCAB_SIZE: int          
    MAX_LENGTH: int          
    EMBEDDING_DIM: int       
    LSTM_UNITS: int              
    DROPOUT: float          
    LEARNING_RATE: float

    CLIPNORM : float