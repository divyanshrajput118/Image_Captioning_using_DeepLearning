from imgCaption.constants import *
from imgCaption.utils.common import read_yaml,create_directories
from imgCaption.entity.config_entity import DataIngestionConfig
from imgCaption.entity.config_entity import PrepareBaseModelConfig

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        
        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),

            # ResNet50 params (replacing VGG16 ones)
            params_image_size=self.params.IMAGE_SIZE,       # kept same
            params_weights=self.params.WEIGHTS,             # kept same
            params_include_top=self.params.INCLUDE_TOP,     # kept same
            params_pooling=self.params.POOLING,             # replaces CLASSES

            # LSTM Decoder params (new)
            params_vocab_size=self.params.VOCAB_SIZE,
            params_max_length=self.params.MAX_LENGTH,
            params_embedding_dim=self.params.EMBEDDING_DIM,
            params_units=self.params.UNITS,
            params_dropout=self.params.DROPOUT,
            params_learning_rate=self.params.LEARNING_RATE,
        )

        return prepare_base_model_config