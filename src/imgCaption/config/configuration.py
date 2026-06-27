from imgCaption.constants import *
from imgCaption.utils.common import read_yaml,create_directories
from imgCaption.entity.config_entity import DataIngestionConfig
from imgCaption.entity.config_entity import PrepareBaseModelConfig
from imgCaption.entity.config_entity import DataTransformationConfig

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
    

    def get_data_transformation_config(self) -> DataTransformationConfig:

        config = self.config.data_transformation
        params = self.params.data_transformation_params

        data_transformation_config = DataTransformationConfig(
                                                root_dir=config.root_dir,
                                                images_dir=config.images_dir,
                                                captions_file=config.captions_file,
                                                train_img_id_path=config.train_img_id_path,
                                                val_img_id_path=config.val_img_id_path,
                                                test_img_id_path=config.test_img_id_path,
                                                train_imagesid_captions_path=config.train_imagesid_captions_path,
                                                val_imagesid_captions_path=config.val_imagesid_captions_path,
                                                test_imagesid_captions_path=config.test_imagesid_captions_path,
                                                TRAIN_SPLIT=params.TRAIN_SPLIT,
                                                TEST_SPLIT=params.TEST_SPLIT,
                                                RANDOM_STATE=params.RANDOM_STATE
                                                              )
        
        return data_transformation_config
    

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        params = self.params.prepare_base_model_params

        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
                                                root_dir=config.root_dir,
                                                base_model_path=config.base_model_path,
                                                updated_base_model_path=config.updated_base_model_path,
                                                IMAGE_SIZE=params.IMAGE_SIZE,
                                                WEIGHTS=params.WEIGHTS,
                                                INCLUDE_TOP=params.INCLUDE_TOP,
                                                POOLING=params.POOLING,
                                                CNN_DIM=params.CNN_DIM,
                                                VOCAB_SIZE=params.VOCAB_SIZE,
                                                MAX_LENGTH=params.MAX_LENGTH,
                                                EMBEDDING_DIM=params.EMBEDDING_DIM,
                                                LSTM_UNITS=params.LSTM_UNITS,
                                                DROPOUT=params.DROPOUT,
                                                LEARNING_RATE=params.LEARNING_RATE,
                                                CLIPNORM=params.CLIPNORM
                                            )

        return prepare_base_model_config