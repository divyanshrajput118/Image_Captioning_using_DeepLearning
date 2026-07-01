from imgCaption.config.configuration import ConfigurationManager
from imgCaption.components.data_transformation import DataTransformation
from imgCaption import logger


STAGE_NAME = "Data Transformation Stage"


class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
            config = ConfigurationManager()
            data_transformation_config = config.get_data_transformation_config()
            data_transformation = DataTransformation(config=data_transformation_config)
            cleaned_captions,all_captions = data_transformation.cleaned_captions()
            data_transformation.build_tokenizer(all_captions)
            data_transformation.split_images_ID()

            splits = [
                (data_transformation_config.train_img_id_path, data_transformation_config.train_imagesid_captions_path),
                (data_transformation_config.val_img_id_path,   data_transformation_config.val_imagesid_captions_path)
                    ]

            for img_ids_path, data_path in splits:
                data_transformation.create_mapping_dict(cleaned_captions=cleaned_captions,
                                                        img_ids_path=img_ids_path,
                                                        data_path=data_path)
    

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataTransformationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e