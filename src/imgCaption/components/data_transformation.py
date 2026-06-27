import os
import re
import json
from pathlib import Path
from imgCaption import logger
from sklearn.model_selection import train_test_split
from imgCaption.entity.config_entity import DataTransformationConfig




class DataTransformation:
    def __init__(self,config : DataTransformationConfig):
        self.config = config

    def split_images_ID(self):
        images_Id = os.listdir(self.config.images_dir)
        train_image_Id, val_test_Id = train_test_split(images_Id, test_size=self.config.TRAIN_SPLIT, random_state=self.config.RANDOM_STATE)
        val_image_Id, test_image_Id = train_test_split(val_test_Id, test_size=self.config.TEST_SPLIT, random_state=self.config.RANDOM_STATE)

        logger.info(f"Total images     : {len(images_Id)}")
        logger.info(f"Train images     : {len(train_image_Id)}")
        logger.info(f"Validation images: {len(val_image_Id)}")
        logger.info(f"Test images      : {len(test_image_Id)}")

        os.makedirs(self.config.root_dir, exist_ok=True)

        with open(self.config.train_img_id_path,"w") as f:
            for img_id in train_image_Id:
                f.write(f"{img_id}\n")

        logger.info("train_img_id_path created")

        with open(self.config.val_img_id_path,"w") as f:
            for img_id in val_image_Id:
                f.write(f"{img_id}\n")

        logger.info("val_img_id_path created")

        with open(self.config.test_img_id_path,"w") as f:
            for img_id in test_image_Id:
                f.write(f"{img_id}\n")

        logger.info("test_img_id_path created")

    def cleaned_captions(self):
        with open(self.config.captions_file) as f:
            full_captions = f.readlines()

        captions = [cap.lower().split(',')[1] for cap in full_captions[1:]]

        cleaned_captions = [
        re.sub(r'\s+', ' ',          
        re.sub(r'\d+', '',            
        re.sub(r'[^\w\s]', '', cap)   
        )).strip().lower()            
        for cap in captions
        ]

        images_ID = [ID.split(',')[0] for ID in full_captions[1:]]

        full_cleaned_captions = []

        for img_id, caption in zip(images_ID, cleaned_captions):
            formatted_str = f"{img_id}\tstartseq {caption} endseq"
            full_cleaned_captions.append(formatted_str)
    
    
        logger.info("captions cleaned and tokenized successfully")

        return full_cleaned_captions
    
    def create_mapping_dict(self, cleaned_captions: list, img_ids_path: Path, data_path: Path):
       
        with open(img_ids_path, "r") as f:
            target_ids = set(line.strip() for line in f.readlines())

        map = {}

        for entry in cleaned_captions:

            img_id,caption = entry.split('\t')

            if img_id in target_ids:
                if img_id not in map:
                    map[img_id] = []

                map[img_id].append(caption)

        logger.info(f"Successfully made {Path(img_ids_path).name} a dictionary")

        with open(data_path, 'w') as f:
            json.dump(map, f, indent=2)

        logger.info(f"File : {Path(data_path).name} has saved successfully")
