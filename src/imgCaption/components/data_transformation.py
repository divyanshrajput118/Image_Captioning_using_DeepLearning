import os
import re
import json
import pickle
from pathlib import Path
from imgCaption import logger
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from imgCaption.entity.config_entity import DataTransformationConfig




class DataTransformation:
    def __init__(self,config : DataTransformationConfig):
        self.config = config


    def cleaned_captions(self):
        with open(self.config.captions_file) as f:
            raw_caption = f.readlines()

        captions = [cap.lower().split(',')[1] for cap in raw_caption[1:]]

        cleaned_captions = [
        re.sub(r'\s+', ' ',          
        re.sub(r'\d+', '',            
        re.sub(r'[^\w\s]', '', cap)   
        )).strip().lower()            
        for cap in captions
        ]

        images_ID = [ID.split(',')[0] for ID in raw_caption[1:]]

        full_cleaned_captions = []
        all_captions = []

        for img_id, caption in zip(images_ID, cleaned_captions):
            formatted_str = f"{img_id}\tstartseq {caption} endseq"
            cap_str = f"startseq {caption} endseq"
            full_cleaned_captions.append(formatted_str)
            all_captions.append(cap_str)
    
    
        logger.info("captions cleaned and start & end tokens added successfully")

        return full_cleaned_captions,all_captions


    def build_tokenizer(self, all_captions : list):
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(all_captions)

        with open(self.config.tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)

        logger.info(f"Tokenizer saved at: {self.config.tokenizer_path}")

        vocab_size = len(self.tokenizer.word_index) + 1    
        max_length = max(len(cap.split()) for cap in all_captions)

        logger.info(f"VOCAB_SIZE: {vocab_size}")
        logger.info(f"MAX_LENGTH: {max_length}")

        
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

        logger.info("train_img_id_file created")

        with open(self.config.val_img_id_path,"w") as f:
            for img_id in val_image_Id:
                f.write(f"{img_id}\n")

        logger.info("val_img_id_file created")

        with open(self.config.test_img_id_path,"w") as f:
            for img_id in test_image_Id:
                f.write(f"{img_id}\n")

        logger.info("test_img_id_file created")

    
    def create_mapping_dict(self, cleaned_captions: list, img_ids_path: Path, data_path: Path):
       
        with open(img_ids_path, "r") as f:
            target_ids = set(line.strip() for line in f.readlines())

        caption_map = {}

        for entry in cleaned_captions:

            img_id,caption = entry.split('\t')

            if img_id in target_ids:
                if img_id not in caption_map:
                    caption_map[img_id] = []

                caption_map[img_id].append(caption)

        logger.info(f"Successfully made {Path(img_ids_path).name} a dictionary")

        with open(data_path, 'w') as f:
            json.dump(caption_map, f, indent=2)

        logger.info(f"File : {Path(data_path).name} has saved successfully")
