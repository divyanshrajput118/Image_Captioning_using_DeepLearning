import os
import tensorflow as tf
import tensorflow as tf
from tensorflow.keras.layers import Input,Dense,LSTM,Embedding,Dropout,Concatenate                            
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from imgCaption import logger
from pathlib import Path
from imgCaption.entity.config_entity import PrepareBaseModelConfig


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    def get_base_model(self):
        self.model = tf.keras.applications.resnet50.ResNet50(
                                input_shape=self.config.IMAGE_SIZE,  
                                weights=self.config.WEIGHTS,                 
                                include_top=self.config.INCLUDE_TOP,         
                                pooling=self.config.POOLING                  
                                                            )

        self.model.trainable = False

        self.save_model(path=self.config.base_model_path, model=self.model)

    @staticmethod
    def prepare_full_model(cnn_dim, vocab_size,max_caption_length,embedding_dim,
                            lstm_units,dropout_rate,learning_rate,clipnorm):
        
        input_img_features = Input(shape=(cnn_dim,), name = "Img_Features_Input")
        img_drop_layer = Dropout(dropout_rate, name="img_drop_layer")(input_img_features)
        img_output_tensor = Dense(embedding_dim, activation='relu', name="img_output_tensor")(img_drop_layer)
        
        input_caption = Input(shape=(max_caption_length,), name="Caption_Input")
        cap_embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True, name="cap_embedding_layer")(input_caption)
        cap_drop_layer = Dropout(dropout_rate, name="cap_drop_layer")(cap_embedding_layer)
        cap_output_tensor = LSTM(lstm_units, name="cap_output_tensor_LSTM")(cap_drop_layer)

        merged_tensor = Concatenate(axis=-1)([img_output_tensor,cap_output_tensor])

        merged_layer = Dense(embedding_dim, activation='relu', name="merged_layer")(merged_tensor)
        merged_drop_layer = Dropout(dropout_rate, name="merged_drop_layer")(merged_layer)
        output_layer = Dense(vocab_size, activation='softmax', name="Output_Layer_Final")(merged_drop_layer)

        full_model = Model(
            inputs=[input_img_features, input_caption],
            outputs=output_layer,
            name='Image_Captioning'
        )

        full_model.compile(
            optimizer=Adam(learning_rate=learning_rate, clipnorm=clipnorm),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        full_model.summary()
        return full_model

    def update_base_model(self):

        self.full_model = self.prepare_full_model(
                            cnn_dim=self.config.CNN_DIM,                               
                            vocab_size=self.config.VOCAB_SIZE,
                            max_caption_length=self.config.MAX_LENGTH,
                            embedding_dim=self.config.EMBEDDING_DIM,
                            lstm_units=self.config.LSTM_UNITS,
                            dropout_rate=self.config.DROPOUT,
                            learning_rate=self.config.LEARNING_RATE,
                            clipnorm=self.config.CLIPNORM
                            )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)
        logger.info(f"Full caption model saved at: {self.config.updated_base_model_path}")