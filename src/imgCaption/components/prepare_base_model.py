import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Embedding,
    Dropout, concatenate
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from pathlib import Path
from imgCaption.entity.config_entity import PrepareBaseModelConfig


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.applications.resnet50.ResNet50(
            input_shape=self.config.params_image_size,  # (224, 224, 3)
            weights=self.config.params_weights,                 # "imagenet"
            include_top=self.config.params_include_top,         # False
            pooling=self.config.params_pooling                  # "avg" → 2048-dim
        )

        # Freeze ResNet50 — we don't retrain it
        self.model.trainable = False

        self.save_model(path=self.config.base_model_path, model=self.model)
        # logger.info(f"Base ResNet50 saved at: {self.config.base_model_path}")

    @staticmethod
    def _prepare_full_model(
        cnn_output_dim,         # 2048 from ResNet50 avg pool
        vocab_size,
        max_caption_length,
        embedding_dim,
        lstm_units,
        dropout_rate,
        learning_rate
    ):
        # 1. Image Feature Extractor branch
        input_image = Input(shape=(cnn_output_dim,), name='Features_Input')
        fe1 = Dropout(dropout_rate)(input_image)
        fe2 = Dense(embedding_dim, activation='relu')(fe1)

        # 2. Sequence Processor branch
        input_caption = Input(shape=(max_caption_length,), name='Sequence_Input')
        se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(input_caption)
        se2 = Dropout(dropout_rate)(se1)
        se3 = LSTM(lstm_units)(se2)

        # 3. Merge both branches
        decoder1 = concatenate([fe2, se3])

        # 4. Dense bottleneck
        decoder2 = Dense(embedding_dim, activation='relu')(decoder1)
        decoder2 = Dropout(dropout_rate)(decoder2)

        # 5. Output layer
        outputs = Dense(vocab_size, activation='softmax', name='Output_Layer')(decoder2)

        # Final model
        full_model = Model(
            inputs=[input_image, input_caption],
            outputs=outputs,
            name='Image_Captioning'
        )

        full_model.compile(
            optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        full_model.summary()
        return full_model

    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            cnn_output_dim=2048,                               # ResNet50 avg pool always 2048
            vocab_size=self.config.params_vocab_size,
            max_caption_length=self.config.params_max_length,
            embedding_dim=self.config.params_embedding_dim,
            lstm_units=self.config.params_units,
            dropout_rate=self.config.params_dropout,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)
        # logger.info(f"Full caption model saved at: {self.config.updated_base_model_path}")

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)