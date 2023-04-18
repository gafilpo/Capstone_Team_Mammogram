import pandas as pd

import gc

import tensorflow.keras as keras
from tensorflow.keras import backend as K
import tensorflow_addons as tfa

import keras_tuner as kt

import tensorflow as tf

class MyHyperModel(kt.HyperModel):
    # Regularization hyperparameter helper function

    def __init__(self, model, autoencoder=False):
        self.model = model
        self.autoencoder = autoencoder
    
    def build(self, hp):

        model = self.model(hp)
        
        #Create loss: if normal model, BinaryCrossentrophy. For autoencoder, MSE
        if self.autoencoder:
            loss = tf.keras.losses.MeanSquaredError() 
            metrics = [tf.keras.metrics.CosineSimilarity()]
        else:
            loss = tf.keras.losses.BinaryCrossentropy()
            metrics = [tf.keras.metrics.AUC(),
                        tf.keras.metrics.Precision(),
                        tf.keras.metrics.Recall(),
                        tfa.metrics.F1Score(num_classes=1, threshold=0.5, average='weighted')
                        ]
        
        #Compile the model
        model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                            loss=loss,
                            metrics=metrics
                            )
        return model

    def fit(self, hp, model, generator_object, autoencoder, resizing, class_weight, epochs, callbacks=None, **kwargs):
        # Convert the datasets to tf.data.Dataset.
        batch_size = hp.Int('batch_size', 16, 128, step=16, default=64)

        
        if autoencoder:
            train = generator_object.create_ds_autoencoder('train', resizing=resizing)
            val = generator_object.create_ds_autoencoder('validate', resizing=resizing)
        else:
            train = generator_object.create_ds('train', resizing=resizing)
            val = generator_object.create_ds('validate', resizing=resizing)

        train = generator_object.configure_for_performance(train, batch_size=batch_size, train=True)
        val = generator_object.configure_for_performance(val, batch_size=batch_size, train=False)

        fitting = model.fit(train,
                    validation_data=val,
                    class_weight=class_weight,
                    epochs=epochs,
                    callbacks=callbacks)
        
        # Clean up
        del train, val
        K.clear_session()
        tf.compat.v1.reset_default_graph()
        gc.collect()

        return fitting
        
