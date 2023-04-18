import pandas as pd

import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import tensorflow_addons as tfa

import keras_tuner as kt
from keras_tuner.tuners import BayesianOptimization

from sklearn.utils.class_weight import compute_class_weight

from create_datasets import create_datasets 
from hypermodel import MyHyperModel

import gc

import tensorflow as tf

# Train and evaluate the model
class model_train_eval():
    
    'Generates the model, kfold train, validation performance, and test performance'
    def __init__(self, dfs, model_name, basepath, splits=5, img_size=(1000, 1000), n_channels=1, 
                    n_classes=1, batch_size=10, random_state=42, n_epochs=5, double_input=False,
                    target_var='cancer', img_var='path', exclude_cols=[]):
        
        'Initialization'
        self.dfs = dfs
        self.model_name = model_name
        self.basepath = basepath
        self.splits = splits   
        self.img_size = img_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.random_state = random_state
        self.n_epochs = n_epochs 
        self.double_input = double_input
        self.target_var = target_var
        self.img_var = img_var
        self.exclude_cols = exclude_cols

    #Create callbacks for the model
    def get_callbacks(self, model_name, patience_lr, autoencoder=False):

        #Save best model for later
        mcp_save = ModelCheckpoint(model_name, save_best_only=True, monitor='val_loss', mode='min')
        #Reduce learning rate dynamically when learning rate plateaus
        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=patience_lr, verbose=1, min_delta=1e-4, mode='min')
        #Early stopping to reduce overfitting - for autoencoder, which has low losses, the min_delta is lower
        if autoencoder:
            es = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.01, patience=10)
        else:
            es = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.15, patience=10)

        # Tensorboard
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir='TENSORBOARD/'+model_name+'/logs/', histogram_freq=0, write_graph=True, write_images=False)

        return [mcp_save, reduce_lr_loss, es, tensorboard]

    #Train the model and test it on the validation set
    def train_validate(self, model, resizing=False, disk_cache=False, autoencoder=False):

        history = {}

        AUTOTUNE = tf.data.AUTOTUNE

        if self.splits == 1:
            iterations = [0]
        else:
            iterations = range(self.splits)
            
        for fold in iterations:
            print('>>>> TRAINING FOLD', fold)
            self.model = model 

            keras.backend.clear_session()

            #Initialize the generators
            create = create_datasets(self.dfs, fold=fold, target_var=self.target_var, 
                            img_var=self.img_var, exclude_cols=self.exclude_cols, 
                            double_input=self.double_input)

            if autoencoder:
                train = create.create_ds_autoencoder('train', resizing=resizing)
                val = create.create_ds_autoencoder('validate', resizing=resizing)
            else:
                train = create.create_ds('train', resizing=resizing)
                val = create.create_ds('validate', resizing=resizing)

            train = create.configure_for_performance(train, batch_size=self.batch_size, train=True, disk_cache=disk_cache)
            val = create.configure_for_performance(val, batch_size=self.batch_size, train=False, disk_cache=False)

            #Create the weights if normal model
            if autoencoder:
                weights = {}
            else:
                w = compute_class_weight(class_weight='balanced', classes=[0, 1], y=self.dfs[fold]['train'][self.target_var])
                weights = {0: w[0], 1: w[1]}

            #Create the callbacks
            name = self.model_name + '_fold' + str(fold) + '.h5'
            callbacks = self.get_callbacks(model_name=name, patience_lr=5, autoencoder=autoencoder)

            #Create loss: if normal model, BinaryCrossentrophy. For autoencoder, MSE
            if autoencoder:
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
            self.model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                                loss=loss,
                                metrics=metrics
                                )

            # Train the model
            history[fold] = self.model.fit(train,
                            validation_data=val,
                            class_weight=weights,
                            epochs=self.n_epochs,
                            callbacks=callbacks)

            # Clean up
            del self.model, train, val, create
            K.clear_session()
            tf.compat.v1.reset_default_graph()
            gc.collect()

        self.H = history

        return history

        #Train the model and test it on the validation set
    def train_validate_hyper(self, model, resizing=False, disk_cache=False, autoencoder=False):

        tuners = {}
        AUTOTUNE = tf.data.AUTOTUNE
        keras.backend.clear_session()
        gc.collect()

        for fold in range(self.splits):
            print('>>>> TRAINING FOLD', fold)

            self.model = model

            keras.backend.clear_session()

            #Initialize the generators
            create = create_datasets(self.dfs, fold=fold, target_var=self.target_var, 
                            img_var=self.img_var, exclude_cols=self.exclude_cols, 
                            double_input=self.double_input)

            #Create the weights if normal model
            if autoencoder:
                weights = {}
            else:
                w = compute_class_weight(class_weight='balanced', classes=[0, 1], y=self.dfs[fold]['train'][self.target_var])
                weights = {0: w[0], 1: w[1]}
                
            #Create the callbacks
            name = self.model_name + '_fold' + str(fold) + '.h5'
            callbacks = self.get_callbacks(model_name=name, patience_lr=5, autoencoder=autoencoder)

            #Tuner
            tuner = BayesianOptimization(MyHyperModel(self.model, autoencoder),
                                        objective='val_loss',
                                        max_trials=10,
                                        executions_per_trial=1,
                                        overwrite=True,
                                        directory=self.model_name + '_fold' + str(fold) + '_hyperparameters'
                                        )
            
            # Train the model
            tuner.search(generator_object=create,
                         autoencoder=autoencoder,
                         resizing=resizing,
                         class_weight=weights,
                         epochs=self.n_epochs,
                         callbacks=callbacks
                         )
        
            tuners[fold] = tuner

            # Clean up
            del tuner, create, self.model
            K.clear_session()
            tf.compat.v1.reset_default_graph()
            gc.collect()

        return tuners