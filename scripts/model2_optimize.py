

# import libraries
import os
import pandas as pd
import numpy as np
import gc
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from scipy import stats
import random
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import *
import warnings
import optuna
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
import pickle
from sklearn.model_selection import train_test_split
import joblib


#get training data
n_features = 300
features = [f'f_{i}' for i in range(n_features)]
train = pd.read_pickle('./train.pkl')

#prepare data
investment_id = train.pop("investment_id")
_ = train.pop("time_id")
y = train.pop("target")

# Use Keras interger lookup on the ID to prepare ID categorical parameters for either embedding layer or dense layer
investment_ids = list(investment_id.unique())
investment_id_size = len(investment_ids) + 1
investment_id_lookup_layer = layers.IntegerLookup(max_tokens=investment_id_size)
investment_id_lookup_layer.adapt(pd.DataFrame({"investment_ids":investment_ids}))

# Making Tesorflow dataset
def preprocess(X, y):
    return X, y
def make_dataset(feature, investment_id, y, batch_size=1024, mode="train"):
    ds = tf.data.Dataset.from_tensor_slices(((investment_id, feature), y))
    ds = ds.map(preprocess)
    if mode == "train":
        ds = ds.shuffle(4096)
    ds = ds.batch(batch_size).cache().prefetch(tf.data.experimental.AUTOTUNE)
    return ds

def get_model(trial):
    # function returns DNN model and passes optuna trials

    #optuna parameters
    dropout_1 = trial.suggest_float("dropout_1", 0.1, 0.9, log=True)
    dropout_2 = trial.suggest_float("dropout_2", 0.1, 0.9, log=True)
    
    #NN layers
    # Branch 1
    investment_id_inputs = tf.keras.Input((1, ), dtype=tf.uint16)
    features_inputs = tf.keras.Input((300, ), dtype=tf.float16)
    
    # Turns positive integers (indexes) into dense vectors of fixed size
    investment_id_x = investment_id_lookup_layer(investment_id_inputs)
    investment_id_x = layers.Embedding(investment_id_size, 32, input_length=1)(investment_id_x)
    investment_id_x = layers.Reshape((-1, ))(investment_id_x)
    investment_id_x = layers.Dense(64, activation='swish')(investment_id_x)    
    investment_id_x = layers.Dense(64, activation='swish')(investment_id_x)
    investment_id_x = layers.Dense(64, activation='swish')(investment_id_x)
    investment_id_x = layers.Dense(64, activation='swish')(investment_id_x)   
    
    #Branch 2
    feature_x = layers.Dense(256, activation='swish')(features_inputs)
    feature_x = layers.Dense(256, activation='swish')(feature_x)
    feature_x = layers.Dense(256, activation='swish')(feature_x)
    feature_x = layers.Dense(256, activation='swish')(feature_x)
    feature_x = layers.Dropout(dropout_1)(feature_x)
    
    # Takes as input a list of tensors and returns a single tensor that is the concatenation of all inputs
    x = layers.Concatenate(axis=1)([investment_id_x, feature_x])
    x = layers.Dense(512, activation='swish', kernel_regularizer="l2")(x)
    x = layers.Dense(128, activation='swish', kernel_regularizer="l2")(x)
    x = layers.Dense(32, activation='swish', kernel_regularizer="l2")(x)
    x = layers.Dense(32, activation='swish', kernel_regularizer="l2")(x)
    x = layers.Dropout(dropout_2)(x)
    
    output = layers.Dense(1)(x)
    
    rmse = keras.metrics.RootMeanSquaredError(name="rmse")
    
    model = tf.keras.Model(inputs=[investment_id_inputs, features_inputs], outputs=[output])
    model.compile(optimizer=tf.optimizers.Adam(0.001), loss='mse', metrics=['mse'])
    
    return model
   

# store scores and models

models = []
scores = []

# objective function for optuna
def objective(trial):
    # objective function which passes optuna trial
    # returns the mean MSE of the scores


    kfold = StratifiedKFold(5, shuffle=True)
    
    for index, (train_indices, valid_indices) in enumerate(kfold.split(train, investment_id)):
        X_train, X_val = train.iloc[train_indices], train.iloc[valid_indices]
        investment_id_train = investment_id[train_indices]
        y_train, y_val = y.iloc[train_indices], y.iloc[valid_indices]
        investment_id_val = investment_id[valid_indices]
        train_ds = make_dataset(X_train, investment_id_train, y_train)
        valid_ds = make_dataset(X_val, investment_id_val, y_val, mode="valid")

        model = get_model(trial)
        checkpoint = keras.callbacks.ModelCheckpoint(f"model_{index}", save_best_only=True)
        early_stop = keras.callbacks.EarlyStopping(patience=2)
        history = model.fit(train_ds, epochs=4, validation_data=valid_ds, callbacks=[checkpoint, early_stop])
        model = keras.models.load_model(f"model_{index}")
        score = model.evaluate(valid_ds)
        models.append(model)
        scores.append(score[1])
        mean_scores = np.mean(scores)

        pearson_score = stats.pearsonr(model.predict(valid_ds).ravel(), y_val.values)[0]
        print('Pearson:', pearson_score)
        #pd.DataFrame(history.history, columns=["mse", "val_mse"]).plot()
        #plt.title("MSE")
        #plt.show()
        del investment_id_train
        del investment_id_val
        del X_train
        del X_val
        del y_train
        del y_val
        del train_ds
        del valid_ds
        gc.collect()
        
        # use 1 fold iteration 
        if index == 0: 
             break
        
    return mean_scores


# optimize and save outputs
study_name = 'model_2_dropouts'  # Unique identifier of the study.
study = optuna.create_study(study_name=study_name, direction="minimize", storage='sqlite:///studies.db', load_if_exists = True)
study.optimize(objective, n_trials=100,  timeout = 21600)   
df = study.trials_dataframe()
df.to_csv("model_2_droputs.csv")
joblib.dump(study_name, 'model_2_droputs')
