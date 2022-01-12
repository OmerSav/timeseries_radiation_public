# -*- coding: utf-8 -*-
"""
@author: omersav
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt

gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

#%% Constants

BATCH_SIZE = 128
EPOCHS = 150
VALIDATION_SIZE = 500 
DATA_PATH = "./radiation.csv"
RANDOM_SEED = 5

#%% In this section, the data is organized and made ready to enter traning.

df = pd.read_csv(DATA_PATH)

radyasyon = df.loc[:, "RADYASYON"]

bulutluluk = df.loc[:, 'Bulutluluk_Miktari_8_Okta']

# Months are selected and their values are fixed to 0 and one-hot-encoding with 12 classes is done.
ay = df.loc[:,'AY'] -1
ay = tf.one_hot(ay, 12).numpy()

# 17 class one-hot-encoding was applied, with 17 measurements taken from 02 to 18 for the hours, fixed to zero.
saat = df.loc[:,'SAAT'] - 2
saat = tf.one_hot(saat, 17).numpy()

# The days were fixed at 0 and coded as one-hot-encoding with 31 classes, respectively.
gun = df.loc[:,'GUN'] - 1
gun = tf.one_hot(gun, 31).numpy()

# All data was collected in a numpy array, with predictor variables in the first 4 columns and labels in the last column.
series = np.concatenate(( np.expand_dims(bulutluluk, -1), ay, gun, saat,
                        np.expand_dims(radyasyon, -1) ), axis=1)

num_feature = series.shape[1] -1 # Number of features

pre_dataset = [series[i,:] for i in range(series.shape[0])]

random.Random(RANDOM_SEED).shuffle(pre_dataset)

pre_training = pre_dataset[:-VALIDATION_SIZE]
pre_validation = pre_dataset[-VALIDATION_SIZE:]

def prepare_dataset(pre_dataset, batch_size):
    ''' sets the data generic to enter the model splits it into bachs

    '''
    dataset = tf.data.Dataset.from_tensor_slices(pre_dataset)
    dataset = dataset.map(lambda x: (x[:-1], x[-1]))    
    dataset = dataset.batch(batch_size)
    return dataset

training_dataset = prepare_dataset(pre_training, BATCH_SIZE)
validation_dataset = prepare_dataset(pre_validation, BATCH_SIZE)

print(training_dataset)

#%% model, loss, obtimizer ve metric belirlenir model compile edilir.

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(50, activation="relu", input_shape=(num_feature,)),
  tf.keras.layers.Dropout(.2),
  tf.keras.layers.Dense(50, activation="relu"),
  tf.keras.layers.Dense(1),
  tf.keras.layers.Lambda(lambda x: x * 50.0)
])

def custom_mse(y_true, y_pred):
    ''' Custom MSE. Unlike normal MSE, the result is divided by the true value and weighting is done to make the model
    more sensitive to low values.
    '''
    return tf.math.square(y_true-y_pred)/y_true
    
optimizer = tf.keras.optimizers.Adam(0.001)

model.compile(loss=custom_mse, optimizer=optimizer, metrics=["mse"])

#%% Train model
history = model.fit(training_dataset, epochs=EPOCHS, validation_data=validation_dataset)


#%% Visualization and results

def plot_hist(lab, epochs, tr_value, val_value):
    plt.plot(epochs, tr_value, label = lab["tr_name"] )
    plt.plot(epochs, val_value, label = lab["val_name"])
    plt.legend()
    plt.xlabel(lab["x"])
    plt.ylabel(lab["y"])
    plt.title(lab["tit"])
    plt.show()
    
epochs = range(1,EPOCHS+1)

# plot losses
loss_lab = {"x":"Epoch Sayısı", "y":"Loss Değerleri",
           "tit":"Epoch'a Göre Loss Değerleri",
           "tr_name":"Training Loss", "val_name": "Validation Loss" }
tr_losses = history.history['loss']
val_losses = history.history['val_loss']
plot_hist(loss_lab, epochs[10:], tr_losses[10:], val_losses[10:])

# plot metric (MSE)
loss_lab = {"x":"Epoch Sayısı", "y":"MSE Değerleri",
           "tit":"Epoch'a Göre MSE Değerleri",
           "tr_name":"Training MSE", "val_name": "Validation MSE" }
tr_mse = history.history['mse']
val_mse = history.history['val_mse']
plot_hist(loss_lab, epochs[10:], tr_mse[10:], val_mse[10:])

y_true = np.concatenate([y for x,y in validation_dataset], axis=0)
y_pred = model.predict(validation_dataset).reshape(-1)

prediction_size = range(len(y_true))

plt.figure(figsize=(21, 9))
plt.plot(prediction_size, y_true, color="blue")
plt.plot(prediction_size, y_pred, color="orange")
plt.grid(True)
plt.show()

df_pred = pd.DataFrame(np.concatenate((y_true.reshape(-1,1), 
                                         y_pred.reshape(-1,1)), axis = 1), 
                         columns=("Gerçek Değerler", "Tahmin edilenler"))
#pd.set_option("display.max_rows", None, "display.max_columns", None)
print(df_pred)

#%% Save model
model.save("radiationANN.h5")

#%% ____________________Load saved model _______________________
model = tf.keras.models.load_model("./radiationANN.h5", 
    custom_objects={"custom_mse": custom_mse})

# Make prediction
def predict_on_input(bulutluluk, ay, gun, saat, model = model):
    """input for utc +3 so extracted 3 from hours"""
    ay_ohe = tf.one_hot(ay-1, 12).numpy()
    gun_ohe = tf.one_hot(gun-1, 31).numpy()
    saat_ohe = tf.one_hot(saat-2-3, 17).numpy()
    features = np.concatenate((np.array([bulutluluk], dtype = np.float32), 
                               ay_ohe, gun_ohe, saat_ohe), axis = 0)
    result = model.predict(features)
    return result.numpy()

predict_on_input(0, 1,1,2, model) # radiation expected prediction is 10
    

