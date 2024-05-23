import numpy as np
import pandas as pd
from PIL import Image
import io
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelBinarizer

# Supondo que IMG_WIDTH e IMG_HEIGHT sejam definidos anteriormente
IMG_WIDTH, IMG_HEIGHT = 28, 28  # substitua pelas dimensões reais

# Função para converter bytes de PNG em array NumPy redimensionado e em grayscale
def get_png_array(df_row):
    png_bytes = df_row['bytes']
    image = Image.open(io.BytesIO(png_bytes))
    grayscale_image = image.convert('L')
    resized_image = grayscale_image.resize((IMG_WIDTH, IMG_HEIGHT))
    image_array = np.array(resized_image)
    return image_array

# Aplicando a função nas colunas do DataFrame
df_train['image_array'] = df_train['image'].apply(get_png_array)
df_validation['image_array'] = df_validation['image'].apply(get_png_array)

# Convertendo para arrays NumPy e normalizando
x_train = np.stack(df_train['image_array'].values).astype(np.float32) / 255.0
x_val = np.stack(df_validation['image_array'].values).astype(np.float32) / 255.0

# Flattening the images: convertendo para arrays 2D (n_samples, n_features)
x_train = x_train.reshape((x_train.shape[0], IMG_WIDTH * IMG_HEIGHT))
x_val = x_val.reshape((x_val.shape[0], IMG_WIDTH * IMG_HEIGHT))

# Labels
y_train = df_train['doc_category']
y_val = df_validation['doc_category']

# One-hot encoding das labels
label_binarizer = LabelBinarizer()
y_train_encoded = label_binarizer.fit_transform(y_train)
y_val_encoded = label_binarizer.transform(y_val)

# Criando tf.data.Dataset
batch_size = 500

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train_encoded))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val_encoded))
val_dataset = val_dataset.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# Definindo o modelo
neural_network_modelv2 = Sequential([
    Flatten(input_shape=(IMG_WIDTH * IMG_HEIGHT,)),
    Dropout(0.2),
    Dense(200, activation='relu'),
    Dropout(0.2),
    Dense(100, activation='relu'),
    Dropout(0.2),
    Dense(50, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
])

# Compilando o modelo
adam = Adam(learning_rate=0.001)
neural_network_modelv2.compile(optimizer=adam, loss="categorical_crossentropy", metrics=['accuracy'])

# Sumário do modelo
neural_network_modelv2.summary()

# Treinando o modelo
callback = EarlyStopping(monitor='loss', patience=5)
neural_network_historyv2 = neural_network_modelv2.fit(
    train_dataset,
    epochs=100,
    verbose=0,
    shuffle=True,
    validation_data=val_dataset,
    callbacks=[callback]
)
