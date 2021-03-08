import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# load pickle files
pfilename = '../SR-ARE-train/names_onehots.pickle'
pickle_data = np.load(pfilename, allow_pickle=True)
train_data = pickle_data['onehots']

# load labels
lfilename = '../SR-ARE-train/names_labels.txt'
file = np.genfromtxt(lfilename, dtype='str')
train_labels = []
for i, data in enumerate(file):
    data = data.split(',')
    train_labels.append(int(data[1]))
train_labels = np.array(train_labels)

# early stop training
callback = keras.callbacks.EarlyStopping(monitor='loss', patience=3)

# balance data
def balance_set(train_data, train_labels):
    positive = np.count_nonzero(train_labels)
    negative = len(train_labels) - positive
    y_train = np.empty((0, 1), int)
    x_train = np.empty((0, 70, 325), int)
    if positive < negative:
        negative_index = np.where(train_labels == 0)[0]
        negative_index = np.random.choice(negative_index, positive)
        for i in range(len(train_labels)):
            if train_labels[i] == 1 or i in negative_index:
                y_train = np.append(y_train, train_labels[i])
                x_train = np.append(x_train, train_data[i])
    elif negative < positive:
        positive_index = np.nonzero(y_train)
        positive_index = np.random.choice(positive_index, negative)
        for i in range(len(train_labels)):
            if train_labels[i] == 0 or i in positive_index:
                y_train = np.append(y_train, train_labels[i])
                x_train = np.append(x_train, train_data[i])
    x_train = np.reshape(x_train, (int(len(x_train) / 70 / 325), 70, 325))
    return x_train, y_train

train_data, train_labels = balance_set(train_data, train_labels)

# build model
def get_model():

    model = keras.Sequential()
    model.add(layers.Reshape((70, 325, 1), input_shape=(70, 325)))
    model.add(layers.Conv2D(32, 5, strides=2, activation="relu"))
    model.add(layers.Conv2D(64, 3, activation="relu"))
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1, activation='sigmoid', name="predictions"))

    # compile model
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy()],
    )
    return model

model = get_model()

model.summary()

# fit model
model.fit(
    train_data,
    train_labels,
    # train_dataset,
    batch_size=64,
    epochs=100,
    # class_weight=class_weight,
    callbacks=[callback]
)

# save model
model.save('trained_model30.h5')

# # evaluate model
# model.evaluate(test_data, test_labels, batch_size=7167)



