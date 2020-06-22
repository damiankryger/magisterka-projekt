from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import LSTM, Dropout, Dense, Activation
from tensorflow.python.layers.normalization import BatchNorm


class Network():
    def __init__(self):
        self.callbacks = []

        self.epochs = 100
        self.batch_size = 32

    def create(self, shape: tuple, dictionary_size: int):
        model = Sequential()
        model.add(LSTM(
            512,
            input_shape=shape,
            recurrent_dropout=0.3,
            return_sequences=True
        ))
        model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3, ))
        model.add(LSTM(512))
        model.add(BatchNorm())
        model.add(Dropout(0.3))
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(BatchNorm())
        model.add(Dropout(0.3))
        model.add(Dense(dictionary_size))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

        self.model = model

    def add_callback(self, callback: Callback):
        self.callbacks.append(callback)

    def train(self, train_input, train_output):
        self.model.fit(train_input, train_output, epochs=self.epochs, batch_size=self.batch_size,
                       callbacks=self.callbacks)

    def load(self, filepath: str):
        self.model.load_weights(filepath)

    def predict(self, predict_input):
        return self.model.predict(predict_input, verbose=0)
