from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense, Dropout, Conv1D, GlobalMaxPooling1D, Bidirectional

def build_lstm(vocab_size=20000, embedding_dim=100, maxlen=120, units=128, dropout=0.2):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=maxlen),
        LSTM(units, dropout=dropout, recurrent_dropout=dropout),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def build_gru(vocab_size=20000, embedding_dim=100, maxlen=120, units=128, dropout=0.2):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=maxlen),
        GRU(units, dropout=dropout, recurrent_dropout=dropout),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def build_cnn(vocab_size=20000, embedding_dim=100, maxlen=120, filters=128, kernel=5):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=maxlen),
        Conv1D(filters=filters, kernel_size=kernel, activation="relu"),
        GlobalMaxPooling1D(),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def build_bilstm(vocab_size=20000, embedding_dim=100, maxlen=120, units=128, dropout=0.2):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=maxlen),
        Bidirectional(LSTM(units, dropout=dropout, recurrent_dropout=dropout)),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model
