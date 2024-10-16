import pandas as pd
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Lambda, Dense
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    Quora = pd.read_csv('./kaggle/input/quora-question-pairs/train.csv.zip')
    questions1 = Quora['question1']
    questions2 = Quora['question2']
    labels = Quora['is_duplicate']

    questions1 = questions1.fillna('')
    questions2 = questions2.fillna('')

    max_sequence_length = 80
    embedding_dim = 300
    questions = Quora['question1'].astype(str) + ' ' + Quora['question2'].astype(str)

    tokens = [word for sentence in questions for word in sentence.split()]

    # Računanje dužine vokabulara
    vocabulary_size = len(set(tokens))
    questions = (questions1 + ' ' + questions2).astype(str)
    tokenizer = Tokenizer(num_words=vocabulary_size)
    tokenizer.fit_on_texts(questions)

    sequences1 = tokenizer.texts_to_sequences(questions1)
    sequences2 = tokenizer.texts_to_sequences(questions2)
    padded_sequences1 = pad_sequences(sequences1, maxlen=max_sequence_length)
    padded_sequences2 = pad_sequences(sequences2, maxlen=max_sequence_length)

    input_layer1 = Input(shape=(max_sequence_length,))
    input_layer2 = Input(shape=(max_sequence_length,))

    embedding_layer = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim)

    lstm_layer = LSTM(units=50)

    x1 = embedding_layer(input_layer1)
    x1 = lstm_layer(x1)

    x2 = embedding_layer(input_layer2)
    x2 = lstm_layer(x2)

    distance_layer = Lambda(lambda x: tf.keras.backend.abs(x[0] - x[1]),
                            output_shape=lambda _: (1,))([x1, x2])

    output_layer = Dense(units=1, activation='sigmoid')(distance_layer)

    siamese_model = Model(inputs=[input_layer1, input_layer2], outputs=output_layer)

    siamese_model.compile(optimizer=Adam(0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True),
        ModelCheckpoint(filepath='siamese_model_weights.h5', save_best_only=True)
    ]

    siamese_model.fit([padded_sequences1, padded_sequences2], labels, epochs=5, batch_size=32, validation_split=0.2,
                      callbacks=callbacks)

