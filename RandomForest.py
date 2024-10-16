import pandas as pd
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np


if __name__ == '__main__':

    Quora = pd.read_csv('./kaggle/input/quora-question-pairs/train.csv.zip')
    X = Quora[['question1', 'question2']]
    y = Quora['is_duplicate']

    # Tokenizovanje rečenica
    tokenized_sentences_q1 = X['question1'].apply(lambda x: str(x).split())
    tokenized_sentences_q2 = X['question2'].apply(lambda x: str(x).split())

    # Treniranje Word2Vec modela
    word2vec_model_q1 = Word2Vec(sentences=tokenized_sentences_q1, vector_size=100, window=5, min_count=1, workers=4)
    word2vec_model_q2 = Word2Vec(sentences=tokenized_sentences_q2, vector_size=100, window=5, min_count=1, workers=4)


    # Funkcija za dobijanje vektorske predstave rečenice
    def get_sentence_vector(sentence, model):
        vector = np.zeros(model.vector_size)
        count = 0
        for word in sentence:
            if word in model.wv:
                vector += model.wv[word]
                count += 1
        if count != 0:
            vector /= count
        return vector


    # Kreiranje vektora karakteristika za svako pitanje
    X_q1 = np.array([get_sentence_vector(sentence, word2vec_model_q1) for sentence in tokenized_sentences_q1])
    X_q2 = np.array([get_sentence_vector(sentence, word2vec_model_q2) for sentence in tokenized_sentences_q2])

    # Konkatenacija vektora karakteristika
    X_combined = np.concatenate((X_q1, X_q2), axis=1)

    # Raspodela podataka
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

    # Random Forest bez sredjivanja
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)

    # Predikcije
    y_pred = rfc.predict(X_test)

    # Evaluacija
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
