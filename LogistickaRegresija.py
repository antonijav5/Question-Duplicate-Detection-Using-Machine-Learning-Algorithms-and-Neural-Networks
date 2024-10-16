import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
if __name__ == '__main__':
 Quora = pd.read_csv('./kaggle/input/quora-question-pairs/train.csv.zip')
 X = Quora[['question1', 'question2']]
 y = Quora['is_duplicate']

 # Kombinovanje 'question1' and 'question2' u jedinstven string za svaki red
 X['combined'] = X['question1'].astype(str) + ' ' + X['question2'].astype(str)

 # Za vrednosti koje ne postoje ubacuje se prazan string
 X['combined'] = X['combined'].fillna('')

# Podela podataka za train funkciju
 X_train, X_test, y_train, y_test = train_test_split(X['combined'], y, test_size=0.2, random_state=42)

# TF-IDF vektorizacija (fit transform)
 vectorizer = TfidfVectorizer()
 X_train_tfidf = vectorizer.fit_transform(X_train)
 X_test_tfidf = vectorizer.transform(X_test)

 # Uveravanje da se broj primeraka poklapa
 print(X_train_tfidf.shape, '  ', len(X_train))

 # Logisticka regresija
 classifier = LogisticRegression()
 classifier.fit(X_train_tfidf, y_train)

 # Predikcije
 predictions = classifier.predict(X_test_tfidf)

 # Evaluacija
 accuracy = accuracy_score(y_test, predictions)
 print(f"Accuracy: {accuracy}")
 print(confusion_matrix(y_test, predictions))
