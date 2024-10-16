import pandas as pd
import seaborn as sns
import matplotlib
from nltk import jaccard_distance

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


def roc_curve__confusion_matrix(model):
    from sklearn.metrics import roc_curve, auc
    y_prob_test = model.predict_proba(X_test_tfidf)[:, 1]
    y_prob_train = model.predict_proba(X_train_tfidf)[:, 1]
    prediction = model.predict(X_test_tfidf)

    fpr_test, tpr_test, thresholds = roc_curve(y_test, y_prob_test)
    fpr_train, tpr_train, thresholds1 = roc_curve(y_train, y_prob_train)

    roc_auc_test = auc(fpr_test, tpr_test)
    roc_auc_train = auc(fpr_train, tpr_train)

    # Racunanje matrice konfuzije za test set
    confusion_matrix_test = confusion_matrix(y_test, prediction)

    # Kreiranje subplotova
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Crtanje ROC krivih na prvom subplotu
    axs[0].plot(fpr_train, tpr_train, color='green', label='ROC curve train (AUC = %0.2f)' % roc_auc_train)
    axs[0].plot(fpr_test, tpr_test, color='blue', label='ROC curve test (AUC = %0.2f)' % roc_auc_test)
    axs[0].plot([0, 1], [0, 1], color='red', linestyle='--')
    axs[0].set_xlabel('False Positive Rate')
    axs[0].set_ylabel('True Positive Rate')
    axs[0].set_title('Receiver Operating Characteristic (ROC) Curve')
    axs[0].legend(loc="lower right")

    sns.set(font_scale=1.4)
    sns.heatmap(confusion_matrix_test, annot=True, fmt='g', cmap='Blues', ax=axs[1])
    axs[1].set_xlabel('Predicted label')
    axs[1].set_ylabel('True label')
    axs[1].set_title('Confusion Matrix (Test Set)model');

    plt.tight_layout()
    plt.show()




if __name__ == '__main__':

    Quora = pd.read_csv('./kaggle/input/quora-question-pairs/train.csv.zip')
    X = Quora[['question1', 'question2']]
    y = Quora['is_duplicate']

    X['combined'] = X['question1'].astype(str) + ' ' + X['question2'].astype(str)

    X['combined'] = X['combined'].fillna('')
    X_train, X_test, y_train, y_test = train_test_split(X['combined'], y, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # XGBoost klasifikator
    classifier = XGBClassifier(verbose=1)

    # Fit modela
    classifier.fit(X_train_tfidf, y_train)

    # Predikcije nad test skupom podataka
    predictions = classifier.predict(X_test_tfidf)

    # Evaluacija
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy}")
    print(confusion_matrix(y_test, predictions))
    roc_curve__confusion_matrix(classifier)







