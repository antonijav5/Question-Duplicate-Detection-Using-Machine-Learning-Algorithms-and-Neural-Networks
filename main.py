
#!pip install seaborn wordcloud
import numpy as np
import pandas as pd
import matplotlib
from nltk import jaccard_distance
from sklearn.feature_extraction.text import CountVectorizer

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import re # Koriste se regularni izrazi za čišćenje podataka
import warnings
warnings.filterwarnings('ignore')
plt.style.use('ggplot')
pd.set_option('display.max_colwidth', None)

from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')


import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")


def generate_wordcloud(data_q1_duplicate, data_q1_non_duplicate, data_q2_duplicate, data_q2_non_duplicate):
    # Generisanje oblaka reči za pitanje 1 i pitanje 2
    wordcloud_q1_duplicate = WordCloud(width=400, height=200, background_color='white').generate(
        ' '.join(data_q1_duplicate.astype(str)))
    wordcloud_q1_non_duplicate = WordCloud(width=400, height=200, background_color='white').generate(
        ' '.join(data_q1_non_duplicate.astype(str)))
    wordcloud_q2_duplicate = WordCloud(width=400, height=200, background_color='white').generate(
        ' '.join(data_q2_duplicate.astype(str)))
    wordcloud_q2_non_duplicate = WordCloud(width=400, height=200, background_color='white').generate(
        ' '.join(data_q2_non_duplicate.astype(str)))

    # Crtanje oblaka reči u 2x2 plotu
    plt.figure(figsize=(15, 10))

    # Crtanje pitanja 1 gde je is_duplicate = 1
    plt.subplot(2, 2, 1)
    plt.imshow(wordcloud_q1_duplicate, interpolation='bilinear')
    plt.title('Question1 - Duplicate')
    plt.axis('off')

    # Crtanje pitanja 1 gde je is_duplicate = 0
    plt.subplot(2, 2, 2)
    plt.imshow(wordcloud_q1_non_duplicate, interpolation='bilinear')
    plt.title('Question1 - Non-duplicate')
    plt.axis('off')

    # Crtanje pitanja 2 gde je is_duplicate = 1
    plt.subplot(2, 2, 3)
    plt.imshow(wordcloud_q2_duplicate, interpolation='bilinear')
    plt.title('Question2 - Duplicate')
    plt.axis('off')

    # Crtanje pitanja 2 gde je is_duplicate = 0
    plt.subplot(2, 2, 4)
    plt.imshow(wordcloud_q2_non_duplicate, interpolation='bilinear')
    plt.title('Question2 - Non-duplicate')
    plt.axis('off')

    plt.tight_layout()
    #plt.show()

if __name__ == '__main__':

    Quora = pd.read_csv('./kaggle/input/quora-question-pairs/train.csv.zip')
    print(Quora.isna().sum())
    Quora.dropna(inplace=True)
    Quora

    temp = Quora.is_duplicate.value_counts()
    df_class = pd.DataFrame({'labels': temp.index,
                             'values': temp.values / len(Quora)})
    plt.figure(figsize=(4, 4))
    plt.title('defaut')
    sns.set_color_codes("pastel")
    sns.barplot(x='labels', y="values", data=df_class)
    locs, labels = plt.xticks()
    #plt.show()
    # Odvojiti podatke za duplirana i neduplirana pitanja
    duplicate_data_q1 = Quora[Quora['is_duplicate'] == 1]['question1']
    non_duplicate_data_q1 = Quora[Quora['is_duplicate'] == 0]['question1']
    duplicate_data_q2 = Quora[Quora['is_duplicate'] == 1]['question2']
    non_duplicate_data_q2 = Quora[Quora['is_duplicate'] == 0]['question2']

    # Generisanje i crtanja oblaka reči za 'question1' (pitanje 1) i'question2' (pitanje 2)
    generate_wordcloud(duplicate_data_q1, non_duplicate_data_q1, duplicate_data_q2, non_duplicate_data_q2)
    # Distribucija dužine teksta za question1
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    sns.histplot(data=Quora, x=Quora['question1'].astype('str').apply(len), bins=50, kde=True, color='skyblue')
    plt.title('Distribution of Text Length for Question1')
    plt.xlabel('Text Length')
    plt.ylabel('Frequency')
    # Text length distribution for question2
    plt.subplot(2, 2, 2)
    sns.histplot(data=Quora, x=Quora['question2'].astype('str').apply(len), bins=50, kde=True, color='red')
    plt.title('Distribution of Text Length for Question2')
    plt.xlabel('Text Length')
    plt.ylabel('Frequency')
    #plt.show()
    # Kreiranje CountVectorizer za bigrame
    vectorizer = CountVectorizer(ngram_range=(2, 2), stop_words='english')
    bigrams_q1 = vectorizer.fit_transform(Quora['question1'])

    # Dohvatanje imena atributa (bigrami)
    feature_names = vectorizer.get_feature_names_out()

    top_bigrams = pd.DataFrame(bigrams_q1.sum(axis=0).tolist()[0], index=feature_names, columns=['Count'])
    top_bigrams_q1 = top_bigrams.sort_values(by='Count', ascending=False).head(20)
    bigrams_q2 = vectorizer.fit_transform(Quora['question2'])
    feature_names = vectorizer.get_feature_names_out()
    top_bigrams = pd.DataFrame(bigrams_q2.sum(axis=0).tolist()[0], index=feature_names, columns=['Count'])
    top_bigrams_q2 = top_bigrams.sort_values(by='Count', ascending=False).head(20)
    # Stilizovanje crteža
    plt.style.use('ggplot')
    # Crtanje histograma za najčešće bigrame u okviru question1
    plt.figure(figsize=(10, 5))
    plt.bar(top_bigrams_q1.index, top_bigrams_q1['Count'], color='skyblue')
    plt.title('Top 20 Bigrams in Question1')
    plt.xlabel('Bigrams')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    #plt.show()
    # Crtanje histograma za najčešće bigrame u okviru question2
    plt.figure(figsize=(10, 5))
    plt.bar(top_bigrams_q2.index, top_bigrams_q2['Count'], color='salmon')
    plt.title('Top 20 Bigrams in Question2')
    plt.xlabel('Bigrams')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()






