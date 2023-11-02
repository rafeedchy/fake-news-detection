import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix



df = pd.read_csv("dataset/news.csv")

print(df.shape)
print(df.head())


labels = df.label
print(labels.head())

x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size = 0.2, random_state = 7)
article = [df.iloc[127].text]
#print(article)

#vectorizing text data
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

#fit and transform train, test set
tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)
tfidf_article = tfidf_vectorizer.transform(article)

model = PassiveAggressiveClassifier(max_iter=50)
model.fit(tfidf_train, y_train)

y_pred = model.predict(tfidf_test)
print(y_pred)
score = accuracy_score(y_test, y_pred)
print(f'Accuracy: {round(score*100,2)}%')

#confusion matrix
print(confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL']))

article_pred = model.predict(tfidf_article)
print(article_pred)
