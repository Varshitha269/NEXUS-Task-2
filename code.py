import pandas as pd
import re
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv("Sentiment Analysis Dataset.csv", encoding='latin1')

# Explore dataset structure, features, and size
print(df.info())
print(df.head())

# Identify key variables
text_content = df['SentimentText']
sentiment_labels = df['Sentiment']

# Text preprocessing functions
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join([WordNetLemmatizer().lemmatize(word) for word in text.split() if word not in stopwords.words('english')])
    return text

# Apply preprocessing to the text content
df['ProcessedText'] = df['SentimentText'].apply(preprocess_text)

# Distribution of sentiment labels
sentiment_distribution = df['Sentiment'].value_counts()
sentiment_distribution.plot(kind='bar', title='Sentiment Label Distribution')
plt.show()

# Text vectorization using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X = tfidf_vectorizer.fit_transform(df['ProcessedText'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, sentiment_labels, test_size=0.2, random_state=42)

# Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Predictions
y_pred = nb_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(classification_report(y_test, y_pred))

# Cross-validation
cv_scores = cross_val_score(nb_model, X, sentiment_labels, cv=5)
print(f'Cross-validation Scores: {cv_scores}')
print(f'Mean CV Score: {cv_scores.mean()}')

# Feature importance (for Naive Bayes)
feature_names = tfidf_vectorizer.get_feature_names()
feature_importance = nb_model.coef_[0]

# Display top N important features
top_features = sorted(zip(feature_importance, feature_names), reverse=True)[:10]
print(f'Top 10 important features: {top_features}')
