import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("data/imdb.csv")
df = df.sample(2000) 

# Clean text (remove <br />)
df['review'] = df['review'].str.replace('<br />', ' ', regex=False)

# Features and labels
X = df['review']
y = df['sentiment']

# Convert text → numbers
vectorizer = CountVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_vec, y)

# Save model
pickle.dump(model, open("model/model.pkl", "wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))

print("Model trained and saved successfully!")