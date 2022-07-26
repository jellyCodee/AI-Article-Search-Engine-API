from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

app = FastAPI(title="Article Search Engine API")


# @app.on_event("startup")
# def on_startup():
#     nltk.download('stopwords')
#     nltk.download('wordnet')
#     nltk.download('omw-1.4')


app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://aiarticle.netlify.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



data = pd.read_csv('static/articles.csv')
data['Title'] = data['Title'].apply(lambda x: x.lower())


stopwords = nltk.corpus.stopwords.words('english')
extra = ['!', '(', ')', '-', '[', ']', '{', '}', ';', ':', '"', ',',
        '<', '>', '/', '?', '@', '#', '$', '%', '^', '&', '*', '_', '~', "'"]

lemmatizer = WordNetLemmatizer()


def stopword_tokenize(sentence):
    # tokenize
    word_list = word_tokenize(sentence)
    # remove stop words
    stop_list = [
        word for word in word_list if word not in stopwords and word not in extra and word.isalpha()]
    # lemmatize result
    final_list = [lemmatizer.lemmatize(word) for word in stop_list]
    return final_list


vectorizer = TfidfVectorizer(tokenizer=stopword_tokenize)


def cosine_sim(text1, query):
    tfidf = vectorizer.fit_transform([text1, query])
    return round(((tfidf * tfidf.T).A)[0, 1], 2)




@app.get("/api")
def get_result(query: str) -> list:
    data['Cosine_sim'] = data['Title'].apply(lambda x: cosine_sim(x, query))
    result = data.sort_values(by='Cosine_sim', ascending=False).head(10).values.tolist()
    return result
