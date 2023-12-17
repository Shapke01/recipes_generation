import numpy as np
import pandas as pd
import pickle

from sklearn.metrics.pairwise import cosine_similarity


class Vectorizer():
    def __init__(self, datasetPath: str, vectorizerPath: str,
                 tfidfRecipesPath: str, similarity_f):
        self.loadDataset(datasetPath)
        self.loadTFIDFrecipes(tfidfRecipesPath)
        self.loadVectorizer(vectorizerPath)
        self.similarity_f = similarity_f

    def loadDataset(self, datasetPath):
        self.dataset = pd.read_csv(datasetPath)

    def loadVectorizer(self, vectorizerPath):
        self.vectorizer = pickle.load(open(vectorizerPath, "rb"))

    def loadTFIDFrecipes(self, tfidfRecipesPath):
        self.tfidf_recipes = pd.read_csv(tfidfRecipesPath,
                                         index_col=0)
        print(self.tfidf_recipes.shape)

    def predict(self, inputSet, nBestMatches=3):
        transformed = self.vectorizer.transform([" ".join(inputSet)])
        print(transformed)
        self.vectorizer.transform([" ".join(inputSet)]).toarray()
        print(transformed.shape)
        similarities = self.similarity_f(transformed,
                                         self.tfidf_recipes).flatten()
        sorted_idx = np.argsort(similarities)[::-1].tolist()[:nBestMatches]
        print([self.dataset["title"].iloc[i] for i in sorted_idx])
        return [self.dataset["ingredients"].iloc[i] for i in sorted_idx]


if __name__ == "__main__":
    datasetPath = "../../data/processed/modified_dataset.csv"
    vectorizerPath = "../../models/tfidf_vectorizer.pkl"
    tfidfRecipesPath = "../../data/processed/tfidf_recipes.csv"

    model = Vectorizer(datasetPath, vectorizerPath,
                       tfidfRecipesPath, cosine_similarity)

    inputSet = ['eggs', 'pepper', 'salt',
                'flour', 'milk', 'sugar']

    results = model.predict(inputSet, nBestMatches=10)
    for recipe in results:
        print(recipe)
