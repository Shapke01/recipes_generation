import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Vectorizer():
    def __init__(self, datasetPath, vectorizerPath, tfidfRecipesPath, similarity_f):
        self.loadDataset(datasetPath)
        self.loadVectorizer(vectorizerPath)
        self.loadTFIDFrecipes(tfidfRecipesPath)
        self.similarity_f = similarity_f

    def loadDataset(self, datasetPath):
        self.dataset = pd.read_csv(datasetPath)

    def loadVectorizer(self, vectorizerPath):
        self.vectorizer = None #TODO

    def loadTFIDFrecipes(self, tfidfRecipesPath):
        self.tfidf_recipes = None #TODO

    def predict(self, inputSet, nBestMatches = 3):
        transformed = self.vectorizer.transform([" ".join(inputSet)]).toarray()
        similarities = self.similarity_f(transformed, self.tfidf_recipes).flatten()
        sorted_idx = np.argsort(similarities)[::-1].tolist()[:nBestMatches]
        return [self.dataset["directions"].iloc[i] for i in sorted_idx]


if __name__ == "__main__":
    datasetPath = "../../data/processed/dataset.csv"
    vectorizerPath "../../models/tfidf_vectorizer.pkl"
    tfidfRecipesPath = "../../data/processed/tfidf_recipes.csv"

    model = Vectorizer(datasetPath, vectorizerPath, tfidfRecipesPath, cosine_similarity)

    inputSet = ['eggs', 'pepper', 'salt', 'pork_sausage']

    results = model.predict(inputSet)
    for recipe in results:
        print(recipe)
    