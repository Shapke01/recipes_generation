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

    def predict(self, inputSet, nBestMatches=3):
        transformed = self.vectorizer.transform([" ".join(inputSet)]).toarray()
        similarities = self.similarity_f(transformed,
                                         self.tfidf_recipes).flatten()
        sorted_idx = np.argsort(similarities)[::-1].tolist()[:nBestMatches]
        return [self.dataset["directions"].iloc[i] for i in sorted_idx]


if __name__ == "__main__":
    datasetPath = "../../data/processed/dataset.csv"
    vectorizerPath = "../../models/tfidf_vectorizer.pkl"
    tfidfRecipesPath = "../../data/processed/tfidf_recipes.csv"

    model = Vectorizer(datasetPath, vectorizerPath,
                       tfidfRecipesPath, cosine_similarity)

    inputSet = ['eggs', 'pepper', 'salt', 'pork_sausage']

    results = model.predict(inputSet)
    for recipe in results:
        print(recipe)
