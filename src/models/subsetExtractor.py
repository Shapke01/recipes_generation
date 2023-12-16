import numpy as np
import pandas as pd
import json
import pathlib


class Extractor():
    
    def __init__(self, path: str, threshold: float = 0.01):
        self.loadCorrelationDict(path)
        self.threshold = threshold

    def loadCorrelationDict(self, path):
        
        with open(path, "r") as infile:
            d = json.load(infile)
        self.corr_dict = d

    def getIngredients(self):
        return list(self.corr_dict.keys())

    def getCorrelation(self, ing1, ing2):
        return self.corr_dict[ing1][ing2]
    
    def predict(self, ingredients):
        subset = []
        for ing1 in ingredients:
            ing1_sum = 0
            for ing2 in ingredients:
                if ing1!=ing2:
                    ing1_sum += self.getCorrelation(ing1, ing2)

            average_corr = ing1_sum / (len(ingredients)-1)
            # print(ing1 + ": " + str(average_corr))
            if average_corr > self.threshold:
                subset.append(ing1)

        return subset


if __name__ == "__main__":

    path = pathlib.Path(__file__).parents[2]
    path = path.joinpath("models/ingredients_correlation.json")
    
    model = Extractor(path=path)

    inputSet = ['eggs', 'pepper', 'salt', 'pork_sausage']

    results = model.predict(inputSet)
    for recipe in results:
        print(recipe)
