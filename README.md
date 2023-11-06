# recipes_generation

## Generating recipes based on available ingredients

### Project title

Generating recipes based on available ingredients.
### Tested hypothesis/thesis
>We are able to generate recipes based on the most suitable subset of available ingredients.

### Major milestones

1. Create a synthetical dataset for evaluation of ingredients subsets using existing ingredients list
2. Create the most suitable subset from input ingredients using FlavorGraph.
3. Set up FlavorGraph repo.
4. Load FlavorGraph
5. Design multiple pairings recommendations to obtain a subset of ingredients. See if there is a need to train a dedicated model using embeddings.
6. Process the subset of ingredients via Eat_PIM embeddings to generate a FlowGraph (predict links).
7. Convert the flow graph to a recipe in natural language.
Evaluate against a TFIDF baseline

### Additional literature
A survey on food computing

### Acquisition of the dataset
Get ingredients from an available recipes data set, e.g. Recipe1M and later extend them with random ingredients to construct a synthetic data set from subset evaluation.

### Technologies

```
Python
rdflib
numpy
pandas
scikit-learn
SPARQLWrapper
spacy
networkx
torch
```


