# Zeroshot classification POC using ruGPT3

Solution to AI4Humanities track of AI Journey 2020

## Fetching the model

Use the src/utils/download_weights.py to download the model and tokenizer.

```
python ./src/utils/download_weights.py sberbank-ai/rugpt3large_based_on_gpt2 ./model
```

## Evaluation

To reproduce the nplus1 evaluation figures from the PowerPoint:

1. Download and unzip the Readability corpus from [Taiga](https://tatianashavrina.github.io/taiga_site/downloads) 
2. Fetch the model weights
3. Run src/evaluate.py number_of_samples seed

```
python ./src/evaluate.py ../NPlus1 ./model 500 0
```

This should produce the following metrics:
```
              precision    recall  f1-score   support

     Гаджеты       0.00      0.00      0.00        24
      Космос       0.33      0.03      0.06        31
       Наука       0.82      0.85      0.84       238
     Не знаю       0.00      0.00      0.00         0
      Оружие       0.98      0.44      0.61        97
  Технологии       0.37      0.44      0.40        85
   Транспорт       0.44      0.16      0.24        25

    accuracy                           0.58       500
   macro avg       0.42      0.27      0.31       500
weighted avg       0.69      0.58      0.60       500
```

An (untuned) logreg over BoW trained on all (6994) but these 500 samples achieves 0.82 weighted 
precision/recall, achieving F1 of 0.60 when trained on about 500 supervised samples, while a dummy classifier 
(stratified) yields 0.31/0.32.

While it is admittedly a bit of cheating (the model probably saw these texts on self-supervised pre-training), 
it was never provided any supervision. Unfortunately, there aren't any widely used Russian datasets like 
20newsgroups.

## Interactive classifier

To use the classifier interactively, run app.py with streamlit. The app expects the model fetched to `./model`

Works best with `large` model.

```
streamlit run ./src/app.py
```

## References

Shavrina T., Shapovalova O. (2017) TO THE METHODOLOGY OF CORPUS CONSTRUCTION FOR MACHINE LEARNING: «TAIGA» SYNTAX TREE CORPUS AND PARSER. in proc. of “CORPORA2017”, international conference , Saint-Petersbourg, 2017.