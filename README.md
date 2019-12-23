# Classification of job positions by level
This is a project to classify job positions using machine learning, more specifically, supervised learning. The main goal is to get a classifier that receives a job position in the form of a sentence, written in natural language, for example `CEO and Founder` and returns the job level for that position. The different levels (labels for the classification) are:  
* C-Suite
* VP
* Director
* Manager
* Other


In the example, `CEO and Founder` would return `C-Suite`.

There is an analogous project but it classifies according to the area of the position: [Classification of job positions by area](https://github.com/rootstrap/ai-job-title-area-classification). If you use pip and virtual environments, you can install easily the named library: `$pip install -r requirements.txt`.


## Content
You can see the explanation in the area classifier documentation.

## Script execution
The steps are the same as the classification by level:
1. Run `data_process/tsv_file_to_dataframe.py` to extract the data from the tsv file and split the dataset.
2. Run `<clf name>_fit_tune_classifier.py` to fit and tune the classifier. `fit` is to learn and fit the model to the train set, and `tune` is to search for the optimal combination of the hyperparams, the ones that achieves better results(tuning may take a while).
3. Run `<clf name>_test_classifier.py` to test the trained classifiers and show the results. Besides, a classified example set is stored in `test_data/<clf name>_results.tsv`.

Note: `<clf name>` can be `mlp` or `sgd`, depending on the classifier.

## Results
These are the results of each classifier:  

### MLP
```
                precision    recall  f1-score   support

     C-Suite       1.00      0.97      0.98        32
          VP       1.00      1.00      1.00        23
    Director       1.00      1.00      1.00        38
     Manager       0.91      0.95      0.93        21
       Other       0.94      0.94      0.94        36

    accuracy                           0.97       150
   macro avg       0.97      0.97      0.97       150
weighted avg       0.97      0.97      0.97       150
```

### SGD
```
                precision    recall  f1-score   support

     C-Suite       0.94      0.94      0.94        32
          VP       0.92      0.96      0.94        23
    Director       0.97      0.97      0.97        38
     Manager       1.00      0.95      0.98        21
       Other       0.97      0.97      0.97        36

    accuracy                           0.96       150
   macro avg       0.96      0.96      0.96       150
weighted avg       0.96      0.96      0.96       150
```
