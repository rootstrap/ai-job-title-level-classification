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
2. Run `<clf name>_fit_tune_classifier.py` to fit and tune the classifier. `fit` is to learn and fit the model to the train set. `tune` is to search for the optimal combination of the hyperparams, the ones that achieves better results (tuning may take a while).
3. Run `<clf name>_test_classifier.py` to test the trained classifiers and show the results. Besides, a classified example set is stored in `test_data/<clf name>_results.tsv`.

Note: `<clf name>` can be `mlp` or `sgd`, depending on the classifier.

## Results
These are the results of each classifier:  

### MLP
```
                precision    recall  f1-score   support

     C-Suite       0.90      0.93      0.92        29
    Director       0.97      0.92      0.94        37
     Manager       0.90      0.95      0.92        19
       Other       0.93      0.95      0.94        41
          VP       1.00      0.96      0.98        24

    accuracy                           0.94       150
   macro avg       0.94      0.94      0.94       150
weighted avg       0.94      0.94      0.94       150
```

### SGD
```
                precision    recall  f1-score   support

     C-Suite       0.97      0.97      0.97        29
    Director       0.97      0.92      0.94        37
     Manager       1.00      1.00      1.00        19
       Other       0.95      1.00      0.98        41
          VP       1.00      1.00      1.00        24

    accuracy                           0.97       150
   macro avg       0.98      0.98      0.98       150
weighted avg       0.97      0.97      0.97       150
```
