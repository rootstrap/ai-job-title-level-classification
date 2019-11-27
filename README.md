# Classification of job positions by level
This is a project to classify job positions using machine learning, more specifically, supervised learning. The main goal is to get a classifier that receives a job position in the form of a sentence, for example `CEO and Founder` and returns the job level for that position. The different levels (labels for the classification) are:  
* C-Suite
* VP
* Director
* Manager
* Other


In the example, `CEO and Founder` would return `C-Suite`.
Two algorithms are studied:  
* [Stochastic gradient descent (SGD)](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
* [Multi-layer Perceptron (MLP)](https://en.wikipedia.org/wiki/Multilayer_perceptron)

There is an analogous project but it classifies according to the level of the position: [Classification of job positions by area](https://github.com/rootstrap/ai-job-title-area-classification). If you use pip and virtual environments, you can install easily the named library: `$pip install -r requirements.txt`.


## Implementation
This project is programed using the [Python language](https://www.python.org). The trained classifiers are implemented in the [Scikit Learn library](https://scikit-learn.org), a set of tools for machine learning in Python.

## Training and general process
To see the explanation of the training of the classifiers, and the general process, take a look at the [documentation of the classification of job positions by area](https://github.com/rootstrap/ai-job-title-area-classification). The processes are the same.

## Script execution
The steps are the same as the classification by area:
1. Run `data_process/tsv_file_to_list.py` to transform the table into lists.
2. Run `data_process/normalize_list.py` to normalize the list, and transform the labels into integers.
3. Run `tuning_mlp_classifier.py` and `tuning_sgd_classifier.py` to find the best params for each algorithm.
4. Run `mlp_classifier.py` and `sgd_classifier.py` to fit the classifiers, run the tests, and store the results of the classification of new data, to see how it works. Finally, each classifier is dumped using pickle, in a file with its corresponding name. To load the instance, you can execute:  
```
with open('sgd.pkl', 'rb') as sgdfile:
    sgd_loaded = pickle.load(sgdfile)
```

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
