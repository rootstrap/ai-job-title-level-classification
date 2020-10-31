FROM deepeshac280/aipythonimg
CMD ["python","/data_process/tsv_to_dataframe.py"]

FROM deepeshac280/aipythonimg
CMD ["python","sgd_fit_tune_classifier.py"]

FROM deepeshac280/aipythonimg
CMD ["python","sgd_test_classifier.py"]