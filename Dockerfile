FROM tiangolo/uwsgi-nginx-flask:python3.6 as step1
COPY . /app
RUN pip install --upgrade pip
RUN pip install -r ./requirements.txt
ENV ENVIRONMENT production
CMD ["python","/data_process/tsv_to_dataframe.py"]

FROM tiangolo/uwsgi-nginx-flask:python3.6 as step2
COPY --from=step1 /app /app
RUN pip install --upgrade pip
RUN pip install -r ./requirements.txt
ENV ENVIRONMENT production
CMD ["python","sgd_fit_tune_classifier.py"]

FROM tiangolo/uwsgi-nginx-flask:python3.6 
COPY --from=step2 /app /app
RUN pip install --upgrade pip
RUN pip install -r ./requirements.txt
ENV ENVIRONMENT production
CMD ["python","sgd_test_classifier.py"]