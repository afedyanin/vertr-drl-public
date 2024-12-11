FROM apache/airflow:2.10.2
ENV AIRFLOW__CORE__LOAD_EXAMPLES=False
COPY requirements.txt /
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" -r /requirements.txt
