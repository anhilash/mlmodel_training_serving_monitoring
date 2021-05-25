# Tarin, Deploy and Monitor a Machine Learning Model using Docker, Flask,Prometheus and Grafana
A common practice for deploying Machine Learning (ML) models into production environments - e.g. ML models trained using the SciKit Learn or Keras packages (for Python), that are ready to provide predictions on new data - is to expose these ML models as RESTful API microservices, hosted from within Docker containers. These can then deployed to a cloud environment for handling everything required for maintaining continuous availability - e.g. fault-tolerance, auto-scaling, load balancing and rolling service updates.
In the code above we have trained and deployed text catogorisation model using Multinominal Naive Bayes Classifier. The trained model will be persisted on the container that code is running and we use the same to make predictions. We can achieve the same using Kubeflow as well.

## Containerising a Simple ML Model  Training and Scoring Service using Flask and Docker
Below is the detailed explination of dockerising the whole process of model training and deployement using Flask for simple Python ML model scoring REST API in server.py , together with the dockerfile, both within the api directory, whose core contents are as follows,
```bash
api/
 | Dockerfile
 | model.py
 | requirements.txt
 | server.py
 | test.jsonl.gz
 | train.jsonl.gz
 | wsgi.py
monitoring/
docker-compose.yml
README.md
```
Trainig and persisting ML Model in the `model.py` Module. For evaluating the model we can use AUC ROC. And the model is persisted accordingly in the

```python
class Model:
    def __init__(self):
        self.vec = CountVectorizer()
        self.tfidf = TfidfTransformer()
        self.model = MultinomialNB()

    #Model training and persisting
    def train(self, train_data: Iterable[dict]):
        counts = self.vec.fit_transform([x["text"] for x in train_data])
        tfidf = self.tfidf.fit_transform(counts)
        self.model.fit(tfidf, [x["label"] for x in train_data])
        with open("./model.joblib", 'wb') as model_jb:
            joblib.dump((self.vec, self.tfidf, self.model), model_jb)

    def predict(self, data: Iterable[dict]):
        return self.model.predict(
            self.tfidf.transform(self.vec.transform([x["text"] for x in data]))
        )
```
Defining the Flask Service in the `server.py` Module along with Prometheus linked to the Flask application for collecting metrics.
```python
import joblib
import logging
from flask import Flask, Response, request, make_response
from flask import jsonify
from prometheus_flask_exporter import PrometheusMetrics

# Defining Flask and Promehteus
logging.basicConfig(level=logging.INFO)
logging.info("Setting LOGLEVEL to INFO")
api = Flask(__name__)
metrics = PrometheusMetrics(api)
metrics.info("app_info", "ML Model Metrics", version="1.0.0")


@api.route("/prediction", methods=['POST'])
def predict() -> Response:
    """"
    This API endpoint expects a JSON payload with a field called `text`
    containing text to send to the model.
    This data is parsed into Python dict and made available via
    `request.json`
    If text cannot be found in the parsed JSON data, then an exception
    will be raised. Otherwise, it will return a JSON payload with the
    `label` field containing the model's prediction.
    """

    try:
        with open('./model.joblib', 'rb') as model_jbl:

            vectorizer, tfidf, model = joblib.load(model_jbl)

        features = request.json
        prediction = model.predict(tfidf.transform(vectorizer.transform([x["text"] for x in [features]])))[0]
        return make_response(jsonify({'label': prediction}), 200)
    except Exception:
        return make_response(jsonify(), 405)


if __name__ == "__main__":
    api.run()
```

Defining the Docker Image with the `Dockerfile`

## Containerising Metrics API for inspecting current metrics of the service along with Prometheus and Grafana Dashboards.
In monitoring module we have all the yml specification files that are required to start the containers for Proetheus and Grafana. The contents of the module are as below.
```bash
api/
monitoring/
    | config.monitoring
    | datasource.yml
    | grafana_dashboard.json
    | promehteus.yml
docker-compose.yml
README.md
```
Prometheus is a time-series database that is extremely popular as a metrics and monitoring database, specially with Kubernetes. Prometheus is really cool because it is designed to scrape metrics from your application, instead of your application having to send metrics to it actively. Coupled with Grafana, this stack turns in to a powerful metrics tracking/monitoring tool, which is used in applications the world over.
For Prometheus, you need a `prometheus.yml`.
In this example, we see that Prometheus is watching two endpoints, itself, example-prometheus:9090, and the Flask api, flask-api(container).
Note that we’re creating our own Docker network and putting all our applications on it, which allows them to talk to each other. However, this would be the same if you didn’t specify a network at all. Also important to note is that am using wsgi to run the Flask application.
Once Grafana is up, you should be able to log in and configure Prometheus as a datasource.
Once that’s done, you can use the example dashboard from the creator of the prometheus_flask_exporter library (use Import->JSON) which can be found here: https://github.com/rycus86/prometheus_flask_exporter/blob/master/examples/sample-signals/grafana/dashboards/example.json

### Set up and run everything using docker-compose
Make sure docker is installed in your local system before running the application. On running the below command all the three container will start running and the trained model will be persisted to serve the requests.
```
docker-compose up
```
### Access

* API: http://localhost:5000/prediction/  #Refefer the postman collection for sample request
* Prometheus: http://localhost:5000/metrcis #API Endpoint for Metrics
* Grafana: http://localhost:3000 `[username: admin, password: pass@123]` #URL for accessing Grafana Dashboard
