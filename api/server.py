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
