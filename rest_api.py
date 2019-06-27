import json
import pickle
import re

from flask import Flask, jsonify, request,Response
from flask_restful import reqparse, abort, Api, Resource
from textblob import TextBlob

with open('naive_bayes_classifier.pickle', 'rb') as f:
    classifier = pickle.load(f)

with open('vocabulary.pickle', 'rb') as f:
    vocabulary = pickle.load(f)

app = Flask(__name__)
api = Api(app)

class Predictor(Resource):
    def post(self):
        json_data = request.get_json(force=True)

        result = []
        for tweetData in json_data:
            tweet = tweetData['tweet']
            date = tweetData['date']

            tweetMatrix = naiveBayesSentimentCalculator(tweet)
            # tweetSentiment = sess.run(prediction, {input_data: tweetMatrix})[0]
            if (tweetMatrix == 'positive'):
                sentiment = 1;
            else:
                sentiment = 0;
            res = {"date":date, "sentiment":sentiment}
            result.append(res)
        return Response(json.dumps(result),  mimetype='application/json')

api.add_resource(Predictor, '/nbpredict')

class PredictorTB(Resource):
    def post(self):
        json_data = request.get_json(force=True)

        result = []
        for tweetData in json_data:
            tweet = tweetData['tweet']
            date = tweetData['date']

            tweetMatrix = analyze_sentiment(tweet)
            res = {"date":date, "sentiment":tweetMatrix}
            result.append(res)
        return Response(json.dumps(result),  mimetype='application/json')
api.add_resource(PredictorTB, '/tbpredict')

def naiveBayesSentimentCalculator(review):
  problemInstance = review.split()
  problemFeatures = extract_features(problemInstance)
  return classifier.classify(problemFeatures)

def extract_features(review):
  review_words=set(review)
  features={}
  for word in vocabulary:
      features[word]=(word in review_words)
  return features

def clean_tweet(text):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split())

def analyze_sentiment(text):
    analysis = TextBlob(clean_tweet(text))

    if analysis.sentiment.polarity > 0:
        return 1
    elif analysis.sentiment.polarity == 0:
        return 0
    else:
        return -1

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5460 ,debug=True)