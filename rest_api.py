import json
import pickle

from flask import Flask, jsonify, request,Response
from flask_restful import reqparse, abort, Api, Resource

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
            return result

        return Response(json.dumps(result),  mimetype='application/json')

api.add_resource(Predictor, '/nbpredict')

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


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5444 ,debug=True)