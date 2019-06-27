import nltk
import pickle

positiveReviewsFileName = "applenews-rt-polarity.pos"
negativeReviewsFileName = "applenews-rt-polarity.neg"

positiveReviewsFileNameNasdaq = "nasdaq-rt-polarity.pos"
negativeReviewsFileNameNasdaq = "nasdaq-rt-polarity.neg"
positiveReviews = []
negativeReviews = []

# with open(positiveReviewsFileName,'r',encoding="latin-1") as f:
#     positiveReviews = f.readlines()
#
# with open(negativeReviewsFileName,'r',encoding="latin-1") as f:
#     negativeReviews = f.readlines()


with open(positiveReviewsFileNameNasdaq,'r',encoding="latin-1") as f:
    poslines = f.readlines()
    positiveReviews = positiveReviews+ poslines

with open(negativeReviewsFileNameNasdaq,'r',encoding="latin-1") as f:
    neglines = f.readlines()
    negativeReviews = negativeReviews + neglines


testTrainingSplitIndex = 2500

testNegativeReviews = negativeReviews[testTrainingSplitIndex+1:]
testPositiveReviews = positiveReviews[testTrainingSplitIndex+1:]

trainingNegativeReviews = negativeReviews[:testTrainingSplitIndex]
trainingPositiveReviews = positiveReviews[:testTrainingSplitIndex]



def getTestReviewSentiments(naiveBayesSentimentCalculator):
  testNegResults = [naiveBayesSentimentCalculator(review) for review in testNegativeReviews]
  testPosResults = [naiveBayesSentimentCalculator(review) for review in testPositiveReviews]
  labelToNum = {'positive':1,'negative':-1}
  numericNegResults = [labelToNum[x] for x in testNegResults]
  numericPosResults = [labelToNum[x] for x in testPosResults]
  return {'results-on-positive':numericPosResults, 'results-on-negative':numericNegResults}


def runDiagnostics(reviewResult):
  positiveReviewsResult = reviewResult['results-on-positive']
  negativeReviewsResult = reviewResult['results-on-negative']
  numTruePositive = sum(x > 0 for x in positiveReviewsResult)
  numTrueNegative = sum(x < 0 for x in negativeReviewsResult)
  pctTruePositive = float(numTruePositive)/len(positiveReviewsResult)
  pctTrueNegative = float(numTrueNegative)/len(negativeReviewsResult)  
  totalAccurate = numTruePositive + numTrueNegative
  total = len(positiveReviewsResult) + len(negativeReviewsResult)
  print("Accuracy on positive reviews = " +"%.2f" % (pctTruePositive*100) + "%")
  print("Accurance on negative reviews = " +"%.2f" % (pctTrueNegative*100) + "%")
  print("Overall accuracy = " + "%.2f" % (totalAccurate*100/total) + "%")



def getVocabulary():
  positiveWordList = [word for line in trainingPositiveReviews if type(line) == str for word in line.split()]
  negativeWordList = [word for line in trainingNegativeReviews if type(line) == str for word in line.split()]
  allWordList = [item for sublist in [positiveWordList,negativeWordList] for item in sublist]
  allWordSet = list(set(allWordList))
  vocabulary = allWordSet
  return vocabulary

def getTrainingData():
  negTaggedTrainingReviewList = [{'review':oneReview.split(),'label':'negative'} for oneReview in trainingNegativeReviews if type(oneReview) == str]
  posTaggedTrainingReviewList = [{'review':oneReview.split(),'label':'positive'} for oneReview in trainingPositiveReviews if type(oneReview) == str]
  fullTaggedTrainingData = [item for sublist in [negTaggedTrainingReviewList,posTaggedTrainingReviewList] for item in sublist]
  trainingData = [(review['review'],review['label']) for review in fullTaggedTrainingData]
  return trainingData




def extract_features(review):
  review_words=set(review)
  features={}
  for word in vocabulary:
      features[word]=(word in review_words)
  return features 


def getTrainedNaiveBayesClassifier(extract_features, trainingData):
  trainingFeatures=nltk.classify.apply_features(extract_features, trainingData)
  trainedNBClassifier=nltk.NaiveBayesClassifier.train(trainingFeatures)
  return trainedNBClassifier

def save_vocabulary(vocabulary):
  with open('vocabulary.pickle', 'wb') as f:
    pickle.dump(vocabulary, f)

def load_vocabulary():
  with open('vocabulary.pickle', 'rb') as f:
    vocabulary = pickle.load(f)
    return vocabulary

def save_classifier(classifier):
  with open('naive_bayes_classifier.pickle', 'wb') as f:
    pickle.dump(classifier, f)

def load_classifier():
  with open('naive_bayes_classifier.pickle', 'rb') as f:
    classifier = pickle.load(f)
    return classifier

vocabulary = getVocabulary()
trainingData = getTrainingData()
trainedNBClassifier = getTrainedNaiveBayesClassifier(extract_features,trainingData)

save_vocabulary(vocabulary)
save_classifier(trainedNBClassifier)

def naiveBayesSentimentCalculator(review):
  problemInstance = review.split()
  problemFeatures = extract_features(problemInstance)
  return trainedNBClassifier.classify(problemFeatures)

naiveBayesSentimentCalculator("What an awesome movie")
naiveBayesSentimentCalculator("What a terrible movie")

runDiagnostics(getTestReviewSentiments(naiveBayesSentimentCalculator))



