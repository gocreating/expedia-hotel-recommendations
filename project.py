from pyspark import SparkContext, SparkConf
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint

conf = SparkConf().setAppName("project")
sc = SparkContext(conf=conf)

# constants definition
TRAIN_DATE_TIME = 0
TRAIN_SITE_NAME = 1
TRAIN_POSA_CONTINENT = 2
TRAIN_USER_LOCATION_COUNTRY = 3
TRAIN_USER_LOCATION_REGION = 4
TRAIN_USER_LOCATION_CITY = 5
TRAIN_ORIG_DESTINATION_DISTANCE = 6
TRAIN_USER_ID = 7
TRAIN_IS_MOBILE = 8
TRAIN_IS_PACKAGE = 9
TRAIN_CHANNEL = 10
TRAIN_SRCH_CI = 11
TRAIN_SRCH_CO = 12
TRAIN_SRCH_ADULTS_CNT = 13
TRAIN_SRCH_CHILDREN_CNT = 14
TRAIN_SRCH_RM_CNT = 15
TRAIN_SRCH_DESTINATION_ID = 16
TRAIN_SRCH_DESTINATION_TYPE_ID = 17
TRAIN_IS_BOOKING = 18
TRAIN_CNT = 19
TRAIN_HOTEL_CONTINENT = 20
TRAIN_HOTEL_COUNTRY = 21
TRAIN_HOTEL_MARKET = 22
TRAIN_HOTEL_CLUSTER = 23

TEST_ID = 0
TEST_DATE_TIME = 1
TEST_SITE_NAME = 2
TEST_POSA_CONTINENT = 3
TEST_USER_LOCATION_COUNTRY = 4
TEST_USER_LOCATION_REGION = 5
TEST_USER_LOCATION_CITY = 6
TEST_ORIG_DESTINATION_DISTANCE = 7
TEST_USER_ID = 8
TEST_IS_MOBILE = 9
TEST_IS_PACKAGE = 10
TEST_CHANNEL = 11
TEST_SRCH_CI = 12
TEST_SRCH_CO = 13
TEST_SRCH_ADULTS_CNT = 14
TEST_SRCH_CHILDREN_CNT = 15
TEST_SRCH_RM_CNT = 16
TEST_SRCH_DESTINATION_ID = 17
TEST_SRCH_DESTINATION_TYPE_ID = 18
TEST_HOTEL_CONTINENT = 19
TEST_HOTEL_COUNTRY = 20
TEST_HOTEL_MARKET = 21

def split(line):
    return line.split(',')

def pickBooking(parts):
    return int(parts[18]) == 1

def parseTrain(parts):
    label = float(parts[TRAIN_HOTEL_CLUSTER])
    features = [
        parts[TRAIN_SITE_NAME],
        parts[TRAIN_POSA_CONTINENT],
        parts[TRAIN_USER_LOCATION_COUNTRY],
        parts[TRAIN_USER_LOCATION_REGION],
        parts[TRAIN_USER_LOCATION_CITY],
        parts[TRAIN_IS_MOBILE],
        parts[TRAIN_IS_PACKAGE],
        parts[TRAIN_CHANNEL],
        parts[TRAIN_SRCH_ADULTS_CNT],
        parts[TRAIN_SRCH_CHILDREN_CNT],
        parts[TRAIN_SRCH_RM_CNT],
        parts[TRAIN_SRCH_DESTINATION_ID],
        parts[TRAIN_SRCH_DESTINATION_TYPE_ID],
        parts[TRAIN_HOTEL_CONTINENT],
        parts[TRAIN_HOTEL_COUNTRY],
        parts[TRAIN_HOTEL_MARKET],
	]
    return LabeledPoint(label, features)

def parseTest(parts):
    label = parts[TEST_ID]
    features = [
        parts[TEST_SITE_NAME],
        parts[TEST_POSA_CONTINENT],
        parts[TEST_USER_LOCATION_COUNTRY],
        parts[TEST_USER_LOCATION_REGION],
        parts[TEST_USER_LOCATION_CITY],
        parts[TEST_IS_MOBILE],
        parts[TEST_IS_PACKAGE],
        parts[TEST_CHANNEL],
        parts[TEST_SRCH_ADULTS_CNT],
        parts[TEST_SRCH_CHILDREN_CNT],
        parts[TEST_SRCH_RM_CNT],
        parts[TEST_SRCH_DESTINATION_ID],
        parts[TEST_SRCH_DESTINATION_TYPE_ID],
        parts[TEST_HOTEL_CONTINENT],
        parts[TEST_HOTEL_COUNTRY],
        parts[TEST_HOTEL_MARKET],
    ]
    return LabeledPoint(label, features)

def toCSVLine(data):
    return ','.join(str(d) for d in data)

def main():
    # prepare training data
    RDDTrainData = sc.textFile('train_100.csv')
    RDDTrainHeader = RDDTrainData.take(1)[0]
    trainData = RDDTrainData.filter(lambda line: line != RDDTrainHeader)\
                            .map(split)\
                            .filter(pickBooking)\
                            .map(parseTrain)

    # prepare testing data
    RDDTestData = sc.textFile('test_100.csv')
    RDDTestHeader = RDDTestData.take(1)[0]
    testData = RDDTestData.filter(lambda line: line != RDDTestHeader)\
                          .map(split)\
                          .map(parseTest)

    # do prediction
    model = NaiveBayes.train(trainData, 1.0)
    predictionData = testData.map(lambda d: (int(d.label), int(model.predict(d.features))))
    CSVPredictData = predictionData.map(toCSVLine)
    CSVPredictData.saveAsTextFile('result')

main()
