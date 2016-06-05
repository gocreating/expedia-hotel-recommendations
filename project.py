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
    p = parts
    label = float(p[TRAIN_HOTEL_CLUSTER])
    features = [
        p[TRAIN_SITE_NAME],
        p[TRAIN_POSA_CONTINENT],
        p[TRAIN_USER_LOCATION_COUNTRY],
        p[TRAIN_USER_LOCATION_REGION],
        p[TRAIN_USER_LOCATION_CITY],
        p[TRAIN_IS_MOBILE],
        p[TRAIN_IS_PACKAGE],
        p[TRAIN_CHANNEL],
        p[TRAIN_SRCH_ADULTS_CNT],
        p[TRAIN_SRCH_CHILDREN_CNT],
        p[TRAIN_SRCH_RM_CNT],
        p[TRAIN_SRCH_DESTINATION_ID],
        p[TRAIN_SRCH_DESTINATION_TYPE_ID],
        p[TRAIN_HOTEL_CONTINENT],
        p[TRAIN_HOTEL_COUNTRY],
        p[TRAIN_HOTEL_MARKET],
	]
    return LabeledPoint(label, features)

def parseTest(parts):
    p = parts
    label = p[TEST_ID]
    features = [
        p[TEST_SITE_NAME],
        p[TEST_POSA_CONTINENT],
        p[TEST_USER_LOCATION_COUNTRY],
        p[TEST_USER_LOCATION_REGION],
        p[TEST_USER_LOCATION_CITY],
        p[TEST_IS_MOBILE],
        p[TEST_IS_PACKAGE],
        p[TEST_CHANNEL],
        p[TEST_SRCH_ADULTS_CNT],
        p[TEST_SRCH_CHILDREN_CNT],
        p[TEST_SRCH_RM_CNT],
        p[TEST_SRCH_DESTINATION_ID],
        p[TEST_SRCH_DESTINATION_TYPE_ID],
        p[TEST_HOTEL_CONTINENT],
        p[TEST_HOTEL_COUNTRY],
        p[TEST_HOTEL_MARKET],
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
