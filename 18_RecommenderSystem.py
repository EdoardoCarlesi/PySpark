from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler


spark = SparkSession.builder.appName('recSystem').getOrCreate()

base_path = 'data/Spark_for_Machine_Learning/Recommender_Systems/'
file_name = 'movielens_ratings.csv'
data = spark.read.csv(base_path + file_name, header=True, inferSchema=True)

data.head(1)
data.printSchema()
data.describe().show()

training, test = data.randomSplit([0.8, 0.2])

# ALS: Alternating Least Squares Recommender System based on matrix factorization.
# The most important part in rec. systems in Spark is getting the data into user / item / rating format
als = ALS(maxIter = 5, regParam = 0.01, userCol = 'userId', itemCol  = 'movieId', ratingCol = 'rating')

model = als.fit(training)

predictions = model.transform(test)
predictions.show()

evaluator = RegressionEvaluator(metricName = 'rmse', labelCol = 'rating', predictionCol = 'prediction')

rmse = evaluator.evaluate(predictions)

print('RMSE: ', rmse)

single_user = test.filter(test['userId'] == 11).select(['movieId', 'userId'])
single_user.show()

recommendations = model.transform(single_user)
recommendations.orderBy('prediction', ascending=False).show()


