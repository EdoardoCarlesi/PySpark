import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StringType, IntegerType, StructType
from pyspark.sql.functions import countDistinct, avg, stddev, format_number, mean, max, min, corr
from pyspark.sql.functions import (dayofmonth, hour, dayofyear, month, 
                                     format_number, date_format, year)

from pyspark.ml.regression import LinearRegression

spark = SparkSession.builder.appName('lrex').getOrCreate()

base_path = '/home/edoardo/Udemy/PySpark/Python-and-Spark-for-Big-Data-master/Spark_for_Machine_Learning/Linear_Regression/'
file_name = 'sample_linear_regression_data.txt'

all_data = spark.read.format('libsvm').load(base_path + file_name)
#training = spark.read.format('libsvm').load(base_path + file_name)
#training.show()

train_data, test_data = all_data.randomSplit([0.7, 0.3])
lr = LinearRegression(featuresCol='features', labelCol='label', predictionCol='prediction')

#print(train_data.describe().show())
#print(test_data.describe().show())

lrModel = lr.fit(train_data)
test_results = lrModel.evaluate(test_data)

#test_results.residuals.show()
#print(test_results.rootMeanSquaredError)

unlabeled_data = test_data.select('features')

#unlabeled_data.show()

predictions = lrModel.transform(unlabeled_data)

'''
lr = LinearRegression(featuresCol='features', labelCol='label', predictionCol='prediction')
lrModel = lr.fit(training)
print(lrModel.coefficients)
print(lrModel.intercept)
training_summary = lrModel.summary
print(training_summary.r2)
'''


