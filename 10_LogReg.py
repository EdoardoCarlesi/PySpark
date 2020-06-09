from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import (BinaryClassificationEvaluator,
                                    MulticlassClassificationEvaluator)

spark = SparkSession.builder.appName('logReg').getOrCreate()

base_path = '/home/edoardo/Udemy/PySpark/Python-and-Spark-for-Big-Data-master/Spark_for_Machine_Learning/Logistic_Regression/'
file_name = 'sample_libsvm_data.txt'

data = spark.read.format('libsvm').load(base_path + file_name)
data.show()

# Basic fit 
logRegModel = LogisticRegression()
fit_logRegModel = logRegModel.fit(data)
fit_logRegModel.summary.predictions.show()

# Split train - test and train the model
train_data, test_data = data.randomSplit([0.7, 0.3])
final_model = LogisticRegression()
fit_final = final_model.fit(train_data)
pred = fit_final.evaluate(test_data)
pred.predictions.show()

# Binary classification
evaluator = BinaryClassificationEvaluator()
final_roc = evaluator.evaluate(pred.predictions)
print(final_roc)




