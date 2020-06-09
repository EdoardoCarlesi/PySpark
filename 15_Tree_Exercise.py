from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier, DecisionTreeClassifier
from pyspark.ml.regression import RandomForestRegressor, GBTRegressor, DecisionTreeRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, StringIndexer


spark = SparkSession.builder.appName('Tree').getOrCreate()

base_path = '/home/edoardo/Udemy/PySpark/Python-and-Spark-for-Big-Data-master/Spark_for_Machine_Learning/Tree_Methods/'
file_name = 'dog_food.csv'
data = spark.read.csv(base_path + file_name, header=True, inferSchema=True)
data.printSchema()

data.show()

cols = ['A', 'B', 'C', 'D']

data.show()

assembler = VectorAssembler(inputCols=cols, outputCol='features')
output = assembler.transform(data)

dtc = DecisionTreeClassifier(labelCol='Spoiled', featuresCol='features')
dtc_model = dtc.fit(output)

rfc = RandomForestClassifier(labelCol='Spoiled', featuresCol='features')
rfc_model = rfc.fit(output)

print(dtc_model.featureImportances)
print(rfc_model.featureImportances)
