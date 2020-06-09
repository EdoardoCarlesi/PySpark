from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.sql.functions import corr

spark = SparkSession.builder.appName('lr_ex').getOrCreate()

base_path = '/home/edoardo/Udemy/PySpark/Python-and-Spark-for-Big-Data-master/Spark_for_Machine_Learning/Linear_Regression/'
file_name = 'cruise_ship_info.csv'

data = spark.read.csv(base_path + file_name, inferSchema=True, header=True)
data.printSchema()

data.select(corr('crew', 'passengers')).show()

#print(data.columns)
#print(data.groupBy('Cruise_line').count())

# This one transforms the strings into numbers
indexer = StringIndexer(inputCol='Cruise_line', outputCol='Cruise_cat')
indexed = indexer.fit(data).transform(data)

inCols = ['Age', 'Tonnage', 'passengers', 'length', 'cabins', 'passenger_density', 'Cruise_cat']

# Including Cruse_cat makes things worse ?!?!
#inCols = ['Age', 'Tonnage', 'passengers', 'length', 'cabins', 'passenger_density']
assembler = VectorAssembler(inputCols=inCols, outputCol='features')
output = assembler.transform(indexed)
indexed.show()

final_data = output.select('features', 'crew')
train_data, test_data = final_data.randomSplit([0.6, 0.4])

lr = LinearRegression(labelCol='crew')
lr_model = lr.fit(train_data)
test_result = lr_model.evaluate(test_data)

print('RMS: ', test_result.rootMeanSquaredError)
print('R2: ', test_result.r2)
final_data.describe().show()


#stringIndex = StringIndexer(inputCol='Cruise_line', outputCol='CruiseLine_Index')
#print(stringIndex)
