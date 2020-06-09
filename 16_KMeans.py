from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler


spark = SparkSession.builder.appName('cluster').getOrCreate()

base_path = 'data/Spark_for_Machine_Learning/Clustering/'
file_name = 'seeds_dataset.csv'
data = spark.read.csv(base_path + file_name, header=True, inferSchema=True)

data.head(1)
data.printSchema()
data.show()

cols = data.columns 

assembler = VectorAssembler(inputCols=cols, outputCol='features')
output = assembler.transform(data)

# We need to scale all the features - there are only features here nothing to compare
scaler = StandardScaler(inputCol='features', outputCol='scaledFeatures')

scaler_model = scaler.fit(output)
output = scaler_model.transform(output)

print(output.head(1))

kmeans = KMeans(featuresCol = 'scaledFeatures', k = 3)

model = kmeans.fit(output)

print('WSSE')
print(model.computeCost(output))

centers = model.clusterCenters()

print(centers)

model.transform(output).select('prediction').show()


