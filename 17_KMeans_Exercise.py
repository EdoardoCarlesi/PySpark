from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler


spark = SparkSession.builder.appName('cluster').getOrCreate()

base_path = 'data/Spark_for_Machine_Learning/Clustering/'
file_name = 'hack_data.csv'
data = spark.read.csv(base_path + file_name, header=True, inferSchema=True)

data.head(1)
data.printSchema()
data.show()

cols = data.columns
new_cols = cols[:5]
new_cols.append(cols[-1])
#print(new_cols)

assembler = VectorAssembler(inputCols=new_cols, outputCol='features')
output = assembler.transform(data)

# We need to scale all the features - there are only features here nothing to compare
scaler = StandardScaler(inputCol='features', outputCol='scaledFeatures')

scaler_model = scaler.fit(output)
output = scaler_model.transform(output)

print(output.head(1))

kmeans = KMeans(featuresCol = 'scaledFeatures', k = 2)

model = kmeans.fit(output)

centers = model.clusterCenters()
print(centers)


preds = model.transform(output).select('prediction') #.show()

print(preds.filter("prediction == 0").count())
print(preds.filter("prediction == 1").count())
print(preds.filter("prediction == 2").count())


print('WSSE')
print(model.computeCost(output))

