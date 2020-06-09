import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StringType, IntegerType, StructType


base_path = '/home/edoardo/Udemy/PySpark/Python-and-Spark-for-Big-Data-master/Spark_DataFrames/'
file_name = 'people.json'

spark = SparkSession.builder.appName('Basics').getOrCreate()

#df = spark.read.json(base_path + file_name)

'''
print(df.show())
print(df.printSchema())
print(df.columns)
print(df.describe().show())
'''

# This is reading from column AGE assuming is value Integer type, True is to say that it is null
data_schema = [StructField('age', IntegerType(), True),
                StructField('name', StringType(), True)]

final_struct = StructType(fields = data_schema)


df = spark.read.json(base_path + file_name, schema = final_struct)

print(df.show())
print(df.printSchema())
print(df.columns)
print(df.describe().show())
