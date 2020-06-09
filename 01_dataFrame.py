import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StringType, IntegerType, StructType


base_path = '/home/edoardo/Udemy/PySpark/Python-and-Spark-for-Big-Data-master/Spark_DataFrames/'
file_name = 'people.json'

spark = SparkSession.builder.appName('Basics').getOrCreate()


# This is reading from column AGE assuming is value Integer type, True is to say that it is null
data_schema = [StructField('age', IntegerType(), True),
                StructField('name', StringType(), True)]

final_struct = StructType(fields = data_schema)


df = spark.read.json(base_path + file_name, schema = final_struct)

# select() method returns a DataFrame of a single column, using ['age'] we return a pyspark.Column object
print(df.select('age').show())

# withColumn() ---> creates a new column in the dataframe
print(df.withColumn('newage', df['age'] * 2).show())

# Just rename a column
print(df.withColumnRenamed('age', 'my_new_age').show())

df.createOrReplaceTempView('people')

results = spark.sql("SELECT * FROM people WHERE age=30")

#print(results.show())

'''
print(df.show())
print(df.printSchema())
print(df.columns)
print(df.describe().show())
'''
