import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StringType, IntegerType, StructType
from pyspark.sql.functions import countDistinct, avg, stddev, format_number, mean

spark = SparkSession.builder.appName('aggs').getOrCreate()

base_path = '/home/edoardo/Udemy/PySpark/Python-and-Spark-for-Big-Data-master/Spark_DataFrames/'
file_name = 'ContainsNull.csv'

df = spark.read.csv(base_path + file_name, inferSchema = True, header = True)

# This drops ALL columns with at least one null
#df.na.drop().show()

# This one sets a threshold for how many non null values there must be on the column
#df.na.drop(thresh=2).show()

# This one specifies the way to drop the lines - only if ALL lines have null, or if ANY, or with subset = ['Sales'] i.e. only those rows with Sales missing
#df.na.drop(how='all').show()

# How to fill
df.printSchema()

# if I send a string it will fill the cols automatically, same thing for a number
#df.na.fill('FILL VALUES')
#df.na.fill(0).show()


# Fill values with mean value
#mean_value = df.select(mean(df['Sales'])).collect()
#print(mean_value[0][0])


# Fill it in one line
#df.na.fill(df.select(mean(df['Sales'])).collect()[0][0], ['Sales']).show()


