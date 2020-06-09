import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StringType, IntegerType, StructType

spark = SparkSession.builder.appName('ops').getOrCreate()

base_path = '/home/edoardo/Udemy/PySpark/Python-and-Spark-for-Big-Data-master/Spark_DataFrames/'
file_name = 'appl_stock.csv'

df = spark.read.csv(base_path + file_name, inferSchema = True, header = True)

# SQL kind of syntax for filtering 
#print(df.filter("Close < 500").show())
#print(df.filter("Close < 500").select(['Open', 'Close']).show())

# Python syntax
#print(df.filter(df['Close'] < 500).select('Volume').show())

#df.filter((df['Close'] < 200) & (df['Open'] > 200)).show()

result = df.filter((df['Close'] < 200) & (df['Open'] > 200)).collect()
#print(result, type(result), result[0].asDict())



