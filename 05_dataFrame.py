import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StringType, IntegerType, StructType
from pyspark.sql.functions import countDistinct, avg, stddev, format_number, mean
from pyspark.sql.functions import (dayofmonth, hour, dayofyear, month, 
                                    format_number, date_format, year)

spark = SparkSession.builder.appName('dates').getOrCreate()

base_path = '/home/edoardo/Udemy/PySpark/Python-and-Spark-for-Big-Data-master/Spark_DataFrames/'
file_name = 'appl_stock.csv'

df = spark.read.csv(base_path + file_name, inferSchema = True, header = True)

df.select(['Date', 'Open']).show()

# How to select the date / hour ... etc.
#df.select(dayofmonth(df['Date'])).show()
newdf = df.withColumn("Year", year(df['Date']))

# Average per year
result = newdf.groupBy("Year").mean().select(["Year", "avg(Close)"])

result.select(['Year', format_number("avg(Close)", 2).alias("Avg Close")]).orderBy('Year').show()


