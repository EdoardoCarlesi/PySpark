import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StringType, IntegerType, StructType
from pyspark.sql.functions import countDistinct, avg, stddev, format_number, mean, max, min, corr
from pyspark.sql.functions import (dayofmonth, hour, dayofyear, month, 
                                    format_number, date_format, year)

spark = SparkSession.builder.appName('dates').getOrCreate()

base_path = '/home/edoardo/Udemy/PySpark/Python-and-Spark-for-Big-Data-master/Spark_DataFrame_Project_Exercise/'
file_name = 'walmart_stock.csv'

df = spark.read.csv(base_path + file_name, inferSchema = True, header = True)

#df.show()
#df.printSchema()
#df.describe().show()
#print(df.describe())

#new_df = df.withColumn('HV Ratio', df["High"]/df["Volume"])
#print(new_df.orderBy(new_df['High'].desc()).head(1)[0][0])

df.select(avg('Close')).show()
df.select(max('Volume')).show()
df.select(min('Volume')).show()

#newdf = df.withColumn("Year", year(df['Date']))
#result = newdf.groupBy("Year").max().select(["Year", "max(High)"])

newdf = df.withColumn("Month", month(df['Date']))
result = newdf.groupBy("Month").max().select(["Month", "max(High)"])

result.show()

'''
df.select(['Date', 'Open']).show()

# How to select the date / hour ... etc.
#df.select(dayofmonth(df['Date'])).show()
newdf = df.withColumn("Year", year(df['Date']))

# Average per year
result = newdf.groupBy("Year").mean().select(["Year", "avg(Close)"])

result.select(['Year', format_number("avg(Close)", 2).alias("Avg Close")]).orderBy('Year').show()
'''

