import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StringType, IntegerType, StructType
from pyspark.sql.functions import countDistinct, avg, stddev, format_number

spark = SparkSession.builder.appName('aggs').getOrCreate()

base_path = '/home/edoardo/Udemy/PySpark/Python-and-Spark-for-Big-Data-master/Spark_DataFrames/'
file_name = 'sales_info.csv'

df = spark.read.csv(base_path + file_name, inferSchema = True, header = True)

# count() max() min() mean() ---> methods to work on grouped data object
#df.groupBy("Company").count().show()

# With a dictionary we can pass the name of the column to aggregate and the function we want to use
#df.agg({'Sales':'sum'}).show()

#df.select(countDistinct('Sales')).show()
#df.select(avg('Sales').alias('Avg.Sales')).show()

#sales_std = df.select(stddev("Sales").alias('std'))

# Output column format
#sales_std.select(format_number('std', 2)).show()

# Sort in ascending order
df.orderBy("Sales").show()

# Sort in descending values
df.orderBy(df['Sales'].desc()).show()




