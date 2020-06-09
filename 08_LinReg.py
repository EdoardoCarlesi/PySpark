from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler


spark = SparkSession.builder.appName('lr_ex').getOrCreate()

base_path = '/home/edoardo/Udemy/PySpark/Python-and-Spark-for-Big-Data-master/Spark_for_Machine_Learning/Linear_Regression/'
file_name = 'Ecommerce_Customers.csv'

data = spark.read.csv(base_path + file_name, inferSchema=True, header=True)

'''
data.printSchema()
data.show()
'''

#print(data.columns)

assembler = VectorAssembler(inputCols=['Avg Session Length', 'Time on App', 'Time on Website', 'Length of Membership'], 
        outputCol='features')

output = assembler.transform(data)

#output.printSchema()

final_data = output.select('features', 'Yearly Amount Spent')
#final_data.show()

train_data, test_data = final_data.randomSplit([0.7, 0.3])

lr = LinearRegression(labelCol='Yearly Amount Spent')
lr_model = lr.fit(train_data)
test_result = lr_model.evaluate(test_data)

# Check the goodness of the fit
final_data.describe().show()
#test_results.residuals.show()

print(test_result.rootMeanSquaredError)

print(test_result.r2)

# Let's treat this as new data for which we don't have the values for the yearly amount spent
unlabeled_data = train_data.select('features')

predictions = lr_model.transform(unlabeled_data)

predictions.show()





