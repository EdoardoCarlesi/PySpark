from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import (VectorAssembler, StringIndexer,
                                VectorIndexer, OneHotEncoder)
from pyspark.ml.evaluation import (BinaryClassificationEvaluator,
                                    MulticlassClassificationEvaluator)
from pyspark.ml import Pipeline


spark = SparkSession.builder.appName('logReg').getOrCreate()

base_path = '/home/edoardo/Udemy/PySpark/Python-and-Spark-for-Big-Data-master/Spark_for_Machine_Learning/Logistic_Regression/'
file_name = 'customer_churn.csv'
file_name_new = 'new_customers.csv'

data = spark.read.csv(base_path + file_name, header=True, inferSchema=True)
data_new = spark.read.csv(base_path + file_name_new, header=True, inferSchema=True)

data.printSchema()
#print(data.columns)

#cols = ['Age', 'Total_Purchase', 'Years', 'Num_Sites', 'Onboard_date', 'Churn']
cols = ['Age', 'Total_Purchase', 'Years', 'Num_Sites', 'Churn']
new_cols = ['Age', 'Total_Purchase', 'Years', 'Num_Sites']

data_cols = data.select(cols)
data_cols.show()
final_data = data_cols.na.drop()

assembler = VectorAssembler(inputCols = new_cols, outputCol = 'features')
logreg_churn = LogisticRegression(featuresCol = 'features', labelCol = 'Churn')
pipeline = Pipeline(stages = [assembler, logreg_churn])
train_data, test_data = final_data.randomSplit([0.7, 0.3])

fit_model = pipeline.fit(train_data)

results = fit_model.transform(test_data)

# We just plug in all the data, no need to edit anything
results_new = fit_model.transform(data_new)
evaluate = BinaryClassificationEvaluator(rawPredictionCol='prediction', labelCol='Churn')

# Evaluate the predicted value vs. the real one
#results.select(['Churn', 'prediction']).show()
acc = evaluate.evaluate(results)
#print(acc)

# In the new results the Churn value is unknown - here we predict it
results_new.select(['Company', 'prediction']).show()



