from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier, DecisionTreeClassifier
from pyspark.ml.regression import RandomForestRegressor, GBTRegressor, DecisionTreeRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, StringIndexer


spark = SparkSession.builder.appName('Tree').getOrCreate()

base_path = '/home/edoardo/Udemy/PySpark/Python-and-Spark-for-Big-Data-master/Spark_for_Machine_Learning/Tree_Methods/'
file_name = 'College.csv'

data = spark.read.csv(base_path + file_name, header=True, inferSchema=True)
#data = spark.read.format('libsvm').load(base_path + file_name)
data.printSchema()

cols = ['Apps', 'Accept', 'Enroll', 'Top10perc', 'Top25perc', 'F_Undergrad', 'P_Undergrad', 'Outstate', 'Room_Board', \
        'Books', 'Personal', 'PhD', 'Terminal', 'S_F_Ratio', 'perc_alumni', 'Expend', 'Grad_Rate']

assembler = VectorAssembler(inputCols=cols, outputCol='features')
output = assembler.transform(data)

# Fix the Private column
indexer = StringIndexer(inputCol='Private', outputCol='PrivateIndex')
output_fixed = indexer.fit(output).transform(output)

final_data = output_fixed.select('features', 'PrivateIndex')

train_data, test_data = final_data.randomSplit([0.7, 0.3])

dtc = DecisionTreeClassifier(labelCol='PrivateIndex', featuresCol='features')
gbt = GBTClassifier(labelCol='PrivateIndex', featuresCol='features')
rfc = RandomForestClassifier(labelCol='PrivateIndex', featuresCol='features', numTrees=150)

dtc_model = dtc.fit(train_data)
gbt_model = gbt.fit(train_data)
rfc_model = rfc.fit(train_data)

# Get the predictions
dtc_preds = dtc_model.transform(test_data)
gbt_preds = gbt_model.transform(test_data)
rfc_preds = rfc_model.transform(test_data)

# Show the predictions
dtc_preds.show()
gbt_preds.show()
rfc_preds.show()

# Evaluate the models
binary_eval = BinaryClassificationEvaluator(labelCol='PrivateIndex')

# GBT only outputs predictions, not the raw predictions, so we need to specifiy this in the BinaryClassificationEvaluator
binary_eval_gbt = BinaryClassificationEvaluator(labelCol='PrivateIndex', rawPredictionCol='prediction')

print('DTC: ')
print(binary_eval.evaluate(dtc_preds))
print('RFC: ')
print(binary_eval.evaluate(rfc_preds))
print('GBT: ')
print(binary_eval_gbt.evaluate(gbt_preds))

