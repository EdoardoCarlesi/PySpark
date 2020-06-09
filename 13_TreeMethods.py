from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier, DecisionTreeClassifier
# pyspark.ml.regression also has RandomForest / Tree etc. methods for regression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


spark = SparkSession.builder.appName('Tree').getOrCreate()

base_path = '/home/edoardo/Udemy/PySpark/Python-and-Spark-for-Big-Data-master/Spark_for_Machine_Learning/Tree_Methods/'
file_name = 'sample_libsvm_data.txt'

data = spark.read.format('libsvm').load(base_path + file_name)
data.printSchema()
data.show()

train_data, test_data = data.randomSplit([0.7, 0.3])

# These classifiers are all initialized to their default values!
dtc = DecisionTreeClassifier()
gbt = GBTClassifier()
rfc = RandomForestClassifier(numTrees=100)

# Fit the models
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

# Evaluate the predictions
acc_eval = MulticlassClassificationEvaluator(metricName='accuracy')

print('DTC ACCURACY: ')
print(acc_eval.evaluate(dtc_preds))

print('GBT ACCURACY: ')
print(acc_eval.evaluate(gbt_preds))

print('RFC ACCURACY: ')
print(acc_eval.evaluate(rfc_preds))

# Get the feature importance 
print('RFC Feat Importance: ')
print(rfc_model.featureImportances)



