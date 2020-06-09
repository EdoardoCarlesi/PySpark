from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import (VectorAssembler, StringIndexer,
                                VectorIndexer, OneHotEncoder)
from pyspark.ml.evaluation import (BinaryClassificationEvaluator,
                                    MulticlassClassificationEvaluator)
from pyspark.ml import Pipeline


spark = SparkSession.builder.appName('logReg').getOrCreate()

base_path = '/home/edoardo/Udemy/PySpark/Python-and-Spark-for-Big-Data-master/Spark_for_Machine_Learning/Logistic_Regression/'
file_name = 'titanic.csv'

data = spark.read.csv(base_path + file_name, header=True, inferSchema=True)
data.show()
data.printSchema()
#print(data.columns)

cols = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

data_cols = data.select(cols)
data_cols.show()
final_data = data_cols.na.drop()

# Transform the categorical columns into numbers
gender_indexer = StringIndexer(inputCol = 'Sex', outputCol = 'SexIndex')

# A B C
# 0 1 2 
# One hot encode ----> this is mapping everyting into [1, 0, 0] [0, 1, 0] etc.
gender_encoder = OneHotEncoder(inputCol ='SexIndex', outputCol = 'SexVec')   # ---> each entry will be converted to a vector A = [1, 0] B = [0, 1]

embark_indexer = StringIndexer(inputCol = 'Embarked', outputCol = 'EmbarkedIndex')
embark_encoder = OneHotEncoder(inputCol = 'EmbarkedIndex', outputCol = 'EmbarkedVec')   # ---> each entry will be converted to a vector A = [1, 0] B = [0, 1]

new_cols = ['Pclass', 'SexVec', 'Age', 'SibSp', 'Parch', 'Fare', 'EmbarkedVec']
assembler = VectorAssembler(inputCols = new_cols, outputCol = 'features')

logreg_titanic = LogisticRegression(featuresCol = 'features', labelCol = 'Survived')

pipeline = Pipeline(stages = [gender_indexer, embark_indexer, gender_encoder, embark_encoder, assembler, logreg_titanic])

train_data, test_data = final_data.randomSplit([0.7, 0.3])

fit_model = pipeline.fit(train_data)

results = fit_model.transform(test_data)
evaluate = BinaryClassificationEvaluator(rawPredictionCol='prediction',
                                       labelCol='Survived')

results.select(['Survived', 'prediction']).show()

acc = evaluate.evaluate(results)
print(acc)

'''
'''


