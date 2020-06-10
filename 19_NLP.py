from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, RegexTokenizer
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType 
from pyspark.ml.feature import StopWordsRemover, NGram

# TF: Term Frequency ---> importance of the therm within a given document
# IDF: importance of the term in the corpus (full dictionary of words and documents)

spark = SparkSession.builder.appName('NLP').getOrCreate()

sen_df = spark.createDataFrame([(0, 'Hi I heard about Spark'), (1, 'I whish java could use case classes'), (2, 'Logistic,regression,models,are,neat')], ['id', 'sentence'])

sen_df.show()

tokenizer = Tokenizer(inputCol = 'sentence', outputCol = 'words')
regex_tokenizer = RegexTokenizer(inputCol = 'sentence', outputCol = 'words', pattern = '\\W')

count_tokens = udf(lambda words: len(words), IntegerType())

tokenized = tokenizer.transform(sen_df)
tokenized.withColumn('tokens', count_tokens(col("words"))).show()

# Remvove commas INSIDE the words
rg_tokenized = regex_tokenizer.transform(sen_df)
rg_tokenized.withColumn('tokens', count_tokens(col('words'))).show()

# Remove stop words
sentence_df = spark.createDataFrame([(0, ['I', 'saw', 'the', 'green', 'horse']), (1, ['Mary', 'had', 'a', 'little', 'lamb'])], ['id', 'tokens'])
sentence_df.show()

remover = StopWordsRemover(inputCol = 'tokens', outputCol = 'filtered')
remover.transform(sentence_df).show()

# n-gram
wordDataFrame = spark.createDataFrame([
        (0, ["Hi", "I", "heard", "about", "Spark"]),
            (1, ["I", "wish", "Java", "could", "use", "case", "classes"]),
                (2, ["Logistic", "regression", "models", "are", "neat"])
                ], ["id", "words"])

ngram = NGram(n=2, inputCol='words', outputCol='grams')
ngram.transform(wordDataFrame).select('grams').show(truncate = False)


