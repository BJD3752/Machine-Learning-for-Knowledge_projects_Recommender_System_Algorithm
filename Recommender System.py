# Databricks notebook source
# DBTITLE 1,Recommender System with a chunk of movielens_dataset
import pyspark,pyspark.sql
from pyspark.sql import SparkSession

# COMMAND ----------

spark = SparkSession.builder.appName('Movie_recomder').getOrCreate()

# COMMAND ----------

md = spark.read.csv("/FileStore/tables/movielens_ratings.csv",inferSchema ='True', header='True')

# COMMAND ----------

md.tail(20)

# COMMAND ----------

from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

# COMMAND ----------

md.describe().show()

# COMMAND ----------

# train, test data with 80:20 split
training,test = md.randomSplit([0.8,0.2])

# COMMAND ----------

als = ALS(maxIter=5,regParam =0.01,userCol='userId',itemCol='movieId',ratingCol='rating')

# COMMAND ----------

models = als.fit(training)

# COMMAND ----------

predictions = models.transform(test)

# COMMAND ----------

#to improve the preduction ratings we need work with more data the dataset has only 1501 datapoints which is very less for recomender system. 
predictions.show()

# COMMAND ----------

evaluator = RegressionEvaluator(metricName='rmse',labelCol='rating',predictionCol='prediction')

# COMMAND ----------

#rsme= Root mean squre error
rsme = evaluator.evaluate(predictions)

# COMMAND ----------

print('RMSE: '+str(rsme))

# COMMAND ----------

#for a sigle user how to predict the new movie he like or not:
single_user = test.filter(test['userId']==11).select(['movieId','userId'])

# COMMAND ----------

single_user.show()

# COMMAND ----------

recommendations = models.transform(single_user)

# COMMAND ----------

recommendations.orderBy('prediction', ascending=False).show()

# COMMAND ----------

#in above we can say user 11 like movieID 36 as the prediction is +ve score i.e +2.94 and do not like movie id 77 with prediction -ve score i.e -3.08 prediction for liking(+ve) or not linking(-ve) according to previous behaviour
# all the above compution based on the data given it might go wrong for verious parametes
