from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql.functions import col, dayofweek, sum, lag, lit, to_date
from pyspark.sql.functions import col, to_date, year, month, dayofmonth
from pyspark.sql.window import Window
from datetime import datetime
from datetime import timedelta
import pandas as pd

# 创建SparkSession
spark = SparkSession.builder \
    .appName("RandomForestRegressor_Prediction_Model") \
    .getOrCreate()

# 读取user_balance_table,加载为dataframe
# 取表头，自动推断每列的数据类型
user_balance_df = spark.read.csv("/home/hadoop/Downloads/Purchase_Redemption_Data/user_balance_table.csv", header=True, inferSchema=True)

# 按日期汇总购买和赎回金额,返回新的dataframe
daily_summary_df = user_balance_df.groupBy("report_date") \
    .agg(
        sum("total_purchase_amt").alias("total_purchase_sum"),
        sum("total_redeem_amt").alias("total_redeem_sum")
    )

# 添加日期列
daily_summary_df = daily_summary_df.withColumn(
    "formatted_date",
    to_date(col("report_date").cast("string"), "yyyyMMdd")
)
# 添加年月日
daily_summary_df = daily_summary_df.withColumn("year", year("formatted_date")) \
                 .withColumn("month", month("formatted_date")) \
                 .withColumn("day", dayofmonth("formatted_date"))
# 添加星期几特征
daily_summary_df = daily_summary_df.withColumn(
    "weekday", 
    dayofweek("formatted_date")
)

# 添加滞后特征
window_spec = Window.orderBy("formatted_date")
for i in range(1, 30):
    daily_summary_df = daily_summary_df \
        .withColumn(f"prev_purchase_{i}", lag("total_purchase_sum", i).over(window_spec)) \
        .withColumn(f"prev_redeem_{i}", lag("total_redeem_sum", i).over(window_spec))

# 填充缺失值
daily_summary_df = daily_summary_df.na.fill(0)
#以上都是特征数据准备

# 特征列
feature_columns = ["weekday","year","month","day"] + [f"prev_purchase_{i}" for i in range(1, 30)] + [f"prev_redeem_{i}" for i in range(1, 30)]

# 创建特征向量,feature_columns 是一个包含特征列名的列表,VectorAssembler 会将这些列的数据合并成一个向量,outputCol表示输出的合并后的向量列的名称
vector_assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

# 准备数据
prepared_data_df = daily_summary_df.select(["report_date"] + feature_columns + ["total_purchase_sum", "total_redeem_sum"])

# 特征转换
vectorized_data_df = vector_assembler.transform(prepared_data_df)
purchase_train_data = vectorized_data_df.select("report_date", "features", col("total_purchase_sum"))
redeem_train_data = vectorized_data_df.select("report_date", "features", col("total_redeem_sum"))

# 训练购买量预测模型
purchase_model_rf = RandomForestRegressor(featuresCol="features", labelCol="total_purchase_sum", numTrees=100)
purchase_model_rf = purchase_model_rf.fit(purchase_train_data)

# 训练赎回量预测模型
redeem_model_rf = RandomForestRegressor(featuresCol="features", labelCol="total_redeem_sum", numTrees=100)
redeem_model_rf = redeem_model_rf.fit(redeem_train_data)

# 创建201409预测DataFrame
future_dates = [(datetime(2014, 9, 1) +timedelta(days=x)).strftime("%Y%m%d") for x in range(30)]
prediction_dates_df = spark.createDataFrame([(date,) for date in future_dates], ["report_date"])

# 以下是构建预测集数据
# 添加日期列,年月日和星期几特征
prediction_dates_df = prediction_dates_df.withColumn(
    "formatted_date",
    to_date(col("report_date").cast("string"), "yyyyMMdd")
).withColumn("year", year("formatted_date")) \
                 .withColumn("month", month("formatted_date")) \
                 .withColumn("day", dayofmonth("formatted_date")).withColumn(
    "weekday",
    dayofweek("formatted_date")
)

last_week_data_list = daily_summary_df.orderBy(col("formatted_date").desc()).limit(29) \
                                  .select("total_purchase_sum", "total_redeem_sum") \
                                  .collect()

# 填充滞后特征
for i in range(1, 30):
    prediction_dates_df = prediction_dates_df.withColumn(f"prev_purchase_{i}", lit(float(last_week_data_list[i-1]["total_purchase_sum"])))
    prediction_dates_df = prediction_dates_df.withColumn(f"prev_redeem_{i}", lit(float(last_week_data_list[i-1]["total_redeem_sum"])))

# 填充缺失值
prediction_dates_df = prediction_dates_df.na.fill(0)

# 向量化特征
vector_assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
pred_vectorized_df = vector_assembler.transform(prediction_dates_df)

# 生成预测
purchase_preds_df = purchase_model_rf.transform(pred_vectorized_df)
redeem_preds_df = redeem_model_rf.transform(pred_vectorized_df)

# 合并预测结果
final_predictions_df = purchase_preds_df.select(col("report_date"),col("prediction").alias("predicted_purchase_amt"))\
.join(
    redeem_preds_df.select(col("report_date"),
        col("prediction").alias("predicted_redeem_amt")
    ),
    "report_date"
)

# 转换为Pandas DataFrame并格式化
pandas_predictions_df = final_predictions_df.toPandas()
pandas_predictions_df['report_date'] = pandas_predictions_df['report_date'].astype(str)

# 精度为分,四舍五入并保存为整数
pandas_predictions_df['predicted_purchase_amt'] = pandas_predictions_df['predicted_purchase_amt'].round().astype(int)
pandas_predictions_df['predicted_redeem_amt'] = pandas_predictions_df['predicted_redeem_amt'].round().astype(int)

# 保存预测结果到CSV
final_predictions_df = pandas_predictions_df[['report_date', 'predicted_purchase_amt', 'predicted_redeem_amt']]
final_predictions_df.to_csv('tc_comp_predict_table.csv', index=False)

# 停止SparkSession
spark.stop()
