from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# 初始化 Spark 会话
spark = SparkSession.builder \
    .appName("CityBalanceAnalysis") \
    .getOrCreate()

# 加载 user_balance_table.csv 文件到 DataFrame
balance_df = spark.read.option("header", "true").csv("/home/hadoop/Downloads/Purchase_Redemption_Data/user_balance_table.csv")

# 加载 user_profile_table.csv 文件到 DataFrame
profile_df = spark.read.option("header", "true").csv("/home/hadoop/Downloads/Purchase_Redemption_Data/user_profile_table.csv")

# 数据类型转换
balance_df = balance_df.withColumn("tBalance", balance_df["tBalance"].cast("float")) \
                       .withColumn("user_id", balance_df["user_id"].cast("string"))

profile_df = profile_df.withColumn("user_id", profile_df["user_id"].cast("string")) \
                       .withColumn("city", profile_df["city"].cast("string"))

# 注册临时视图
balance_df.createOrReplaceTempView("user_balance")
profile_df.createOrReplaceTempView("user_profile")

# SQL 查询：根据 user_id 连接两个表，计算2014年3月1日的用户平均余额
query = """
SELECT p.city, AVG(b.tBalance) AS avg_balance
FROM user_balance b
JOIN user_profile p ON b.user_id = p.user_id
WHERE b.report_date = '20140301'
GROUP BY p.city
ORDER BY avg_balance DESC
"""

# 执行 SQL 查询并展示结果
result = spark.sql(query)
result.show()

# 停止 Spark 会话
spark.stop()
