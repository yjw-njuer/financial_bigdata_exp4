from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as _sum, row_number,to_date
from pyspark.sql.window import Window

# 创建SparkSession
spark = SparkSession.builder \
    .appName("CityUserRanking") \
    .getOrCreate()

# 加载 CSV 文件
user_profile_df = spark.read.option("header", "true").csv("/home/hadoop/Downloads/Purchase_Redemption_Data/user_profile_table.csv")
user_balance_df = spark.read.option("header", "true").csv("/home/hadoop/Downloads/Purchase_Redemption_Data/user_balance_table.csv")
#user_balance_df = user_balance_df.withColumn("report_date", to_date(user_balance_df["report_date"], "yyyy-MM-dd"))

# 筛选出2014年8月的数据
#user_balance_df = user_balance_df.filter((col("report_date") >= "20140801") & (col("report_date") <= "20140831"))

# 数据类型转换
user_balance_df = user_balance_df.withColumn("total_purchase_amt", user_balance_df["total_purchase_amt"].cast("double")) \
                                 .withColumn("total_redeem_amt", user_balance_df["total_redeem_amt"].cast("double"))
user_profile_df = user_profile_df.withColumn("user_id", user_profile_df["user_id"].cast("int"))

user_balance_df.createOrReplaceTempView("user_balance")
user_profile_df.createOrReplaceTempView("user_profile")
query = """
    SELECT up.city, ub.user_id, 
           (SUM(ub.total_purchase_amt) + SUM(ub.total_redeem_amt)) AS total_flow
    FROM user_profile AS up
    LEFT JOIN user_balance AS ub
    ON up.user_id = ub.user_id
    WHERE ub.report_date BETWEEN '20140801' AND '20140831'
    GROUP BY up.city, ub.user_id ORDER BY total_flow DESC 
"""

# 执行SQL查询
result_df = spark.sql(query)

# 使用窗口函数进行排名
window_spec = Window.partitionBy("city").orderBy(col("total_flow").desc())

# 为每个城市的用户添加排名
ranked_df = result_df.withColumn("rank", row_number().over(window_spec))

# 筛选出每个城市总流量排名前三的用户
top_users_df = ranked_df.filter(col("rank") <= 3)

# 选择需要的列并显示结果
top_users_df.select("city", "user_id", "total_flow", "rank").show(30)

# 停止 SparkSession
spark.stop()
