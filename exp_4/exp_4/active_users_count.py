from pyspark import SparkContext, SparkConf

# 设置Spark应用程序配置
conf = SparkConf().setAppName("ActiveUserAnalysis")
sc = SparkContext(conf=conf)

# HDFS上的文件路径
file_path = "/home/hadoop/Downloads/Purchase_Redemption_Data/user_balance_table.csv"

# 读取文件，返回一个RDD
data = sc.textFile(file_path)

# 获取第一行作为表头并去除
header = data.first()
data = data.filter(lambda line: line != header)

# 提取用户ID和日期，并解析日期
def parse_line(line):
    parts = line.split(",")
    user_id = parts[0]  # 用户ID在第一列
    date = parts[1]  # 日期在第二列
    return (user_id, date)

# 解析数据
parsed_data = data.map(parse_line)

# 筛选出2014年8月的记录，并为每个用户创建一个包含其活动日期的集合
def filter_august(date):
    return date.startswith("201408")

# 过滤出2014年8月的数据
august_data = parsed_data.filter(lambda x: filter_august(x[1]))

# 为每个用户收集他们在2014年8月的活动日期
user_dates = august_data.map(lambda x: (x[0], x[1])) \
                        .distinct()  # 使用distinct保证每个用户每个日期只统计一次

# 对每个用户，统计他们在2014年8月出现的不同日期数量
user_active_days = user_dates.mapValues(lambda x: 1) \
                             .reduceByKey(lambda a, b: a + b)

# 筛选出活跃用户（在8月有至少5天记录）
active_users = user_active_days.filter(lambda x: x[1] >= 5)

# 统计活跃用户的总数
active_user_count = active_users.count()

# 输出结果
print(f"2014年8月的活跃用户总数：{active_user_count}")

# 停止SparkContext
sc.stop()
