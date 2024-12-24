from pyspark import SparkContext, SparkConf
from datetime import datetime
import csv
# 初始化 Spark 配置和上下文
conf = SparkConf().setAppName("UserBalanceAnalysis")
sc = SparkContext(conf=conf)

# 读取 CSV 文件
file_path = "/home/hadoop/Downloads/Purchase_Redemption_Data/user_balance_table.csv"
data = sc.textFile(file_path) 

# 去除 CSV 文件头（假设文件有头行）
header = data.first()
data = data.filter(lambda line: line != header)

# 处理数据：解析每一行，提取日期、total_purchase_amt 和 total_redeem_amt
def parse_line(line):
    parts = line.split(",")
    
    # CSV 文件中第 5 列为 total_purchase_amt，第 9 列为 total_redeem_amt
    date = parts[1]  # 日期在第2列
    total_purchase_amt = float(parts[4]) if parts[4] else 0.0
    total_redeem_amt = float(parts[8]) if parts[8] else 0.0
    
    return (date, total_purchase_amt, total_redeem_amt)

# 解析数据
parsed_data = data.map(parse_line)

# 按日期聚合资金流入和流出
def aggregate_by_date(accumulated, current):
    in_amt, out_amt = current  # 解包 amt 元组
    accumulated = list(accumulated)
    accumulated[0] += in_amt  # 累加流入金额
    accumulated[1] += out_amt  # 累加流出金额
    return accumulated
# 转换数据格式为（日期，(流入总额, 流出总额)）
aggregated_data = parsed_data.map(lambda x: (x[0], (x[1], x[2]))) .reduceByKey(aggregate_by_date)

# 输出结果
result = aggregated_data.collect()

# 按日期排序
sorted_result = sorted(result, key=lambda x: datetime.strptime(x[0], "%Y%m%d"))

# 打印结果
for date, (total_purchase, total_redeem) in sorted_result:
    print(f"Date: {date}, Total Purchase Amount: {total_purchase}, Total Redeem Amount: {total_redeem}")

# 停止 SparkContext
sc.stop()
with open('flow_count.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Date", "Purchase", "Redeem"])  # 写入表头
    
    for date, (total_purchase, total_redeem) in sorted_result:
        writer.writerow([date, total_purchase, total_redeem])