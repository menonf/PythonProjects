import os
import sys
import common_functions
from pyspark.sql import SparkSession

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

test1 = common_functions.get_max([456, 700, 700.24343, 200, 200])
test2 = common_functions.get_min([456, 700, 700.24343, 200, 200])
testst3 = common_functions.weighted_average([0.5, 0.3, 0.2], [20000, 30000, 40000])

spark = SparkSession.builder.appName('unit_test').getOrCreate()

factor_col = ["Cycle","Factors","Daily","Monthly","Annual"]
factor_data = [("Cyclical","Aerospace", 9,10,8), ("Cyclical","Aerospace", 7,5,2),("Cyclical","Aerospace", 8,6,3),
                ("Cyclical","Banks", 7,5,3),("Cyclical","Banks", 7,4,9),("Cyclical","Banks", 7,0,2),
                ("Non-Cyclical","Biotech", 6,12,8),("Non-Cyclical","Biotech", 7,2,6),("Non-Cyclical","Biotech", 8,1,0),
                ("Non-Cyclical","Chemicals", 8,5,3),("Non-Cyclical","Chemicals", 7,6,2),("Non-Cyclical","Chemicals", 6,4,0)]


factor_table = spark.createDataFrame(data=factor_data, schema = factor_col)
factor_table = factor_table.withColumn("rank_factors_daily_asc",  common_functions.crisp_rank(partition_by=["Factors"], fields_order=["Daily"], sort_direction=["asc"]))
factor_table.show()
