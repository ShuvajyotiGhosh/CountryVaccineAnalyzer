import unittest

from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, TimestampType, LongType

from src.script import VaccinationAnalyzer


class TestVaccinationAnalyzer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder.appName("test").config(conf=SparkConf() \
                                                                .set("spark.driver.memory", "4g") \
                                                                .set("spark.executor.memory", "4g") \
                                                                .set("spark.jars.packages",
                                                                     "com.crealytics:spark-excel_2.12:0.13.7")).getOrCreate()
        cls.vaccine_analyzer = VaccinationAnalyzer()
        cls.combined_df = None

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_population_df_creator(self):
        population_dict = [("USA", 331.4), ("INDIA", 1366.4), ("AUSTRALIA", 25.4)]

        schema = StructType([
            StructField("CountryName", StringType(), True),
            StructField("TotalPopulation", DoubleType(), True)
        ])
        expected_df = self.spark.createDataFrame(population_dict, schema)
        self.assertTrue(self.vaccine_analyzer.population_df_creator().collect() == expected_df.collect())

    def test_xlsx_df_reader(self):
        actual_data = self.vaccine_analyzer.xlsx_df_reader("../test/resources/input/AUS.xlsx")
        self.assertTrue(actual_data.collect()[0][0] == 1.0)
        self.assertTrue(actual_data.collect()[0][1] == "Mike")
        self.assertTrue(actual_data.collect()[0][2] == "LMN")
        self.assertTrue(actual_data.collect()[0][4] == "5/11/22")

    def test_csv_df_reader(self):
        actual_data_csv = self.vaccine_analyzer.csv_df_reader("../test/resources/input/USA.csv")
        self.assertTrue(actual_data_csv.collect()[0][0] == 1)
        self.assertTrue(actual_data_csv.collect()[0][1] == "Sam")
        self.assertTrue(actual_data_csv.collect()[0][2] == "EFG")

    def test_combined_df_creator(self):
        aus_df = self.spark.read.format("com.crealytics.spark.excel") \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .load("../test/resources/input/AUS.xlsx")
        ind_df = self.spark.read.format("csv") \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .load("../test/resources/input/IND.csv")
        usa_df = self.spark.read.format("csv") \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .load("../test/resources/input/USA.csv")

        combined_df = self.vaccine_analyzer.combined_df_creator(aus_df, ind_df, usa_df)
        self.assertTrue((aus_df.count() + ind_df.count() + usa_df.count()) == combined_df.count())

    def test_schema_metrics_1(self):
        aus_df = self.spark.read.format("com.crealytics.spark.excel") \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .load("../test/resources/input/AUS.xlsx")
        ind_df = self.spark.read.format("csv") \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .load("../test/resources/input/IND.csv")
        usa_df = self.spark.read.format("csv") \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .load("../test/resources/input/USA.csv")

        combined_df = self.vaccine_analyzer.combined_df_creator(aus_df, ind_df, usa_df)
        metrics_1_df = self.vaccine_analyzer.metric_1_calculator(combined_df)
        expected_schema = StructType([
            StructField("CountryName", StringType(), nullable=False),
            StructField("VaccinationType", StringType(), nullable=True),
            StructField("No. of vaccinations", LongType(), nullable=False)
        ])
        self.assertTrue(expected_schema == metrics_1_df.schema)

    def test_schema_metrics_2(self):
        aus_df = self.spark.read.format("com.crealytics.spark.excel") \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .load("../test/resources/input/AUS.xlsx")
        ind_df = self.spark.read.format("csv") \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .load("../test/resources/input/IND.csv")
        usa_df = self.spark.read.format("csv") \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .load("../test/resources/input/USA.csv")

        combined_df = self.vaccine_analyzer.combined_df_creator(aus_df, ind_df, usa_df)
        metrics_2_df = self.vaccine_analyzer.metric_2_calculator(combined_df,
                                                                 self.vaccine_analyzer.population_df_creator())
        expected_schema = StructType(
            [StructField('CountryName', StringType(), False), StructField('% Vaccinated', DoubleType(), True)])
        self.assertTrue(expected_schema == metrics_2_df.schema)

    def test_schema_metrics_3(self):
        aus_df = self.spark.read.format("com.crealytics.spark.excel") \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .load("../test/resources/input/AUS.xlsx")
        ind_df = self.spark.read.format("csv") \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .load("../test/resources/input/IND.csv")
        usa_df = self.spark.read.format("csv") \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .load("../test/resources/input/USA.csv")

        combined_df_for_metrics_3 = self.vaccine_analyzer.combined_df_creator(aus_df, ind_df, usa_df)
        metrics_3_df = self.vaccine_analyzer.metric_3_calculator(combined_df_for_metrics_3,self.vaccine_analyzer.population_df_creator())
        expected_schema = StructType([StructField('CountryName', StringType(), False), StructField('% Contribution', DoubleType(), True)])
        self.assertTrue(expected_schema == metrics_3_df.schema)


if __name__ == '__main__':
    unittest.main()
