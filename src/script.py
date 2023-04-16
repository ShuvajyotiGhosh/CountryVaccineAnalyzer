from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from pyspark.sql.functions import col, count, lit, round, sum, DataFrame
import os
import sys
from typing import Optional

os.environ['HADOOP_HOME'] = "<Mention your path to winutils>"
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

conf = SparkConf() \
    .set("spark.driver.memory", "4g") \
    .set("spark.executor.memory", "4g") \
    .set("spark.jars.packages", "com.crealytics:spark-excel_2.12:0.13.7")
# create a SparkSession
spark = SparkSession.builder.appName("Data_Collation").config(conf=conf).getOrCreate()

# Assumed population from each country
total_population = {'USA': 331.4, 'INDIA': 1366.4, 'AUSTRALIA': 25.4}

# input paths
australia_xlsx_path_input = "resources/input/AUS.xlsx"
india_csv_path_input = "resources/input/IND.csv"
usa_csv_path_input = "resources/input/USA.csv"

# output paths
metrics_1_csv_path_output = "../src/resources/output/metrics_1/"
metrics_2_csv_path_output = "../src/resources/output/metrics_2/"
metrics_3_csv_path_output = "../src/resources/output/metrics_3/"


class VaccinationAnalyzer:

    def __init__(self):
        pass

    # Convert the total population to a DataFrame
    @staticmethod
    def population_df_creator() -> DataFrame:
        population = spark.createDataFrame([(k, v) for k, v in total_population.items()],
                                           ['CountryName', 'TotalPopulation'])
        return population

    # read the xlsx file as a DataFrame
    @staticmethod
    def xlsx_df_reader(path) -> Optional[DataFrame]:
        try:
            df_read = spark.read.format("com.crealytics.spark.excel") \
                .option("header", "true") \
                .option("inferSchema", "true") \
                .load(path)
            return df_read
        except Exception as e:
            print("Error reading xlsx file: ", e)
            df_read = None  # set aus_df to None in case of error
            return df_read

    # read the CSV file as a DataFrame
    @staticmethod
    def csv_df_reader(path) -> Optional[DataFrame]:
        try:
            df_read = spark.read.format("csv") \
                .option("header", "true") \
                .option("inferSchema", "true") \
                .load(path)
            return df_read
        except Exception as e:
            print("Error reading csv file: ", e)
            df_read = None  # set aus_df to None in case of error
            return df_read

    # created combined df by combining all the dfs cleaning them and union them
    @staticmethod
    def combined_df_creator(aus_df, ind_df, usa_df) -> Optional[DataFrame]:
        if aus_df is not None and ind_df is not None and usa_df is not None:
            # rename the columns of the AUS DataFrame
            aus_df = aus_df.selectExpr("`Unique ID` as ID", "`Patient Name` as Name",
                                       "`Vaccine Type` as VaccinationType")
            aus_df_cleaned = aus_df.withColumn("CountryName", lit("AUSTRALIA"))
            # select the required columns from the IND DataFrame
            ind_df = ind_df.select("ID", "Name", "VaccinationType")
            ind_df_cleaned = ind_df.withColumn("CountryName", lit("INDIA"))
            # select the required columns from the USA DataFrame
            usa_df = usa_df.select("ID", "Name", "VaccinationType")
            usa_df_cleaned = usa_df.withColumn("CountryName", lit("USA"))
            try:
                combined_df = aus_df_cleaned.union(ind_df_cleaned).union(usa_df_cleaned)
                return combined_df
            except Exception as e:
                print("Error while creating combined df")
                combined_df = None  # set combined_df to None in case of error
                return combined_df

    # Metrics 1 Dataframe creator
    @staticmethod
    def metric_1_calculator(combined_dataframe) -> DataFrame:
        # metrics 1 calculation starting
        metric_1 = combined_dataframe.groupBy("CountryName", "VaccinationType").agg(
            count("*").alias("No. of vaccinations"))
        return metric_1

    # Metrics 2 Dataframe creator
    @staticmethod
    def metric_2_calculator(combined_dataframe, population_dataframe) -> DataFrame:
        total_vaccinated_per_country = combined_dataframe.groupBy(['CountryName']).agg(
            count('ID').alias('TotalVaccinatedPerCountry')).join(population_dataframe, 'CountryName')

        metric_2 = total_vaccinated_per_country.withColumn('% Vaccinated', round(
            (total_vaccinated_per_country['TotalVaccinatedPerCountry'] / (
                    total_vaccinated_per_country['TotalPopulation'])) * 100, 4)).drop(
            col('TotalVaccinatedPerCountry')).drop('TotalPopulation')  ## Keeping result in 4 decimal

        return metric_2

    # Metrics 3 Dataframe creator
    @staticmethod
    def metric_3_calculator(combined_dataframe, population_df) -> DataFrame:
        total_vaccinated_per_country = combined_dataframe.groupBy(['CountryName']).agg(
            count('ID').alias('TotalVaccinatedPerCountry')).join(population_df, 'CountryName')
        total_vaccinated = combined_dataframe.groupBy(['CountryName']).agg(
            count('ID').alias('TotalVaccinatedPerCountry')).agg(
            sum('TotalVaccinatedPerCountry').alias('TotalVaccinated')).collect()[0][0]
        total_vaccinated_per_country_with_total_vaccination = total_vaccinated_per_country.withColumn('TotalVaccinated',
                                                                                                      lit(total_vaccinated))
        metrics_3 = total_vaccinated_per_country_with_total_vaccination.withColumn('% Contribution', round((
                                                                                                                   total_vaccinated_per_country_with_total_vaccination[
                                                                                                                       'TotalVaccinatedPerCountry'] /
                                                                                                                   total_vaccinated_per_country_with_total_vaccination[
                                                                                                                       'TotalVaccinated']) * 100)).drop(
            "TotalVaccinatedPerCountry").drop("TotalPopulation").drop('TotalVaccinated')
        return metrics_3

    # Writing the cleaned metrics as csv file
    @staticmethod
    def metrics_writer(dataframe: DataFrame, path) -> None:
        dataframe.write.mode('overwrite').option('header', 'true').csv(path)


if __name__ == '__main__':
    vaccination_analyzer = VaccinationAnalyzer()
    population_df = vaccination_analyzer.population_df_creator()
    australia_df = vaccination_analyzer.xlsx_df_reader(australia_xlsx_path_input)
    india_df = vaccination_analyzer.csv_df_reader(india_csv_path_input)
    usa_df = vaccination_analyzer.csv_df_reader(usa_csv_path_input)
    combined_df = vaccination_analyzer.combined_df_creator(aus_df=australia_df, ind_df=india_df, usa_df=usa_df)
    if combined_df is not None:
        metric_1_df = vaccination_analyzer.metric_1_calculator(combined_dataframe=combined_df)
        metric_2_df = vaccination_analyzer.metric_2_calculator(combined_dataframe=combined_df,
                                                               population_dataframe=population_df)
        metric_3_df = vaccination_analyzer.metric_3_calculator(combined_dataframe=combined_df,
                                                               population_df=population_df)
        vaccination_analyzer.metrics_writer(metric_1_df, metrics_1_csv_path_output)
        vaccination_analyzer.metrics_writer(metric_2_df, metrics_2_csv_path_output)
        vaccination_analyzer.metrics_writer(metric_3_df, metrics_3_csv_path_output)
