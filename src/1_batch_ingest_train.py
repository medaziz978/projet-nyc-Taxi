from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, dayofweek, unix_timestamp, avg, count
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
import sys
import os

# Configuration PostgreSQL
DB_URL = "jdbc:postgresql://postgres:5432/nyc_data"
DB_PROPERTIES = {
    "user": "admin",
    "password": "password",
    "driver": "org.postgresql.Driver"
}

def get_spark_session(app_name="NYC_Taxi_Batch_Analysis"):
    """Initialise la session Spark avec le driver JDBC."""
    return SparkSession.builder \
        .appName(app_name) \
        .config("spark.jars", "/opt/spark/jars/postgresql-42.6.0.jar") \
        .getOrCreate()

def clean_data(df):
    """Nettoie les données : filtre les valeurs nulles et aberrantes."""
    print("Nettoyage des données...")
    
    # Conversion des timestamps s'ils sont chargés comme strings (dépend du schéma)
    # On suppose ici qu'ils sont détectés ou on les cast
    df = df.withColumn("tpep_pickup_datetime", col("tpep_pickup_datetime").cast("timestamp")) \
           .withColumn("tpep_dropoff_datetime", col("tpep_dropoff_datetime").cast("timestamp")) \
           .withColumn("trip_distance", col("trip_distance").cast("double")) \
           .withColumn("total_amount", col("total_amount").cast("double")) \
           .withColumn("PULocationID", col("PULocationID").cast("integer")) \
           .withColumn("passenger_count", col("passenger_count").cast("integer"))

    # Calcul de la durée en minutes
    df = df.withColumn("duration_min", 
        (unix_timestamp("tpep_dropoff_datetime") - unix_timestamp("tpep_pickup_datetime")) / 60
    )

    # Filtrage des outliers
    # Distance > 0, Prix > 0, Durée entre 1 min et 4 heures
    cleaned_df = df.filter(
        (col("trip_distance") > 0) & 
        (col("trip_distance") < 100) &  # Exclure distances absurdes
        (col("total_amount") > 0) & 
        (col("total_amount") < 500) &   # Exclure montants absurdes
        (col("duration_min") >= 1) & 
        (col("duration_min") <= 240)
    ).dropna()
    
    return cleaned_df

def write_to_postgres(df, table_name, mode="overwrite"):
    """Écrit le DataFrame dans PostgreSQL."""
    print(f"Écriture dans la table {table_name}...")
    df.write.jdbc(url=DB_URL, table=table_name, mode=mode, properties=DB_PROPERTIES)

def main():
    spark = get_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    # Chemin des données
    input_path = "/app/data/*.csv" 
    
    print(f"Chargement des données depuis {input_path}...")
    try:
        # header=True, inferSchema=True pour faciliter, mais attention à la performance sur gros volumes
        raw_df = spark.read.csv(input_path, header=True, inferSchema=True)
    except Exception as e:
        print(f"Erreur lors du chargement des données : {e}")
        # Si pas de données, on crée un DF vide pour tester la structure ou on quitte
        sys.exit(1)

    # 1. Nettoyage
    cleaned_df = clean_data(raw_df)
    cleaned_df.cache()
    
    print(f"Nombre de lignes après nettoyage : {cleaned_df.count()}")

    # 2. Analytics SQL & Export vers Postgres
    
    # KPI 1 : Volume de courses par heure
    stats_hourly = cleaned_df.withColumn("hour", hour("tpep_pickup_datetime")) \
        .groupBy("hour") \
        .agg(
            count("*").alias("trip_count"),
            avg("total_amount").alias("avg_price"),
            avg("trip_distance").alias("avg_distance")
        ).orderBy("hour")
    
    write_to_postgres(stats_hourly, "kpi_hourly_stats")

    # KPI 2 : Prix moyen par Zone de départ
    stats_zone = cleaned_df.groupBy("PULocationID") \
        .agg(
            count("*").alias("trip_count"),
            avg("total_amount").alias("avg_price"),
            avg("duration_min").alias("avg_duration")
        )
    
    write_to_postgres(stats_zone, "kpi_zone_stats")

    # 3. Machine Learning (Régression Prix)
    print("Entraînement du modèle ML...")
    
    # Feature Engineering simplifié
    data_ml = cleaned_df.select("trip_distance", "PULocationID", "total_amount")
    
    assembler = VectorAssembler(
        inputCols=["trip_distance", "PULocationID"],
        outputCol="features"
    )
    
    # Split train/test
    train_data, test_data = data_ml.randomSplit([0.8, 0.2], seed=42)
    
    lr = LinearRegression(featuresCol="features", labelCol="total_amount", predictionCol="prediction")
    
    # Pipeline
    pipeline = Pipeline(stages=[assembler, lr])
    model = pipeline.fit(train_data)
    
    # Évaluation
    predictions = model.transform(test_data)
    evaluator_rmse = RegressionEvaluator(labelCol="total_amount", predictionCol="prediction", metricName="rmse")
    evaluator_r2 = RegressionEvaluator(labelCol="total_amount", predictionCol="prediction", metricName="r2")
    
    rmse = evaluator_rmse.evaluate(predictions)
    r2 = evaluator_r2.evaluate(predictions)
    
    print(f"Modèle entraîné. RMSE: {rmse}, R2: {r2}")
    
    # Sauvegarde du modèle
    model_path = "/app/models/taxi_price_model"
    model.write().overwrite().save(model_path)
    print(f"Modèle sauvegardé dans {model_path}")
    
    # Sauvegarde des métriques ML dans Postgres pour affichage (optionnel, table simple)
    metrics_df = spark.createDataFrame([(rmse, r2)], ["rmse", "r2"])
    write_to_postgres(metrics_df, "ml_model_metrics")

    spark.stop()

if __name__ == "__main__":
    main()
