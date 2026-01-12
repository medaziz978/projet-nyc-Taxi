from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_timestamp, from_json
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, TimestampType
from pyspark.ml import PipelineModel
import sys

# Configuration PostgreSQL
DB_URL = "jdbc:postgresql://postgres:5432/nyc_data"
DB_PROPERTIES = {
    "user": "admin",
    "password": "password",
    "driver": "org.postgresql.Driver"
}

def get_spark_session(app_name="NYC_Taxi_Streaming"):
    return SparkSession.builder \
        .appName(app_name) \
        .config("spark.jars", "/opt/spark/jars/postgresql-42.6.0.jar") \
        .getOrCreate()

# Schéma requis pour le streaming (CSV n'infère pas le schéma en streaming automatiquement sans lecture de fichiers)
# On définit un schéma standard pour les données Yellow Taxi
cols = [
    "VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime", "passenger_count", "trip_distance",
    "RatecodeID", "store_and_fwd_flag", "PULocationID", "DOLocationID", "payment_type", "fare_amount",
    "extra", "mta_tax", "tip_amount", "tolls_amount", "improvement_surcharge", "total_amount", 
    "congestion_surcharge", "Airport_fee" # Champs récents, à adapter si nécessaire
]

# Simplification pour l'exercice : On prend un subset ou on définit tout en String/Double
schema = StructType()
for c in cols:
    if "datetime" in c:
        schema.add(c, TimestampType())
    elif "ID" in c or "count" in c:
        schema.add(c, IntegerType())
    elif "flag" in c:
        schema.add(c, StringType())
    else:
        schema.add(c, DoubleType())

def write_streaming_batch(batch_df, batch_id):
    """Fonction exécutée sur chaque micro-batch."""
    print(f"Processing batch {batch_id} with {batch_df.count()} records")
    if batch_df.count() > 0:
        # Écriture dans Postgres
        # On écrit les prédictions
        batch_df.select("tpep_pickup_datetime", "trip_distance", "total_amount", "prediction") \
                .write.jdbc(url=DB_URL, table="streaming_predictions", mode="append", properties=DB_PROPERTIES)

def main():
    spark = get_spark_session()
    spark.sparkContext.setLogLevel("WARN")
    
    model_path = "/app/models/taxi_price_model"
    print(f"Chargement du modèle depuis {model_path}...")
    try:
        model = PipelineModel.load(model_path)
    except Exception as e:
        print(f"Erreur chargement modèle (avez-vous lancé le batch avant ?) : {e}")
        sys.exit(1)

    # Lecture Stream
    print("Démarrage du Stream depuis /app/data/streaming_input/ ...")
    # On surveille un dossier spécifique
    input_stream = spark.readStream \
        .schema(schema) \
        .option("header", "true") \
        .csv("/app/data/streaming_input")

    # Nettoyage minimal pour le ML (doit matcher le training)
    clean_stream = input_stream \
        .withColumn("PULocationID", col("PULocationID").cast("integer")) \
        .withColumn("trip_distance", col("trip_distance").cast("double")) \
        .filter(col("trip_distance") > 0)

    # Prédiction
    predictions = model.transform(clean_stream)
    
    # Selection pour output
    # On ajoute un timestamp de processing si on veut
    output = predictions.withColumn("processing_time", current_timestamp())

    # Démarrage
    query = output.writeStream \
        .foreachBatch(write_streaming_batch) \
        .trigger(processingTime='5 seconds') \
        .start()

    query.awaitTermination()

if __name__ == "__main__":
    main()
