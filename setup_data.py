import requests
import pandas as pd
import os
import shutil

# URL d'un fichier Yellow Taxi Trip Data (Jan 2023) - Parquet est le standard actuel
# On le convertit en CSV pour l'exercice
DATA_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet"
OUTPUT_DIR = "data"
RAW_FILE = os.path.join(OUTPUT_DIR, "yellow_tripdata_2023-01.parquet")
CSV_FILE = os.path.join(OUTPUT_DIR, "yellow_tripdata_2023-01.csv")

def download_file():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"Téléchargement de {DATA_URL}...")
    response = requests.get(DATA_URL, stream=True)
    with open(RAW_FILE, 'wb') as f:
        shutil.copyfileobj(response.raw, f)
    print("Téléchargement terminé.")

def convert_to_csv():
    print("Conversion Parquet -> CSV...")
    # On charge avec pandas (nécessite pyarrow ou fastparquet)
    # Pour minimiser les dépendances, on espère que l'env local a pandas/pyarrow. 
    # Sinon l'utilisateur devra installer : pip install pandas pyarrow
    try:
        df = pd.read_parquet(RAW_FILE)
        # On ne garde qu'un échantillon pour le dév (100k lignes) pour aller vite
        df_sample = df.sample(n=100000, random_state=42)
        df_sample.to_csv(CSV_FILE, index=False)
        print(f"Fichier CSV créé : {CSV_FILE} ({len(df_sample)} lignes)")
    except Exception as e:
        print(f"Erreur conversion : {e}")
        print("Assurez-vous d'avoir installé : pip install pandas pyarrow")

def prepare_streaming_dir():
    stream_dir = os.path.join(OUTPUT_DIR, "streaming_input")
    if not os.path.exists(stream_dir):
        os.makedirs(stream_dir)
    print(f"Dossier streaming prêt : {stream_dir}")

if __name__ == "__main__":
    download_file()
    convert_to_csv()
    prepare_streaming_dir()
    # Suppression du parquet pour gagner de la place
    if os.path.exists(RAW_FILE):
        os.remove(RAW_FILE)
