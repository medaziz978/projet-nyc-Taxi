FROM apache/spark:3.5.0-python3
ENV PATH="/opt/spark/bin:${PATH}"

USER root

# Installation des dépendances Python
# L'image officielle a déjà Python, on ajoute les libs
RUN pip install --no-cache-dir numpy pandas

# Téléchargement du driver JDBC PostgreSQL
# Le dossier de jars pour l'image officielle est /opt/spark/jars
RUN curl -o /opt/spark/jars/postgresql-42.6.0.jar https://jdbc.postgresql.org/download/postgresql-42.6.0.jar

USER 185
