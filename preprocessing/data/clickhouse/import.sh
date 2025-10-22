#!/usr/bin/bash

python fix_sql.py ./ch_dump/*.sql

clickhouse-client --queries-file ch_dump/*.sql

for file in ch_dump/*.native; do
    # Extrahiere den Tabellennamen aus dem Dateinamen (ohne .native)
    table_name=$(basename "$file" .native)
    # Importiere die Datei in die entsprechende Tabelle
    clickhouse-client --query "INSERT INTO $table_name FORMAT Native" < "$file"
done