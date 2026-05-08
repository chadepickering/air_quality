#!/usr/bin/env bash
# Create Kafka topics for the air quality pipeline.
#
# 19 partitions for raw_air_quality: one per FEM station (14) plus headroom for
# future stations and parallel consumer tasks.  Key = station_id ensures all
# readings for a station land in the same partition, preserving temporal order
# per station without a global ordering guarantee.
#
# processed_air_quality mirrors the same partition count so the consumer can
# write one processed record per raw message without reshuffling.
#
# Run after `docker compose up -d` once the broker is healthy:
#   bash streaming/create_topics.sh

set -euo pipefail

BOOTSTRAP="localhost:9093"
REPLICATION=1
RETENTION_MS=604800000   # 7 days

kafka-topics --bootstrap-server "$BOOTSTRAP" \
  --create --if-not-exists \
  --topic raw_air_quality \
  --partitions 19 \
  --replication-factor "$REPLICATION" \
  --config retention.ms="$RETENTION_MS"

kafka-topics --bootstrap-server "$BOOTSTRAP" \
  --create --if-not-exists \
  --topic processed_air_quality \
  --partitions 19 \
  --replication-factor "$REPLICATION" \
  --config retention.ms="$RETENTION_MS"

echo "Topics created:"
kafka-topics --bootstrap-server "$BOOTSTRAP" --list
