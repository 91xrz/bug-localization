#!/usr/bin/env bash
set -u
project_root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$project_root" || exit 1
while true; do
  if curl -fsS --max-time 3 http://127.0.0.1:9200 > /dev/null; then
    sleep 15
    continue
  fi
  export ES_PATH_CONF="$project_root/.runtime/elasticsearch/config"
  export ES_JAVA_HOME="$project_root/.runtime/elasticsearch/elasticsearch-9.5.5/jdk"
  "$project_root/.runtime/elasticsearch/elasticsearch-9.5.5/bin/elasticsearch" >> "$project_root/results/full_semantic/elasticsearch-supervisor.log" 2>&1
  sleep 15
done
