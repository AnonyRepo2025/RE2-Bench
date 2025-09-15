#!/bin/bash

# Check if a path is given
if [ -z "$1" ]; then
  echo "Usage: $0 <path>"
  exit 1
fi

BASE_PATH="$1"

# Check if it's a valid directory
if [ ! -d "$BASE_PATH" ]; then
  echo "Error: $BASE_PATH is not a directory"
  exit 1
fi

# Iterate through all directories inside the given path
for dir in "$BASE_PATH"/*/; do
  if [ -d "$dir" ]; then
    echo "Running analysis on $dir"
    python3 dynamic-type-analysis.py "$dir"
  fi
done

