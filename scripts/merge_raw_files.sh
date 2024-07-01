#!/usr/bin/env bash

python3 -c "from src.data import merge_raw_files, build_final_files; merge_raw_files(); build_final_files()"