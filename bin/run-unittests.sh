#!/usr/bin/env bash
#
#Run unit tests.
#
#Run in container with:
#/app/bin/run-unittests.sh

pytest /app/tests/test_preprocess.py
pytest /app/tests/test_train.py
pytest /app/tests/test_predict.py
