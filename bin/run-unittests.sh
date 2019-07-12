#!/usr/bin/env bash
#
#Run unit tests.
#
#Run in container with:
#/app/bin/run-unittests.sh

pytest /app/tests/test_disambiguate_sense.py
