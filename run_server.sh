#!/bin/bash

export FLASK_APP=server/main.py
PYTHONPATH=. python3 -m flask run --host=0.0.0.0
