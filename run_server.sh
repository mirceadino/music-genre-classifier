#!/bin/bash

export FLASK_APP=server/main.py
export FLASK_DEBUG=1
PYTHONPATH=. python3 -m flask run
