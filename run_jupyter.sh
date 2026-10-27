#!/usr/bin/env bash
cd "$(dirname "$0")"
exec .venv/bin/jupyter lab analysis.ipynb
