#!/bin/bash
pip install uv
uv pip install pipreqs
pipreqs . --savepath requirements.txt --force
uv pip install -r requirements.txt
uv init
uv add -r requirements.txt --frozen