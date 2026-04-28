setup:
  pip install uv
  uv pip install pipreqs
  pipreqs . --savepath requirements.txt --force
  uv pip install -r requirements.txt

run:
  uv run python main.py

clean:
  rm -rf __pycache__/ */__pycache__/
  