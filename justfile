test:
  uv run pytest

fetch:
  ./fetch_data.

synth-run method:
  uv run scripts/synth.py run --method {{method}} --target_risk 0.4 --num_trials 10000 --batch_size 1000 --out_file output/synth_{{method}}.csv

synth-analyze method:
  uv run scripts/synth_analyze.py run --method {{method}} --target_risk 0.4

heteroskedastic-run method:
  uv run scripts/heteroskedastic.py run --method {{method}} --target_risk 0.1 --n 200 --num_trials 10000 --batch_size 1000 --seed 1 --out_file output/heteroskedastic_{{method}}.csv

heteroskedastic-analyze method:
  uv run scripts/heteroskedastic_analyze.py run --method {{method}} --target_risk 0.1
