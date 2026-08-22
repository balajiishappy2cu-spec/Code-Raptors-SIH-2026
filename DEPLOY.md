# Deploying the console to Streamlit Community Cloud

The repository is prepared for deployment. The two steps that need a human are the ones
that publish something or touch an account: pushing to GitHub, and connecting the app on
share.streamlit.io.

## What ships, and why it is small

The working dataset cannot travel with the app. Raw downloads are ~4.6 GB, the processed
cache ~863 MB, and the Turing dataset is gated so the deployed app cannot fetch it at run
time either. Community Cloud also runs on roughly 1 GB of RAM.

So the repository ships a **demonstration bundle** instead:

| Item | Size | Notes |
|---|---|---|
| `data/demo/` | 21 MB | 12 pulse trains (4 per split), 60,000-pulse window each |
| `models/artifacts/activity_predictor.joblib` | 0.67 MB | the trained model, loaded not retrained |
| `results/*.json` | 0.6 MB | the recorded full-run metrics the scorecard cites |
| code, config, requirements | < 1 MB | |
| **total tracked** | **~24 MB** | |

Excluded by `.gitignore`: `data/raw/`, `data/processed/`, `*.npz` feature matrices, and
every model artifact except the shipped one — the random-forest comparison artifact alone
is 92 MB.

**The demo bundle is a demonstration subset, not the experiment.** Every metric quoted in
the README comes from the full 55-train run and is read from `results/`, which ships as
recorded. The console labels single-environment readouts as such and prints the
split-wide figure beneath them.

Regenerate the bundle at any time:

```bash
python scripts/build_demo_bundle.py --config config.yaml --trains-per-split 4 --max-pulses 60000
```

## How the app finds its data

`dashboard/app.py` prefers the manifest named in `config.yaml` and falls back to
`data/demo/manifest.json` when that is absent. Locally the full manifest exists and is
used; on Community Cloud only the bundle ships, so the bundle is used. One configuration
works in both places, with no second config file to keep in step.

Bundle manifest paths are stored **relative to the repository root**, so they resolve on
any checkout regardless of where it sits on disk.

## Steps

1. **Review what would be committed.** The remote is a shared team repository
   (`Code-Raptors-SIH-2026`), so check the diff before pushing:

   ```bash
   git status
   git diff --stat HEAD
   ```

2. **Push.** Consider a branch rather than committing straight to `main` if others are
   working in the repo:

   ```bash
   git push origin main
   ```

3. **Create the app** at [share.streamlit.io](https://share.streamlit.io) → *New app*:

   | Field | Value |
   |---|---|
   | Repository | `balajiishappy2cu-spec/Code-Raptors-SIH-2026` |
   | Branch | `main` |
   | Main file path | `dashboard/app.py` |

4. **Nothing else is required.** `requirements.txt` covers the dependencies and
   `.streamlit/config.toml` pins the dark console theme, so the deployed app matches
   local rendering.

## Notes for the deployed environment

- **No GPU.** `xgb_params.device: cuda` in `config.yaml` probes for a usable CUDA device
  and falls back to CPU with a warning, so the same config runs unmodified. The model is
  loaded rather than trained, so this only affects anyone retraining from the deployment.
- **Simulations run live.** Every control change re-runs the schedulers. At the default
  4,000 cycles across three strategies this takes a few seconds on Community Cloud
  hardware; lower `DURATION (CYCLES)` in the sidebar if it feels slow.
- **No secrets needed.** The app reads no Hugging Face token — the data it uses is already
  bundled. `.env` and `.streamlit/secrets.toml` are gitignored; never commit a token.
- **Repository is public.** Community Cloud's free tier requires it. Everything shipped is
  derived from an openly published research dataset and contains no credentials.
