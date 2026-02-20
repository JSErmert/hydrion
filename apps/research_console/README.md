# Hydrion Research Console (v0)

A thin, research-grade UI over `HydrionEnv`.

## Install

From the **repo root**:

```bash
python -m pip install -e .
python -m pip install streamlit
```

## Run

```bash
python -m streamlit run apps/research_console/streamlit_app.py
```

## What it does

- **Scenarios**: browse YAML configs + show diffs
- **Run Console**: step the environment (manual or deterministic-random actions)
- **Observatory**: generate plots from `EpisodeHistory` recorded in the session
- **Export**: download run history JSON/CSV for offline analysis

## Why this exists

The example script writes separate PNGs into `outputs/…`, which feels "fragmented".
This console makes those same visuals feel like a real research instrument panel, without rewriting the backend.
