# GitHub Pages Setup

Use this only after reviewing the files you intend to commit.

1. Open the repository on GitHub.
2. Go to `Settings`.
3. Go to `Pages`.
4. Under `Build and deployment`, set `Source` to `Deploy from a branch`.
5. Set `Branch` to `main`.
6. Set `Folder` to `/docs`.
7. Save.
8. Wait for GitHub Pages to publish the site.

Safety notes:

- The dashboard should contain compact research summaries only.
- Do not commit `.env`, API keys, secrets, downloaded CSVs, raw result `.txt` files, raw JSONL logs, `.venv`, or cache files.
- Do not push until you have reviewed the changed files.
- This dashboard is research/backtesting only and is not financial advice.
