# Quick Setup Guide for Data Collection

## Get a GitHub Token (2 minutes):

1. Go to: https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Give it a name: "LLM Fine-Tuning Project"
4. Select scopes: ✅ `public_repo` (that's all you need)
5. Click "Generate token"
6. **Copy the token immediately** (you won't see it again)

## Add Token to .env File:

Open `.env` file and paste your token:
```
GITHUB_TOKEN=ghp_your_token_here
```

## Run the Collector:

```bash
python scripts/collect_data.py
```

This will collect ~2000 bug reports from:
- Godot Engine (game engine)
- Bevy (Rust game engine)
- Minetest (voxel game)
- OpenRCT2 (RollerCoaster Tycoon 2)

Takes about 10-15 minutes.

## If You Don't Want to Create a Token:

You can run without a token (60 requests/hour limit):
- You'll get ~100-200 bug reports instead of 2000
- Still enough for the project!
