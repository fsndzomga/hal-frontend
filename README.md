# HAL: Holistic Agent Leaderboard Frontend

## Database Update

To update the database:

1. Download the encrypted trace files from HuggingFace
2. Decrypt them using `hal-decrypt` as described in the [hal-harness GitHub repository](https://github.com/benediktstroebl/hal-harness)
3. Use `utils/db.py` to process and update the database