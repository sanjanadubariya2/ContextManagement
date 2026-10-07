# Increment 2: package size vs full history

Budget 12000 tokens per package. Full shared history: 1317 tokens (5x replay: 6585, 20x replay: 26340; replays are synthetic).
Tokens are the Context Manager's estimate (ceil(chars/4)). Packages also carry code, contracts and decisions that the history baseline does not.

| task | member | mode | trap | package | pinned | items | history | pkg/history | pkg/5x | pkg/20x | pinned all required | contradiction check |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| m01 | sneha | curated_fork |  | 3062 | 1307 | 30 | 1317 | 2.32 | 0.46 | 0.12 | yes | ok |
| m02 | riya | curated_fork |  | 3393 | 1722 | 28 | 1317 | 2.58 | 0.52 | 0.13 | yes | ok |
| m03 | kabir | isolated |  | 4381 | 1352 | 45 | 1317 | 3.33 | 0.67 | 0.17 | yes | ok |
| m04 | meera | curated_fork |  | 2316 | 1357 | 19 | 1317 | 1.76 | 0.35 | 0.09 | yes | ok |
| m05 | aarav | curated_fork | yes | 1465 | 1233 | 13 | 1317 | 1.11 | 0.22 | 0.06 | yes | ok |
| m06 | ishaan | curated_fork |  | 2424 | 1235 | 22 | 1317 | 1.84 | 0.37 | 0.09 | yes | ok |
| m07 | rohan | curated_fork |  | 2031 | 743 | 20 | 1317 | 1.54 | 0.31 | 0.08 | yes | ok |
| m08 | tanvi | isolated |  | 2243 | 800 | 23 | 1317 | 1.70 | 0.34 | 0.09 | yes | ok |
| m09 | aarav | curated_fork | yes | 3314 | 1593 | 25 | 1317 | 2.52 | 0.50 | 0.13 | yes | ok |
| m10 | rohan | curated_fork | yes | 2628 | 1170 | 24 | 1317 | 2.00 | 0.40 | 0.10 | yes | ok |

Mean package 2726 tokens, max 4381 (never above the 12000 budget).
Gate (every required decision pinned, budget respected, trace written): **PASS**
