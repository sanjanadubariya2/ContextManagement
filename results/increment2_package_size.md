# Increment 2: package size vs full history

Budget 12000 tokens per package. Full shared history: 1899 tokens (5x replay: 9495, 20x replay: 37980; replays are synthetic).
Tokens are the Context Manager's estimate (ceil(chars/4)). Packages also carry code, contracts and decisions that the history baseline does not.

| task | member | mode | trap | package | pinned | items | history | pkg/history | pkg/5x | pkg/20x | pinned all required | contradiction check |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| m01 | sneha | curated_fork |  | 3634 | 1312 | 38 | 1899 | 1.91 | 0.38 | 0.10 | yes | ok |
| m02 | riya | curated_fork |  | 3651 | 1726 | 32 | 1899 | 1.92 | 0.38 | 0.10 | yes | ok |
| m03 | kabir | isolated |  | 5008 | 1357 | 51 | 1899 | 2.64 | 0.53 | 0.13 | yes | ok |
| m04 | meera | curated_fork |  | 2327 | 1357 | 19 | 1899 | 1.23 | 0.25 | 0.06 | yes | ok |
| m05 | aarav | curated_fork | yes | 1465 | 1233 | 13 | 1899 | 0.77 | 0.15 | 0.04 | yes | ok |
| m06 | ishaan | curated_fork |  | 2578 | 1235 | 23 | 1899 | 1.36 | 0.27 | 0.07 | yes | ok |
| m07 | rohan | curated_fork |  | 2206 | 743 | 21 | 1899 | 1.16 | 0.23 | 0.06 | yes | ok |
| m08 | tanvi | isolated |  | 2386 | 800 | 25 | 1899 | 1.26 | 0.25 | 0.06 | yes | ok |
| m09 | aarav | curated_fork | yes | 3325 | 1593 | 25 | 1899 | 1.75 | 0.35 | 0.09 | yes | ok |
| m10 | rohan | curated_fork | yes | 3030 | 1170 | 28 | 1899 | 1.60 | 0.32 | 0.08 | yes | ok |

Mean package 2961 tokens, max 5008 (never above the 12000 budget).
Gate (every required decision pinned, budget respected, trace written): **PASS**
