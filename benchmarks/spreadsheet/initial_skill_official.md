Inspect the workbook preview before writing. Output JSON only:
{"writes": [{"cell": "A1", "value": 123}]}

Rules:
- Write static evaluated values (numbers/strings), never formulas
- Fill every cell required by the task / answer position
- Read sheet structure from the preview; do not guess cell addresses
