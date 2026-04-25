# Report - Live Testing and Fixes

Use this file as your Day 12 evidence after deployment.

## App Information

- Live URL: `https://<your-render-url>.onrender.com`
- Deploy platform: `Render` (or `Railway`)
- Test date: `YYYY-MM-DD`
- Tester(s): `Your name + friend name`

## Quick Result

- Overall status: `PASS` / `PASS with notes` / `FAIL`
- Main issue found: `<none or short issue>`
- Main fix applied: `<short fix summary>`

## Test Checklist

- Home page opens without server error
- Custom grading prediction returns result
- Curve grading prediction returns result
- Invalid input shows validation error banner (no crash)
- Mobile layout works (inputs, button, results visible)
- Render logs show no active 500/Traceback after tests

## Test Cases Run

1. **Custom mode valid input**
  - Input: `thresholds 90/80/70/60`, normal study data
  - Expected: grade + score shown
  - Actual: `<write result>`
  - Status: `PASS/FAIL`
2. **Curve mode valid input**
  - Input: `pct_a=30, pct_b=40, pct_c=30` and class stats
  - Expected: curve result + percentile shown
  - Actual: `<write result>`
  - Status: `PASS/FAIL`
3. **Curve mode invalid distribution**
  - Input: `pct_a=30, pct_b=40, pct_c=20` (total 90)
  - Expected: validation error shown
  - Actual: `<write result>`
  - Status: `PASS/FAIL`
4. **Custom mode invalid thresholds**
  - Input: `A=50, B=80, C=70, D=60`
  - Expected: threshold order error shown
  - Actual: `<write result>`
  - Status: `PASS/FAIL`

## Logs Review (Render/Railway)

- Checked logs at: `<time>`
- Errors seen: `<none / list>`
- Notes: `<cold start delay on free tier is expected>`

## Fixes Applied Today

- `<fix 1>`
- `<fix 2>`

## What I Learned (Day 12)

- Deployment success is not enough; production testing is required.
- Free tier can have cold start latency after inactivity.
- Validation and clear error messages improve real user experience.

## Interview-Ready Summary

On Day 12, I validated the deployed app in a live environment, tested both valid and invalid user flows on desktop/mobile, reviewed deployment logs, and confirmed reliability after fixing issues.
