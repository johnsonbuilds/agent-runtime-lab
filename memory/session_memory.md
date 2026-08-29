# Agent Context Snapshot

## 1. Work State
### Completed
- R environment fixed (ldpaths symlink issue resolved, R 4.3.3 working)
- Multiple iterations of `/app/ars.R` written and tested (6 versions so far)
- Bugs found and fixed across iterations:
  - Validation incorrectly rejected both-infinite domains (needed for normal)
  - Segment areas didn't respect domain bounds [lower, upper]
  - Log-concavity check was inverted (checking if derivative increased instead of decreased)
  - Initial log-concavity check was too strict (prevented non-log-concave detection test)
  - Inverse CDF sampling formula was numerically unstable
- Debug scripts (`debug_test.R`, `debug_test2.R`) confirmed the insertion bug

### Active (In-Progress)
- **6th version of `/app/ars.R` just written** — completely redesigned upper hull (n-1 intersection points instead of n+1), new insertion logic, new `eval_upper_hull`/`eval_lower_hull`/`compute_segment_areas`/`sample_from_envelope` implementations
- **NOT YET TESTED** — the 6th version has not been run
- The test function in the 6th version has a syntax error: `error = classCondition = function(e)` should be `error = function(e)`

### Blocked / Failure Lessons
- **Insertion bug (5th version)**: When `insert_idx >= length(x)` (appending at end), `x[(insert_idx+1):length(x)]` produced `c(NA, NA)`, poisoning `hx`/`hpx` with NAs, causing `compute_upper_hull` to fail with `"missing value where TRUE/FALSE needed"` in `abs(denom) < 1e-12`. Root cause: R's `c()` with out-of-bounds indices returns NA. Fix: handle `insert_idx >= length(x)` as a separate branch using `c(x, w)`.
- **Log-concavity check direction**: The check must verify `hpx_w` is between left and right neighbor derivatives (non-increasing). The condition `hpx_new > hpx[pos+1] - 1e-6` (checking if derivative decreased too much) was wrong; should be `hpx_new < hpx[pos+1] - 1e-6` (checking if derivative increased, violating log-concavity).
- **Initial log-concavity check too strict**: Checking `diff(hpx) > 1e-6` at initialization prevented the non-log-concave detection test from ever reaching the sampling loop. Should only warn, not error, at initialization.

## 2. Next Move
1. Fix the syntax error in test function: `error = classCondition = function(e)` → `error = function(e)`
2. Run `Rscript -e 'source("/app/ars.R"); test()'` to test the 6th version
3. If failures, debug iteratively — the 6th version has a fundamentally different upper hull structure (n-1 z points instead of n+1) that may have its own bugs in `eval_upper_hull` segment indexing
4. Verify `/app/normal_samples.txt` and `/app/exponential_samples.txt` are generated

## 3. Working Context & Anchors
- **Relevant Files / Artifacts**: `/app/ars.R` (6th version, untested), `/app/debug_test.R`, `/app/debug_test2.R` (old debug scripts)
- **Environment State**: R 4.3.3 installed and working at `/usr/bin/R`, `/app/` is working directory
- **Key parameters**: `n_samples = 10000`, `set.seed(42)` for reproducibility, test distributions: Normal(0,1), Exponential(1), Half-Normal, bimodal mixture (non-log-concave)
- **Output requirements**: `/app/normal_samples.txt`, `/app/exponential_samples.txt`, test output format `"TEST_NAME: PASS"` or `"TEST_NAME: FAIL"` with mean/sd stats
- **Critical constraint**: The `ars` function must detect non-log-concave densities during sampling and throw an error with message containing "Non-log-concave"