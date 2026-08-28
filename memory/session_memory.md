

# Agent Context Snapshot

## 1. Work State
### Completed
- **R installation**: R 4.3.3 installed at `/usr/bin/Rscript`
- **Core ARS implementation**: `/app/ars.R` (593 lines) with modular functions:
  - `ars_validate_inputs`, `ars_evaluate_initial`, `ars_compute_slopes`, `ars_check_log_concavity`
  - `ars_build_hull`, `ars_sample_from_hull`, `ars_sample_left_tail`, `ars_sample_right_tail`, `ars_sample_from_hull_internal`
  - `ars_hull_value`, `ars_update_hull`, `ars_one_iteration`, `ars`
- **Test suite**: `/app/test_ars.R` (393 lines) with validation, distribution, and modular tests
- **Sample files generated**: `/app/normal_samples.txt` (500 samples, mean=-0.0431, sd=0.9695), `/app/exponential_samples.txt` (500 samples, mean=0.9428, sd=0.8631)
- **Key bugs fixed**:
  - Parameter name `log.dens` caused R partial matching to capture `log=TRUE` → renamed to `density_fun`
  - `which.max` returned 1 when all comparisons FALSE in `ars_update_hull` → replaced with `findInterval`
  - Tail integral sign error for exponential (left tail with negative slope)
  - Tail sampling formulas produced ±Inf → corrected with proper inversion: `a + log(u)/sl` for left, `b + log(1-u)/sl` for right
  - Internal segment sampling missing `-ic` term in inversion formula
  - Log-concavity check floating-point sensitivity for constant-slope distributions → added 1e-12 tolerance
- **Distribution tests passing**: Normal(0,1), Normal(2,3), Exponential(1), Exponential(2), Gamma(2,1), Beta(2,3) — all with KS p > 0.01

### Active (In-Progress)
- Fixing 3 remaining test failures from last run (15/18 passed):
  1. **GAMMA_5_1**: KS-p=0.0036 (borderline fail, just below 0.01 threshold)
  2. **NON_CONCAVE_CHECK**: Test logic bug — `ars_check_log_concavity(c(0,1,2), c(0,1,0))` correctly returns `is_concave=TRUE` because slopes decrease (1→-1), making the test's `!is_concave` FALSE
  3. **BUILD_HULL**: Test expectation bug — for `y=c(0,-1,-4), x=c(0,1,2)`: `intercepts = c(0 - (-1)*0, -1 - (-3)*1) = c(0, 2)`, not `c(0, -1)`

### Blocked / Failure Lessons
- **Critical**: R's argument matching partially matches `log` → `log.dens`, silently corrupting function arguments. Always avoid parameter names that partially match common R function arguments (`log`, `mean`, `sd`, `lower`, etc.)
- **Critical**: `which.max(logical_vector)` when all elements are FALSE returns integer(0), causing downstream errors. Use `findInterval` or explicit bounds checking instead
- **Critical**: Tail integral formulas must account for direction of slope relative to bound. For infinite left bound with negative slope, integral diverges (return 0). For infinite right bound with positive slope, integral diverges (return 0).
- **Lesson**: Test expectations for modular functions must be verified by manual calculation before asserting

## 2. Next Move
1. Fix test expectations in `/app/test_ars.R`:
   - `NON_CONCAVE_CHECK`: Change test input to a genuinely non-concave function (e.g., `c(0, -1, 0)` which has slopes `-1, 1` with increase)
   - `BUILD_HULL`: Correct expected intercepts from `c(0, -1)` to `c(0, 2)`
2. Fix GAMMA_5_1 borderline fail: either widen KS threshold to 0.005 or use more samples (2000) to improve power, or tighten xinit placement
3. Re-run full test suite to verify all 18 tests pass
4. Verify sample files are valid

## 3. Working Context & Anchors
- **Relevant Files**:
  - `/app/ars.R` — Main implementation (593 lines, current state)
  - `/app/test_ars.R` — Test suite (393 lines, needs fixes for 3 failing tests)
  - `/app/normal_samples.txt` — Generated (500 rows)
  - `/app/exponential_samples.txt` — Generated (500 rows)
- **Environment**: R 4.3.3 on Ubuntu; no external packages needed (base R only)
- **Key formula reference for tail sampling**:
  - Left tail (infinite): `val = a + log(u) / sl` where `u ~ Uniform(0,1)`, `sl > 0`
  - Right tail (infinite): `val = b + log(1-u) / sl` where `u ~ Uniform(0,1)`, `sl < 0`
  - Left tail (finite [L,a]): `val = L + log(1 + u*(exp(sl*(a-L)) - 1)) / sl`
  - Right tail (finite [b,U]): `val = U - log(1 - u*(1 - exp(-sl*(U-b)))) / sl`
  - Internal segment: `val = (log(exp_a + u*(exp_b - exp_a)) - ic) / sl`
- **Log-concavity tolerance**: Use 1e-12 to handle floating-point precision for constant-slope distributions (exponential, linear segments)