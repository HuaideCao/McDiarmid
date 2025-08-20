# McDiarmid’s Inequality in Lean 4

> A Lean 4 + Mathlib formalisation of **McDiarmid’s inequality**, organised with reusable definitions and axiom-level interfaces. The development follows the standard **Doob martingale → Azuma’s inequality → McDiarmid** route.

---

## Overview

The project contains a single Lean file `Basic.lean` which provides:
- **Core definitions**: independence `Independent`, bounded difference condition `BoundedDifference`, random function `RandomFunction`, Doob sequence interface `DoobSequence`;
- **Axiom-level assumptions**: Doob martingale property, bounded martingale differences, one-sided and two-sided Azuma inequalities, measurability/integrability side conditions, boundary consistency;
- **Bridging lemmas**: key identity \(Y - \mathbb E[Y] = Z_n - Z_0\), event equivalence, measure sub-additivity;
- **Main results**: upper tail, lower tail, and two-sided versions of McDiarmid’s inequality in exponential form.

The design goal is to **isolate technical probability tools** (martingales, conditional expectations, concentration bounds) into explicit axiom interfaces. This keeps the main logical flow readable and checkable in Lean, while allowing future replacement of axioms with actual Mathlib theorems.

---

## Code Structure

The file is organised into four parts (labelled *Part I–IV* in comments):

1. **Part I — Core Definitions**
   - `Independent`: finite independence of random variables
   - `BoundedDifference`: coordinate-wise bounded difference condition
   - `RandomFunction`: \(Y(\omega) = f(X(\omega))\)
   - `DoobSequence` (axiom interface): \(Z_k = \mathbb E[Y \mid X_1, \dots, X_k]\)

2. **Part II — Technical Assumptions (axioms)**
   - `doob_martingale_property` (martingale property: increments have mean 0)
   - `martingale_differences_bounded_axiom` (\(|Z_{k+1} - Z_k| \le c_k\))
   - `azuma_inequality_axiom` (two-sided Azuma)
   - `azuma_inequality_one_sided_axiom` (one-sided Azuma)
   - auxiliary measurability/integrability/boundary axioms

3. **Part III — Bridging Lemmas**
   - `martingale_difference_expectation_zero`
   - `key_identity`: \(Y - \mathbb E[Y] = Z_n - Z_0\)
   - `event_equivalence`: \(\{|Y - \mathbb E[Y]| \ge t\} = \{|Z_n - Z_0| \ge t\}\)
   - `upper_tail_subset_abs`, `abs_event_decomposition`, `measure_union_bound`

4. **Part IV — Main Theorems**
   - `mcddiarmid_upper_tail`
   - `mcddiarmid_lower_tail`
   - `mcddiarmid_inequality_main` (two-sided)
   - `mcddiarmid_from_azuma_direct`

---

## Main Results

- **Upper tail**
  \[
  \mu\!\left(\,Y-\mathbb E[Y]\ge t\,\right)
  \;\le\; \exp\!\Bigl(-\,\tfrac{t^2}{2\sum_i c_i^2}\Bigr).
  \]

- **Lower tail**
  \[
  \mu\!\left(\,\mathbb E[Y]-Y\ge t\,\right)
  \;\le\; \exp\!\Bigl(-\,\tfrac{t^2}{2\sum_i c_i^2}\Bigr).
  \]

- **Two-sided bound**
  \[
  \mu\!\left(\,|Y-\mathbb E[Y]|\ge t\,\right)
  \;\le\; 2\,\exp\!\Bigl(-\,\tfrac{t^2}{2\sum_i c_i^2}\Bigr).
  \]

**Proof strategy.** Use the key identity and event equivalence to translate to \(|Z_n - Z_0|\), apply one-sided Azuma, then combine via event decomposition and union bound.

---

## Installation & Build

1. Install **Lean 4** (recommended via `elan`) and the **VS Code Lean extension**.
2. Add a `lakefile.lean` to the project root, e.g.:

   ```lean
   import Lake
   open Lake DSL

   package McDiarmid where

   require mathlib from git
     "https://github.com/leanprover-community/mathlib4" @ "master"

   @[default_target] lean_lib McDiarmid
   ```

3. Build:
   ```bash
   lake update
   lake build
   ```

4. Open `Basic.lean` in VS Code to interactively step through the proofs.

---

## How to Read & Extend

- **Suggested reading order**: Part I (definitions) → Part II (axioms) → Part III (key lemmas) → Part IV (theorems).
- **Replacing axioms**: When Mathlib (or your own library) provides the needed theorems, replace the `axiom` with a `theorem` and add the proof; higher-level theorems remain unchanged.
- **Extending to other inequalities**: To formalise Bernstein or Freedman bounds, keep the key identity + event decomposition, and swap in the appropriate one-sided tail bound.

---

## License

Released under the **MIT License**.
