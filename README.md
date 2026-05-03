# D3Q19 MRT: classical reference and CPTP (Kraus) relaxation

JAX code for a **decaying Taylor–Green** setup on a periodic **D3Q19** lattice: **classical linear MRT** (`ref_mrt.py`) side by side with the **open-channel** collide step that applies the two-rail **Kraus** map to dissipative moments (`baseline_jax.py`). Please use it to reproduce **PDF figures**, **theorem-audit JSON**, and **stencil-free synthetic checks**.

**Author:** Muhammad Idrees Khan (see module headers for version tags).

### Citation

**Repository:** `[IdreesKhan3/open-channel-dissipation-q-lbm](https://github.com/IdreesKhan3/open-channel-dissipation-q-lbm)`  

**Manuscript — arXiv preprint** [physics.comp-ph, 2604.25429](https://arxiv.org/abs/2604.25429):

```bibtex
@misc{khan2026deterministicrealizationclassicaldissipation,
      title={Deterministic Realization of Classical Dissipation on Quantum Computers},
      author={Muhammad Idrees Khan and Sauro Succi and Hua-Dong Yao},
      year={2026},
      eprint={2604.25429},
      archivePrefix={arXiv},
      primaryClass={physics.comp-ph},
      url={https://arxiv.org/abs/2604.25429},
}
```

---

## Layout


| File                        | Role                                                                                         |
| --------------------------- | -------------------------------------------------------------------------------------------- |
| `ref_mrt.py`                | Classical linear MRT collide–stream (reference).                                             |
| `baseline_jax.py`           | Same stencil and moment closures; dissipative modes via `_build_kraus` / `_apply_kraus_jax`. |
| `viscous_dissipation.py`    | Volume mean `2 ν ⟨S_ij S_ij⟩` from post-stream velocity (periodic central differences).      |
| `generate_paper_results.py` | Builds `theorem_error_dm_only.pdf` and `tgv_ke_dissipation.pdf`.                             |
| `run_theorem_audit.py`      | Full TGV run + endpoint checks → `theorem_audit_table.json`.                                 |
| `run_synthetic_audit.py`    | Random `(δm, λ)` Kraus checks (no lattice) → `synthetic_audit_table.json`.                   |


---

## Requirements

- **Python** 3.10+ recommended  
- **JAX** + **NumPy**; **Matplotlib** for figures

Example install (CPU; follow [JAX install](https://jax.readthedocs.io/en/latest/installation.html) for your OS/GPU):

```bash
pip install "jax[cpu]" numpy matplotlib
```

Optional: `tqdm` for progress text in `ref_mrt.py`.

Each script enables JAX double precision via `jax.config.update("jax_enable_x64", True)`. To force CPU if needed:

```bash
export JAX_PLATFORMS=cpu
```

---

## How to run

Work from the directory that **contains** these Python files (in the repository this is usually `solver_code/`):

```bash
cd /path/to/solver_code
```

Replace `/path/to/solver_code` with the directory where `baseline_jax.py` lives.

### Manuscript figures (defaults: `nx=64`, `timesteps=2000`, `u0=0.1`, `tau=0.5035`)

```bash
python3 generate_paper_results.py
```

Optional:

```bash
python3 generate_paper_results.py --outdir ./final --nx 64 --timesteps 2000 --u0 0.1 --tau 0.5035
```

**Outputs:** `theorem_error_dm_only.pdf`, `tgv_ke_dissipation.pdf` (under `--outdir` or this directory).

### Theorem-style audit (`theorem_audit_table.json`)

```bash
python3 run_theorem_audit.py
```

Flags: `--nx`, `--tau`, `--u0`, `--timesteps` (see `argparse` in file).

### Stencil-free synthetic audit (`synthetic_audit_table.json`)

```bash
python3 run_synthetic_audit.py
```

Large by default; for a fast check reduce e.g. `--n-lambda-grid`, `--n-dm-grid`, `--n-boundary`, etc.

### Classical driver only

```bash
python3 ref_mrt.py
```

Uses module-level defaults at the bottom of `ref_mrt.py`; edit there or wrap `run_ref_mrt(...)` from another script.

---

## Outputs


| Script                      | Typical outputs                                       |
| --------------------------- | ----------------------------------------------------- |
| `generate_paper_results.py` | `theorem_error_dm_only.pdf`, `tgv_ke_dissipation.pdf` |
| `run_theorem_audit.py`      | `theorem_audit_table.json`                            |
| `run_synthetic_audit.py`    | `synthetic_audit_table.json`                          |


---

## License

This software is licensed under the [MIT License](LICENSE).

---

## Contact

Corresponding author: **Muhammad Idrees Khan** — see the article for current email.