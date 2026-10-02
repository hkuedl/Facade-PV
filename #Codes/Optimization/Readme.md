# Optimization code

## Files

- `Functions.py`: time-of-use tariff, representative-day, and facade
  regulation helpers.
- `Load_generate.py`: generates the full-year grid loads for all 102
  cities. Residential, office, and commercial-service shares are assigned
  from grid-level land-use evidence, with a nearest-neighbour fallback where
  a grid has no matched land-use point.
- `Multi_period_planning.py`: solves the 2030–2050 RPV–FPV–storage
  planning model. It reads the saved RPV35/FPV70 main-case table and scales
  the baseline ML capacity and generation arrays before optimization.

## Expected data

The default data root is `../../##Data` relative to this directory:

- `Evaluation/All_input`
- `Evaluation/All_output`
- `Optimization/Electricity_price`
- `Optimization/Loads`
- `Optimization/Other_parameters`
- `Optimization/Fig_input_data`

The load generator writes to
`Optimization/Loads/Landuse_Loads`. The planning model writes to
`Optimization/Planning_results`.

Set `FPV_DATA_ROOT` to use another data root. The scripts also expose
fine-grained environment-variable overrides in their configuration sections.

## Run

```text
python Load_generate.py
python Multi_period_planning.py --city-start 0 --city-end 102
```

For distributed optimization, use non-overlapping half-open index ranges,
for example `--city-start 0 --city-end 17`. Set `FPV_OPT_SOLVER` or pass
`--solver` to select a configured Pyomo solver.
