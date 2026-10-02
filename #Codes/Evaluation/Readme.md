# Evaluation code

## Files

- `UMEP_QGIS/`: QGIS/UMEP preprocessing and irradiance simulations for the
  sampled cities.
- `Sample_feature_read.py`: constructs regression features for the sampled
  grids.
- `Sample_label_read.py`: processes UMEP outputs into regression labels.
- `ML_training.py`: trains the regression models.
- `ML_input_read.py`: prepares model inputs for all 102 cities.
- `ML_output_power.py`: produces baseline grid-level rooftop and facade
  power arrays.
- `ML_output_capacity.py`: produces baseline grid-level rooftop and facade
  capacity arrays.
- `Functions.py`: shared evaluation and tariff helpers.

The public `All_output` archive stores the filtered-facade-AOI ML outputs.
The main planning case does not require regenerating this archive:
`Optimization/Multi_period_planning.py` scales the grid arrays to the city
totals stored in
`main_case_city_totals.csv` (RPV 35%, FPV 70% of filtered AOI).

Most scripts in this directory reproduce the expensive upstream evaluation
stage and expect the corresponding Baidu Netdisk data to be placed in the
working directory described by the root readme.
