# Code structure

- `Evaluation/` contains the UMEP preprocessing, feature and label
  extraction, regression training, and 102-city inference workflow. The large
  public inference archive is used as the grid-level PV potential input for
  the planning workflow.
- `Optimization/` contains the land-use-informed annual load
  generator and the 2030–2050 multi-period RPV–FPV–storage planning model.
- `Figures/` contains the scripts used for the main-text and selected
  supplementary figures.

All public scripts use repository-relative paths by default. Set
`FPV_DATA_ROOT` when the downloaded `##Data` directory is stored outside
the repository root.
