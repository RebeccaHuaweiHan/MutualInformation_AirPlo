# Mutual Information Based Correlation Analysis of Air Pollutants

## Overview

This repository contains the code and resources for the paper *"Mutual Information Based Correlation Analysis of Health-Related Multiple Air Pollutants"* by Huawei Han and Wesley S. Burr, presented at the 6th International Conference on Statistics: Theory and Applications (ICSTA'24). The project focuses on analyzing correlations among air pollutants in Toronto, ON, Canada, using mutual information (MI) and the Maximal Information Coefficient (MIC) to support risk factor selection in multi-pollutant health risk models.

## Project Description

Air pollutants such as ground-level ozone (O3), particulate matter (PM2.5), and nitrogen oxides (NOx) are associated with adverse public health outcomes, including cardiovascular and respiratory diseases. This study addresses challenges in multi-pollutant health risk assessment, such as statistical interactions and collinearity, by applying MI-based correlation analysis to data from the Canadian National Air Pollution Surveillance (NAPS) Program (2000–2021). The analysis compares MIC with traditional Pearson Correlation Coefficients (PCC) to evaluate linear and nonlinear relationships among pollutants, facilitating more effective variable selection for models like Generalized Additive Models (GAMs).

### Key Objectives

- Prepare and preprocess air pollutant datasets (NO, NO2, NOx, CO, SO2, O3, PM2.5) and temperature data for Toronto, ON.
- Compute MI and MIC to capture both linear and nonlinear dependencies among pollutants.
- Compare MIC with PCC to assess their effectiveness in identifying pollutant relationships.
- Analyze seasonal variations (warm: April–September; cold: October–March) to understand context-specific correlations.

## Methodology

1. **Data Preparation**:

   - Source: Canadian National Air Pollution Surveillance (NAPS) Program and Canadian Centre for Climate Services.
   - Pollutants: NO, NO2, NOx, CO, SO2, O3, PM2.5 (PM10 excluded due to missing data).
   - Timeframe: Daily mean concentrations from 2000 to 2021.
   - Preprocessing: Calculated 24-hour mean concentrations, imputed missing data, and removed time dependencies for residual analysis.

2. **Correlation Analysis**:

   - **Mutual Information (MI)**: Measures mutual dependence between variables using entropy.
   - **Maximal Information Coefficient (MIC)**: Normalizes MI to a 0–1 scale for comparable analysis across pollutants.
   - **Pearson Correlation Coefficient (PCC)**: Used for comparison to evaluate linear relationships.
   - Visualizations: Heatmaps and scatter plots to illustrate pollutant relationships and seasonal variations.

3. **Tools and Libraries**:

   - **Python**: Data preprocessing, MI/MIC calculations, and visualization.
   - **Libraries**: NumPy, Pandas, Matplotlib, Scikit-learn (for data handling and analysis).
   - **Data Source**: NAPS database and historical climate data.

## Results

- **MIC vs. PCC**: MIC captures both linear and nonlinear relationships more effectively than PCC, which often underestimates nonlinear dependencies (e.g., between SO2 and O3).
- **Key Findings**:
  - Strong correlations among NO, NO2, and NOx due to shared emission sources (e.g., industrial emissions, vehicles).
  - Temperature shows stronger nonlinear relationships with NO and NOx in cold seasons, as captured by MIC.
  - CO and PM2.5 exhibit moderate correlations with other pollutants across seasons.
- **Seasonal Analysis**: MIC reveals stronger associations in specific seasons (e.g., T–NOx in warm seasons, CO–O3 in warm seasons) compared to PCC.
- **Implications**: MIC provides a robust method for risk factor selection in health risk models, improving the accuracy of multi-pollutant assessments.

## Citation

If you use this code or findings, please cite:

> Han, H., Burr, W. S. (2024). "Mutual Information Based Correlation Analysis of Health-Related Multiple Air Pollutant." *6th International Conference on Statistics: Theory and Applications (ICSTA'24)*.

## Contact

- **Huawei Han**: rebeccahan@trentu.ca
- **GitHub**: github.com/RebeccaHuaweiHan

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
