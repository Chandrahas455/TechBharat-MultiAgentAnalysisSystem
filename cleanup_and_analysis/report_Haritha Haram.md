# Final Report on Haritha Haram Plantation Dataset

## Dataset Overview and Quality Summary
The transformed dataset contains **31 rows and 4 columns** relating to the Haritha Haram plantation initiative. The key attributes are:
- **Districts**: String identifiers for each area.
- **Nurseries**: Number of nurseries available (integer).
- **Seedlings (lakh Nos.)**: Total number of seedlings (in lakh).
- **Plantation (lakh Nos.)**: Total number of plantations done (in lakh).

### Data Quality Metrics
- **Duplicates Removed**: 0
- **Missing Values**: 0
- **Data Validity**: Valid, with no identified issues or warnings.

## Key Trends and Correlations
The analysis of correlations between key variables indicated:
- A strong positive correlation (|r| >= 0.5) exists between **Seedlings (lakh Nos.)** and **Plantation (lakh Nos.)** which indicates that as the number of seedlings planted increases, the amount of plantation also tends to rise.

### Correlation Coefficient:
The correlation coefficient between **Seedlings (lakh Nos.)** and **Plantation (lakh Nos.)** is **strongly positive**, reflecting the dependency.

The plot below visualizes this correlation:

![Seedlings vs Plantation](charts\scatter_Seedlings (lakh Nos.)_vs_Plantation (lakh Nos.)_20250907_181844.png)

## Notable Patterns, Gaps, and Imbalances
- **Nursery Distribution**: Some districts appear to have no nurseries at all which can pose a risk to future plantation efforts.
- **High Dependency**: Districts with low nurseries are directly affecting the output of seedlings and plantations, indicating potential areas for future development.

## Clear, Actionable Recommendations
1. **Invest in Nurseries**: Strengthen nursery capacity in districts with fewer resources, particularly by building new nurseries or enhancing existing ones.
2. **Monitor Outputs**: Regularly analyze the output of seedlings in correlation with the number of nurseries to ensure sustainability and healthy growth.
3. **Focused Initiatives**: Target additional support in districts struggling with seedling production to balance growth across all areas.

With these strategies implemented, the Haritha Haram initiative can ensure greater success and more balanced ecological development through improved plantation efforts.