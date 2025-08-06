# Saturation Point Analysis

## Overview

The saturation point analysis feature helps marketers identify the spending levels at which marketing channels begin to show diminishing returns. This is crucial for budget optimization as it indicates when additional spending becomes less effective.

## What is Saturation?

In marketing mix modeling, saturation occurs when increasing spend on a channel yields progressively smaller incremental returns. The saturation point represents the spending level beyond which the return on investment significantly decreases.

## Usage

### Basic Usage

```python
from robyn.allocator.allocator import BudgetAllocator
from robyn.visualization.allocator_visualizer import AllocatorVisualizer

# Create visualizer from your budget allocator
visualizer = AllocatorVisualizer(budget_allocator)

# Calculate saturation points
saturation_df = visualizer.calculate_saturation_points(
    saturation_percentage=0.8,  # 80% of maximum response
    export_path="saturation_points.csv"
)
```

### Output Format

The function returns a DataFrame with the following columns:

- `year_month`: The time period (YYYY-MM format)
- `spend_name`: The marketing channel name
- `current_spend`: Current spending level
- `saturation_point`: Spending level at which saturation occurs

### Methods for Calculating Saturation

#### 1. Response-Based Method (Default)

This method defines saturation as the point where the channel reaches a certain percentage of its maximum theoretical response.

```python
# Find where channel reaches 80% of maximum possible response
saturation_df = visualizer.calculate_saturation_points(
    method="response_based",
    saturation_percentage=0.8,
    export_path="saturation_80pct.csv"
)
```

**When to use**: When you want to understand how close you are to the theoretical maximum effectiveness of a channel.

#### 2. Marginal Return-Based Method

This method defines saturation as the point where the marginal return (additional response per dollar) drops below a threshold percentage of the current marginal return.

```python
# Find where marginal return drops to 10% of current level
saturation_df = visualizer.calculate_saturation_points(
    method="marginal_based",
    marginal_threshold=0.1,  # 10% of current marginal return
    export_path="saturation_marginal.csv"
)
```

**When to use**: When you want to identify the point where spending efficiency significantly decreases relative to current performance.

## Interpreting Results

### Example Output

```
year_month | spend_name | current_spend | saturation_point
2024-01   | TV         | 50,000        | 150,000
2024-01   | Digital    | 30,000        | 45,000
2024-01   | Radio      | 20,000        | 80,000
```

### Analysis

1. **Under-saturated channels** (current_spend << saturation_point):

   - Have room for increased investment
   - Likely to show good returns on additional spending

2. **Near-saturated channels** (current_spend ≈ saturation_point):

   - Operating at efficient levels
   - Additional spending will show diminishing returns

3. **Over-saturated channels** (current_spend > saturation_point):
   - May benefit from budget reallocation
   - Consider reducing spend and moving to under-saturated channels

## Advanced Analysis

### Comparing Different Saturation Thresholds

```python
# Analyze saturation at different levels
for threshold in [0.7, 0.8, 0.9]:
    df = visualizer.calculate_saturation_points(
        saturation_percentage=threshold,
        export_path=f"saturation_{int(threshold*100)}pct.csv"
    )

    # Calculate distance to saturation
    df['distance_to_saturation'] = df['saturation_point'] - df['current_spend']
    df['saturation_ratio'] = df['current_spend'] / df['saturation_point']

    print(f"At {threshold*100}% saturation level:")
    print(df[['spend_name', 'saturation_ratio', 'distance_to_saturation']])
```

### Marginal Return Analysis

```python
# Compare marginal returns at different thresholds
for threshold in [0.05, 0.1, 0.2]:  # 5%, 10%, 20% of current marginal
    df = visualizer.calculate_saturation_points(
        method="marginal_based",
        marginal_threshold=threshold,
        export_path=f"saturation_marginal_{int(threshold*100)}pct.csv"
    )
    print(f"Saturation points at {threshold*100}% marginal threshold:")
    print(df)
```

## Best Practices

1. **Regular Analysis**: Run saturation analysis periodically (monthly/quarterly) to track changes in channel effectiveness.

2. **Multiple Methods**: Use both response-based and marginal-based methods for a comprehensive view.

3. **Context Matters**: Consider external factors (seasonality, competition, market conditions) when interpreting results.

4. **Gradual Changes**: Don't make drastic budget changes based solely on saturation analysis. Test incremental changes first.

5. **Channel Interactions**: Remember that channels may interact; reducing spend in one channel might affect others.

## Technical Details

### Hill Transformation

The saturation calculation is based on the Hill transformation used in the Robyn model:

```
Response = coeff * (x^alpha / (x^alpha + inflexion^alpha))
```

Where:

- `x` is the adstocked spend
- `alpha` controls the shape of the curve
- `inflexion` is the half-saturation point
- `coeff` is the coefficient scaling factor

### Limitations

1. **Static Analysis**: Saturation points are calculated based on current model parameters and may change over time.

2. **Single Channel View**: The analysis considers each channel independently, not accounting for interaction effects.

3. **Model Dependency**: Results are only as good as the underlying MMM model.

## See Also

- [Budget Allocation Documentation](./budget_allocation.md)
- [Response Curves Documentation](./response_curves.md)
- [Robyn Model Documentation](./robyn_model.md)
