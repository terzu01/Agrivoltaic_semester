from supporting_functions import *
import numpy as np

# Input definition and data extraction
PPFD = 550  # micromol/M^2/s
max_yield_loss = 0.05  # Maximum acceptable yield loss (5%)
power_per_square_meter = 0.15  # kW per square meter
scaling_factor = 0.05  # Cost increases by 5% per meter of height increase
field_size = [400, 400]  # Length and width of the field
panel_width = 2.1  # Panel width in meters
lifetime = 20  # System lifetime in years
weight_LCOE = 16  # Adjust these weights as needed
weight_power = 0  # Adjust these weights as needed
system_type = 'fixed'
op_maint_cost_per_kW = 13  # Operational and maintenance cost per kW per year

# Load irradiation data
glob_irr_fixed, _, _ = load_irradiation_data()

# Optimization
opt_vars = agrivoltaic_optimization_ga(
    system_type, PPFD, glob_irr_fixed, max_yield_loss, field_size, panel_width,
    power_per_square_meter, lifetime,
    op_maint_cost_per_kW, weight_LCOE, weight_power
)

# Results checking and plotting
result_check(
    opt_vars, system_type, PPFD, glob_irr_fixed, max_yield_loss, field_size, panel_width,
    power_per_square_meter, lifetime,
    op_maint_cost_per_kW, weight_LCOE, weight_power
)

# Optional: Plotting LCOE function
inv_cost_func = fit_investment_cost(system_type, plotting=True)
structure_price_height_func = fit_exponential_price_model(plotting=True)

# Define ranges for row distance and current height
row_dist_range = [2, 20]  # Row distance range from 2m to 20m
current_height_range = [1.5, 5]  # Current height range from 1.5m to 5m

lcoe_func = lambda x, y: lcoe_function(
    x, y, inv_cost_func, structure_price_height_func, field_size, panel_width,
    power_per_square_meter, lifetime, op_maint_cost_per_kW,
    weight_LCOE, weight_power, system_type, display=False
)

data = load_excel_data()
pA, pB, pC, pD = fit_polynomial_models(system_type, plotting=False)

constraint_func = lambda x, y: constraint_function(
    x, y, data, pA, pB, pC, pD, PPFD, glob_irr_fixed, max_yield_loss, field_size
)

# Plot the 3D LCOE function with constraints
plot_3d_function(lcoe_func, constraint_func, row_dist_range, current_height_range, opt_vars)
