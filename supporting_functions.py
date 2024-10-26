import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, differential_evolution, NonlinearConstraint, Bounds
from scipy.special import erf
from scipy.constants import c, h, N_A
import matplotlib.pyplot as plt

# OPTIMIZATION
def agrivoltaic_optimization_ga(agri_type, PPFD, avg_solar_irr, max_yield_loss, field_size, panel_width,
                                power_per_square_meter, lifetime,
                                op_maint_cost_per_kW, weight_LCOE, weight_power):
    # Load data from Excel
    inv_cost_func = fit_investment_cost(agri_type, plotting=False)
    data = load_excel_data()
    pA, pB, pC, pD = fit_polynomial_models(agri_type, plotting=False)
    structure_price_height_func = fit_exponential_price_model(plotting=False)

    # Bounds for variables (height and row distance)
    lb = [1.5, 2.0]  # Lower bounds for height and row distance
    ub = [5.0, 20.0]  # Upper bounds for height and row distance
    bounds = Bounds(lb, ub)

    # Define the fitness function
    def fitness(x):
        height, row_dist = x
        lcoe = weight_LCOE * calculate_LCOE(
            inv_cost_func, row_dist, field_size, panel_width,
            height, power_per_square_meter, structure_price_height_func, lifetime,
            op_maint_cost_per_kW, agri_type, display_data=False
        )
        power = weight_power * calculate_rated_power(
            power_per_square_meter,
            calculate_total_panel_area(field_size[1]/row_dist, field_size[0], panel_width)
        )
        return lcoe - power

    # Define the constraint function
    def constraint(x):
        c = constraints_ga(x, data, field_size[1], pA, pB, pC, pD, PPFD, avg_solar_irr, max_yield_loss)
        return c

    # Create a NonlinearConstraint object
    nl_constraint = NonlinearConstraint(constraint, -np.inf, 0.0)  # c <= 0

    # Run differential evolution with constraints
    result = differential_evolution(fitness, bounds=bounds, constraints=(nl_constraint,), strategy='best1bin',
                                    maxiter=100, popsize=50, tol=1e-6, mutation=(0.5, 1), recombination=0.7, disp=True)

    opt_vars = result.x
    n_rows = round(field_size[1] / opt_vars[1])

    # Checking results
    result_check(opt_vars, agri_type, PPFD, avg_solar_irr, max_yield_loss, field_size, panel_width,
                 power_per_square_meter, lifetime,
                 op_maint_cost_per_kW, weight_LCOE, weight_power)
    return opt_vars

def constraints_ga(x, data, field_size, pA, pB, pC, pD, PPFD, avg_irr, max_yield_loss):
    height, row_dist = x
    n_rows = field_size / row_dist + 10

    # Compute shadow parameters
    _, b, c, _ = shadow_parameters(pA, pB, pC, pD, height)

    # Compute shadow shape
    field_x, shade = shadow_shape(1, b, c, 0, row_dist, field_size, n_rows)

    # Compute solar and plant irradiance, then yield loss
    sun_irr = solar_irr(data['lambda'], data['irr'], avg_irr)
    irr_plant = plant_irr(PPFD, data['lambda'], data['irr'], data['quantum_yield'])
    avg_yield_loss, _ = yield_loss(data['lambda'], shade, sun_irr, irr_plant, field_x)

    # Nonlinear inequality constraint (c <= 0)
    c = avg_yield_loss - max_yield_loss
    return c

# SHADOW SHAPE AND CROP YIELD LOSS
def fit_polynomial_models(fit_type, plotting=False):
    # Validate input
    if fit_type == 'fixed':
        file_name = 'fixed_fitting_results.xlsx'
    elif fit_type == 'track':
        file_name = 'track_fitting_results.xlsx'
    else:
        raise ValueError("Invalid input: fit_type must be 'fixed' or 'track'.")

    # Load the data
    data = pd.read_excel(file_name)

    # Extract predictors and responses
    h = data['h'].values
    a = data['a'].values
    b = data['b'].values
    c = data['c'].values
    d = data['d'].values

    # Polynomial degree to use
    poly_degree = 2  # Quadratic model

    # Fit polynomial models
    pA = np.polyfit(h, a, poly_degree)
    pB = np.polyfit(h, b, poly_degree)
    pC = np.polyfit(h, c, poly_degree)
    pD = np.polyfit(h, d, poly_degree)

    # Optionally plot the results
    if plotting:
        plot_fitting(h, a, b, c, d, pA, pB, pC, pD)

    return pA, pB, pC, pD

def shadow_parameters(pA, pB, pC, pD, height):
    # Evaluate the polynomial at the given height
    a_fit = np.polyval(pA, height)
    b_fit = np.polyval(pB, height)
    c_fit = np.polyval(pC, height)
    d_fit = np.polyval(pD, height)
    return a_fit, b_fit, c_fit, d_fit

def plot_fitting(h, a, b, c, d, pA, pB, pC, pD):
    h_plot = np.linspace(min(h), max(h), 100)
    a_fit = np.polyval(pA, h_plot)
    b_fit = np.polyval(pB, h_plot)
    c_fit = np.polyval(pC, h_plot)
    d_fit = np.polyval(pD, h_plot)

    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    plt.plot(h, a, 'ko', markerfacecolor='k')
    plt.plot(h_plot, a_fit, 'r-', linewidth=2)
    plt.title('Model Fit for a')
    plt.xlabel('h')
    plt.ylabel('a')
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(h, b, 'ko', markerfacecolor='k')
    plt.plot(h_plot, b_fit, 'r-', linewidth=2)
    plt.title('Model Fit for b')
    plt.xlabel('h')
    plt.ylabel('b')
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(h, c, 'ko', markerfacecolor='k')
    plt.plot(h_plot, c_fit, 'r-', linewidth=2)
    plt.title('Model Fit for c')
    plt.xlabel('h')
    plt.ylabel('c')
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(h, d, 'ko', markerfacecolor='k')
    plt.plot(h_plot, d_fit, 'r-', linewidth=2)
    plt.title('Model Fit for d')
    plt.xlabel('h')
    plt.ylabel('d')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def shadow_shape(a, b, c, d, raw_dist, field_size, n_rows):
    field_x = np.linspace(-field_size / 2, field_size / 2, 400)
    model = lambda p, x_vals: p[0] * (1 - 0.5 * (erf((x_vals + p[1] / 2 - p[3]) / np.sqrt(p[2])) +
                                                 erf((-x_vals + p[1] / 2 + p[3]) / np.sqrt(p[2]))))

    if n_rows == 1:
        erf_coeff = [a, b, c, d]
        erf_result = model(erf_coeff, field_x)
    else:
        erf_coeff_l = [a, b, c, d - (1 - 0.5) * raw_dist]
        erf_coeff_r = [a, b, c, d + (1 - 0.5) * raw_dist]
        erf_result = model(erf_coeff_l, field_x) + model(erf_coeff_r, field_x) - 1
        for i in range(2, int(n_rows / 2) + 1):
            erf_coeff_l = [a, b, c, d - (i - 0.5) * raw_dist]
            erf_coeff_r = [a, b, c, d + (i - 0.5) * raw_dist]
            erf_result += model(erf_coeff_l, field_x) + model(erf_coeff_r, field_x) - 2

    y = erf_result
    x = field_x
    return x, y

def yield_loss(lambda_vals, shade, sun_irr, plant_irr, field_x):
    yield_loss_vector = np.zeros_like(field_x)
    for i in range(len(field_x)):
        field_matrix = sun_irr * shade[i] - plant_irr
        field_matrix[field_matrix >= 0] = 0
        numerator = integral(lambda_vals, -field_matrix, 300, 800)
        denominator = integral(lambda_vals, plant_irr, 300, 800)
        yield_loss_vector[i] = numerator / denominator

    yield_loss_avg = np.mean(yield_loss_vector)
    return yield_loss_avg, yield_loss_vector

# OTHERS
def integral(x, y, x_min, x_max):
    idx_min = np.searchsorted(x, x_min, side='left')
    idx_max = np.searchsorted(x, x_max, side='right')
    x_integrate = x[idx_min:idx_max]
    y_integrate = y[idx_min:idx_max]
    result = np.trapz(y_integrate, x_integrate)
    return result

# SOLAR AND PLANT IRRADIATION CURVE
def plant_irr(PPFD, lambda_vals, irr, quantum_yield):
    PPFD = PPFD * 1e-6 * N_A  # Convert PPFD to photons per second
    Ph_en = h * c / (lambda_vals * 1e-9)  # Photon energy
    ph_lambda = irr / Ph_en

    lambda_min = 400
    lambda_max = 700
    integr = integral(lambda_vals, ph_lambda, lambda_min, lambda_max)
    x = PPFD / integr

    plant_irr_result = x * quantum_yield * irr
    return plant_irr_result

def solar_irr(lambda_vals, irr, mean_irr):
    normalization_factor = np.trapz(irr, lambda_vals)
    solar_irr_result = irr / normalization_factor * mean_irr
    return solar_irr_result

# DATA LOADING
def load_excel_data():
    # Filename of the Excel sheet
    filename = 'PVL_SpectrumCalculator1.xlsx'

    # Load data from Excel
    data = pd.read_excel(filename, sheet_name='Sheet1')

    data_dict = {
        'lambda': data.iloc[40:110, 0].values,  # Wavelength [nm]
        'irr': data.iloc[40:110, 1].values,     # Sun irradiance [W/m^2/nm]
        'abs': data.iloc[40:110, 2].values,     # Solar cell absorption [%]
        'eqe': data.iloc[40:110, 4].values,     # Solar cell EQE [%]
        'quantum_yield': data.iloc[40:110, 3].values,  # McCree quantum yield [%]
    }
    data_dict['trans'] = np.zeros(70)  # Transparency [%] (initialized but unused)
    return data_dict

def load_irradiation_data():
    glob_fixed = np.zeros((24, 12))
    direct_fixed = np.zeros((24, 12))
    diff_fixed = np.zeros((24, 12))

    glob_track = np.zeros((24, 12))
    direct_track = np.zeros((24, 12))
    diff_track = np.zeros((24, 12))

    glob_tilted = np.zeros((24, 12))
    direct_tilted = np.zeros((24, 12))
    diff_tilted = np.zeros((24, 12))

    temperature = np.zeros((24, 12))

    for month in range(1, 13):
        filename1 = f'{month}.csv'
        filename2 = f'{month}_tilted.csv'

        # Adjust the delimiter and parsing based on your CSV files
        data1 = pd.read_csv(filename1, sep='\t', header=None)
        data2 = pd.read_csv(filename2, sep='\t', header=None)

        table = np.zeros((24, 10))
        for i in range(len(data1)):
            # Split the data based on spaces and tabs
            split = data1.iloc[i, 0].split()
            split2 = data2.iloc[i, 0].split()
            row_data = [float(x) for x in split[1:]] + [float(x) for x in split2[1:]]
            table[i, :] = row_data

        glob_fixed[:, month - 1] = table[:, 0]
        direct_fixed[:, month - 1] = table[:, 1]
        diff_fixed[:, month - 1] = table[:, 2]

        glob_track[:, month - 1] = table[:, 3]
        direct_track[:, month - 1] = table[:, 4]
        diff_track[:, month - 1] = table[:, 5]

        glob_tilted[:, month - 1] = table[:, 6]
        direct_tilted[:, month - 1] = table[:, 7]
        diff_tilted[:, month - 1] = table[:, 8]

        temperature[:, month - 1] = table[:, 9]

    # calculate the annual average (24h)
    glob_fixed_avg = np.mean(glob_fixed, axis=1)
    direct_fixed_avg = np.mean(direct_fixed, axis=1)
    diff_fixed_avg = np.mean(diff_fixed, axis=1)

    glob_track_avg = np.mean(glob_track, axis=1)
    direct_track_avg = np.mean(direct_track, axis=1)
    diff_track_avg = np.mean(diff_track, axis=1)

    glob_tilted_avg = np.mean(glob_tilted, axis=1)
    direct_tilted_avg = np.mean(direct_tilted, axis=1)
    diff_tilted_avg = np.mean(diff_tilted, axis=1)

    temp_avg = np.mean(temperature, axis=1)

    # calculate annual-daily average of irradiation for each system
    fixed = np.mean(glob_fixed_avg[glob_fixed_avg != 0])
    tilted = np.mean(glob_tilted_avg[glob_tilted_avg != 0])
    track = np.mean(glob_track_avg[glob_track_avg != 0])

    return fixed, tilted, track

# FITTING FUNCTIONS
def fit_investment_cost(system_type, plotting=False):
    # Data for fitting
    power = np.array([200, 500, 1000, 5000, 10000, 15000, 20000])
    if system_type == 'track':
        cost = np.array([2.83, 2.09, 1.98, 1.76, 1.68, 1.61, 1.59])
    elif system_type == 'fixed':
        cost = np.array([1.85, 1.65, 1.5, 1.4, 1.35, 1.31, 1.29])
    else:
        print('Error: system type must be either "track" or "fixed".')
        return None

    # Define the power law fit function
    def power_fit_func(x, a, b):
        return a * x ** b

    # Curve fitting
    params, _ = curve_fit(power_fit_func, power, cost, p0=[2, -0.1])

    # Create a function handle for the fitted model
    def cost_func(rated_power):
        return power_fit_func(rated_power, *params)

    # Plotting the results if required
    if plotting:
        plt.figure()
        plt.scatter(power, cost, color='red', label='Data')
        power_range = np.linspace(min(power), max(power), 100)
        plt.plot(power_range, power_fit_func(power_range, *params), 'r-', linewidth=2, label='Fitted Power Law')
        plt.xlabel('Rated Power (kW)')
        plt.ylabel('Cost ($/Wdc)')
        plt.title(f'Power Law Curve Fit for PV + Crops Installation Costs: {system_type}')
        plt.legend()
        plt.grid(True)
        plt.show()

    return cost_func

def fit_exponential_price_model(plotting=False):
    # Data points
    height = np.array([1.0, 2.5, 4.0])
    price = np.array([132, 216, 355])
    base_height = 2.5  # Hypothesized base height

    # Define exponential fit function
    def exp_func(x, a, b):
        return a * np.exp(b * x)

    # Perform the fitting
    params, _ = curve_fit(exp_func, height, price, p0=[1, 0.1])

    # Normalization factor
    normalization_factor = exp_func(base_height, *params)

    # Adjusted function handle
    def structure_price_func(x):
        return exp_func(x, *params) / normalization_factor

    # Plotting if required
    if plotting:
        plt.figure()
        plt.scatter(height, price, label='Data')
        height_range = np.linspace(min(height), max(height), 100)
        plt.plot(height_range, exp_func(height_range, *params), 'r-', label='Fitted Exponential')
        plt.title('Fit of Price vs. Height')
        plt.xlabel('Height (m)')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure()
        normalized_prices = price / normalization_factor
        plt.scatter(height, normalized_prices, label='Normalized Data')
        plt.plot(height_range, structure_price_func(height_range), 'b-', label='Normalized Fitted Function')
        plt.title('Normalized Fit of Price vs. Height')
        plt.xlabel('Height (m)')
        plt.ylabel('Normalized Price')
        plt.legend()
        plt.grid(True)
        plt.show()

    return structure_price_func

# LCOE Calculation
def calculate_LCOE(inv_cost_func, row_dist, field_size, panel_width,
                   current_height, power_per_square_meter, structure_price_height_func, lifetime,
                   op_maint_cost_per_kWh, system_type, display_data=False):
    # Calculate the final investment cost
    num_rows = field_size[1] / row_dist
    final_investment = calculate_final_investment(inv_cost_func, power_per_square_meter, num_rows, field_size[0],
                                                  panel_width, current_height, structure_price_height_func)

    # Calculate the total panel area
    total_panel_area = calculate_total_panel_area(num_rows, field_size[0], panel_width)

    # Calculate the rated power of the system
    rated_power = calculate_rated_power(power_per_square_meter, total_panel_area)

    # Calculate the annual energy production
    annual_energy = calculate_annual_energy(rated_power, system_type)

    # Calculate the total operational and maintenance cost over the system's lifetime
    total_op_maint_cost = calculate_total_op_maint_cost(rated_power, op_maint_cost_per_kWh, lifetime)

    # Calculate the total energy produced over the lifetime
    total_energy_produced = annual_energy * lifetime

    # Calculate the Levelized Cost of Energy (LCOE)
    lcoe = (final_investment + total_op_maint_cost) / total_energy_produced

    if display_data:
        print(f'Final Investment Cost: ${final_investment:.2f}')
        print(f'Total Panel Area: {total_panel_area:.2f} square meters')
        print(f'Rated Power: {rated_power:.2f} kW')
        print(f'Annual Energy Production: {annual_energy:.2f} kWh')
        print(f'Total Operational and Maintenance Cost: ${total_op_maint_cost:.2f}')
        print(f'Total Energy Produced Over Lifetime: {total_energy_produced:.2f} kWh')
        print(f'Levelized Cost of Energy (LCOE): ${lcoe:.3f} per kWh')

    return lcoe

def calculate_final_investment(cost_function, power_per_square_meter, num_rows, row_length, row_width,
                               current_height, exp_price_func):
    total_panel_area = calculate_total_panel_area(num_rows, row_length, row_width)
    rated_power = calculate_rated_power(power_per_square_meter, total_panel_area)
    base_investment = calculate_total_investment(cost_function, rated_power)
    final_investment = scale_cost_for_height(current_height, exp_price_func, base_investment)
    return final_investment

def calculate_total_investment(cost_function, rated_power):
    cost_per_watt_dc = cost_function(rated_power)
    total_investment = cost_per_watt_dc * rated_power * 1000  # Convert kW to W
    return total_investment

def calculate_total_op_maint_cost(rated_power, op_maint_cost_per_kWh, lifetime):
    total_op_maint_cost = rated_power * op_maint_cost_per_kWh * lifetime
    return total_op_maint_cost

def calculate_annual_energy(rated_power, system_type):
    if system_type == 'track':
        annual_production_factor = 1720
    elif system_type == 'fixed':
        annual_production_factor = 1400
    else:
        raise ValueError('System type must be "track" or "fixed".')
    annual_energy = rated_power * annual_production_factor
    return annual_energy

def calculate_total_panel_area(num_rows, row_length, row_width):
    total_panel_area = num_rows * row_length * row_width
    return total_panel_area

def calculate_rated_power(power_per_square_meter, total_panel_area):
    rated_power = power_per_square_meter * total_panel_area
    return rated_power

def scale_cost_for_height(current_height, exp_price_func, base_installation_cost):
    # Normalize the exponential fitted function
    structure_cost_ratio = 0.1
    adjusted_cost = base_installation_cost * (1 + structure_cost_ratio * (exp_price_func(current_height) - 1))
    return adjusted_cost

# DISPLAY RESULTS
def result_check(opt_vars, agri_type, PPFD, avg_solar_irr, max_yield_loss, field_size, panel_width,
                 power_per_square_meter, lifetime,
                 op_maint_cost_per_kW, weight_LCOE, weight_power):
    inv_cost_func = fit_investment_cost(agri_type, plotting=False)
    data = load_excel_data()
    pA, pB, pC, pD = fit_polynomial_models(agri_type, plotting=True)
    structure_price_height_func = fit_exponential_price_model(plotting=False)
    n_rows = round(field_size[1] / opt_vars[1])

    # Checking results
    print(f'Optimal panel height: {opt_vars[0]:.2f} m')
    print(f'Optimal row distance: {opt_vars[1]:.2f} m')
    print(f'Number of rows: {n_rows:.0f}')
    print('Breakdown of the LCOE calculation:')
    calculate_LCOE(inv_cost_func, opt_vars[1], field_size, panel_width,
                   opt_vars[0], power_per_square_meter, structure_price_height_func, lifetime,
                   op_maint_cost_per_kW, agri_type, display_data=True)

    _, b, c, _ = shadow_parameters(pA, pB, pC, pD, opt_vars[0])
    field_x, shade = shadow_shape(1, b, c, 0, opt_vars[1], field_size[1], n_rows)
    sun_irr = solar_irr(data['lambda'], data['irr'], avg_solar_irr)
    irr_plant = plant_irr(PPFD, data['lambda'], data['irr'], data['quantum_yield'])
    avg_yield_loss, yield_loss_vector = yield_loss(data['lambda'], shade, sun_irr, irr_plant, field_x)

    plt.figure()
    plt.plot(field_x, yield_loss_vector * 100, label='Yield loss')
    plt.plot(field_x, np.ones_like(field_x) * avg_yield_loss * 100, 'r--', label='Average yield loss')
    plt.title('Crop yield loss along field length')
    plt.xlabel('Position (m)')
    plt.ylabel('Crop yield decrease (%)')
    plt.legend()
    plt.show()

    print(f'The average crop yield loss is {avg_yield_loss * 100:.3f} %.')

    row_dist_range = [2, 20]  # Row distance range from 2m to 20m
    current_height_range = [1.5, 5]  # Current height range from 1.5m to 5m

    lcoe_func = lambda x, y: lcoe_function(
        x, y, inv_cost_func, structure_price_height_func, field_size,
        panel_width, power_per_square_meter, lifetime, op_maint_cost_per_kW,
        weight_LCOE, weight_power, agri_type, display=False
    )
    constraint_func = lambda x, y: constraint_function(
        x, y, data, pA, pB, pC, pD, PPFD, avg_solar_irr, max_yield_loss, field_size
    )

    plot_3d_function(lcoe_func, constraint_func, row_dist_range, current_height_range, opt_vars)

def lcoe_function(row_dist, current_height, inv_cost_func, structure_price_height_func, field_size,
                  panel_width, power_per_square_meter, lifetime, op_maint_cost_per_kW, weight_LCOE,
                  weight_power, agri_type, display):
    lcoe = weight_LCOE * calculate_LCOE(
        inv_cost_func, row_dist, field_size, panel_width,
        current_height, power_per_square_meter, structure_price_height_func, lifetime,
        op_maint_cost_per_kW, agri_type, display
    ) - weight_power * calculate_rated_power(
        power_per_square_meter,
        calculate_total_panel_area(field_size[1] / row_dist, field_size[0], panel_width)
    )
    return lcoe

def constraint_function(row_dist, current_height, data, pA, pB, pC, pD, PPFD, glob_irr_fixed,
                        max_yield_loss, field_size):
    avg_yield_loss = calculate_constraints(
        current_height, row_dist, data, pA, pB, pC, pD, PPFD, glob_irr_fixed, field_size
    )
    if avg_yield_loss > max_yield_loss:
        constraint_value = 100  # Arbitrary high value for violation
    else:
        constraint_value = np.nan  # Use NaN to indicate no violation
    return constraint_value

def calculate_constraints(height, row_dist, data, pA, pB, pC, pD, PPFD, avg_solar_irr, field_size):
    n_rows = field_size[0] / row_dist + 10
    _, b, c, _ = shadow_parameters(pA, pB, pC, pD, height)
    field_x, shade = shadow_shape(1, b, c, 0, row_dist, field_size[0], n_rows)
    sun_irr = solar_irr(data['lambda'], data['irr'], avg_solar_irr)
    irr_plant = plant_irr(PPFD, data['lambda'], data['irr'], data['quantum_yield'])
    avg_yield_loss, _ = yield_loss(data['lambda'], shade, sun_irr, irr_plant, field_x)
    return avg_yield_loss

def plot_3d_function(lcoe_func, constraint_func, x_range, y_range, optimal_point):
    # Generate x and y values
    x = np.linspace(x_range[0], x_range[1], 50)
    y = np.linspace(y_range[0], y_range[1], 50)
    X, Y = np.meshgrid(x, y)

    # Calculate LCOE and constraints
    Z_lcoe = np.array([[lcoe_func(xi, yi) for xi, yi in zip(row_x, row_y)] for row_x, row_y in zip(X, Y)])
    Z_constraint = np.array([[constraint_func(xi, yi) for xi, yi in zip(row_x, row_y)] for row_x, row_y in zip(X, Y)])

    # Plotting the LCOE surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z_lcoe, cmap='viridis', edgecolor='none', alpha=0.7)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    ax.set_xlabel('Row Distance (m)')
    ax.set_ylabel('Current Height (m)')
    ax.set_zlabel('LCOE ($/kWh)')
    ax.set_title('3D Plot of LCOE with Constraint Violations and Optimal Point')

    # Highlight areas of constraint violations
    violation_indices = np.where(Z_constraint == 100)
    violationX = X[violation_indices]
    violationY = Y[violation_indices]
    violationZ = Z_lcoe[violation_indices]
    ax.scatter(violationX, violationY, violationZ, color='red', marker='x', s=50, label='Constraint Violated')

    # Plot the optimal point
    optimalX = optimal_point[1]
    optimalY = optimal_point[0]
    optimalZ = lcoe_func(optimalX, optimalY)
    ax.scatter(optimalX, optimalY, optimalZ, color='green', marker='o', s=100, label='Optimal Point')

    ax.legend()
    plt.show()
