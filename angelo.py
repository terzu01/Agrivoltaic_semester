import numpy as np
import matplotlib.pyplot as plt
from pvlib import pvsystem, modelchain, irradiance, temperature, iotools, location as pv_location
import pandas as pd
from datetime import datetime
import pytz


def compute_panel_shadows(n_panels, n_rows, panel_length, panel_width,
                          height_from_ground, row_spacing, surface_tilt,
                          surface_azimuth, times, location):
    """
    Computes and visualizes the shadows of solar panels on the ground.

    Parameters:
    - n_panels: Total number of panels.
    - n_rows: Number of rows of panels.
    - panel_length: Length of a panel (along its surface normal direction).
    - panel_width: Width of a panel.
    - height_from_ground: Height of the bottom edge of the panel from the ground.
    - row_spacing: Spacing between rows.
    - surface_tilt: Tilt angle of the panel (degrees from horizontal).
    - surface_azimuth: Azimuth angle of the panel (degrees from North, increasing eastward).
    - times: pandas.DatetimeIndex of the time(s) to compute shadows for.
    - location: pvlib.location.Location object.
    """
    # Convert angles from degrees to radians
    surface_tilt_rad = np.deg2rad(surface_tilt)
    surface_azimuth_rad = np.deg2rad(surface_azimuth)

    # Compute the number of columns
    n_cols = n_panels // n_rows

    # Get solar position data
    solar_position = location.get_solarposition(times)

    # Initialize plot
    plt.figure(figsize=(12, 8))

    # Loop over times
    for idx, time in enumerate(times):
        # Extract zenith and azimuth angles
        zenith = solar_position['zenith'].values[idx]
        azimuth = solar_position['azimuth'].values[idx]

        # Skip times when sun is below the horizon
        if zenith >= 90:
            continue

        # Convert sun angles from degrees to radians
        theta_s_rad = np.deg2rad(zenith)
        phi_s_rad = np.deg2rad(azimuth)

        # Compute sun's direction vector
        d_sun_x = -np.sin(theta_s_rad) * np.sin(phi_s_rad)
        d_sun_y = -np.sin(theta_s_rad) * np.cos(phi_s_rad)
        d_sun_z = -np.cos(theta_s_rad)

        d_sun = np.array([d_sun_x, d_sun_y, d_sun_z])

        # Loop over each panel
        for row in range(n_rows):
            for col in range(n_cols):
                # Panel center position
                x0 = col * (panel_width + 0.5)  # Adjust 0.5 as needed for panel spacing
                y0 = row * (row_spacing + panel_length)
                z0 = height_from_ground

                # Panel normal vector
                n_p_x = np.sin(surface_tilt_rad) * np.sin(surface_azimuth_rad)
                n_p_y = np.sin(surface_tilt_rad) * np.cos(surface_azimuth_rad)
                n_p_z = np.cos(surface_tilt_rad)
                n_p = np.array([n_p_x, n_p_y, n_p_z])

                # Local axes of the panel
                z_unit_vector = np.array([0.0, 0.0, 1.0])  # Ensure this is of float type
                axis_u = np.cross(n_p, z_unit_vector)
                if np.linalg.norm(axis_u) == 0:
                    axis_u = np.array([1.0, 0.0, 0.0])  # Ensure this is of float type
                axis_u /= np.linalg.norm(axis_u)
                axis_v = np.cross(axis_u, n_p)

                # Corner points in local coordinates
                w = panel_width
                l = panel_length
                corners_local = np.array([
                    [-w / 2, -l / 2, 0],
                    [w / 2, -l / 2, 0],
                    [w / 2, l / 2, 0],
                    [-w / 2, l / 2, 0]
                ])

                # Transform to global coordinates
                rotation_matrix = np.column_stack((axis_u, axis_v, n_p))
                corners_global = corners_local @ rotation_matrix.T + np.array([x0, y0, z0])

                # Compute shadow points
                t = -corners_global[:, 2] / d_sun_z
                shadow_points = corners_global[:, :2] + np.outer(t, d_sun[:2])

                # Plot panel (projected onto ground for visualization)
                plt.fill(corners_global[:, 0], corners_global[:, 1], 'skyblue', alpha=0.5,
                         label='Panel' if row == 0 and col == 0 and idx == 0 else "")

                # Plot shadow
                plt.fill(shadow_points[:, 0], shadow_points[:, 1], 'gray', alpha=0.5,
                         label='Shadow' if row == 0 and col == 0 and idx == 0 else "")

        # Formatting the plot
        plt.title(f"Shadows of Solar Panels at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        plt.xlabel('East-West Position (m)')
        plt.ylabel('North-South Position (m)')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()


# Define the location and environment
latitude = 45.17088421209007
longitude = 9.19305983636286
site_location = pv_location.Location(latitude, longitude, tz='Europe/Rome', altitude=70)

# Parameters of the PV System
sandia_modules = pvsystem.retrieve_sam('SandiaMod')
sapm_inverters = pvsystem.retrieve_sam('cecinverter')
module = sandia_modules['Canadian_Solar_CS5P_220M___2009_']
inverter = sapm_inverters['ABB__MICRO_0_25_I_OUTD_US_208__208V_']
temperature_model_parameters = temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']

# Retrieve meteorological data using PVGIS
weather = iotools.get_pvgis_hourly(latitude, longitude, start="2021-07-01", end="2021-08-01")[0]

weather['poa_diffuse'] = weather['poa_sky_diffuse'] + weather['poa_ground_diffuse']
weather['poa_global'] = weather['poa_diffuse'] + weather['poa_direct']

# Define the number of panels and topology of the PV system
n_panels = 15
n_rows = 3
n_cols = n_panels // n_rows
panel_length = 1.65  # meters
panel_width = 0.99  # meters
height_from_ground = 10  # meters above ground
row_spacing = 2.0  # spacing between rows in meters

# Define PV system and model chain
from pvlib.pvsystem import PVSystem, Array, FixedMount
from pvlib.modelchain import ModelChain

mount = FixedMount(surface_tilt=0, surface_azimuth=180)
array = Array(
    mount=mount,
    module_parameters=module,
    temperature_model_parameters=temperature_model_parameters,
)
system = PVSystem(arrays=[array], inverter_parameters=inverter)
mc = ModelChain(system, site_location)

# Run ModelChain to estimate energy production
mc.run_model_from_poa(weather)
energy_output = mc.results.ac

# Get the time when the PV system is generating maximum power
max_power_time = energy_output.idxmax()

# For testing purposes, let's pick the time when the PV system is generating maximum power
times = pd.DatetimeIndex([max_power_time])

# Now, call the compute_panel_shadows function
compute_panel_shadows(n_panels, n_rows, panel_length, panel_width,
                      height_from_ground, row_spacing, mount.surface_tilt,
                      mount.surface_azimuth, times, site_location)



