# solar_clustering_app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go  # Add this import
import pvlib
from geopy.geocoders import Nominatim
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from io import StringIO
import time
import warnings

# ------------------------------
# Streamlit App Configuration
# ------------------------------
st.set_page_config(
    page_title="Solar Irradiance Clustering",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------
# Suppress FutureWarnings (Optional)
# ------------------------------
warnings.simplefilter(action='ignore', category=FutureWarning)

# ------------------------------
# Helper Functions
# ------------------------------

def get_coordinates(location_name):
    """
    Get latitude and longitude for a given location name using geopy.
    """
    try:
        # Update the User-Agent to comply with Nominatim's policy
        geolocator = Nominatim(user_agent="SolarClusteringApp/1.0 (your_email@example.com)")
        location = geolocator.geocode(location_name)
        if location:
            return location.latitude, location.longitude
        else:
            st.error(f"Location '{location_name}' not found. Please enter a valid location.")
            return None, None
    except Exception as e:
        st.error(f"An error occurred during geocoding: {e}")
        return None, None

def get_solar_data_pvgis(latitude, longitude, year, hourly=True):
    """
    Fetch solar irradiation data from PVGIS for the specified location and year.
    Returns a pandas DataFrame.
    """
    try:
        if hourly:
            # Fetch data in 'csv' format
            data, metadata, inputs = pvlib.iotools.get_pvgis_hourly(
                latitude=latitude,
                longitude=longitude,
                start=year,
                end=year,
                components=True,
                usehorizon=True,
                pvcalculation=False,  # Set to False to get irradiance components
                outputformat='csv'      # Fetch data in CSV format
            )
            # 'data' is a string in CSV format
            if isinstance(data, str):
                # Parse CSV string into DataFrame
                df = pd.read_csv(StringIO(data))
            elif isinstance(data, pd.DataFrame):
                # If data is already a DataFrame
                df = data
            else:
                st.error("Unexpected data format returned from PVGIS.")
                return None
        else:
            # Currently, only hourly data is supported
            st.warning("Only hourly data is supported in this app.")
            return None
        st.success(f"✅ Solar data for {year} successfully retrieved.")
        return df
    except Exception as e:
        st.error(f"An error occurred while fetching solar data for {year}: {e}")
        return None

def preprocess_data(solar_data):
    """
    Preprocess the solar irradiation data.
    """
    # Handle missing values using forward and backward fill
    solar_data = solar_data.ffill().bfill()

    # Check if 'time' column exists
    if 'time' in solar_data.columns:
        solar_data['time'] = pd.to_datetime(solar_data['time'])
        solar_data.set_index('time', inplace=True)
    elif solar_data.index.name == 'time':
        # If 'time' is already the index
        solar_data.index = pd.to_datetime(solar_data.index)
    else:
        # Check for necessary time components
        required_time_cols = ['Year', 'Month', 'Day', 'Hour']
        if all(col in solar_data.columns for col in required_time_cols):
            solar_data['time'] = pd.to_datetime(solar_data[['Year', 'Month', 'Day', 'Hour']])
            solar_data.set_index('time', inplace=True)
            solar_data.drop(columns=required_time_cols, inplace=True)
        else:
            st.error("The solar data does not contain a 'time' column or the necessary time components.")
            return None

    # Resample to hourly data if not already
    try:
        solar_data = solar_data.resample('H').mean()  # 'H' is standard for hourly
    except Exception as e:
        st.error(f"An error occurred during resampling: {e}")
        return None

    return solar_data

def calculate_ghi_dni_dhi(solar_data):
    """
    Calculate GHI, DNI, and DHI from POA irradiance data.
    """
    # Check if required columns exist
    required_columns = ['solar_elevation', 'poa_direct', 'poa_sky_diffuse', 'poa_ground_diffuse']
    if not all(col in solar_data.columns for col in required_columns):
        st.error(f"Solar data is missing required columns: {required_columns}")
        return None

    # Calculate solar zenith angle
    theta_z = 90 - solar_data['solar_elevation']

    # Convert solar zenith angle to radians for cosine calculation
    theta_z_rad = np.radians(theta_z)

    # Avoid division by zero by setting a minimum angle (e.g., 1 degree in radians)
    theta_z_rad = theta_z_rad.where(theta_z_rad > 0.01745, 0.01745)  # 0.01745 radians ≈ 1 degree

    # Calculate DNI
    solar_data['DNI'] = solar_data['poa_direct'] / np.cos(theta_z_rad)

    # Calculate DHI
    solar_data['DHI'] = solar_data['poa_sky_diffuse'] + solar_data['poa_ground_diffuse']

    # Calculate GHI
    solar_data['GHI'] = solar_data['DHI'] + solar_data['DNI'] * np.cos(theta_z_rad)

    return solar_data

def feature_engineering(solar_data):
    """
    Engineer features for clustering.
    """
    # Extract temporal features
    solar_data['Hour'] = solar_data.index.hour
    solar_data['DayOfYear'] = solar_data.index.dayofyear
    solar_data['Month'] = solar_data.index.month

    # Categorize seasons
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Autumn'

    solar_data['Season'] = solar_data['Month'].apply(get_season)

    # Encode categorical features
    solar_data = pd.get_dummies(solar_data, columns=['Season'], drop_first=True)

    # Select features for clustering
    # Using GHI, DNI, DHI as primary irradiance features
    features = solar_data[['GHI', 'DNI', 'DHI', 'Hour', 'DayOfYear']]

    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    return features_scaled, scaler, features

def determine_optimal_clusters(features_scaled, max_k=10):
    """
    Determine the optimal number of clusters using the Silhouette Score.
    """
    silhouette_scores = []
    K = range(2, max_k + 1)

    for k in K:
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=1000)
        cluster_labels = kmeans.fit_predict(features_scaled)
        silhouette_avg = silhouette_score(features_scaled, cluster_labels)
        silhouette_scores.append(silhouette_avg)

    # Create a DataFrame for plotting
    silhouette_df = pd.DataFrame({
        'k': list(K),
        'Silhouette Score': silhouette_scores
    })

    # Plot Silhouette Scores
    fig = px.line(
        silhouette_df,
        x='k',
        y='Silhouette Score',
        title='Silhouette Score vs. Number of Clusters (k)',
        markers=True
    )
    st.plotly_chart(fig, use_container_width=True)

    # Determine optimal k (k with maximum silhouette score)
    optimal_k = silhouette_df.loc[silhouette_df['Silhouette Score'].idxmax(), 'k']
    st.markdown(f"**Optimal number of clusters (k): {optimal_k}** based on the highest Silhouette Score.")

    return int(optimal_k)

def perform_mini_batch_clustering(features_scaled, n_clusters):
    """
    Perform Mini-Batch K-Means clustering.
    """
    mb_kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1000)
    cluster_labels = mb_kmeans.fit_predict(features_scaled)
    return cluster_labels, mb_kmeans

def compute_cluster_centroids(features, cluster_labels, n_clusters):
    """
    Compute the centroids of each cluster.
    """
    centroids_list = []
    for cluster in range(n_clusters):
        cluster_data = features[cluster_labels == cluster]
        centroid = cluster_data.mean()
        centroids_list.append(centroid)
    
    # Concatenate all centroids into a single DataFrame
    centroids = pd.concat(centroids_list, axis=1).T.reset_index(drop=True)
    centroids['Cluster'] = centroids.index
    return centroids

# ------------------------------
# Streamlit App Layout
# ------------------------------

def main():
    st.title("🌞 Solar Irradiance Clustering and Interactive Visualization")

    # Sidebar Inputs
    st.sidebar.header("1. Input Parameters")

    location_name = st.sidebar.text_input("Enter Location Name (e.g., 'Siena, Italy')", "Siena, Italy")
    start_year = st.sidebar.number_input("Enter Start Year (e.g., 2018)", min_value=2000, max_value=2100, value=2018, step=1)
    end_year = st.sidebar.number_input("Enter End Year (e.g., 2020)", min_value=2000, max_value=2100, value=2020, step=1)

    if end_year < start_year:
        st.sidebar.error("End Year must be greater than or equal to Start Year.")
        st.stop()

    # Button to Fetch and Process Data
    if st.sidebar.button("📥 Fetch and Process Data"):
        with st.spinner("Fetching solar data..."):
            latitude, longitude = get_coordinates(location_name)
            if latitude is None or longitude is None:
                st.error("Unable to retrieve coordinates. Please check the location name.")
                st.stop()

            # Initialize an empty list to store data for multiple years
            all_years_data = []

            for year in range(start_year, end_year + 1):
                solar_data = get_solar_data_pvgis(latitude, longitude, year, hourly=True)
                if solar_data is not None:
                    processed_data = preprocess_data(solar_data)
                    if processed_data is not None:
                        processed_data = calculate_ghi_dni_dhi(processed_data)
                        if processed_data is not None:
                            all_years_data.append(processed_data)
                time.sleep(1)  # To prevent overwhelming the server

            if not all_years_data:
                st.error("No solar data retrieved for the specified years.")
                st.stop()

            # Concatenate all years' data
            solar_data_combined = pd.concat(all_years_data)
            st.success("✅ Solar data fetched and processed successfully!")

        with st.spinner("Performing feature engineering..."):
            features_scaled, scaler, features = feature_engineering(solar_data_combined)
            st.success("✅ Feature engineering completed!")

        with st.spinner("Determining the optimal number of clusters..."):
            optimal_k = determine_optimal_clusters(features_scaled, max_k=10)

        with st.spinner("Performing clustering..."):
            cluster_labels, kmeans_model = perform_mini_batch_clustering(features_scaled, optimal_k)
            solar_data_combined['Cluster'] = cluster_labels
            st.success("✅ Clustering completed!")

        with st.spinner("Computing cluster centroids..."):
            cluster_centroids = compute_cluster_centroids(features, cluster_labels, optimal_k)
            st.success("✅ Cluster centroids computed!")

        # Store data in session state for later use
        st.session_state['solar_data'] = solar_data_combined
        st.session_state['features'] = features
        st.session_state['cluster_centroids'] = cluster_centroids

    # Check if data is available
    if 'solar_data' in st.session_state:
        solar_data = st.session_state['solar_data']
        features = st.session_state['features']
        cluster_centroids = st.session_state['cluster_centroids']

        st.header("📊 Clustering Results")

        # Display cluster distribution
        st.subheader("🔢 Cluster Distribution")
        cluster_counts = solar_data['Cluster'].value_counts().sort_index()
        fig_cluster = px.bar(
            x=cluster_counts.index.astype(str),
            y=cluster_counts.values,
            labels={'x': 'Cluster', 'y': 'Number of Days'},
            title='Number of Days per Cluster'
        )
        st.plotly_chart(fig_cluster, use_container_width=True)

        # Download clustered data
        csv_clustered = solar_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Clustered Data as CSV",
            data=csv_clustered,
            file_name='clustered_solar_data.csv',
            mime='text/csv',
        )

        # Download cluster centroids
        csv_centroids = cluster_centroids.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Cluster Centroids as CSV",
            data=csv_centroids,
            file_name='cluster_centroids.csv',
            mime='text/csv',
        )

        # Interactive Visualization
        st.header("🔍 Interactive Cluster Visualization")

        st.markdown("""
        Select any two variables to visualize the relationship between them. The data points are colored based on their cluster assignments, and cluster centroids are marked for reference.
        """)

        # Dropdowns for selecting variables
        all_features = ['GHI', 'DNI', 'DHI', 'Hour', 'DayOfYear']
        col1, col2 = st.columns(2)

        with col1:
            plot_x = st.selectbox("Select X-axis Variable", all_features, index=0)
        with col2:
            plot_y = st.selectbox("Select Y-axis Variable", all_features, index=1)

        if plot_x == plot_y:
            st.warning("⚠️ Please select different variables for X and Y axes.")
        else:
            # Create scatter plot for the main data
            fig = px.scatter(
                solar_data,
                x=plot_x,
                y=plot_y,
                color=solar_data['Cluster'].astype(str),
                title=f"Clusters Visualized on {plot_x} vs {plot_y}",
                labels={'color': 'Cluster'},
                opacity=0.6,
                hover_data=['Cluster']
            )

            # Add cluster centroids using graph_objects.Scatter
            fig.add_trace(
                go.Scatter(
                    x=cluster_centroids[plot_x],
                    y=cluster_centroids[plot_y],
                    mode='markers',
                    marker=dict(color='black', size=12, symbol='x'),
                    name='Centroids'
                )
            )

            # Update layout for better aesthetics
            fig.update_layout(
                legend_title_text='Cluster',
                legend=dict(
                    itemsizing='constant'
                )
            )

            st.plotly_chart(fig, use_container_width=True)

            # Display centroids
            st.subheader("📌 Cluster Centroids")
            st.dataframe(cluster_centroids[[plot_x, plot_y, 'Cluster']])

            # Download centroids specific to selected variables
            centroids_selected = cluster_centroids[[plot_x, plot_y, 'Cluster']]
            csv_centroids_selected = centroids_selected.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Selected Centroids as CSV",
                data=csv_centroids_selected,
                file_name='cluster_centroids_selected.csv',
                mime='text/csv',
            )

    else:
        st.info("🔍 Please fetch and process data to view clustering results.")

# ------------------------------
# Run the App
# ------------------------------
if __name__ == "__main__":
    main()
