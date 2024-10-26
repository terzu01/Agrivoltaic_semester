import pvlib
import pandas as pd
from geopy.geocoders import Nominatim
import pytz
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time

def get_coordinates(location_name):
    """
    Get latitude and longitude for a given location name using geopy.
    """
    try:
        # Update the User-Agent to comply with Nominatim's policy
        geolocator = Nominatim(user_agent="AgrivoltaicOptimizer/1.0 (your_email@example.com)")
        location = geolocator.geocode(location_name)
        if location:
            print(f"Coordinates of '{location_name}': ({location.latitude}, {location.longitude})")
            return location.latitude, location.longitude
        else:
            raise ValueError(f"Location '{location_name}' not found.")
    except Exception as e:
        print(f"An error occurred during geocoding: {e}")
        raise

def get_solar_data_pvgis(latitude, longitude, year, hourly=True):
    """
    Fetch solar irradiation data from PVGIS for the specified location and year.
    """
    try:
        if hourly:
            data, metadata, inputs = pvlib.iotools.get_pvgis_hourly(
                latitude=latitude,
                longitude=longitude,
                start=year,
                end=year,
                components=True,
                usehorizon=True,
                pvcalculation=False,  # Set to False to get irradiance components
                outputformat='json'
                # Removed 'session' parameter
            )
        else:
            data, metadata = pvlib.iotools.get_pvgis_tmy(
                latitude=latitude,
                longitude=longitude,
                outputformat='json'
                # Removed 'session' parameter
            )
        print(f"Solar data for {year} successfully retrieved.")
        return data
    except Exception as e:
        print(f"An error occurred while fetching solar data for {year}: {e}")
        return None

def preprocess_data(solar_data):
    """
    Preprocess the solar irradiation data.
    """
    # Handle missing values using forward fill
    solar_data = solar_data.ffill()

    # Ensure consistent hourly data with lowercase 'h'
    solar_data = solar_data.resample('h').mean()

    return solar_data

def calculate_ghi_dni_dhi(solar_data):
    """
    Calculate GHI, DNI, and DHI from POA irradiance data.
    """
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

    # Drop 'Season' column if it still exists
    if 'Season' in solar_data.columns:
        solar_data = solar_data.drop('Season', axis=1)

    # Select features for clustering
    # Using GHI, DNI, DHI as primary irradiance features
    features = solar_data[['GHI', 'DNI', 'DHI', 'Hour', 'DayOfYear']]

    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    return features_scaled, scaler

def determine_optimal_clusters(features_scaled, max_k=10):
    """
    Determine the optimal number of clusters using Elbow and Silhouette methods.
    """
    wcss = []
    silhouette_scores = []
    K = range(2, max_k+1)

    for k in K:
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=1000)
        kmeans.fit(features_scaled)
        wcss.append(kmeans.inertia_)

        cluster_labels = kmeans.labels_
        silhouette_avg = silhouette_score(features_scaled, cluster_labels)
        silhouette_scores.append(silhouette_avg)

    # Plot Elbow and Silhouette Scores
    plt.figure(figsize=(14, 6))

    # Elbow Method
    plt.subplot(1, 2, 1)
    sns.lineplot(x=list(K), y=wcss, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.title('Elbow Method for Optimal k')

    # Silhouette Scores
    plt.subplot(1, 2, 2)
    sns.lineplot(x=list(K), y=silhouette_scores, marker='o', color='orange')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Optimal k')

    plt.tight_layout()
    plt.show()

    # Return the optimal k based on the maximum silhouette score
    optimal_k = K[silhouette_scores.index(max(silhouette_scores))]
    print(f"Optimal number of clusters based on Silhouette Score: {optimal_k}")
    return optimal_k

def perform_mini_batch_clustering(features_scaled, n_clusters):
    """
    Perform Mini-Batch K-Means clustering.
    """
    mb_kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1000)
    mb_kmeans.fit(features_scaled)
    cluster_labels = mb_kmeans.labels_
    return cluster_labels, mb_kmeans

def visualize_clusters(features_scaled, cluster_labels, kmeans, n_components=2):
    """
    Visualize clusters using PCA.
    """
    pca = PCA(n_components=n_components, random_state=42)
    principal_components = pca.fit_transform(features_scaled)

    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=principal_components[:,0], y=principal_components[:,1], hue=cluster_labels, palette='viridis', s=50, alpha=0.6)

    # Plot cluster centers
    centers = pca.transform(kmeans.cluster_centers_)
    plt.scatter(centers[:,0], centers[:,1], c='red', s=200, alpha=0.75, marker='X', label='Centroids')

    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('Clusters Visualization with PCA')
    plt.legend()
    plt.tight_layout()
    plt.show()

def visualize_clusters_tsne(features_scaled, cluster_labels, n_components=2, sample_size=10000, perplexity=30, n_iter=500):
    """
    Visualize clusters using t-SNE with sampling and optimized parameters.
    """
    if len(features_scaled) > sample_size:
        sample_indices = np.random.choice(len(features_scaled), sample_size, replace=False)
        features_sampled = features_scaled[sample_indices]
        cluster_labels_sampled = cluster_labels[sample_indices]
    else:
        features_sampled = features_scaled
        cluster_labels_sampled = cluster_labels

    # Perform PCA before t-SNE for speed
    pca_n_components = min(50, features_sampled.shape[1])
    pca = PCA(n_components=pca_n_components, random_state=42)
    features_pca = pca.fit_transform(features_sampled)

    # Perform t-SNE
    tsne = TSNE(n_components=n_components, random_state=42, perplexity=perplexity, n_iter=n_iter)
    tsne_results = tsne.fit_transform(features_pca)

    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=tsne_results[:,0], y=tsne_results[:,1], hue=cluster_labels_sampled, palette='viridis', s=50, alpha=0.6)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('Clusters Visualization with t-SNE')
    plt.legend(title='Cluster')
    plt.tight_layout()
    plt.show()

def plot_daily_dni_heatmap(aggregated_daily_dni_sorted, save_path='daily_dni_heatmap.png'):
    """
    Plots a heatmap of daily average DNI sorted by month and day.
    
    Parameters:
    - aggregated_daily_dni_sorted (pd.DataFrame): Aggregated daily data sorted by DNI with 'DNI', 'Month', and 'Day' columns.
    - save_path (str): File path to save the plot image.
    """
    # Pivot the data to create a heatmap (Month vs. Day)
    pivot_table = aggregated_daily_dni_sorted.pivot(index='Month', columns='Day', values='DNI')
    print(f"Pivot table shape: {pivot_table.shape}")  # Should be (12, 31)
    
    # Handle missing days (e.g., April 31st doesn't exist) by filling NaNs with zeros or another appropriate value
    pivot_table = pivot_table.fillna(0)  # Alternatively, use pivot_table.fillna(pivot_table.mean())
    
    # Plotting
    plt.figure(figsize=(20, 10))
    sns.heatmap(pivot_table, cmap='viridis', linewidths=.5, cbar_kws={'label': 'Average DNI'})
    plt.xlabel('Day of Month')
    plt.ylabel('Month')
    plt.title('Heatmap of Daily Average DNI')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Heatmap saved to {save_path}.")

def get_cluster_representatives(aggregated_data, cluster_labels, n_clusters):
    """
    Get representative data for each cluster.
    """
    aggregated_data = aggregated_data.copy()
    aggregated_data['Cluster'] = cluster_labels

    # Select only numeric columns
    numeric_cols = aggregated_data.select_dtypes(include=[np.number]).columns
    representatives = aggregated_data.groupby('Cluster')[numeric_cols].mean()

    # Add cluster sizes
    cluster_sizes = aggregated_data['Cluster'].value_counts().sort_index()
    representatives['Size'] = cluster_sizes

    return representatives

def main():
    start_time = time.time()
    
    # Input
    location_name = input("Enter the location name (e.g., 'Chamoson, Switzerland'): ")
    start_year = int(input("Enter the start year (e.g., 2018): "))
    end_year = int(input("Enter the end year (e.g., 2020): "))
    
    # Step 1: Get Coordinates
    latitude, longitude = get_coordinates(location_name)
    
    # Step 2: Fetch Solar Data for Multiple Years
    all_years_data = []
    for year in range(start_year, end_year + 1):
        yearly_data = get_solar_data_pvgis(latitude, longitude, year, hourly=True)
        if yearly_data is not None:
            print(f"Year {year} data shape before preprocessing: {yearly_data.shape}")
            all_years_data.append(yearly_data)
    
    if not all_years_data:
        print("No solar data retrieved for the specified years.")
        return
    
    # Concatenate all years' data
    solar_data = pd.concat(all_years_data)
    print(f"Total data points collected after concatenation: {len(solar_data)}")
    
    # Step 3: Preprocess Data
    solar_data = preprocess_data(solar_data)
    print(f"Data shape after preprocessing: {solar_data.shape}")
    
    # Step 4: Calculate GHI, DNI, DHI
    solar_data = calculate_ghi_dni_dhi(solar_data)
    print(f"Data shape after calculating GHI, DNI, DHI: {solar_data.shape}")
    
    # Step 5: Handle Leap Years by Excluding February 29th
    # This ensures each year has 365 days, resulting in 8760 hours
    solar_data = solar_data[~((solar_data.index.month == 2) & (solar_data.index.day == 29))]
    print(f"Data shape after removing February 29th: {solar_data.shape}")
    
    # Step 6: Feature Engineering
    features_scaled, scaler = feature_engineering(solar_data)
    print(f"Features scaled shape: {features_scaled.shape}")
    
    # Step 7: Aggregate Data by Averaging Each Hour Across Multiple Years
    # Create 'DayOfYear' and 'Hour' columns if not already present
    solar_data['DayOfYear'] = solar_data.index.dayofyear
    solar_data['Hour'] = solar_data.index.hour
    
    # Group by 'DayOfYear' and 'Hour' to compute the average across years
    aggregated_data = solar_data.groupby(['DayOfYear', 'Hour']).agg({
        'GHI': 'mean',
        'DNI': 'mean',
        'DHI': 'mean'
    }).reset_index()
    
    print(f"Aggregated data shape: {aggregated_data.shape}")  # Should be 365 days * 24 hours = 8760 rows
    
    # Optional: Add 'Month' and 'Day' for visualization purposes
    # Create a representative date (e.g., year=2020) to map DayOfYear to Month and Day
    representative_year = 2020  # Leap year, but Feb 29th has been excluded
    aggregated_data['Date'] = pd.to_datetime(aggregated_data['DayOfYear'], format='%j').apply(lambda x: x.replace(year=representative_year))
    aggregated_data['Month'] = aggregated_data['Date'].dt.month
    aggregated_data['Day'] = aggregated_data['Date'].dt.day
    
    # Drop the 'Date' column as it's no longer needed
    aggregated_data = aggregated_data.drop('Date', axis=1)
    
    # Update features for clustering
    # You can include 'Hour' and 'DayOfYear' as features or exclude them based on your analysis needs
    clustering_features = aggregated_data[['GHI', 'DNI', 'DHI', 'Hour', 'DayOfYear']]
    
    # Normalize the features
    scaler = StandardScaler()
    features_scaled_aggregated = scaler.fit_transform(clustering_features)
    
    # Step 8: Determine Optimal Clusters
    optimal_k = determine_optimal_clusters(features_scaled_aggregated, max_k=10)
    
    # Step 9: Perform Clustering using Mini-Batch K-Means
    cluster_labels, kmeans = perform_mini_batch_clustering(features_scaled_aggregated, n_clusters=optimal_k)
    print(f"Cluster labels assigned: {len(cluster_labels)}")
    
    # Step 10: Visualize Clusters with PCA
    visualize_clusters(features_scaled_aggregated, cluster_labels, kmeans, n_components=2)
    
    # Step 11: Visualize Clusters with t-SNE
    visualize_clusters_tsne(features_scaled_aggregated, cluster_labels, n_components=2, sample_size=10000, perplexity=30, n_iter=500)
    
    # Step 12: Get Cluster Representatives
    aggregated_data['Cluster'] = cluster_labels
    cluster_reps = get_cluster_representatives(aggregated_data, cluster_labels, n_clusters=optimal_k)
    print("Cluster Representatives:")
    print(cluster_reps)
    
    # Step 13: Aggregate Daily DNI for Heatmap
    # Group by 'Month' and 'Day' to calculate daily average DNI
    aggregated_daily_dni = aggregated_data.groupby(['Month', 'Day']).agg({'DNI': 'mean'}).reset_index()
    print(f"Aggregated daily DNI shape: {aggregated_daily_dni.shape}")  # Should be up to 365 rows
    
    # Step 14: Sort Aggregated Data by DNI from High to Low
    aggregated_daily_dni_sorted = aggregated_daily_dni.sort_values(by='DNI', ascending=False).reset_index(drop=True)
    
    # Step 15: Plot Daily DNI with Cluster Comparison (Heatmap)
    plot_daily_dni_heatmap(aggregated_daily_dni_sorted, save_path='daily_dni_heatmap.png')
    
    # Step 16: Save the Representatives to a CSV
    cluster_reps.to_csv('cluster_representatives.csv')
    print("Cluster representatives saved to 'cluster_representatives.csv'.")
    
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

if __name__ == '__main__':
    main()
