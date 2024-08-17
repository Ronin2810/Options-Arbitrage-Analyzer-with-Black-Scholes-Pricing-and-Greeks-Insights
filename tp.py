import requests

# Replace 'YOUR_API_KEY' with your actual FRED API key
api_key = 'facfdd6e5c8599cdbf0598a7434a1bf4'
series_id = 'DGS10'  # Series ID for 10-Year Treasury Constant Maturity Rate
base_url = 'https://api.stlouisfed.org/fred/series/observations'

# Parameters for the API request
params = {
    'series_id': series_id,
    'api_key': api_key,
    'file_type': 'json',
    'sort_order': 'desc',
    'limit': 1  # Get the most recent data point
}

# Send the request to the FRED API
response = requests.get(base_url, params=params)
data = response.json()

# Extract the 10-year Treasury rate
if 'observations' in data and data['observations']:
    rate = data['observations'][0]['value']
    print(f"10-Year Treasury Bill Rate: {rate}%")
else:
    print("Data not found or unable to retrieve the rate.")
