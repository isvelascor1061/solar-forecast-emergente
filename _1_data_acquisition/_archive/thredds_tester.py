import requests
from datetime import datetime, timedelta
from bs4 import BeautifulSoup

# 🔹 Base THREDDS URL for NCAR RDA GFS data
base_thredds_url = "https://thredds.rda.ucar.edu/thredds/catalog/files/g/ds084.1"

# 🔹 Define the forecast date and time
forecast_date = "20150101"  # YYYYMMDD
forecast_hour = "00"  # GFS model run time (00, 06, 12, 18)

# 🔹 Forecast time steps to check (every hour for 24 hours)
forecast_steps = [str(i).zfill(3) for i in range(1, 25)]  # "001" to "024"

# 🔹 Construct the THREDDS catalog URL for this date
year = forecast_date[:4]
catalog_url = f"{base_thredds_url}/{year}/{forecast_date}/catalog.html"

# 🔹 Request the catalog page
response = requests.get(catalog_url)

if response.status_code == 200:
    # 🔹 Parse the catalog HTML
    soup = BeautifulSoup(response.text, "html.parser")

    # 🔹 Extract all available forecast files
    available_files = [link.text for link in soup.find_all("a") if link.text.endswith(".grib2")]

    # 🔹 Check if DSWRF1 data is available for all 24 hours
    available_steps = []
    for step in forecast_steps:
        filename = f"gfs.0p25.{forecast_date}{forecast_hour}.f{step}.grib2"
        if filename in available_files:
            available_steps.append(step)

    # 🔹 Print results
    if len(available_steps) == 24:
        print(f"✅ All 24 forecast hours are available for DSWRF1 on {forecast_date} at {forecast_hour} UTC!")
    else:
        print(f"⚠️ Only {len(available_steps)}/24 forecast hours are available for DSWRF1 on {forecast_date} at {forecast_hour} UTC.")
        print(f"Available forecast steps: {available_steps}")

else:
    print(f"❌ Failed to access THREDDS server. HTTP Status Code: {response.status_code}")


