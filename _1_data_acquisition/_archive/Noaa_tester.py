#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 10:31:01 2025

@author: leonardmerl
"""

import s3fs

# 🔹 AWS S3 Public Bucket for NOAA GFS Data
S3_BUCKET = "noaa-gfs-bdp-pds"

# 🔹 Define the forecast date and time
forecast_date = "20210411"  # YYYYMMDD format
forecast_hour = "00"  # Model run time (00, 06, 12, 18)

# 🔹 Forecast time steps to check (every hour for 24 hours)
forecast_steps = [str(i).zfill(3) for i in range(1, 25)]  # "001" to "024"

# 🔹 Connect to AWS S3 (public access, no credentials needed)
fs = s3fs.S3FileSystem(anon=True)

# 🔹 List available forecast files in the AWS S3 bucket
s3_path = f"{S3_BUCKET}/gfs.{forecast_date}/{forecast_hour}/atmos/"
try:
    available_files = fs.ls(s3_path)
except Exception as e:
    print(f"❌ Failed to access AWS S3: {e}")
    available_files = []

# 🔹 Check if DSWRF1 is available for all 24 hours
available_steps = []
for step in forecast_steps:
    filename = f"gfs.t{forecast_hour}z.pgrb2.0p25.f{step}"
    if any(filename in f for f in available_files):
        available_steps.append(step)

# 🔹 Print results
if len(available_steps) == 24:
    print(f"✅ All 24 forecast hours are available for DSWRF1 on {forecast_date} at {forecast_hour} UTC!")
else:
    print(f"⚠️ Only {len(available_steps)}/24 forecast hours are available for DSWRF1 on {forecast_date} at {forecast_hour} UTC.")
    print(f"Available forecast steps: {available_steps}")
