#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 17:32:24 2025

@author: leonardmerl
"""
import s3fs

# 🔹 Connect to AWS S3 (public, no credentials needed)
fs = s3fs.S3FileSystem(anon=True)

# 🔹 List available GFS forecast directories
s3_path = "noaa-gfs-bdp-pds/"
available_dates = fs.ls(s3_path)

# 🔹 Print first few dates to see what's available
print("✅ Available GFS forecast dates in AWS S3:")
print(available_dates[:10])  # Show the first 10 available dates





