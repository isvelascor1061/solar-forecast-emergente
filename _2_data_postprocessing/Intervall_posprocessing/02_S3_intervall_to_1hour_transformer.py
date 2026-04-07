#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert a cumulative GFS radiation field to strict hourly values.

Block logic (0-6, 6-12, 12-18, 18-24) is unchanged; only the variable
names are now user-definable via VAR_IN / VAR_OUT in main().
"""

from pathlib import Path
import xarray as xr
import numpy as np


class HourlyConverter:
    def __init__(self, input_dir, output_dir, var_in: str, var_out: str):
        """
        Parameters
        ----------
        input_dir  : folder with merged NetCDF files (24-step cumulative field)
        output_dir : where the hourly NetCDFs will be written
        var_in     : name of the cumulative variable inside the files
        var_out    : name of the hourly variable to create
        """
        self.input_dir   = Path(input_dir)
        self.output_dir  = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.var_in  = var_in    # e.g. "sdlwrf"  or "sdswrf"
        self.var_out = var_out   # e.g. "dlwrf1"  or "dswrf1"

    # ------------------------------------------------------------------ #
    @staticmethod
    def block_start(hour):
        """Return the start hour of the 6-h block containing *hour*."""
        if   1 <= hour <= 6:   return 0
        elif 7 <= hour <= 12:  return 6
        elif 13 <= hour <= 18: return 12
        elif 19 <= hour <= 24: return 18
        raise ValueError(hour)

    # ------------------------------------------------------------------ #
    def _to_hourly(self, ds):
        """
        Build *var_out* from *var_in*; keeps dimensionality (1D, 3D, …).
        """
        hours = (ds["step"] / np.timedelta64(1, "h")).astype(int)  # 1…24
        cum   = ds[self.var_in]

        hourly = xr.zeros_like(cum, dtype=cum.dtype)
        hourly.name = self.var_out
        ndim = cum.ndim

        for i, h in enumerate(hours):
            b = self.block_start(int(h))

            if i == 0 or h == b + 1:          # first step of block
                hourly[i] = cum[i]
                continue

            h_prev  = int(hours[i - 1])
            b_prev  = self.block_start(h_prev)

            factor_now  = (h      - b)
            factor_prev = (h_prev - b_prev)

            cur_val  = factor_now  * cum[i]
            prev_val = factor_prev * cum[i - 1]
            diff     = (cur_val - prev_val).clip(min=0)

            hourly[i] = diff

        return hourly

    # ------------------------------------------------------------------ #
    def _process_file(self, nc_path: Path):
        ds      = xr.open_dataset(nc_path)
        out_da  = self._to_hourly(ds)

        out_ds  = xr.Dataset({self.var_out: out_da}, coords=ds.coords)

        out_name = nc_path.name.replace(self.var_in, self.var_out)
        out_ds.to_netcdf(self.output_dir / out_name)

        ds.close(); out_ds.close()
        print("✓", out_name)

    # ------------------------------------------------------------------ #
    def convert_all(self, launch_tag: str):
        """
        launch_tag: e.g. "0100", "0700" – selects only those files.
        """
        pattern = f"{self.var_in}_*{launch_tag}.nc"
        files   = sorted(self.input_dir.glob(pattern))

        if not files:
            print(f"No files matching {pattern} in {self.input_dir}")
            return

        for f in files:
            try:
                self._process_file(f)
            except Exception as e:
                print(f"✖ {f.name}: {e}")


# ---------------------------------------------------------------------- #
def main():
    # ------------ USER SETTINGS ------------------------------------- #
    INPUT_DIR  = "_2_data_postprocessing/_01_merged_Radrawdata/dswrf/merged_raw_dswrf_0100"
    OUTPUT_DIR = "_2_data_postprocessing/_02_merged_Rad1data/dswrf/merged_raw_dswrf1_0100"
    
    VAR_IN  = "sdswrf"      #name of the variable in the merged files -> all sdswrf
    VAR_OUT = "dswrf1"      #name of the variable in the output file -> dswrf1 launch time is added in script
    
    LAUNCH_TIME = "0100"    # z.B. "0100", "0700", ...
        # ---------------------------------------------------------------- #

    conv = HourlyConverter(INPUT_DIR, OUTPUT_DIR, VAR_IN, VAR_OUT)
    conv.convert_all(launch_tag=LAUNCH_TIME)


if __name__ == "__main__":
    main()
    """
    This code is the second step of the intervall Data Postprocessing 
    it converts the downloaded intervall values back to 1 hour intervalls
    -> the values on the server are mean values resetting in a 6-Hour Block 
    for the first forecasthour the inervall is over 1 hour 
    for the second its the intervall from 1-2 
    for the sixth its the intervall from 1-6
    and then it resets 
    so the value for hour 6 is 6xM(6) -5xM(5)  (M(6) is the mean intervall value on the server)
    The code does the following:
    1. It takes the daily values from the server (Merged in the previous script)
    2. Uses the explained logic to reverse the medial allegation
    3. Changes the variable to dswrf1_launch_time (launch time being 0100,0700..)
    (1 meaning 1 hour intervall) and then saves them as daily data again 
    this step is very important, because later on we use this info in the feature Vektor script 
    to iterate over all launc times
    4. Input dir is the folder with the daily merged files of 1 launch time (eg 0100)
       Output dir is the folder for the new reverse medial allegated daily values
    5. var in is the name of the variable in the merged raw files
        var out is the name of the variable in the new reverse allegated files -> dswrf1
    6. launch time needs to correspond with 1 of the 4 runs you are doing right now -> 
    this script needs to be executed ones for each run over the 4 years 
    
    Important -> At this point this pipeline only works with 1 tile in the grid 
    -> to work with multiple it would need to subset with each corresponding tile 
    -> Code to do that would need to be added 
    """
   
   
   
   
   
   
   
   
   
   
   

