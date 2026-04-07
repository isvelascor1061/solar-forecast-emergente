import xarray as xr
import numpy as np
from pathlib import Path

class DSWRF1Converter:
    """
    Loads merged NetCDF files that contain DSWRF with a 24-step 'observation_time' dimension
    and a 'step' coordinate (timedelta64[ns] -> hours).
    Converts the cumulative DSWRF data (sdswrf) into strictly hourly dswrf1 values.

    - Block-based approach:
       hours 1..6   => 0..6
       hours 7..12  => 6..12
       hours 13..18 => 12..18
       hours 19..24 => 18..24

    For each file:
       - If hour h == block_start(h)+1 (z.B. 7==6+1),
         we take ds.sdswrf(h) as is (the model already provides a 1-hour average).
       - Otherwise we compute:
         dswrf1[h] = (h - block)*sdswrf[h] - ((h-1 - block)*sdswrf[h-1])
         negative values are clipped to 0
    """

    def __init__(self, input_dir, output_dir):
        """
        Parameters
        ----------
        input_dir : str or Path
            Folder with merged raw NetCDF files (e.g. dswrf_YYYYmmdd_HH00.nc),
            each containing 24 steps of DSWRF data (time, lat, lon or time).
        output_dir : str or Path
            Folder where we save the new DSWRF1 NetCDF files.
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def block_start(self, hour):
        """
        Returns the block's start hour for a given forecast hour h.
        1..6   => 0
        7..12  => 6
        13..18 => 12
        19..24 => 18
        """
        if 1 <= hour <= 6:
            return 0
        elif 7 <= hour <= 12:
            return 6
        elif 13 <= hour <= 18:
            return 12
        elif 19 <= hour <= 24:
            return 18
        else:
            raise ValueError(f"Invalid forecast hour: {hour}")

    def convert_sdswrf_to_hourly(self, ds):
        """
        Creates 'dswrf1' from 'sdswrf'.
        
        ds must have:
         - 'step' coordinate in hours (1..24),
         - 'sdswrf' as a data variable, shape can be:
            * (observation_time=24, lat, lon) 3D
            * (observation_time=24) 1D if only a single lat-lon point was loaded
        """

        # hours = array([1,2,...,24])
        hours_float = ds["step"].values / np.timedelta64(1, "h")
        hours = hours_float.astype(int)  # shape e.g. (24,)

        # This is your cumulative DSWRF
        sdswrf_data = ds["sdswrf"]  # shape could be (24,) or (24, lat, lon)

        ndim = sdswrf_data.ndim  # 2 or 3, depending on if lat/lon are included

        # Create a NEW array dswrf1 with the same shape/type
        dswrf1 = xr.zeros_like(sdswrf_data)
        dswrf1.name = "dswrf1"

        for i in range(len(hours)):  # i=0..23
            h = hours[i]
            b = self.block_start(h)

            if i == 0:
                # Der erste Zeitschritt (h=1) muss ohnehin "so" genommen werden
                dswrf1[i] = sdswrf_data[i]
                continue

            if h == b + 1:
                # Blockanfang => DSWRF1 = sdswrf(h) directly
                if ndim == 1:
                    dswrf1[i] = sdswrf_data[i]
                elif ndim == 2:
                    # shape (24, lat) -> quite unusual, but let's handle anyway
                    dswrf1[i, :] = sdswrf_data[i, :]
                elif ndim == 3:
                    dswrf1[i, :, :] = sdswrf_data[i, :, :]
            else:
                # wir holen hour h und hour h-1
                hprev = hours[i-1]
                bprev = self.block_start(hprev)

                # Je nach Anzahl Dimensionen
                if ndim == 1:
                    current_val = (h - b) * sdswrf_data[i]
                    prev_val = (hprev - bprev) * sdswrf_data[i-1]
                elif ndim == 2:
                    current_val = (h - b) * sdswrf_data[i, :]
                    prev_val = (hprev - bprev) * sdswrf_data[i-1, :]
                elif ndim == 3:
                    current_val = (h - b) * sdswrf_data[i, :, :]
                    prev_val = (hprev - bprev) * sdswrf_data[i-1, :, :]

                diff = current_val - prev_val
                # clip negatives to 0
                diff = diff.clip(min=0)

                if ndim == 1:
                    dswrf1[i] = diff
                elif ndim == 2:
                    dswrf1[i, :] = diff
                else:
                    dswrf1[i, :, :] = diff

        return dswrf1

    def process_file(self, file_path):
        """
        Loads a single file, computes dswrf1, returns a new Dataset with dswrf1.
        """
        ds = xr.open_dataset(file_path)
        dswrf1_da = self.convert_sdswrf_to_hourly(ds)

        # Build a new dataset with dswrf1
        out_ds = xr.Dataset({"dswrf1": dswrf1_da})

        # Copy relevant coords if needed
        for c in ds.coords:
            if c not in out_ds.coords:
                out_ds.coords[c] = ds.coords[c]

        return out_ds

    def convert_all_files(self, launch_time):
        # launch_time als String erwarten, z.B. "0100"
        nc_files = sorted(self.input_dir.glob(f"sdswrf_*{launch_time}.nc"))
    
        for file_path in nc_files:
            print(f"Processing: {file_path.name}")
            try:
                out_ds = self.process_file(file_path)
                out_filename = file_path.name.replace("sdswrf", "dswrf1")
                out_path = self.output_dir / out_filename
    
                out_ds.to_netcdf(out_path)
                print(f"  Saved: {out_path.name}")
            except Exception as e:
                print(f"  Error processing {file_path.name}: {e}")

def main():
    input_dir = "_2_data_postprocessing/_01_merged_Radrawdata/merged_raw_dswrf_1900"
    output_dir = "_2_data_postprocessing/_02_DSWRF1_merged_data/merged_raw_dswrf1_1900"
    LAUNCH_TIME = "1900"

    converter = DSWRF1Converter(input_dir, output_dir)
    converter.convert_all_files(launch_time=LAUNCH_TIME)

if __name__ == "__main__":
    main()
