import subprocess,sys
from pathlib import Path
r=Path(__file__).parent
for s in ['run_chemical_bands.py','run_band_error.py']:
 print('RUN',s,flush=True)
 subprocess.run([sys.executable,str(r/'scripts'/s)],check=True)
