# autoCCT
This script makes CCT matrix using x-rite or spydercheckr 24 colorcheckers to correct camera colors.
To install requirements, use the command:
  
  pip install -r requirements.txt
  
Connect and authorise phone to adb to automatically get new photos or use --nophone to get colorchecker photo locally (save it to last_photo.jpg in the folder from you're starting script)

Normally with adb connected phone you should use script like that:

  python autoCubes.py --from0 --matrix
 
 or
  
  python autoCubes.py --from0 --gcam --matrix
  for gcam
  
After that the script will save customCCT.txt to PhotonCamera folder and will save it locally to customCCT_autoCubes.txt
There's also different keys, that i'll explain later.
