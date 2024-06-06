#!/bin/bash
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

# HBW
echo -e "\nYou need to register at https://shapy.is.tue.mpg.de"
read -p "Username (SMPL-X):" username
read -p "Password (SMPL-X):" password
username=$(urle $username)
password=$(urle $password)

# mkdir -p ./HBW
# wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=shapy&resume=1&sfile=HBW.zip' -O './HBW.zip' --no-check-certificate --continue
# unzip ./HBW.zip -d ./HBW
# rm ./HBW.zip

# ModelAgencyData
# mkdir -p ./ModelAgencyData
# wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=shapy&resume=1&sfile=ModelAgencyData.zip' -O './ModelAgencyData.zip' --no-check-certificate --continue
# unzip ./ModelAgencyData.zip -d ./ModelAgencyData
# rm ./ModelAgencyData.zip

# SSP-3D
mkdir -p  ./SSP-3D
wget https://github.com/akashsengupta1997/SSP-3D/raw/master/ssp_3d.zip 
unzip ssp_3d.zip -d ./SSP-3D
rm ssp_3d.zip
