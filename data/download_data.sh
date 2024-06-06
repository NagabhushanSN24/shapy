#!/bin/bash
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

# SMPL-X model
echo -e "\nYou need to register at https://smpl-x.is.tue.mpg.de"
read -p "Username (SMPL-X):" username
read -p "Password (SMPL-X):" password
username=$(urle $username)
password=$(urle $password)

mkdir -p ./body_models
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=models_smplx_v1_1.zip' -O './body_models/smplx.zip' --no-check-certificate --continue
# wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=SMPLX_NEUTRAL_2020.npz' -O './body_models/smplx_neutral.npz' --no-check-certificate --continue
unzip ./body_models/smplx.zip -d ./body_models/smplx1
mv ./body_models/smplx1/models/smplx ./body_models/smplx
rmdir ./body_models/smplx1/models
rmdir ./body_models/smplx1

# SMPL model
# echo -e "\nYou need to register at https://smpl.is.tue.mpg.de"
# read -p "Username (SMPL-X):" username
# read -p "Password (SMPL-X):" password
# username=$(urle $username)
# password=$(urle $password)

# mkdir -p ./body_models
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_python_v.1.1.0.zip' -O './body_models/smpl.zip' --no-check-certificate --continue
unzip ./body_models/smpl.zip -d ./body_models/smpl1
mv ./body_models/smpl1/SMPL_python_v.1.1.0/smpl ./body_models/smpl
rmdir ./body_models/smpl1/SMPL_python_v.1.1.0
rmdir ./body_models/smpl1


# SHAPY data
# echo -e "\nYou need to register at https://shapy.is.tue.mpg.de"
# read -p "Username:" username
# read -p "Password:" -s password
# username=$(urle $username)
# password=$(urle $password)

wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=shapy&resume=1&sfile=shapy_data.zip' -O 'shapy_data.zip' --no-check-certificate --continue
unzip shapy_data.zip
