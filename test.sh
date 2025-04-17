# conda create -n graphcast python=3.10
# conda activate graphcast

# pip install --upgrade https://github.com/deepmind/graphcast/archive/master.zip
# pip uninstall -y jax jaxlib
# pip install --upgrade "jax[cuda12]==0.4.29" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# ./download_era5.sh
# python convert_era5.py

python graphcast_demo.py