# conda create -n graphcast python=3.10
# conda activate graphcast

# pip install --upgrade https://github.com/deepmind/graphcast/archive/master.zip
# pip uninstall -y jax jaxlib
# pip install --upgrade "jax[cuda12]==0.4.29" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# pip install tenacity

# ./download_era5.sh
# cd /opt/dlami/nvme/ && python /home/ubuntu/graphcast/convert_era5.py

# pip install gsutil
# /opt/pytorch/bin/gsutil -m cp -r \
#   "gs://dm_graphcast/LICENSE" \
#   "gs://dm_graphcast/dataset" \
#   "gs://dm_graphcast/gencast" \
#   "gs://dm_graphcast/graphcast" \
#   "gs://dm_graphcast/params" \
#   "gs://dm_graphcast/stats" \
#   .

# pip install s5cmd
# s5cmd cp s3://datalab/nsf-ncar-era5/surface/surface_*.nc /opt/dlami/nvme/surface/
# s5cmd cp s3://datalab/nsf-ncar-era5/upper/upper_*.nc /opt/dlami/nvme/upper/

# python merge_era5.py

# python graphcast_demo.py
python graphcast_multi_demo.py