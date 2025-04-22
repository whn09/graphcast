import os
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import xarray

import io
import s3fs
import tempfile
from tqdm import tqdm
import multiprocessing as mp
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from botocore.exceptions import NoCredentialsError, ClientError

import torch


def get_last_day_of_month(date_string):
    # 将字符串转换为 datetime 对象
    date = datetime.strptime(date_string, '%Y%m%d')

    # 获取下个月的第一天
    if date.month == 12:
        next_month = datetime(date.year + 1, 1, 1)
    else:
        next_month = datetime(date.year, date.month + 1, 1)

    # 下个月第一天减去一天，就是本月最后一天
    last_day = next_month - timedelta(days=1)

    # 返回天数作为字符串
    return f'{last_day.day:02d}'

def get_last_month(select_month):
    year = int(select_month[:4])
    month = int(select_month[4:])
    
    if month == 1:
        last_month = '12'
        last_year = str(year - 1)
    else:
        last_month = str(month - 1).zfill(2)
        last_year = str(year)
        
    return last_year + last_month

def get_next_month(select_month):
    year = int(select_month[:4])
    month = int(select_month[4:])
    
    if month == 12:
        next_month = '01'
        next_year = str(year + 1)
    else:
        next_month = str(month + 1).zfill(2)
        next_year = str(year)
        
    return next_year + next_month

@retry(
    stop=stop_after_attempt(5),  # 最多重试5次
    wait=wait_exponential(multiplier=1, min=4, max=60),  # 指数退避，最小4秒，最大60秒
    retry=retry_if_exception_type((NoCredentialsError, ClientError)),  # 只在特定异常时重试
    reraise=True  # 如果所有重试都失败，重新抛出最后一个异常
)
def open_s3_dataset(path):
    local_path = path.replace(f's3://{s3_bucket}/', '')
    if os.path.exists(local_path):
        ds = xarray.open_dataset(local_path)
        return ds
    
    s3 = s3fs.S3FileSystem(anon=False)  # 使用 AWS 凭证
    
    # 创建临时文件
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_path = temp_file.name
    
    try:
        # 从S3下载文件到临时位置
        s3.get(path, temp_path)
        # 使用xarray打开本地文件
        ds = xarray.open_dataset(temp_path)
        return ds
    except (NoCredentialsError, ClientError) as e:
        print(f"Error accessing S3: {str(e)}. Retrying...", path)
        raise  # 重新抛出异常，触发重试
    finally:
        # 确保临时文件被删除
        if os.path.exists(temp_path):
            os.remove(temp_path)

def create_actual_time(ds):
    # 创建网格化的时间
    init_times = ds.forecast_initial_time.values[:, np.newaxis]  # 添加一个维度以便广播
    forecast_hours = pd.Timedelta(hours=1) * (ds.forecast_hour.values-1)  # 转换小时为时间差
    
    # 广播并相加得到实际时间
    actual_times = init_times + forecast_hours
    
    # 重塑数据，将forecast_initial_time和forecast_hour合并为单个time维度
    ds_new = ds.stack(time=('forecast_initial_time', 'forecast_hour'))
    
    # 将计算出的实际时间赋值给新维度
    ds_new.coords['time'] = actual_times.ravel()
    
    # # 删除原始的forecast维度
    # ds_new = ds_new.drop(['forecast_initial_time', 'forecast_hour'])
    
    # 调整维度顺序为 (time, latitude, longitude)
    ds_new = ds_new.transpose('time', 'latitude', 'longitude')
    
    return ds_new

def process_month(select_month):
    select_month_end = get_last_day_of_month(select_month+'01')
    last_month = get_last_month(select_month)
    next_month = get_next_month(select_month)
    print(select_month, select_month_end, last_month, next_month)

    # if os.path.exists(f'surface/surface_{select_month}.nc'):
    #     continue

    load_start = time.time()
    msl_ds = open_s3_dataset(f's3://{s3_bucket}/{s3_prefix}/e5.oper.an.sfc/{select_month}/e5.oper.an.sfc.128_151_msl.ll025sc.{select_month}0100_{select_month}{select_month_end}23.nc')
    load_end = time.time()
    print('load msl time:', load_end-load_start)
    u10_ds = open_s3_dataset(f's3://{s3_bucket}/{s3_prefix}/e5.oper.an.sfc/{select_month}/e5.oper.an.sfc.128_165_10u.ll025sc.{select_month}0100_{select_month}{select_month_end}23.nc')
    load_end = time.time()
    print('load msl+u10 time:', load_end-load_start)
    v10_ds = open_s3_dataset(f's3://{s3_bucket}/{s3_prefix}/e5.oper.an.sfc/{select_month}/e5.oper.an.sfc.128_166_10v.ll025sc.{select_month}0100_{select_month}{select_month_end}23.nc')
    load_end = time.time()
    print('load msl+u10+v10 time:', load_end-load_start)
    t2m_ds = open_s3_dataset(f's3://{s3_bucket}/{s3_prefix}/e5.oper.an.sfc/{select_month}/e5.oper.an.sfc.128_167_2t.ll025sc.{select_month}0100_{select_month}{select_month_end}23.nc')
    load_end = time.time()
    print('load msl+u10+v10+t2m time:', load_end-load_start)
    mtpr_ds0 = open_s3_dataset(f's3://{s3_bucket}/{s3_prefix}/e5.oper.fc.sfc.meanflux/{last_month}/e5.oper.fc.sfc.meanflux.235_055_mtpr.ll025sc.{last_month}1606_{select_month}0106.nc')
    mtpr_ds1 = open_s3_dataset(f's3://{s3_bucket}/{s3_prefix}/e5.oper.fc.sfc.meanflux/{select_month}/e5.oper.fc.sfc.meanflux.235_055_mtpr.ll025sc.{select_month}0106_{select_month}1606.nc')
    mtpr_ds2 = open_s3_dataset(f's3://{s3_bucket}/{s3_prefix}/e5.oper.fc.sfc.meanflux/{select_month}/e5.oper.fc.sfc.meanflux.235_055_mtpr.ll025sc.{select_month}1606_{next_month}0106.nc')
    mtpr_ds = xarray.merge([mtpr_ds0, mtpr_ds1, mtpr_ds2])
    mtpr_ds = create_actual_time(mtpr_ds)  # TODO: how to get total_precipitation_6hr?
    load_end = time.time()
    print('load msl+u10+v10+t2m+mtpr time:', load_end-load_start)
    tisr_ds0 = open_s3_dataset(f's3://{s3_bucket}/{s3_prefix}/e5.oper.fc.sfc.accumu/{last_month}/e5.oper.fc.sfc.accumu.128_212_tisr.ll025sc.{last_month}1606_{select_month}0106.nc')
    tisr_ds1 = open_s3_dataset(f's3://{s3_bucket}/{s3_prefix}/e5.oper.fc.sfc.accumu/{select_month}/e5.oper.fc.sfc.accumu.128_212_tisr.ll025sc.{select_month}0106_{select_month}1606.nc')
    tisr_ds2 = open_s3_dataset(f's3://{s3_bucket}/{s3_prefix}/e5.oper.fc.sfc.accumu/{select_month}/e5.oper.fc.sfc.accumu.128_212_tisr.ll025sc.{select_month}1606_{next_month}0106.nc')
    tisr_ds = xarray.merge([tisr_ds0, tisr_ds1, tisr_ds2])
    tisr_ds = create_actual_time(tisr_ds)
    load_end = time.time()
    print('load time:', load_end-load_start)

    merge_start = time.time()
    surface_ds = xarray.merge([msl_ds.rename({'MSL': 'mean_sea_level_pressure'}), u10_ds.rename(
        {'VAR_10U': '10m_u_component_of_wind'}), v10_ds.rename({'VAR_10V': '10m_v_component_of_wind'}), t2m_ds.rename({'VAR_2T': '2m_temperature'}), mtpr_ds.rename({'MTPR': 'total_precipitation_6hr'}), tisr_ds.rename({'TISR': 'toa_incident_solar_radiation'})], compat='override').drop_vars(['utc_date'])
    # surface_ds.to_netcdf(f'surface/surface_{select_month}.nc')
    merge_end = time.time()
    print('merge time:', merge_end-merge_start)

    s3 = s3fs.S3FileSystem(anon=False)
    for d in tqdm(range(1, int(select_month_end)+1)):
        if d < 10:
            d = '0'+str(d)
        else:
            d = str(d)
        select_date = select_month+d
        for h in range(24):
            if h < 10:
                h = '0'+str(h)
            else:
                h = str(h)
            select_hour = select_date+h
            select_hour_datetime = pd.to_datetime(
                select_hour, format='%Y%m%d%H')

            select_surface_ds = surface_ds.sel(time=select_hour_datetime)
            select_surface_ds.to_netcdf(f'surface/surface_{select_hour}.nc')
            
            # # 将结果保存到S3
            # buffer = io.BytesIO()
            # torch.save(select_surface_ds, buffer)
            # buffer.seek(0)
            # with s3.open(f's3://{s3_bucket}/{s3_prefix}/surface/surface_{select_hour}.nc', 'wb') as f:
            #     f.write(buffer.getvalue())
                
def process_date(select_date):
    select_month = select_date[:6]
    select_month_end = get_last_day_of_month(select_date)
    print(select_date, select_month, select_month_end)

    load_start = time.time()
    z_ds = open_s3_dataset(
        f's3://{s3_bucket}/{s3_prefix}/e5.oper.an.pl/{select_month}/e5.oper.an.pl.128_129_z.ll025sc.{select_date}00_{select_date}23.nc')
    load_end = time.time()
    print('load z time:', load_end-load_start)
    q_ds = open_s3_dataset(
        f's3://{s3_bucket}/{s3_prefix}/e5.oper.an.pl/{select_month}/e5.oper.an.pl.128_133_q.ll025sc.{select_date}00_{select_date}23.nc')
    load_end = time.time()
    print('load z+q time:', load_end-load_start)
    t_ds = open_s3_dataset(
        f's3://{s3_bucket}/{s3_prefix}/e5.oper.an.pl/{select_month}/e5.oper.an.pl.128_130_t.ll025sc.{select_date}00_{select_date}23.nc')
    load_end = time.time()
    print('load z+q+t time:', load_end-load_start)
    u_ds = open_s3_dataset(
        f's3://{s3_bucket}/{s3_prefix}/e5.oper.an.pl/{select_month}/e5.oper.an.pl.128_131_u.ll025uv.{select_date}00_{select_date}23.nc')
    load_end = time.time()
    print('load z+q+t+u time:', load_end-load_start)
    v_ds = open_s3_dataset(
        f's3://{s3_bucket}/{s3_prefix}/e5.oper.an.pl/{select_month}/e5.oper.an.pl.128_132_v.ll025uv.{select_date}00_{select_date}23.nc')
    load_end = time.time()
    print('load z+q+t+u+v time:', load_end-load_start)
    w_ds = open_s3_dataset(
        f's3://{s3_bucket}/{s3_prefix}/e5.oper.an.pl/{select_month}/e5.oper.an.pl.128_135_w.ll025sc.{select_date}00_{select_date}23.nc')
    load_end = time.time()
    print('load time:', load_end-load_start)
    
    z_ds = z_ds.sel(level=pressure_levels)
    q_ds = q_ds.sel(level=pressure_levels)
    t_ds = t_ds.sel(level=pressure_levels)
    u_ds = u_ds.sel(level=pressure_levels)
    v_ds = v_ds.sel(level=pressure_levels)
    w_ds = w_ds.sel(level=pressure_levels)

    upper_ds = xarray.merge([z_ds.rename({'Z': 'geopotential'}), q_ds.rename({'Q': 'specific_humidity'}), t_ds.rename(
        {'T': 'temperature'}), u_ds.rename({'U': 'u_component_of_wind'}), v_ds.rename({'V': 'v_component_of_wind'}), w_ds.rename({'W': 'vertical_velocity'})]).drop_vars(['utc_date'])
    # upper_ds.to_netcdf(f'upper/upper_{select_date}.nc')

    s3 = s3fs.S3FileSystem(anon=False)
    for h in tqdm(range(24)):
        if h < 10:
            h = '0'+str(h)
        else:
            h = str(h)
        select_hour = select_date+h
        select_hour_datetime = pd.to_datetime(select_hour, format='%Y%m%d%H')
        
        select_upper_ds = upper_ds.sel(time=select_hour_datetime)
        select_upper_ds.to_netcdf(f'upper/upper_{select_hour}.nc')
        
        # # 将结果保存到S3
        # buffer = io.BytesIO()
        # torch.save(select_upper_ds, buffer)
        # buffer.seek(0)
        # with s3.open(f's3://{s3_bucket}/{s3_prefix}/upper/upper_{select_hour}.nc', 'wb') as f:
        #     f.write(buffer.getvalue())


s3_bucket = "datalab"
s3_prefix = "nsf-ncar-era5"

pressure_levels = [1000, 925, 850, 700, 600,
                   500, 400, 300, 250, 200, 150, 100, 50]
startDate = '20150101'
endDate = '20241031'
select_dates = list(pd.date_range(start=startDate, end=endDate, freq='1D'))
select_dates = [date.strftime('%Y%m%d') for date in select_dates]
# select_months = set([select_date[:6] for select_date in select_dates])
select_months = list(pd.date_range(start=startDate, end=endDate, freq='1ME'))
select_months = [date.strftime('%Y%m') for date in select_months]

print('select_dates:', len(select_dates))
print('select_months:', len(select_months))

os.system('mkdir -p surface')
os.system('mkdir -p upper')

# 设置进程数，可以根据你的CPU核心数进行调整
num_processes = 1  # mp.cpu_count()  # 使用所有可用的CPU核心

# 使用进程池并行处理
with mp.Pool(num_processes) as pool:
    list(tqdm(pool.imap(process_month, select_months), total=len(select_months)))

# 设置进程数，可以根据你的CPU核心数进行调整
num_processes = 60  # mp.cpu_count()  # 使用所有可用的CPU核心

# 使用进程池并行处理
with mp.Pool(num_processes) as pool:
    list(tqdm(pool.imap(process_date, select_dates), total=len(select_dates)))
