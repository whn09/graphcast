import xarray
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta

def merge_surface_upper(surface_ds, upper_ds):
    # 1. 重命名维度
    surface_ds = surface_ds.rename({'latitude': 'lat', 'longitude': 'lon'})
    upper_ds = upper_ds.rename({'latitude': 'lat', 'longitude': 'lon'})

    # 2. 调整lat维度顺序（从90到-90）,level维度顺序（从50到1000）
    surface_ds = surface_ds.reindex(lat=list(reversed(surface_ds.lat)))
    upper_ds = upper_ds.reindex(lat=list(reversed(upper_ds.lat)))
    upper_ds = upper_ds.reindex(level=list(reversed(upper_ds.level)))

    # 3. 扩展维度
    # 添加batch维度，使用expand_dims的axis参数而不是dim参数
    surface_ds = surface_ds.expand_dims({'batch': 1}, axis=0)
    upper_ds = upper_ds.expand_dims({'batch': 1}, axis=0)

    # 根据time坐标创建time维度数组
    time_coords = [surface_ds.time.values]

    # 添加time维度
    surface_ds = surface_ds.expand_dims(dim={'time': time_coords})
    upper_ds = upper_ds.expand_dims(dim={'time': time_coords})

    # 4. 调整维度顺序
    surface_vars = list(surface_ds.data_vars)
    for var in surface_vars:
        if len(surface_ds[var].dims) == 2:  # lat, lon
            surface_ds[var] = surface_ds[var]
        else:  # batch, time, lat, lon
            surface_ds[var] = surface_ds[var].transpose('batch', 'time', 'lat', 'lon')

    upper_vars = list(upper_ds.data_vars)
    for var in upper_vars:
        upper_ds[var] = upper_ds[var].transpose('batch', 'time', 'level', 'lat', 'lon')

    # 5. 合并数据集
    merged_ds = xarray.merge([surface_ds, upper_ds])

    # 6. 添加datetime坐标
    merged_ds = merged_ds.assign_coords({
        'datetime': (('batch', 'time'), [surface_ds.time.values])
    })

    # 7. 确保level坐标是整数类型
    merged_ds = merged_ds.assign_coords({
        'level': merged_ds.level.astype('int32')
    })
    return merged_ds

def get_input_ds(merged_dss, geopotential_at_surface, land_sea_mask):
    input_ds = xarray.merge(merged_dss)
    reference_time = input_ds.time.values[0]  # 获取第一个时间点作为参考时间
    input_ds = input_ds.assign_coords(time=(input_ds.time - reference_time))
    input_ds = xarray.merge([input_ds, geopotential_at_surface, land_sea_mask])
    return input_ds

def accumulate_precipitation(base_ds, other_datasets):
    """
    累加多个Dataset中的降水数据
    
    Parameters:
    -----------
    base_ds : xarray.Dataset
        基础数据集
    other_datasets : List[xarray.Dataset]
        需要累加的其他数据集
    
    Returns:
    --------
    xarray.Dataset
        累加后的数据集
    """
    # 创建副本以避免修改原始数据
    result_ds = base_ds.copy()
    
    # 累加每个数据集的降水量
    for ds in other_datasets:
        result_ds['total_precipitation_6hr'].values += ds['total_precipitation_6hr'].values
    
    return result_ds

def compare_datasets(ds1, ds2, rtol=1e-05, atol=1e-08):
    """
    全面对比两个xarray Dataset的异同
    
    Parameters:
    -----------
    ds1, ds2 : xarray.Dataset
        需要对比的两个数据集
    rtol : float, optional
        相对误差容限 (默认: 1e-05)
    atol : float, optional
        绝对误差容限 (默认: 1e-08)
    
    Returns:
    --------
    dict
        包含对比结果的字典
    """
    results = {
        'identical': False,
        'equals': False,
        'dims_equal': False,
        'coords_equal': False,
        'vars_equal': False,
        'detailed_differences': {}
    }
    
    # 检查完全相同（包括元数据）
    results['identical'] = ds1.identical(ds2)
    
    # 检查数值相等
    results['equals'] = ds1.equals(ds2)
    
    # 检查维度
    results['dims_equal'] = ds1.dims == ds2.dims
    if not results['dims_equal']:
        results['detailed_differences']['dims'] = {
            'ds1_dims': dict(ds1.dims),
            'ds2_dims': dict(ds2.dims)
        }
    
    # 检查坐标
    coords1 = set(ds1.coords)
    coords2 = set(ds2.coords)
    results['coords_equal'] = coords1 == coords2
    if not results['coords_equal']:
        results['detailed_differences']['coords'] = {
            'only_in_ds1': list(coords1 - coords2),
            'only_in_ds2': list(coords2 - coords1)
        }
    
    # 检查变量
    vars1 = set(ds1.data_vars)
    vars2 = set(ds2.data_vars)
    results['vars_equal'] = vars1 == vars2
    if not results['vars_equal']:
        results['detailed_differences']['variables'] = {
            'only_in_ds1': list(vars1 - vars2),
            'only_in_ds2': list(vars2 - vars1)
        }
    
    # 对比共同变量的值
    common_vars = vars1 & vars2
    results['detailed_differences']['variable_values'] = {}
    results['percent_differences'] = {}  # 新增：存储百分比差异
    
    for var in common_vars:
        try:
            xarray.testing.assert_allclose(ds1[var], ds2[var], rtol=rtol, atol=atol)
            results['detailed_differences']['variable_values'][var] = 'equal'
            results['percent_differences'][var] = 0.0  # 如果值相等，百分比差异为0
        except AssertionError as e:
            results['detailed_differences']['variable_values'][var] = str(e)
            
            # 计算百分比差异
            # 1. 计算绝对差异
            abs_diff = abs(ds1[var] - ds2[var])
            
            # 2. 计算百分比差异（相对于ds1）
            # 避免除以0：当ds1的值为0时，使用小数替代
            denominator = ds1[var].where(ds1[var] != 0, 1e-10)
            percent_diff = (abs_diff / abs(denominator)) * 100
            
            # 3. 计算统计数据
            percent_stats = {
                'mean': float(percent_diff.mean().values),
                'median': float(percent_diff.median().values) if hasattr(percent_diff, 'median') else float(np.nanmedian(percent_diff.values)),
                'max': float(percent_diff.max().values),
                'min': float(percent_diff.min().values),
                'std': float(percent_diff.std().values) if hasattr(percent_diff, 'std') else float(np.nanstd(percent_diff.values))
            }
            
            # 4. 存储结果
            results['percent_differences'][var] = percent_stats
            
            # 5. 可选：标记差异较大的区域
            # 如果要标识差异大于某个阈值的区域，可以取消下面的注释
            # large_diff = percent_diff > 10  # 例如标记差异超过10%的区域
            # if large_diff.any():
            #     results['percent_differences'][var]['large_diff_locations'] = large_diff
    
    return results

def print_comparison_results(results):
    """
    打印对比结果的辅助函数
    
    Parameters:
    -----------
    results : dict
        compare_datasets函数返回的结果字典
    """
    print("数据集对比结果:")
    print("-" * 50)
    print(f"完全相同（包括元数据）: {results['identical']}")
    print(f"数值相等: {results['equals']}")
    print(f"维度相同: {results['dims_equal']}")
    print(f"坐标相同: {results['coords_equal']}")
    print(f"变量相同: {results['vars_equal']}")
    
    if not results['dims_equal']:
        print("\n维度差异:")
        print(f"Dataset1 维度: {results['detailed_differences']['dims']['ds1_dims']}")
        print(f"Dataset2 维度: {results['detailed_differences']['dims']['ds2_dims']}")
    
    if not results['coords_equal']:
        print("\n坐标差异:")
        print(f"仅在Dataset1中的坐标: {results['detailed_differences']['coords']['only_in_ds1']}")
        print(f"仅在Dataset2中的坐标: {results['detailed_differences']['coords']['only_in_ds2']}")
    
    if not results['vars_equal']:
        print("\n变量差异:")
        print(f"仅在Dataset1中的变量: {results['detailed_differences']['variables']['only_in_ds1']}")
        print(f"仅在Dataset2中的变量: {results['detailed_differences']['variables']['only_in_ds2']}")
    
    print("\n变量值对比:")
    for var, result in results['detailed_differences']['variable_values'].items():
        print(f"{var}: {'相同' if result == 'equal' else '不同'}")
        if result != 'equal':
            print(f"  差异详情: {result}")
            
    # 打印各变量的百分比差异
    for var, diff in results['percent_differences'].items():
        if diff == 0.0:
            print(f"{var}: 完全相同")
        else:
            print(f"{var}: 平均差异 {diff['mean']:.2f}%, 最大差异 {diff['max']:.2f}%")

if __name__ == '__main__':
    ds = xarray.open_dataset("dataset/source-era5_date-2022-01-01_res-0.25_levels-13_steps-01.nc")

    ds['geopotential_at_surface'].to_netcdf('geopotential_at_surface-0.25.nc')
    ds['land_sea_mask'].to_netcdf('land_sea_mask-0.25.nc')
    
    geopotential_at_surface = xarray.open_dataset('geopotential_at_surface-0.25.nc')
    land_sea_mask = xarray.open_dataset('land_sea_mask-0.25.nc')
    
    start_date = '20240101'  # '20240101'
    end_date = '20241031'  # '20241031'
    lead_time = 0.5  # 10 is too slow
    horizon = 6
    steps = int(lead_time*24//horizon)
    
    # 转换日期字符串为datetime对象
    start_dt = datetime.strptime(start_date, '%Y%m%d')
    end_dt = datetime.strptime(end_date, '%Y%m%d')
    
    # 存储所有结果
    results = {}
    
    # 遍历每一天
    current_dt = start_dt
    while current_dt <= end_dt:
        current_dt_str = current_dt.strftime('%Y-%m-%d')
        print(f"处理起始日期: {current_dt_str}")
        
        # 处理当天及其后lead_time-1天的数据

        merged_dss = []
        
        for i in tqdm(range(steps+2)):
            current_time = current_dt+pd.Timedelta(hours=horizon*i)
            current_time_str = current_time.strftime('%Y%m%d%H')
            upper_ds1 = xarray.open_dataset(f'/opt/dlami/nvme/upper/upper_{current_time_str}.nc')
            surface_ds1 = xarray.open_dataset(f'/opt/dlami/nvme/surface/surface_{current_time_str}.nc')
            other_surface_dss = []
            surface_ds1_1 = xarray.open_dataset(f'/opt/dlami/nvme/surface/surface_{(current_time-pd.Timedelta(hours=5)).strftime('%Y%m%d%H')}.nc')
            other_surface_dss.append(surface_ds1_1)
            surface_ds1_2 = xarray.open_dataset(f'/opt/dlami/nvme/surface/surface_{(current_time-pd.Timedelta(hours=4)).strftime('%Y%m%d%H')}.nc')
            other_surface_dss.append(surface_ds1_2)
            surface_ds1_3 = xarray.open_dataset(f'/opt/dlami/nvme/surface/surface_{(current_time-pd.Timedelta(hours=3)).strftime('%Y%m%d%H')}.nc')
            other_surface_dss.append(surface_ds1_3)
            surface_ds1_4 = xarray.open_dataset(f'/opt/dlami/nvme/surface/surface_{(current_time-pd.Timedelta(hours=2)).strftime('%Y%m%d%H')}.nc')
            other_surface_dss.append(surface_ds1_4)
            surface_ds1_5 = xarray.open_dataset(f'/opt/dlami/nvme/surface/surface_{(current_time-pd.Timedelta(hours=1)).strftime('%Y%m%d%H')}.nc')
            other_surface_dss.append(surface_ds1_5)
            surface_ds1 = accumulate_precipitation(surface_ds1, other_surface_dss)
            # print('surface_ds1:', surface_ds1.total_precipitation_6hr)
            merged_ds1 = merge_surface_upper(surface_ds1, upper_ds1)
            merged_dss.append(merged_ds1)

        input_ds = get_input_ds(merged_dss, geopotential_at_surface, land_sea_mask)

        input_ds.to_netcdf(f'dataset/source-era5_date-{current_dt_str}_res-0.25_levels-13_steps-{steps:02d}.nc', engine='netcdf4')
        
        # results = compare_datasets(input_ds, ds)
        # print_comparison_results(results)
        
        current_dt = current_dt+pd.Timedelta(days=1)