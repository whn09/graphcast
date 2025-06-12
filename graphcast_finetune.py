# GraphCast Fine-tuning Demo - Multi-GPU Version
# 支持单机多卡训练的GraphCast微调代码

# @title Pip install graphcast and dependencies
# pip install --upgrade https://github.com/deepmind/graphcast/archive/master.zip
# pip install optax
# pip install h5netcdf

# @title Imports (添加多卡支持)
import dataclasses
import datetime
import functools
import math
import re
from typing import Optional
import pickle
import os
import time

import cartopy.crs as ccrs
from graphcast import autoregressive
from graphcast import casting
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import graphcast
from graphcast import normalization
from graphcast import rollout
from graphcast import xarray_jax
from graphcast import xarray_tree
from IPython.display import HTML
import ipywidgets as widgets
import haiku as hk
import jax
import jax.numpy as jnp
from jax import random
# from jax.experimental import maps
from jax.experimental import pjit
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import xarray
import optax

# 多卡配置
def setup_multi_gpu():
    """配置多GPU环境"""
    # 获取可用设备
    devices = jax.devices()
    n_devices = len(devices)
    print(f"发现 {n_devices} 个设备: {devices}")
    
    if n_devices > 1:
        print("启用多GPU训练")
        # 设置并行策略
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"
        return True, n_devices
    else:
        print("使用单GPU训练")
        return False, 1

multi_gpu_enabled, n_devices = setup_multi_gpu()

# @title Fine-tuning Setup and Training Functions
class GraphCastFineTuner:
    def __init__(self, params, state, model_config, task_config, 
                 diffs_stddev_by_level, mean_by_level, stddev_by_level,
                 learning_rate=1e-5):
        self.params = params
        self.state = state
        self.model_config = model_config
        self.task_config = task_config
        self.diffs_stddev_by_level = diffs_stddev_by_level
        self.mean_by_level = mean_by_level
        self.stddev_by_level = stddev_by_level
        self.learning_rate = learning_rate
        
        # 设置优化器
        self.optimizer = optax.adam(learning_rate=learning_rate)
        self.opt_state = self.optimizer.init(params)
        
        # 构建模型函数
        self._build_model_functions()
    
    def _build_model_functions(self):
        """构建模型前向传播和损失函数"""
        
        def construct_wrapped_graphcast(model_config, task_config):
            predictor = graphcast.GraphCast(model_config, task_config)
            predictor = casting.Bfloat16Cast(predictor)
            predictor = normalization.InputsAndResiduals(
                predictor,
                diffs_stddev_by_level=self.diffs_stddev_by_level,
                mean_by_level=self.mean_by_level,
                stddev_by_level=self.stddev_by_level)
            predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
            return predictor

        @hk.transform_with_state
        def run_forward(model_config, task_config, inputs, targets_template, forcings):
            predictor = construct_wrapped_graphcast(model_config, task_config)
            return predictor(inputs, targets_template=targets_template, forcings=forcings)

        @hk.transform_with_state  
        def loss_fn(model_config, task_config, inputs, targets, forcings):
            predictor = construct_wrapped_graphcast(model_config, task_config)
            loss, diagnostics = predictor.loss(inputs, targets, forcings)
            return xarray_tree.map_structure(
                lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True),
                (loss, diagnostics))
        
        self.run_forward = run_forward
        self.loss_fn = loss_fn
        
        # 定义训练步骤
        def train_step(params, opt_state, state, inputs, targets, forcings):
            def compute_loss(params):
                (loss, diagnostics), next_state = self.loss_fn.apply(
                    params, state, jax.random.PRNGKey(0), 
                    self.model_config, self.task_config,
                    inputs, targets, forcings)
                return loss, (diagnostics, next_state)
            
            (loss, (diagnostics, next_state)), grads = jax.value_and_grad(
                compute_loss, has_aux=True)(params)
            
            # 梯度裁剪
            grads = optax.clip_by_global_norm(1.0)(grads)
            
            # 更新参数
            updates, new_opt_state = self.optimizer.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)
            
            return new_params, new_opt_state, next_state, loss, diagnostics
        
        self.train_step_fn = jax.jit(train_step)
    
    def fine_tune(self, train_inputs, train_targets, train_forcings,
                  eval_inputs, eval_targets, eval_forcings, 
                  num_epochs=5, print_every=1):
        """执行微调训练"""
        print("开始微调训练...")
        
        current_params = self.params
        current_opt_state = self.opt_state
        current_state = self.state
        
        for epoch in range(num_epochs):
            # 训练步骤
            current_params, current_opt_state, current_state, train_loss, train_diagnostics = \
                self.train_step_fn(
                    current_params, current_opt_state, current_state,
                    train_inputs, train_targets, train_forcings)
            
            if (epoch + 1) % print_every == 0:
                # 评估步骤
                (eval_loss, eval_diagnostics), _ = self.loss_fn.apply(
                    current_params, current_state, jax.random.PRNGKey(0),
                    self.model_config, self.task_config,
                    eval_inputs, eval_targets, eval_forcings)
                
                print(f"Epoch {epoch+1}/{num_epochs}")
                print(f"  Train Loss: {float(train_loss):.6f}")
                print(f"  Eval Loss:  {float(eval_loss):.6f}")
        
        # 更新最终参数
        self.params = current_params
        self.opt_state = current_opt_state
        self.state = current_state
        print("微调完成!")
        
        return current_params, current_state
    
    def predict(self, inputs, targets_template, forcings):
        """使用微调后的模型进行预测"""
        predictions, _ = self.run_forward.apply(
            self.params, self.state, jax.random.PRNGKey(0),
            self.model_config, self.task_config,
            inputs, targets_template, forcings)
        return predictions
    
    def save_checkpoint(self, filepath):
        """保存微调后的模型"""
        checkpoint_data = {
            "params": self.params,
            "state": self.state,
            "model_config": self.model_config,
            "task_config": self.task_config,
            "optimizer_state": self.opt_state
        }
        with open(filepath, "wb") as f:
            pickle.dump(checkpoint_data, f)
        print(f"模型已保存到: {filepath}")

# @title Multi-GPU GraphCast Fine-tuner
class MultiGPUGraphCastFineTuner:
    def __init__(self, params, state, model_config, task_config, 
                diffs_stddev_by_level, mean_by_level, stddev_by_level,
                learning_rate=1e-5, use_multi_gpu=True):
        self.params = params
        self.state = state
        self.model_config = model_config
        self.task_config = task_config
        self.diffs_stddev_by_level = diffs_stddev_by_level
        self.mean_by_level = mean_by_level
        self.stddev_by_level = stddev_by_level
        self.learning_rate = learning_rate
        self.use_multi_gpu = use_multi_gpu and multi_gpu_enabled
        
        # 获取设备信息
        self.devices = jax.devices()
        self.n_devices = len(self.devices)
        
        if self.use_multi_gpu:
            print(f"使用多GPU训练，设备数量: {self.n_devices}")
            # 创建设备网格
            self.device_mesh = np.array(self.devices).reshape(self.n_devices, 1)
        else:
            print("使用单GPU训练")
            self.device_mesh = None
        
        # 设置优化器
        self.optimizer = optax.adam(learning_rate=learning_rate)
        
        # 复制参数到所有设备
        if self.use_multi_gpu:
            self.params = jax.device_put_replicated(params, self.devices)
            self.opt_state = self.optimizer.init(self.params)
        else:
            self.opt_state = self.optimizer.init(params)
        
        # 构建模型函数
        self._build_model_functions()
    
    def _build_model_functions(self):
        """构建支持多GPU的模型函数"""
        
        def construct_wrapped_graphcast(model_config, task_config):
            predictor = graphcast.GraphCast(model_config, task_config)
            predictor = casting.Bfloat16Cast(predictor)
            predictor = normalization.InputsAndResiduals(
                predictor,
                diffs_stddev_by_level=self.diffs_stddev_by_level,
                mean_by_level=self.mean_by_level,
                stddev_by_level=self.stddev_by_level)
            predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
            return predictor

        @hk.transform_with_state
        def run_forward(model_config, task_config, inputs, targets_template, forcings):
            predictor = construct_wrapped_graphcast(model_config, task_config)
            return predictor(inputs, targets_template=targets_template, forcings=forcings)

        @hk.transform_with_state  
        def loss_fn(model_config, task_config, inputs, targets, forcings):
            predictor = construct_wrapped_graphcast(model_config, task_config)
            loss, diagnostics = predictor.loss(inputs, targets, forcings)
            return xarray_tree.map_structure(
                lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True),
                (loss, diagnostics))
        
        self.run_forward = run_forward
        self.loss_fn = loss_fn
        
        # 定义训练步骤
        def train_step(params, opt_state, state, inputs, targets, forcings, rng_key):
            def compute_loss(params):
                (loss, diagnostics), next_state = self.loss_fn.apply(
                    params, state, rng_key, 
                    self.model_config, self.task_config,
                    inputs, targets, forcings)
                return loss, (diagnostics, next_state)
            
            (loss, (diagnostics, next_state)), grads = jax.value_and_grad(
                compute_loss, has_aux=True)(params)
            
            # 梯度裁剪
            grads = optax.clip_by_global_norm(1.0)(grads)
            
            # 更新参数
            updates, new_opt_state = self.optimizer.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)
            
            return new_params, new_opt_state, next_state, loss, diagnostics
        
        if self.use_multi_gpu:
            # 多GPU版本：使用pmap进行数据并行
            self.train_step_fn = jax.pmap(
                train_step, 
                axis_name='devices',
                in_axes=(0, 0, 0, 0, 0, 0, 0),
                out_axes=(0, 0, 0, 0, 0)
            )
        else:
            # 单GPU版本
            self.train_step_fn = jax.jit(train_step)
    
    def _replicate_data(self, data):
        """将数据复制到所有设备"""
        if self.use_multi_gpu:
            return jax.device_put_replicated(data, self.devices)
        else:
            return data
    
    def _split_batch(self, data, batch_size=None):
        """将批次数据分割到多个设备"""
        if not self.use_multi_gpu:
            return data
        
        if batch_size is None:
            batch_size = self.n_devices
        
        # 如果数据没有batch维度，添加一个
        if 'batch' not in data.dims:
            data = data.expand_dims('batch', axis=0)
        
        # 确保batch大小能被设备数整除
        original_batch_size = data.sizes['batch']
        if original_batch_size < self.n_devices:
            # 重复数据以匹配设备数
            repeat_factor = (self.n_devices + original_batch_size - 1) // original_batch_size
            data = xarray.concat([data] * repeat_factor, dim='batch')
        
        # 截取到设备数的倍数
        new_batch_size = (data.sizes['batch'] // self.n_devices) * self.n_devices
        data = data.isel(batch=slice(0, new_batch_size))
        
        # 重新reshape为 (n_devices, local_batch_size, ...)
        local_batch_size = data.sizes['batch'] // self.n_devices
        
        # 将数据分割到各个设备
        split_data = []
        for i in range(self.n_devices):
            start_idx = i * local_batch_size
            end_idx = start_idx + local_batch_size
            device_data = data.isel(batch=slice(start_idx, end_idx))
            split_data.append(device_data)
        
        return split_data
    
    def fine_tune(self, train_inputs, train_targets, train_forcings,
                eval_inputs, eval_targets, eval_forcings, 
                num_epochs=5, print_every=1):
        """执行多GPU微调训练"""
        print("开始多GPU微调训练..." if self.use_multi_gpu else "开始单GPU微调训练...")
        
        current_params = self.params
        current_opt_state = self.opt_state
        current_state = self.state
        
        # 生成随机数种子
        rng_key = random.PRNGKey(42)
        
        for epoch in range(num_epochs):
            rng_key, step_key = random.split(rng_key)
            
            if self.use_multi_gpu:
                # 多GPU训练
                # 准备数据
                train_inputs_split = self._split_batch(train_inputs)
                train_targets_split = self._split_batch(train_targets)
                train_forcings_split = self._split_batch(train_forcings)
                
                # 为每个设备生成不同的随机数种子
                step_keys = random.split(step_key, self.n_devices)
                
                # 训练步骤
                current_params, current_opt_state, current_state, train_loss, train_diagnostics = \
                    self.train_step_fn(
                        current_params, current_opt_state, current_state,
                        train_inputs_split, train_targets_split, train_forcings_split,
                        step_keys)
                
                # 平均多设备的损失
                train_loss = jnp.mean(train_loss)
                
            else:
                # 单GPU训练
                current_params, current_opt_state, current_state, train_loss, train_diagnostics = \
                    self.train_step_fn(
                        current_params, current_opt_state, current_state,
                        train_inputs, train_targets, train_forcings, step_key)
            
            if (epoch + 1) % print_every == 0:
                # 评估步骤
                eval_params = current_params[0] if self.use_multi_gpu else current_params
                eval_state = current_state[0] if self.use_multi_gpu else current_state
                
                (eval_loss, eval_diagnostics), _ = self.loss_fn.apply(
                    eval_params, eval_state, random.PRNGKey(0),
                    self.model_config, self.task_config,
                    eval_inputs, eval_targets, eval_forcings)
                
                print(f"Epoch {epoch+1}/{num_epochs}")
                print(f"  Train Loss: {float(train_loss):.6f}")
                print(f"  Eval Loss:  {float(eval_loss):.6f}")
                
                if self.use_multi_gpu:
                    # 显示设备使用情况
                    print(f"  使用设备数: {self.n_devices}")
        
        # 更新最终参数（如果是多GPU，取第一个设备的参数）
        if self.use_multi_gpu:
            self.params = current_params[0]
            self.opt_state = current_opt_state[0]
            self.state = current_state[0]
        else:
            self.params = current_params
            self.opt_state = current_opt_state
            self.state = current_state
            
        print("多GPU微调完成!" if self.use_multi_gpu else "单GPU微调完成!")
        
        return self.params, self.state
    
    def predict(self, inputs, targets_template, forcings):
        """使用微调后的模型进行预测"""
        predictions, _ = self.run_forward.apply(
            self.params, self.state, random.PRNGKey(0),
            self.model_config, self.task_config,
            inputs, targets_template, forcings)
        return predictions
    
    def save_checkpoint(self, filepath):
        """保存微调后的模型"""
        checkpoint_data = {
            "params": self.params,
            "state": self.state,
            "model_config": self.model_config,
            "task_config": self.task_config,
            "optimizer_state": self.opt_state
        }
        with open(filepath, "wb") as f:
            pickle.dump(checkpoint_data, f)
        print(f"模型已保存到: {filepath}")

# @title 数据加载和预处理部分保持不变
# [这里插入之前的数据加载代码，从加载模型开始到数据准备结束]

# 为了演示，我简化了数据加载部分，您需要插入完整的数据加载代码
# gcs_client = storage.Client.create_anonymous_client()
# gcs_bucket = gcs_client.get_bucket("dm_graphcast")
dir_prefix = "graphcast/"

# @title Plotting functions

def select(
    data: xarray.Dataset,
    variable: str,
    level: Optional[int] = None,
    max_steps: Optional[int] = None
    ) -> xarray.Dataset:
  data = data[variable]
  if "batch" in data.dims:
    data = data.isel(batch=0)
  if max_steps is not None and "time" in data.sizes and max_steps < data.sizes["time"]:
    data = data.isel(time=range(0, max_steps))
  if level is not None and "level" in data.coords:
    data = data.sel(level=level)
  return data

def scale(
    data: xarray.Dataset,
    center: Optional[float] = None,
    robust: bool = False,
    ) -> tuple[xarray.Dataset, matplotlib.colors.Normalize, str]:
  vmin = np.nanpercentile(data, (2 if robust else 0))
  vmax = np.nanpercentile(data, (98 if robust else 100))
  if center is not None:
    diff = max(vmax - center, center - vmin)
    vmin = center - diff
    vmax = center + diff
  return (data, matplotlib.colors.Normalize(vmin, vmax),
          ("RdBu_r" if center is not None else "viridis"))

def plot_data(
    data: dict[str, xarray.Dataset],
    fig_title: str,
    plot_size: float = 5,
    robust: bool = False,
    cols: int = 4
    ) -> tuple[xarray.Dataset, matplotlib.colors.Normalize, str]:

  first_data = next(iter(data.values()))[0]
  max_steps = first_data.sizes.get("time", 1)
  assert all(max_steps == d.sizes.get("time", 1) for d, _, _ in data.values())

  cols = min(cols, len(data))
  rows = math.ceil(len(data) / cols)
  figure = plt.figure(figsize=(plot_size * 2 * cols,
                               plot_size * rows))
  figure.suptitle(fig_title, fontsize=16)
  figure.subplots_adjust(wspace=0, hspace=0)
  figure.tight_layout()

  images = []
  for i, (title, (plot_data, norm, cmap)) in enumerate(data.items()):
    ax = figure.add_subplot(rows, cols, i+1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    im = ax.imshow(
        plot_data.isel(time=0, missing_dims="ignore"), norm=norm,
        origin="lower", cmap=cmap)
    plt.colorbar(
        mappable=im,
        ax=ax,
        orientation="vertical",
        pad=0.02,
        aspect=16,
        shrink=0.75,
        cmap=cmap,
        extend=("both" if robust else "neither"))
    images.append(im)

  def update(frame):
    if "time" in first_data.dims:
      td = datetime.timedelta(microseconds=first_data["time"][frame].item() / 1000)
      figure.suptitle(f"{fig_title}, {td}", fontsize=16)
    else:
      figure.suptitle(fig_title, fontsize=16)
    for im, (plot_data, norm, cmap) in zip(images, data.values()):
      im.set_data(plot_data.isel(time=frame, missing_dims="ignore"))

  ani = animation.FuncAnimation(
      fig=figure, func=update, frames=max_steps, interval=250)
  plt.close(figure.number)
  return HTML(ani.to_jshtml())

# 加载模型
params_file = 'GraphCast_operational - ERA5-HRES 1979-2021 - resolution 0.25 - pressure levels 13 - mesh 2to6 - precipitation output only.npz'
with open(f"params/{params_file}", "rb") as f:
    ckpt = checkpoint.load(f, graphcast.CheckPoint)
params = ckpt.params
state = {}

model_config = ckpt.model_config
task_config = ckpt.task_config
print("Model description:\n", ckpt.description, "\n")
print("Model license:\n", ckpt.license, "\n")

dataset_file = 'source-era5new_date-2018-01-02_res-0.25_levels-13_steps-02.nc'
with open(f"dataset/{dataset_file}", "rb") as f:
    example_batch = xarray.load_dataset(f).compute()

with open("stats/diffs_stddev_by_level.nc", "rb") as f:
    diffs_stddev_by_level = xarray.load_dataset(f).compute()
with open("stats/mean_by_level.nc", "rb") as f:
    mean_by_level = xarray.load_dataset(f).compute()
with open("stats/stddev_by_level.nc", "rb") as f:
    stddev_by_level = xarray.load_dataset(f).compute()

train_steps = 1
eval_steps = 1
train_inputs, train_targets, train_forcings = data_utils.extract_inputs_targets_forcings(
    example_batch, target_lead_times=slice("6h", f"{train_steps*6}h"),
    **dataclasses.asdict(task_config))
eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
    example_batch, target_lead_times=slice("6h", f"{eval_steps*6}h"),
    **dataclasses.asdict(task_config))

# @title Initialize Multi-GPU Fine-tuner
print("初始化多GPU微调器...")

# @title 训练配置
learning_rate_widget = widgets.FloatText(value=1e-5, description="Learning Rate:")
num_epochs_widget = widgets.IntSlider(value=5, min=1, max=20, description="Epochs:")
use_multi_gpu_widget = widgets.Checkbox(value=multi_gpu_enabled, description="使用多GPU:")

widgets.VBox([
    learning_rate_widget,
    num_epochs_widget,
    use_multi_gpu_widget,
    widgets.Label(value=f"检测到 {n_devices} 个设备，建议启用多GPU训练")
])

# @title 性能监控工具
class PerformanceMonitor:
    def __init__(self):
        self.start_time = None
        self.epoch_times = []
    
    def start_epoch(self):
        self.start_time = time.time()
    
    def end_epoch(self):
        if self.start_time:
            epoch_time = time.time() - self.start_time
            self.epoch_times.append(epoch_time)
            return epoch_time
        return 0
    
    def get_average_time(self):
        return np.mean(self.epoch_times) if self.epoch_times else 0
    
    def estimate_remaining_time(self, current_epoch, total_epochs):
        if len(self.epoch_times) > 0:
            avg_time = self.get_average_time()
            remaining_epochs = total_epochs - current_epoch
            return avg_time * remaining_epochs
        return 0

# @title 内存使用监控
def monitor_memory_usage():
    """监控GPU内存使用情况"""
    if jax.devices()[0].platform == 'gpu':
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            for i, gpu in enumerate(gpus):
                print(f"GPU {i}: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB ({gpu.memoryPercent:.1f}%)")
        except ImportError:
            print("安装GPUtil以监控GPU内存: pip install gputil")
    
    # JAX内存使用
    try:
        backend = jax.lib.xla_bridge.get_backend()
        print(f"JAX后端: {backend.platform}")
        if hasattr(backend, 'device_count'):
            print(f"设备数量: {backend.device_count()}")
    except:
        pass

# @title 运行多GPU训练示例
def run_multi_gpu_training_example():
    """运行多GPU训练的完整示例"""
    print("=== 多GPU GraphCast微调示例 ===")
    
    # 1. 检查设备
    devices = jax.devices()
    print(f"可用设备: {len(devices)} 个")
    for i, device in enumerate(devices):
        print(f"  设备 {i}: {device}")
    
    # 2. 内存监控
    print("\n=== 内存使用情况 ===")
    monitor_memory_usage()
    
    # 3. 设置多GPU训练
    if len(devices) > 1:
        print(f"\n启用多GPU训练，使用 {len(devices)} 个设备")
        # 这里会插入实际的训练代码
        multi_gpu_fine_tuner = MultiGPUGraphCastFineTuner(
                            params=params,
                            state=state,
                            model_config=model_config,
                            task_config=task_config,
                            diffs_stddev_by_level=diffs_stddev_by_level,
                            mean_by_level=mean_by_level,
                            stddev_by_level=stddev_by_level,
                            learning_rate=1e-5,
                            use_multi_gpu=True
                        )
        multi_gpu_fine_tuner.fine_tune(train_inputs, train_targets, train_forcings, eval_inputs, eval_targets, eval_forcings)
    else:
        print("\n当前环境只有一个设备，使用单GPU训练")
        fine_tuner = GraphCastFineTuner(
                            params=params,
                            state=state,
                            model_config=model_config,
                            task_config=task_config,
                            diffs_stddev_by_level=diffs_stddev_by_level,
                            mean_by_level=mean_by_level,
                            stddev_by_level=stddev_by_level,
                            learning_rate=1e-5
                        )
        fine_tuner.fine_tune(train_inputs, train_targets, train_forcings, eval_inputs, eval_targets, eval_forcings)
    
    print("\n=== 训练完成 ===")

# 调用示例函数
run_multi_gpu_training_example()