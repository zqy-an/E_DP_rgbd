# E_DP_rgbd


## 环境配置
### 1. 创建Conda环境

```bash
conda create -n E_DP python=3.10.15
conda activate E_DP
```
### 2. 安装ManiSkill
```bash
pip install --upgrade git+https://github.com/haosulab/ManiSkill.git
```
or
```bash
git clone https://github.com/haosulab/ManiSkill.git
cd ManiSkill
pip install -e .
```

### 3. 安装Theia

```bash
git clone https://github.com/bdaiinstitute/theia.git
cd theia
pip install -e .
```

### 4. 安装PyTorch及其他依赖项

```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.45.2 zarr==2.16.1 timm==1.0.11
pip install scikit-learn
```

## 运行

### maniskill数据集制作
```bash
python mani_skill.utils.download_demo "StackCube-v1"

python -m mani_skill.trajectory.replay_trajectory \
  --traj-path ~/.maniskill/demos/StackCube-v1/motionplanning/trajectory.h5 \
  --use-first-env-state -c pd_ee_delta_pose -o rgbd \
  --save-traj --num-procs 10 -b cpu
```

### 代码运行
```bash
python 3_dp_v2_all_rgbdd.py --env-id $env_id \
  --demo-path $demo_path \
  --theia_model_path "theaiinstitute/theia-tiny-patch16-224-cddsv" \
  --control-mode "pd_ee_delta_pose" --sim-backend "cpu" --num-demos 150 \
  --total_iters 400_000
```
