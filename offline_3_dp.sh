#!/bin/bash

# 定义要使用的 env_id 列表
env_ids=("StackCube-v1" "PegInsertionSide-v1" "PullCubeTool-v1" "PushCube-v1")
demo_paths=(
    "/home/zqy/code/RL2/dataset/demos/StackCube-v1/motionplanning/trajectory.rgbd.pd_ee_delta_pose.cpu.h5"
    "/home/zqy/code/RL2/dataset/demos/PegInsertionSide-v1/motionplanning/trajectory.rgbd.pd_ee_delta_pose.cpu.h5"
    "/home/zqy/code/RL2/dataset/demos/PullCubeTool-v1/motionplanning/trajectory.rgbd.pd_ee_delta_pose.cpu.h5"
    "/home/zqy/code/RL2/dataset/demos/PushCube-v1/motionplanning/trajectory.rgbd.pd_ee_delta_pose.cpu.h5"
)

# 启动四个并行的训练任务
for i in "${!env_ids[@]}"; do
    env_id=${env_ids[$i]}
    demo_path=${demo_paths[$i]}
    python 3_dp_v2_Conv_rgbdd.py --env-id $env_id \
      --demo-path $demo_path \
      --control-mode "pd_ee_delta_pose" --sim-backend "cpu" --num-demos 150 \
      --total_iters 500_000
done

# 等待所有后台任务完成
wait

echo "All training processes have completed."
