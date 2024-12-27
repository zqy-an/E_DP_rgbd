#!/bin/bash

# 定义要使用的 env_id 列表
env_ids=("PegInsertionSide-v1" "LiftPegUpright-v1")
demo_paths=(
    "/home/zqy/code/RL2/dataset/demos/PegInsertionSide-v1/motionplanning/trajectory.rgbd.pd_ee_delta_pose.cpu.h5"
    "/home/zqy/code/RL2/dataset/demos/LiftPegUpright-v1/motionplanning/trajectory.rgbd.pd_ee_delta_pose.cpu.h5"
)
cudas=('cuda:5' 'cuda:5' 'cuda:5' 'cuda:5')
demos=(250 250)
emb_dims=(512 512)

for i in "${!env_ids[@]}"; do
    env_id=${env_ids[$i]}
    demo_path=${demo_paths[$i]}
    cuda=${cudas[$i]}
    num_demos=${demos[$i]}

    python 4_dp_v2_all_rgbdd.py --env-id $env_id \
      --demo-path $demo_path --device $cuda --seed 1 \
      --control-mode "pd_ee_delta_pose"  --num-demos $num_demos \
      --total_iters 460_000 --batch_size 64 --emb_dim ${emb_dims[$i]}
done

# 等待所有后台任务完成
wait

echo "All training processes have completed."
