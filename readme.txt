"commit code"
开发在本地分支，例如lerobot043
提交代码时：
git switch master
git pull gitee master

git switch lerobot043
如果有冲突，先解决冲突
git merge master --allow-unrelated-histories

git add .
git commit -m "xxxx"
git push gitee lerobot043:chenjun
chenjun分支是我建的，也可以建自己的分支。


lerobot-train   --policy.pretrained_path=/home/hpc/VLA/lerobot/weights/xvla \
--dataset.repo_id=/home/hpc/VLA/data_12_17_train  \
--policy.type=xvla  \
--output_dir=/home/hpc/VLA/lerobot/result_12_17   \
--job_name=lerobot_training   \
--policy.device=cuda   \
--policy.repo_id=lerobot/xvla-base  \
 --wandb.enable=true



 lerobot-train \
  --dataset.repo_id=/home/hpc/VLA/data_test \
  --output_dir=/home/hpc/VLA/lerobot/result_12_29 \
  --job_name=xvla_training_agilex \
  --policy.path=/home/hpc/VLA/lerobot/weights/xvla_agibot_world \
  --policy.repo_id="H1412205371/xvla-cup-robot" \
  --policy.dtype=bfloat16 \
  --policy.action_mode=auto \
  --steps=20000 \
  --policy.device=cuda \
  --policy.freeze_vision_encoder=false \
  --policy.freeze_language_encoder=false \
  --policy.train_policy_transformer=true \
  --policy.train_soft_prompts=true \
  --wandb.enable=true


lerobot-train \
  --dataset.repo_id=/home/hpc/VLA/data_endpos12_5_3_1 \
  --output_dir=/home/hpc/VLA/lerobot/result_12_24 \
  --job_name=xvla_training_test \
  --policy.path=/home/hpc/VLA/lerobot/weights/xvla_agibot_world \
  --policy.repo_id="H1412205371/xvla-cup-robot" \
  --policy.dtype=bfloat16 \
  --policy.action_mode=auto \
  --steps=20000 \
  --policy.device=cuda \
  --policy.freeze_vision_encoder=false \
  --policy.freeze_language_encoder=false \
  --policy.train_policy_transformer=true \
  --policy.train_soft_prompts=true \
  --wandb.enable=true

  

  # Using a multi-GPU setup
accelerate launch \
  --multi_gpu \
  --num_processes=1 \
  $(which lerobot-train) \
  --output_dir=/home/hpc/VLA/lerobot/result_12_23_groot \
  --save_checkpoint=true \
  --batch_size=4 \
  --steps=200000 \
  --save_freq=20000 \
  --log_freq=1000 \
  --policy.push_to_hub=true \
  --policy.type=groot \
  --policy.repo_id=nvidia/GR00T-N1.5-3B \
  --policy.tune_diffusion_model=false \
  --dataset.repo_id=/home/hpc/VLA/data_12_17_train \
  --wandb.enable=true \
  --wandb.disable_artifact=true \
  --job_name=groot_1.5


#数据格式转换
python -m lerobot.datasets.v30.convert_dataset_v21_to_v30  --repo-id=/home/hpc/VLA/data_endpos12_5_3_1


lerobot-train \
  --dataset.repo_id=  \
  --dataset.root=/home/hpc/VLA/package_data/single/merge_data/merge_data_1_8_922 \
  --policy.type=act \
  --output_dir=/home/hpc/VLA/lerobot_models/act/merge_data_1_8_922 \
  --job_name=merge_data_1_8_922 \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.push_to_hub=False \
  --policy.device=cuda \
  --batch_size=16 \
  --steps=800000 \
  --save_freq=10000 \
  --log_freq=1000



lerobot-edit-dataset  --repo_id=data_test --new_repo_id=data_test_delete --operation.type=delete_episodes --operation.episode_indices="[41]" --root=/home/hpc/VLA/data_test_delete --push_to_hub=false


#要修改末端位姿训练，在/home/hpc/VLA/lerobot/src/lerobot/datasets/lerobot_dataset.py 的 
class LeRobotDataset(torch.utils.data.Dataset):
 def __getitem__



 lerobot-train \
  --dataset.repo_id=  \
  --dataset.root=../package_data/single/merge_data_1_4 \
  --policy.type=act \
  --output_dir=/home/hpc/VLA/lerobot_models/act/merge_data_1_4 \
  --job_name=act_graps_1_4 \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.push_to_hub=False \
  --policy.device=cuda \
  --batch_size=8 \
  --steps=200000 \
  --save_freq=10000 \
  --log_freq=1000

#ACT末端位姿训练命令
 lerobot-train \
  --dataset.repo_id=  \
  --dataset.root=../package_data/single/merge_data_1_4_eef \
  --policy.type=act \
  --output_dir=/home/hpc/VLA/lerobot_models/act/merge_data_1_4_eef \
  --job_name=act_graps_1_5 \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.push_to_hub=False \
  --policy.device=cuda \
  --batch_size=8 \
  --steps=200000 \
  --save_freq=10000 \
  --log_freq=1000