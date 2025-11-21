# ==== 改进版 SAC Hopper 训练 ====
import os
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# 1) Directory Settings
base_dir = "./sac_local"
drive_model_path = os.path.join(base_dir, "sac_models")
save_path = os.path.join(drive_model_path, "sac_hopper")
log_dir = os.path.join(base_dir, "sac_logs", "sac_default")

for d in [base_dir, drive_model_path, save_path, log_dir]:
    os.makedirs(d, exist_ok=True)

# 2) environment
ENV_ID = "Hopper-v4"

def make_env():
    env = gym.make(ENV_ID)
    env = Monitor(env)  
    return env

env = DummyVecEnv([make_env])
eval_env = DummyVecEnv([make_env])

# 3) SAC model
model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    seed=0,
    learning_rate=3e-4,
    buffer_size=300_000,       
    batch_size=256,
    gamma=0.99,
    tau=0.005,                  
    ent_coef="auto",
    train_freq=1,               
    gradient_steps=1,
    tensorboard_log=log_dir,
    policy_kwargs=dict(net_arch=[256, 256]) 
)

# 4) callback function
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=save_path,
    log_path=log_dir,
    eval_freq=10_000,            
    n_eval_episodes=10,          
    deterministic=True,
    render=False,
)

# Save checkpoints regularly (to prevent training interruptions)
checkpoint_callback = CheckpointCallback(
    save_freq=10_000,
    save_path=save_path,
    name_prefix="sac_checkpoint"
)

# 5) train
TOTAL_STEPS = 500_000  #  Increase to 500,000 steps（about 2-3h）

print(f"Start training {TOTAL_STEPS:,} step...")
model.learn(
    total_timesteps=TOTAL_STEPS,
    callback=[eval_callback, checkpoint_callback],
    progress_bar=True  # Show progress bar
)

# 6) save model
model.save(os.path.join(save_path, "sac_hopper"))
print(f"Training is complete! The model is saved in: {save_path}")
print(f"TensorBoard: tensorboard --logdir {log_dir}")