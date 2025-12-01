# This script trains a Twin Delayed Deep Deterministic Policy Gradient (TD3) agent on the Hopper-v4 environment for continuous-control performance.
# ============================================
# 1) Create directories for saving models & logs
# ============================================
import os
import gymnasium as gym

from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np

base_dir = "./td3_local"
drive_model_path = os.path.join(base_dir, "td3_models")
save_path = os.path.join(drive_model_path, "td3_hopper")
log_dir = os.path.join(base_dir, "td3_logs", "td3_default")

for d in [base_dir, drive_model_path, save_path, log_dir]:
    os.makedirs(d, exist_ok=True)


# ============================================
# 2) Build training & evaluation environments
# ============================================
ENV_ID = "Hopper-v4"

def make_env():
    """
    Create a Hopper-v4 env wrapped with Monitor for logging.
    """
    env = gym.make(ENV_ID)
    env = Monitor(env)
    return env

env = DummyVecEnv([make_env])
eval_env = DummyVecEnv([make_env])


# ============================================
# 3) Create action noise for TD3 exploration
# ============================================
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(
    mean=np.zeros(n_actions),
    sigma=0.1 * np.ones(n_actions),
)


# ============================================
# 4) Build TD3 model with hyperparameters
# ============================================
model = TD3(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    seed=0,
    learning_rate=3e-4,
    buffer_size=300_000,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    train_freq=1,
    gradient_steps=1,
    action_noise=action_noise,
    tensorboard_log=log_dir,
    policy_kwargs=dict(
        net_arch=dict(pi=[256, 256], qf=[256, 256])
    )
)


# ============================================
# 5) Define callbacks for evaluation & checkpoint saving
# ============================================
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=save_path,
    log_path=log_dir,
    eval_freq=10_000,
    n_eval_episodes=10,
    deterministic=True,
    render=False,
)

checkpoint_callback = CheckpointCallback(
    save_freq=10_000,
    save_path=save_path,
    name_prefix="td3_checkpoint",
)


# ============================================
# 6) Train TD3 agent
# ============================================
TOTAL_STEPS = 500_000

print(f"Start training TD3 for {TOTAL_STEPS:,} steps...")
model.learn(
    total_timesteps=TOTAL_STEPS,
    callback=[eval_callback, checkpoint_callback],
    progress_bar=True
)


# ============================================
# 7) Save final trained TD3 model
# ============================================
final_model_path = os.path.join(save_path, "td3_hopper")
model.save(final_model_path)
print(f"Training complete! Saved to: {final_model_path}")
print(f"TensorBoard: tensorboard --logdir {log_dir}")
