import os
import argparse
import json
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import imageio
import random
import matplotlib.pyplot as plt
import torch.nn.functional as F

import csv
import time

# ----------------------------
# Config / 하이퍼파라미터
# ----------------------------
DEFAULT_CONFIG = {
    "env_id": "LunarLander-v3",
    "episodes": 10000,
    "gamma": 0.99,
    "lam": 0.95,
    "ppo_epochs": 8,
    "lr": 1e-4,
    "clip_eps": 0.2,
    "entropy_coef": 0.01,
    "value_coef": 0.5,
    "max_grad_norm": 0.5,
    "save_gif_every": 500,
    "gif_fps": 30,
    "seed": 0,
    "hidden_size": 256,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "render_mode": None,  # or "rgb_array"
    "frame_stack": 1,
    # Box2D / environment physics controls
    "box2d_gravity": -10.0,
    "enable_wind": True,
    "wind_power": 10.0,
    "turbulence_power": 1.0,
    # optional per-episode randomization
    "randomize_physics": False,
    "physics_ranges": {  # uniform sampling용 (기존)
        "box2d_gravity": [-15.0, -5.0],
        "wind_power": [0.0, 20.0],
        "turbulence_power": [0.0, 5.0]
    },
    # Normal distribution 파라미터 추가
    "physics_normal": {  
        "box2d_gravity": {"mean": -5.0, "std": 2.5, "clip": [-20.0, -2.0]},  # 평균 -5, 표준편차 2.5, 클리핑 범위 -20 ~ -2
        "wind_power": {"mean": 15.0, "std": 3.5, "clip": [0.0, 25.0]},  # 평균 15, 표준편차 3.5, 클리핑 범위 0 ~ 25
        "turbulence_power": {"mean": 1.5, "std": 0.6, "clip": [0.0, 3.0]} # 평균 1.5, 표준편차 0.6, 클리핑 범위 0 ~ 3
    },
    "use_normal_distribution": True  # True면 normal, False면 uniform
}


def sample_physics_from_ranges(cfg):
    sampled = {}
    
    if cfg.get("use_normal_distribution", False):
        # Normal distribution으로 샘플링
        pn = cfg.get("physics_normal", {})
        
        if "box2d_gravity" in pn:
            params = pn["box2d_gravity"]
            gravity = np.random.normal(params["mean"], params["std"])
            # 클리핑 적용
            if "clip" in params:
                gravity = np.clip(gravity, params["clip"][0], params["clip"][1])
            sampled["box2d_gravity"] = gravity
            
        if "wind_power" in pn:
            params = pn["wind_power"]
            wind = np.random.normal(params["mean"], params["std"])
            if "clip" in params:
                wind = np.clip(wind, params["clip"][0], params["clip"][1])
            sampled["wind_power"] = wind
            
        if "turbulence_power" in pn:
            params = pn["turbulence_power"]
            turb = np.random.normal(params["mean"], params["std"])
            if "clip" in params:
                turb = np.clip(turb, params["clip"][0], params["clip"][1])
            sampled["turbulence_power"] = turb
    else:
        # 기존 uniform distribution
        pr = cfg.get("physics_ranges", {})
        if "box2d_gravity" in pr:
            sampled["box2d_gravity"] = np.random.uniform(pr["box2d_gravity"][0], pr["box2d_gravity"][1])
        if "wind_power" in pr:
            sampled["wind_power"] = np.random.uniform(pr["wind_power"][0], pr["wind_power"][1])
        if "turbulence_power" in pr:
            sampled["turbulence_power"] = np.random.uniform(pr["turbulence_power"][0], pr["turbulence_power"][1])
    
    return sampled

def load_config(path):
    cfg = DEFAULT_CONFIG.copy()
    if path and os.path.exists(path):
        with open(path, "r") as f:
            cfg.update(json.load(f))
    return cfg

# ----------------------------
# RNN Actor-Critic 정의
# ----------------------------
class RecurrentActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_size=256, device="cpu"):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.device = device

        # input -> GRU -> shared hidden
        self.input_linear = nn.Linear(obs_dim, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.actor = nn.Linear(hidden_size, action_dim)  # logits
        self.critic = nn.Linear(hidden_size, 1)

        self.to(self.device)

    def init_hidden(self, batch_size=1):
        return torch.zeros(1, batch_size, self.hidden_size, device=self.device)

    def forward(self, states_seq, hidden=None):
        if states_seq.dim() == 2:
            seq = states_seq.unsqueeze(0)  # (1, T, obs_dim)
        elif states_seq.dim() == 3:
            seq = states_seq
        else:
            raise ValueError("states_seq must be 2D or 3D tensor")

        x = torch.relu(self.input_linear(seq))
        h0 = self.init_hidden(batch_size=x.size(0)) if hidden is None else hidden
        out, hn = self.gru(x, h0)
        logits = self.actor(out)
        values = self.critic(out)
        return logits, values.squeeze(-1), hn

    def act(self, state, hidden=None):
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        logits, value, hidden = self.forward(state, hidden)
        
        # Logits 안정화
        logits = torch.clamp(logits, min=-20, max=20)
        probs = F.softmax(logits, dim=-1)
        
        # 확률 클리핑으로 numerical stability 확보
        probs = torch.clamp(probs, min=1e-8, max=1.0)
        
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()  # entropy 계산 추가
        
        return action.item(), log_prob, value.squeeze(), hidden, entropy 


# ----------------------------
# GAE 계산
# ----------------------------
def compute_gae(rewards, values, dones, gamma, lam):
    values = values + [0.0]
    gae = 0.0
    returns = []
    advantages = []  # advantages 리스트 추가
    
    for step in reversed(range(len(rewards))):
        mask = 0.0 if dones[step] else 1.0
        delta = rewards[step] + gamma * values[step + 1] * mask - values[step]
        gae = delta + gamma * lam * mask * gae
        returns.insert(0, gae + values[step])
        advantages.insert(0, gae)  # advantages에 gae 추가
        
    # Advantage normalization 추가
    if len(advantages) > 1:
        advantages = np.array(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return returns

def moving_average(x, window):
    """Running (causal) moving average with same-length output (pads start with smaller-window means)."""
    x = np.asarray(x, dtype=float)
    w = int(window)
    if x.size == 0 or w <= 1:
        return x.copy()
    # cumulative sum with leading zero for easy range sums
    cumsum = np.cumsum(np.insert(x, 0, 0.0))
    ret = np.empty_like(x)
    for i in range(x.size):
        start = max(0, i - w + 1)
        ret[i] = (cumsum[i + 1] - cumsum[start]) / (i - start + 1)
    return ret

# ----------------------------
# 환경 물리 파라미터 적용 / 샘플링
# ----------------------------
def set_env_physics(env, gravity=None, enable_wind=None, wind_power=None, turbulence_power=None):
    ue = env.unwrapped
    if gravity is not None:
        if hasattr(ue, "world") and hasattr(ue.world, "gravity"):
            try:
                ue.world.gravity = (0.0, gravity)
            except Exception:
                try:
                    ue.world.gravity.y = gravity
                except Exception:
                    pass
    if enable_wind is not None and hasattr(ue, "enable_wind"):
        try:
            ue.enable_wind = bool(enable_wind)
        except Exception:
            pass
    if wind_power is not None and hasattr(ue, "wind_power"):
        try:
            ue.wind_power = float(wind_power)
        except Exception:
            pass
    if turbulence_power is not None and hasattr(ue, "turbulence_power"):
        try:
            ue.turbulence_power = float(turbulence_power)
        except Exception:
            pass

# ----------------------------
# Training loop
# ----------------------------
def train(config_path=None):
    cfg = load_config(config_path)
    device = torch.device(cfg["device"])
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])

    os.makedirs("logs", exist_ok=True)
    metrics_csv_path = os.path.join("logs", f"episode_metrics_{int(time.time())}.csv")
    metrics_fields = ["episode","total_reward","length","avg50","mean_entropy","physics","timestamp"]
    metrics_file = open(metrics_csv_path, "w", newline="")
    metrics_writer = csv.DictWriter(metrics_file, fieldnames=metrics_fields)
    metrics_writer.writeheader()
    # per-episode detailed data saved to logs/episodes/
    os.makedirs("logs/episodes", exist_ok=True)

    env_kwargs = {}
    if cfg.get("render_mode"):
        env_kwargs["render_mode"] = cfg["render_mode"]
    env = gym.make(cfg["env_id"], **env_kwargs)

    # Warn when render_mode isn't rgb_array (no frames captured) and advise how to enable GIF saving.
    if cfg.get("render_mode") != "rgb_array":
        print("Warning: render_mode != 'rgb_array'. GIF frames won't be captured. Set render_mode='rgb_array' in config to enable GIF saving.")

    # apply initial physics from config
    set_env_physics(env,
                    gravity=cfg.get("box2d_gravity", None),
                    enable_wind=cfg.get("enable_wind", None),
                    wind_power=cfg.get("wind_power", None),
                    turbulence_power=cfg.get("turbulence_power", None))

    obs_shape = env.observation_space.shape
    assert obs_shape is not None, "Obs shape must be defined"
    obs_dim = obs_shape[0] * cfg["frame_stack"]
    assert isinstance(env.action_space, gym.spaces.Discrete), "This trainer supports Discrete action spaces only"
    action_dim = env.action_space.n

    policy = RecurrentActorCritic(obs_dim, action_dim, hidden_size=cfg["hidden_size"], device=device)
    optimizer = optim.Adam(policy.parameters(), lr=cfg["lr"])

    episode_rewards = []
    policy_losses = []
    value_losses = []
    entropies = []

    for episode in range(cfg["episodes"]):
        # optionally randomize physics per-episode
        if cfg.get("randomize_physics", False):
            sampled = sample_physics_from_ranges(cfg)
            set_env_physics(env,
                            gravity=sampled.get("box2d_gravity", cfg.get("box2d_gravity")),
                            wind_power=sampled.get("wind_power", cfg.get("wind_power")),
                            turbulence_power=sampled.get("turbulence_power", cfg.get("turbulence_power")))
            print(f"[Episode {episode}] Sampled physics: {sampled}")

        state, _ = env.reset()
        policy_hidden = policy.init_hidden(batch_size=1)

        total_reward = 0.0
        done = False

        state_buffer = deque(maxlen=cfg["frame_stack"])
        states = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        values = []
        entropies_episode = []
        frames = []
        returns = []

        while not done:
            if cfg.get("render_mode") == "rgb_array" and (episode % cfg.get("save_gif_every", 100) == 0):
                frames.append(env.render())

            state_buffer.append(state)
            if len(state_buffer) < cfg["frame_stack"]:
                pad = [np.zeros_like(state) for _ in range(cfg["frame_stack"] - len(state_buffer))]
                stacked = np.concatenate(pad + list(state_buffer), axis=0)
            else:
                stacked = np.concatenate(list(state_buffer), axis=0)

            action, logp, value, policy_hidden, ent = policy.act(stacked, policy_hidden)

            # ensure discrete action
            if not isinstance(action, int):
                action = int(action)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done_flag = terminated or truncated

            states.append(stacked)
            actions.append(action)
            log_probs.append(logp.detach().cpu().item())
            rewards.append(float(reward))
            dones.append(done_flag)
            values.append(float(value.detach().cpu().item()))
            entropies_episode.append(ent.detach().cpu().item() if torch.is_tensor(ent) else ent)

            state = next_state
            total_reward += reward
            done = done_flag

        episode_rewards.append(total_reward)
        # compute some summary stats for this episode
        ep_len = len(rewards)
        mean_ent = float(np.mean(entropies_episode)) if entropies_episode else float("nan")
        avg50 = np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 1 else total_reward
        # physics state capture
        physics_info = {}
        try:
            ue = env.unwrapped
            if hasattr(ue, "world") and hasattr(ue.world, "gravity"):
                physics_info["gravity"] = tuple(ue.world.gravity)
            if hasattr(ue, "enable_wind"):
                physics_info["enable_wind"] = bool(getattr(ue, "enable_wind", None))
            if hasattr(ue, "wind_power"):
                physics_info["wind_power"] = float(getattr(ue, "wind_power", None))
            if hasattr(ue, "turbulence_power"):
                physics_info["turbulence_power"] = float(getattr(ue, "turbulence_power", None))
        except Exception:
            physics_info = {}
        # write summary row
        metrics_writer.writerow({
            "episode": episode,
            "total_reward": float(total_reward),
            "length": int(ep_len),
            "avg50": float(avg50),
            "mean_entropy": float(mean_ent),
            "physics": json.dumps(physics_info),
            "timestamp": int(time.time())
        })
        metrics_file.flush()
        # save detailed per-episode JSON for later analysis (frames/actions/returns)
        if episode % max(1, cfg.get("save_gif_every", 100)) == 0:
            detail = {
                "episode": episode,
                "total_reward": float(total_reward),
                "length": int(ep_len),
                "states": None,   # not saving full states by default to limit size
                "actions": actions,
                "rewards": rewards,
                "returns": returns,
                "values": values,
                "log_probs": log_probs,
                "entropies": entropies_episode,
                "physics": physics_info
            }
            with open(os.path.join("logs","episodes", f"episode_{episode}.json"), "w") as jf:
                json.dump(detail, jf)

        if cfg.get("render_mode") == "rgb_array" and len(frames) > 0 and (episode % cfg.get("save_gif_every", 100) == 0):
            os.makedirs("gifs", exist_ok=True)
            imageio.mimsave(f'gifs/episode_{episode}.gif', frames, fps=cfg["gif_fps"])
            try:
                os.makedirs("gifs", exist_ok=True)
                clean_frames = []
                for f in frames:
                    ff = np.array(f)
                    # scale float frames in [0,1] to uint8
                    if np.issubdtype(ff.dtype, np.floating):
                        ff = (np.clip(ff, 0.0, 1.0) * 255.0).astype(np.uint8)
                    else:
                        ff = ff.astype(np.uint8)
                    clean_frames.append(ff)
                imageio.mimsave(f'gifs/episode_{episode}.gif', clean_frames, fps=cfg["gif_fps"])
                print(f"Saved GIF: gifs/episode_{episode}.gif (frames={len(clean_frames)})")
            except Exception as e:
                print("Failed to save GIF:", e)
        returns = compute_gae(rewards, values, dones, cfg["gamma"], cfg["lam"])
        returns_t = torch.FloatTensor(np.array(returns)).to(device)
        values_t = torch.FloatTensor(np.array(values)).to(device)
        advantages = returns_t - values_t
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # save detailed per-episode JSON for later analysis (frames/actions/returns)
        if episode % max(1, cfg.get("save_gif_every", 100)) == 0:
           detail = {
               "episode": episode,
               "total_reward": float(total_reward),
               "length": int(len(rewards)),
               "states": None,
                "actions": actions,
                "rewards": rewards,
                "returns": returns,
                "values": values,
                "log_probs": log_probs,
                "entropies": entropies_episode,
                "physics": physics_info
            }
        with open(os.path.join("logs","episodes", f"episode_{episode}.json"), "w") as jf:
                json.dump(detail, jf)

        states_np = np.array(states)
        states_t = torch.FloatTensor(states_np).to(device)
        actions_t = torch.LongTensor(np.array(actions)).to(device)
        old_log_probs_t = torch.FloatTensor(np.array(log_probs)).to(device)

        for _ in range(cfg["ppo_epochs"]):
            logits_seq, value_preds_seq, _ = policy(states_t)
            if logits_seq.dim() == 3:
                logits = logits_seq.squeeze(0)
                value_preds = value_preds_seq.squeeze(0)
            elif logits_seq.dim() == 2:
                logits = logits_seq
                value_preds = value_preds_seq
            else:
                raise RuntimeError("Unexpected logits_seq dim")

            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions_t)
            entropy = dist.entropy().mean()

            # Ratio 계산 시 안정화 추가
            ratio = (new_log_probs - old_log_probs_t).exp()
            ratio = torch.clamp(ratio, min=1e-6, max=1e6)  # 극단값 방지
            
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - cfg["clip_eps"], 1.0 + cfg["clip_eps"]) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # NaN 체크 추가
            if torch.isnan(policy_loss):
                print(f"Warning: Policy loss is NaN at episode {episode}, skipping update")
                continue
                
            value_loss = nn.MSELoss()(value_preds, returns_t)
            loss = policy_loss + cfg["value_coef"] * value_loss - cfg["entropy_coef"] * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg["max_grad_norm"])
            optimizer.step()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropies.append(entropy.item())

        if episode % 500 == 0:
            avg500 = np.mean(episode_rewards[-500:]) if len(episode_rewards) >= 500 else np.mean(episode_rewards)  # MA500으로 변경
            last_ent = entropies[-1] if entropies else np.nan
            last_policy_loss = policy_losses[-1] if policy_losses else np.nan
            last_value_loss = value_losses[-1] if value_losses else np.nan
            print(f"Ep {episode:4d} Reward {total_reward:7.2f} Avg500 {avg500:7.2f} "
                  f"PL {last_policy_loss:.4f} VL {last_value_loss:.2f} Ent {last_ent:.3f}")

    os.makedirs("models", exist_ok=True)
    # --- 학습 지표 시각화 저장 (간단한 PNG) ---
    try:
        plt.figure(figsize=(12,8))
        if episode_rewards:
            plt.subplot(2,2,1)
            plt.plot(episode_rewards, label="Episode Reward")
            plt.title("Episode Reward"); plt.grid(); plt.legend()
        if policy_losses:
            plt.subplot(2,2,2)
            plt.plot(policy_losses, label="Policy Loss")
            plt.title("Policy Loss"); plt.grid(); plt.legend()
        if value_losses:
            plt.subplot(2,2,3)
            plt.plot(value_losses, label="Value Loss")
            plt.title("Value Loss"); plt.grid(); plt.legend()
        if entropies:
            plt.subplot(2,2,4)
            plt.plot(entropies, label="Entropy")
            plt.title("Entropy"); plt.grid(); plt.legend()
        plt.tight_layout()
        plt.savefig("ppo_rnn_metrics.png")
        print("Saved metrics to ppo_rnn_metrics.png")
        plt.figure(figsize=(12,8))
        ma_win = 50
        if episode_rewards:
            ma_rewards = moving_average(episode_rewards, ma_win)
            plt.subplot(2,2,1)
            plt.plot(episode_rewards, color="lightgray", label="Episode Reward")
            plt.plot(ma_rewards, color="tab:blue", label=f"MA{ma_win}")
            plt.title("Episode Reward"); plt.grid(); plt.legend()
        if policy_losses:
            ma_policy = moving_average(policy_losses, ma_win)
            plt.subplot(2,2,2)
            plt.plot(policy_losses, color="lightgray", label="Policy Loss")
            plt.plot(ma_policy, color="tab:orange", label=f"MA{ma_win}")
            plt.title("Policy Loss"); plt.grid(); plt.legend()
        if value_losses:
            ma_value = moving_average(value_losses, ma_win)
            plt.subplot(2,2,3)
            plt.plot(value_losses, color="lightgray", label="Value Loss")
            plt.plot(ma_value, color="tab:green", label=f"MA{ma_win}")
            plt.title("Value Loss"); plt.grid(); plt.legend()
        if entropies:
            ma_ent = moving_average(entropies, ma_win)
            plt.subplot(2,2,4)
            plt.plot(entropies, color="lightgray", label="Entropy")
            plt.plot(ma_ent, color="tab:red", label=f"MA{ma_win}")
            plt.title("Entropy"); plt.grid(); plt.legend()
        plt.tight_layout()
        plt.savefig("ppo_rnn_metrics.png")
        print("Saved metrics to ppo_rnn_metrics.png")
    except Exception as e:
        print("Failed to save metrics plot:", e)

        # close metrics file
    try:
        metrics_file.close()
        print(f"Saved per-episode metrics to {metrics_csv_path}")
    except Exception:
        pass

    torch.save(policy.state_dict(), "models/ppo_rnn_lunar.pt")
    print("Saved model to models/ppo_rnn_lunar.pt")

    # optional: automatically record a few episodes with the trained policy
    try:
        record_out = os.path.join("models", "recording_after_training.mp4")
        record_policy(cfg["env_id"], policy, out_path=record_out, episodes=3, fps=30, device=device)
    except Exception as e:
        print("Auto-record failed:", e)

    env.close()

# ← 여기에 바로 추가 (train 함수 끝난 직후)
def record_policy(env_id, model, out_path="recording.mp4", episodes=3, fps=30, device="cpu"):
    from collections import deque as _deque
    # create env with rgb_array render
    env = gym.make(env_id, render_mode="rgb_array")
    set_env_physics(env,
                    gravity=DEFAULT_CONFIG.get("box2d_gravity", None),
                    enable_wind=DEFAULT_CONFIG.get("enable_wind", None),
                    wind_power=DEFAULT_CONFIG.get("wind_power", None),
                    turbulence_power=DEFAULT_CONFIG.get("turbulence_power", None))
    frames_all = []
    model.to(device)
    model.eval()
    
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        hidden = model.init_hidden(batch_size=1)
        state_buffer = _deque(maxlen=DEFAULT_CONFIG.get("frame_stack",1))
        ep_frames = []
        
        while not done:
            state_buffer.append(state)
            if len(state_buffer) < DEFAULT_CONFIG.get("frame_stack",1):
                pad = [ np.zeros_like(state) for _ in range(DEFAULT_CONFIG.get("frame_stack",1) - len(state_buffer)) ]
                stacked = np.concatenate(pad + list(state_buffer), axis=0)
            else:
                stacked = np.concatenate(list(state_buffer), axis=0)
                
            with torch.no_grad():
                st_t = torch.FloatTensor(np.array(stacked)).to(device).unsqueeze(0)  # (1, obs_dim)
                logits_seq, _, _ = model(st_t)
                if logits_seq.dim() == 3:
                    logits_step = logits_seq.squeeze(0)[-1]
                elif logits_seq.dim() == 2:
                    logits_step = logits_seq[-1]
                else:
                    logits_step = logits_seq
                action = int(torch.argmax(logits_step).cpu().item())
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            frame = env.render()
            ff = np.array(frame)
            if np.issubdtype(ff.dtype, np.floating):
                ff = (np.clip(ff, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                ff = ff.astype(np.uint8)
            ep_frames.append(ff)
            state = next_state
        frames_all.extend(ep_frames)
    env.close()
    try:
        writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8)
        for f in frames_all:
            writer.append_data(f)
        writer.close()
        print(f"Saved recording to {out_path} (frames={len(frames_all)})")
    except Exception as e:
        print("Failed to save MP4, trying GIF fallback:", e)
        try:
            gif_path = out_path.rsplit(".",1)[0] + ".gif"
            imageio.mimsave(gif_path, frames_all, fps=fps)
            print(f"Saved GIF fallback to {gif_path}")
        except Exception as e2:
            print("Failed to save GIF fallback:", e2)

def load_and_record(model_path, cfg_path=None, out_path="recording_from_saved.mp4", episodes=3, fps=30):
    cfg = load_config(cfg_path)
    device = torch.device(cfg.get("device", "cpu"))
    obs_shape = gym.make(cfg["env_id"]).observation_space.shape
    obs_dim = obs_shape[0] * cfg.get("frame_stack",1)
    action_dim = gym.make(cfg["env_id"]).action_space.n
    model = RecurrentActorCritic(obs_dim, action_dim, hidden_size=cfg.get("hidden_size",256), device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    record_policy(cfg["env_id"], model, out_path=out_path, episodes=episodes, fps=fps, device=device)

# ----------------------------
# Entry point
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="path to json config")
    args = parser.parse_args()
    train(args.config)