# Hopper-v4 SAC  
Soft Actor-Critic (SAC) implementation for the **Hopper-v4** MuJoCo environment.

---

## 🧠 Overview  
- Train and evaluate a SAC agent using **Stable-Baselines3** and **Gymnasium (MuJoCo)**  
- Visualize learning performance and reward statistics  
- Generate evaluation plots and summary metrics  

---

## ⚙️ Installation  
```bash
pip install stable-baselines3[extra]
pip install "gymnasium[mujoco]"
pip install matplotlib seaborn
```

---

## 🚀 Usage  
```bash
python train.py
```

---


## 🧩 Future Work  
- Extend to **Hopper-v5**  
- Compare SAC with **PPO** and **TD3**  
- Add robustness tests and reward shaping  

---

## 📚 References  
- Haarnoja et al., *Soft Actor-Critic*, ICML 2018. DOI: [10.48550/arXiv.1801.01290](https://doi.org/10.48550/arXiv.1801.01290)  
- Brockman et al., *OpenAI Gym*, arXiv:1606.01540, 2016  
- Todorov et al., *MuJoCo: A physics engine for model-based control*, IROS 2012  
- Raffin et al., *Stable Baselines3*, JMLR 2021  
