# DQN

## Q-Learning + Experience Replay
[![Paper](https://img.shields.io/badge/paper-DQN-red)](https://arxiv.org/abs/1312.5602)
[![Implementation](https://img.shields.io/badge/implementation-PyTorch-blue)](https://pytorch.org/)
<hr style="border: none; border-top: 1px solid #eaecef;">

**Playing Atari with Deep Reinforcement Learning**. NIPS Deep Learning Workshop 2013. [Paper](https://arxiv.org/abs/1312.5602)

*Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, Martin Riedmiller*

This implementation uses a **Convolutional Neural Network** to process preprocessed raw pixel frames. It applies **Q-Learning** to calculate action-value functions for each action given a specific state, and employs a **Fully Connected Network** to approximate the Q-learning function. **Experience Replay** has been implemented to support the learning process.

### 500-Episodes
![best_quality_breakout-DQN-episode-Early](https://github.com/user-attachments/assets/5254ae38-16c4-42fc-9080-066381c3de6b)
### 5000-Episodes
![best_quality_breakout-DQN-episode-Middle](https://github.com/user-attachments/assets/3370253c-92e5-4a57-a0e6-c062859fae80)
### 10000-Episodes
![best_quality_breakout-DQN-episode-Last](https://github.com/user-attachments/assets/74505f09-6a91-4331-9e29-3c8b0ca74d23)

