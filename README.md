## Q-Learning + Experience Replay
[![Paper](https://img.shields.io/badge/paper-DQN-red)](https://arxiv.org/abs/1312.5602)
[![Implementation](https://img.shields.io/badge/implementation-PyTorch-blue)](https://pytorch.org/)
<hr style="border: none; border-top: 1px solid #eaecef;">

**Playing Atari with Deep Reinforcement Learning**. NIPS Deep Learning Workshop 2013. [Paper](https://arxiv.org/abs/1312.5602)

*Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, Martin Riedmiller*

This implementation uses a **Convolutional Neural Network** to process preprocessed raw pixel frames. It applies **Q-Learning** to calculate action-value functions for each action given a specific state, and employs a **Fully Connected Network** to approximate the Q-learning function. **Experience Replay** has been implemented to support the learning process.
### Result
<img width="1500" height="500" alt="10000_graph" src="https://github.com/user-attachments/assets/02bc30b3-06ba-409e-9ae3-446e437376ce" />

### 500-Episodes
https://github.com/user-attachments/assets/91c9f606-83f2-442c-b67a-b9ec76fb376a
### 5000-Episodes
https://github.com/user-attachments/assets/374424b4-9db0-4b68-818b-d520a3f14b47
### 10000-Episodes
https://github.com/user-attachments/assets/ca1249df-7905-452f-8fcc-79d91d2d007c


## Q-Learning + Prioritized Experience Replay
[![Paper](https://img.shields.io/badge/paper-DQN-red)](https://arxiv.org/abs/1312.5602)
[![Paper](https://img.shields.io/badge/paper-PER-red)](https://arxiv.org/abs/1312.5602)
[![Implementation](https://img.shields.io/badge/implementation-PyTorch-blue)](https://pytorch.org/)
<hr style="border: none; border-top: 1px solid #eaecef;">

**Playing Atari with Deep Reinforcement Learning**. NIPS Deep Learning Workshop 2013. [Paper](https://arxiv.org/abs/1312.5602)

*Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, Martin Riedmiller*

**Prioritized Experience Replay**. ICLR 2016. [Paper](https://arxiv.org/abs/1511.05952)

*Tom Schaul, John Quan, Ioannis Antonoglou, David Silver*

This implementation adds **Prioritized Experience Replay** to DQN. It replaces the standard **Experience Replay**. Even with half the buffer size, it achieves better performance.
### Result
<img width="1500" height="500" alt="10000_graph" src="https://github.com/user-attachments/assets/64091d53-c34f-42ac-96df-9012f8ce4758" />

### 500-Episodes
https://github.com/user-attachments/assets/8c600f7d-7b91-463a-a748-b202c3b37750
### 5000-Episodes
https://github.com/user-attachments/assets/0861738c-5745-45c5-a2c7-133200208c00
### 10000-Episodes
https://github.com/user-attachments/assets/0eba3a3f-5b04-40e4-b444-a57b36014d8c
