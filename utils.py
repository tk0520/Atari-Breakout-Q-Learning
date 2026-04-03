from gymnasium.wrappers import RecordVideo, AtariPreprocessing, FrameStackObservation
import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
import ale_py
import torch
import copy

import settings

def record_stat(episode_rewards, episode_losses):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # --- 1. Reward Plot ---
    ax1.plot(episode_rewards[::10], color='blue', linewidth=1)
    ax1.set_title("Total Rewards (per 10 Episode)")
    ax1.set_xlabel("Per 10 Episodes")
    ax1.set_ylabel("Total Reward")

    # --- 2. Loss Plot ---
    ax2.plot(episode_losses[::10], color='red', linewidth=1)
    ax2.set_title("Average Loss (per 10 Episode)")
    ax2.set_xlabel("Per 10 Episodes")
    ax2.set_ylabel("Loss")
    # ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig(f"./stats/{len(episode_rewards)}_graph.png")
    plt.close()

def record_video(eval_env, eval_agent):
    observation, info = eval_env.reset()
    lives = info["lives"]
    done = False

    print("=" * 100)
    print("Info:", info)
    print("Evaluation Env Start...!")

    observation, reward, terminated, truncated, info = eval_env.step(1)
    print("Observation:", observation.ndim)

    while not done:
        action_tensor = eval_agent.act_test(observation)
        print("Action Tensor:", action_tensor)

        action = action_tensor.argmax().item()
        print("Action:", action)

        observation, reward, terminated, truncated, info = eval_env.step(action)
        done = terminated or truncated

        current_lives = info["lives"]
        if not lives == current_lives:
            observation, reward, terminated, truncated, info = eval_env.step(1)
            lives = current_lives
            print("Lives:", lives)
            continue
            
    print("Evaluation Env End...!")
    print("=" * 100)


def train(env, eval_env, agent):
    step_count = 0
    completed_episodes = 0
    episode_rewards = []
    episode_losses = []
    
    # 각 환경별 누적 보상 추적
    current_rewards = np.zeros(settings.ENVS_NUM)
    temp_losses = []

    observations, infos = env.reset()

    while completed_episodes < settings.NUM_EPISODES:
        # 1. Action 선택
        actions = agent.act(observations) 
        
        # 2. Env Step
        next_observations, rewards, terminated, truncated, infos = env.step(actions)
        
        # 3. 데이터 저장 및 에피소드 관리
        for i in range(settings.ENVS_NUM):
            done = terminated[i] or truncated[i]
            experience_done = terminated[i]
            
            # 개별 경험 저장
            experience = (observations[i].astype(np.uint8), actions[i], np.clip(rewards[i], -1, 1), next_observations[i], experience_done)
            agent.replay_memory.store(experience)
            
            current_rewards[i] += rewards[i]
            step_count += 1 

            if done: 
                completed_episodes += 1
                episode_rewards.append(current_rewards[i])
                
                # Loss 기록
                if len(temp_losses) > 0:
                    avg_loss = sum(temp_losses) / len(temp_losses)
                    episode_losses.append(avg_loss)
                    temp_losses = []
                else:
                    episode_losses.append(0)

                if completed_episodes > 0:
                    # 50 에피소드마다 그래프 저장
                    if completed_episodes % settings.STAT_INTERVAL == 0:
                        record_stat(episode_rewards, episode_losses)
                    
                    # 모델 저장
                    if completed_episodes % settings.MODEL_SAVE_INTERVAL == 0:
                        torch.save(agent.current_model.state_dict(), f"models/DQN_PER_{completed_episodes}.pt")

                    # 비디오 녹화
                    if completed_episodes % settings.VIDEO_INTERVAL == 0:
                        record_video(eval_env, agent)
                    
                    # 로그 출력
                    print(f"Episode {completed_episodes} | Env {i} Reward: {current_rewards[i]} Step: {step_count}")
                    print(f"Epsilon: {agent.epsilon}")

                # 리워드 초기화
                current_rewards[i] = 0

        if len(agent.replay_memory) > settings.INITIAL_MEMORY and step_count % (settings.ENVS_NUM * 1) == 0:
            # samples_indices = agent.replay_memory.get_samples_indices()
            samples_batch, samples_weights, tree_indices, samples_indices = agent.replay_memory.get_samples()
            loss_batch, raw_loss_batch = agent.learner.get_loss_batch(samples_batch, samples_weights)

            # =======================================================
            # samples_batch = agent.replay_memory.get_samples()
            # loss_batch = agent.learner.get_loss_batch(samples_batch)
            temp_losses.append(loss_batch.item())

            agent.optimizer.zero_grad()
            loss_batch.backward()
            agent.optimizer.step()
            # =======================================================

            agent.replay_memory.update_priorities(samples_indices, raw_loss_batch)
            
            if step_count % settings.UPDATE_INTERVAL == 0:
                agent.update_target_model()

        if step_count > settings.EPSILON_DECAY_COUNT:
            agent.epsilon_decay()

        observations = next_observations
    
