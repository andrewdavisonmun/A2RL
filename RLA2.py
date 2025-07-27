#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 14:29:12 2025

@author: andrewdavison
"""

import numpy as np
import matplotlib.pyplot as plt
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Grid and MDP setup
n_rows, n_cols = 5, 5                   # Grid dimensions
n_states = n_rows * n_cols              # Total number of states
gamma = 0.95                            # Discount factor
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Actions: up, down, left, right
action_prob = 0.25                      # Uniform random policy

# Special states with rewards and transitions
blue = 1
green = 4
red = 17
yellow = 24

# Transition probability matrix and reward vector
P = np.zeros((n_states, n_states))     # Transition matrix: P[s, s']
R = np.zeros(n_states)                 # Reward vector: R[s]

# Helper functions to convert between state index and (row, col)
def state_to_coords(s):
    return divmod(s, n_cols)

def coords_to_state(r, c):
    return r * n_cols + c

# Populate P and R
for s in range(n_states):
    if s == blue:
        R[s] = 5
        P[s, red] = 1
        continue
    elif s == green:
        R[s] = 2.5
        P[s, red] = 0.5
        P[s, yellow] = 0.5
        continue

    r, c = state_to_coords(s)
    for dr, dc in actions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < n_rows and 0 <= nc < n_cols:
            s_prime = coords_to_state(nr, nc)
            P[s, s_prime] += action_prob
        else:
            P[s, s] += action_prob
            R[s] += action_prob * -0.5  # Penalty for hitting wall

# Solve Bellman equation directly for exact value function
I = np.eye(n_states)
V_exact = np.linalg.solve(I - gamma * P, R)

# Iterative policy evaluation using uniform random policy
V_iter = np.zeros(n_states)
threshold = 1e-4
delta = float('inf')

while delta > threshold:
    delta = 0
    V_new = np.zeros_like(V_iter)
    
    for s in range(n_states):
        v = 0
        if s == blue:
            v = 5 + gamma * V_iter[red]
        elif s == green:
            v = 2.5 + gamma * 0.5 * (V_iter[red] + V_iter[yellow])
        else:
            r, c = state_to_coords(s)
            for dr, dc in actions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n_rows and 0 <= nc < n_cols:
                    s_prime = coords_to_state(nr, nc)
                    reward = 0
                else:
                    s_prime = s
                    reward = -0.5
                v += action_prob * (reward + gamma * V_iter[s_prime])
        V_new[s] = v
        delta = max(delta, abs(V_iter[s] - v))
    
    V_iter = V_new.copy()

# Display results from exact and iterative evaluation
print("Value function (Exact Solution):")
print(V_exact.reshape(5, 5).round(2))

print("\nValue function (Iterative Policy Evaluation):")
print(V_iter.reshape(5, 5).round(2))

# Reshape for plotting
V_exact = V_exact.reshape(5, 5).round(2)
V_iter = V_iter.reshape(5, 5).round(2)

# Plot exact value function
plt.figure(figsize=(6, 6))
plt.imshow(V_exact, cmap='coolwarm', interpolation='nearest')
plt.colorbar(label="Value")
plt.grid(False)
plt.xticks(np.arange(5), labels=np.arange(5))
plt.yticks(np.arange(5), labels=np.arange(5))
for i in range(5):
    for j in range(5):
        plt.text(j, i, f'{V_exact[i, j]:.2f}', ha='center', va='center', color='black')
plt.title("Exact Value Function (Bellman Equation)")
plt.xlabel("Column")
plt.ylabel("Row")
plt.show()

# Plot iterative value function
plt.figure(figsize=(6, 6))
plt.imshow(V_iter, cmap='coolwarm', interpolation='nearest')
plt.colorbar(label="Value")
plt.grid(False)
plt.xticks(np.arange(5), labels=np.arange(5))
plt.yticks(np.arange(5), labels=np.arange(5))
for i in range(5):
    for j in range(5):
        plt.text(j, i, f'{V_iter[i, j]:.2f}', ha='center', va='center', color='black')
plt.title("Iterative Value Function (Policy Evaluation)")
plt.xlabel("Column")
plt.ylabel("Row")
plt.show()

# Value Iteration to find optimal policy
V_opt = np.zeros(n_states)
policy_opt = np.zeros(n_states, dtype=int)

delta = float('inf')
threshold = 1e-4

while delta > threshold:
    delta = 0
    V_new = np.zeros_like(V_opt)
    
    for s in range(n_states):
        if s == blue:
            best_value = 5 + gamma * V_opt[red]
        elif s == green:
            best_value = 2.5 + gamma * 0.5 * (V_opt[red] + V_opt[yellow])
        else:
            r, c = state_to_coords(s)
            values = []
            for a_idx, (dr, dc) in enumerate(actions):
                nr, nc = r + dr, c + dc
                if 0 <= nr < n_rows and 0 <= nc < n_cols:
                    s_prime = coords_to_state(nr, nc)
                    reward = 0
                else:
                    s_prime = s
                    reward = -0.5
                values.append(reward + gamma * V_opt[s_prime])
            best_value = max(values)
            policy_opt[s] = np.argmax(values)

        delta = max(delta, abs(V_opt[s] - best_value))
        V_new[s] = best_value
    
    V_opt = V_new.copy()

# Plot optimal value function
V_opt_grid = V_opt.reshape(5, 5).round(2)
policy_opt_grid = policy_opt.reshape(5, 5)

plt.figure(figsize=(6, 6))
plt.imshow(V_opt_grid, cmap='viridis', interpolation='nearest')
plt.colorbar(label="Value")
plt.xticks(np.arange(5))
plt.yticks(np.arange(5))
plt.title("Optimal Value Function (Value Iteration)")
for i in range(5):
    for j in range(5):
        plt.text(j, i, f'{V_opt_grid[i, j]:.2f}', ha='center', va='center', color='white')
plt.grid(False)
plt.show()

# Plot optimal policy (value iteration)
arrow_map = {0: '↑', 1: '↓', 2: '←', 3: '→'}

plt.figure(figsize=(6, 6))
plt.imshow(np.zeros((5, 5)), cmap='gray', alpha=0)
for i in range(5):
    for j in range(5):
        s = coords_to_state(i, j)
        if s == blue:
            txt = 'B'
        elif s == green:
            txt = 'G'
        elif s == red:
            txt = 'R'
        elif s == yellow:
            txt = 'Y'
        else:
            txt = arrow_map[policy_opt[s]]
        plt.text(j, i, txt, ha='center', va='center', fontsize=14)
plt.xticks(np.arange(5))
plt.yticks(np.arange(5))
plt.grid(False)
plt.title("Optimal Policy (Value Iteration)")
plt.show()

# Policy Iteration
policy = np.random.choice([0, 1, 2, 3], size=n_states)
V_pi = np.zeros(n_states)

is_policy_stable = False

while not is_policy_stable:
    # Policy Evaluation
    while True:
        delta = 0
        V_new = np.zeros_like(V_pi)
        for s in range(n_states):
            a = policy[s]
            dr, dc = actions[a]
            r, c = state_to_coords(s)

            if s == blue:
                V_new[s] = 5 + gamma * V_pi[red]
            elif s == green:
                V_new[s] = 2.5 + gamma * 0.5 * (V_pi[red] + V_pi[yellow])
            else:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n_rows and 0 <= nc < n_cols:
                    s_prime = coords_to_state(nr, nc)
                    reward = 0
                else:
                    s_prime = s
                    reward = -0.5
                V_new[s] = reward + gamma * V_pi[s_prime]
            delta = max(delta, abs(V_new[s] - V_pi[s]))

        V_pi = V_new.copy()
        if delta < 1e-4:
            break

    # Policy Improvement
    is_policy_stable = True
    for s in range(n_states):
        old_action = policy[s]
        r, c = state_to_coords(s)
        action_values = []

        for a_idx, (dr, dc) in enumerate(actions):
            if s == blue:
                value = 5 + gamma * V_pi[red]
            elif s == green:
                value = 2.5 + gamma * 0.5 * (V_pi[red] + V_pi[yellow])
            else:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n_rows and 0 <= nc < n_cols:
                    s_prime = coords_to_state(nr, nc)
                    reward = 0
                else:
                    s_prime = s
                    reward = -0.5
                value = reward + gamma * V_pi[s_prime]
            action_values.append(value)

        best_action = np.argmax(action_values)
        policy[s] = best_action
        if best_action != old_action:
            is_policy_stable = False

# Plot policy iteration results
V_pi_grid = V_pi.reshape(5, 5).round(2)
policy_pi_grid = policy.reshape(5, 5)

plt.figure(figsize=(6, 6))
plt.imshow(V_pi_grid, cmap='YlGn', interpolation='nearest')
plt.colorbar(label="Value")
plt.title("Optimal Value Function (Policy Iteration)")
plt.xticks(np.arange(5))
plt.yticks(np.arange(5))
for i in range(5):
    for j in range(5):
        plt.text(j, i, f'{V_pi_grid[i, j]:.2f}', ha='center', va='center', color='black')
plt.grid(False)
plt.show()

# Plot policy from policy iteration
plt.figure(figsize=(6, 6))
plt.imshow(np.zeros((5, 5)), cmap='gray', alpha=0)
for i in range(5):
    for j in range(5):
        s = coords_to_state(i, j)
        if s == blue:
            txt = 'B'
        elif s == green:
            txt = 'G'
        elif s == red:
            txt = 'R'
        elif s == yellow:
            txt = 'Y'
        else:
            txt = arrow_map[policy[s]]
        plt.text(j, i, txt, ha='center', va='center', fontsize=14)
plt.xticks(np.arange(5))
plt.yticks(np.arange(5))
plt.grid(False)
plt.title("Optimal Policy (Policy Iteration)")
plt.show()

# Summary
print("\n--- Value Function Comparison ---")
print("Exact solution (random policy):")
print(V_exact.reshape(5, 5).round(2))
print("\nIterative policy evaluation (random policy):")
print(V_iter.reshape(5, 5).round(2))
print("\nValue Iteration (optimal policy):")
print(V_opt.reshape(5, 5).round(2))
print("\nPolicy Iteration (optimal policy):")
print(V_pi.reshape(5, 5).round(2))

print("\n--- Optimal Policies ---")
print("Policy from Value Iteration:")
print(policy_opt.reshape(5, 5))
print("\nPolicy from Policy Iteration:")
print(policy.reshape(5, 5))

# Print policy as arrows
def print_policy(policy_grid):
    for row in policy_grid:
        print(' '.join(arrow_map[a] if a in arrow_map else 'X' for a in row))

print("\nPolicy from Value Iteration:")
print_policy(policy_opt.reshape(5, 5))
print("\nPolicy from Policy Iteration:")
print_policy(policy.reshape(5, 5))

#Part 2

# Define terminal states (absorbing states that end an episode)
terminal_states = [coords_to_state(r, c) for r, c in [(2, 0), (2, 4), (4, 0)]]

# Step function to simulate environment dynamics
def step(state, action):
    r, c = state_to_coords(state)
    dr, dc = actions[action]
    nr, nc = r + dr, c + dc
    
    # If action goes out of bounds, stay in place with penalty
    if not (0 <= nr < n_rows and 0 <= nc < n_cols):
        next_state = state
        reward = -0.5
        done = False
    else:
        next_state = coords_to_state(nr, nc)
        if next_state in terminal_states:
            reward = 0
            done = True
        else:
            reward = -0.2  # Slight negative reward for each step
            done = False
    return next_state, reward, done

def mc_exploring_starts(num_episodes=10000, gamma=0.95, max_steps=100):
    Q = np.zeros((n_states, len(actions)))  # Action-value estimates
    returns_count = np.zeros((n_states, len(actions)))  # Visit counts
    policy = np.ones((n_states, len(actions))) / len(actions)  # Uniform policy

    for episode_num in range(num_episodes):
        # Start from a random non-terminal state and random action (exploring starts)
        while True:
            state = random.randint(0, n_states - 1)
            if state not in terminal_states:
                break
        action = random.choice(range(len(actions)))

        episode = []
        done = False
        cur_state, cur_action = state, action
        steps = 0

        # Generate episode using the current policy
        while not done and steps < max_steps:
            next_state, reward, done = step(cur_state, cur_action)
            episode.append((cur_state, cur_action, reward))
            cur_state = next_state
            steps += 1
            if not done:
                cur_action = np.random.choice(range(len(actions)), p=policy[cur_state])

        # Monte Carlo return calculation (first-visit)
        G = 0
        visited = set()
        for s, a, r in reversed(episode):
            G = gamma * G + r
            if (s, a) not in visited:
                returns_count[s, a] += 1
                Q[s, a] += (G - Q[s, a]) / returns_count[s, a]
                visited.add((s, a))

                # Improve policy greedily
                best_action = np.argmax(Q[s])
                policy[s] = np.eye(len(actions))[best_action]

        if episode_num % 1000 == 0:
            print(f"[Exploring Starts] Episode {episode_num}")

    return policy, Q

def mc_e_soft(num_episodes=10000, gamma=0.95, epsilon=0.1, max_steps=100):
    Q = np.zeros((n_states, len(actions)))
    returns_count = np.zeros((n_states, len(actions)))
    policy = np.ones((n_states, len(actions))) * (epsilon / len(actions))  # ε-soft init

    # Initialize policy to be slightly biased toward best action (currently 0s)
    for s in range(n_states):
        best_action = np.argmax(Q[s])
        policy[s, best_action] += 1 - epsilon

    for episode_num in range(num_episodes):
        state = random.randint(0, n_states - 1)
        if state in terminal_states:
            continue

        episode = []
        done = False
        cur_state = state
        steps = 0

        # Generate episode using ε-soft policy
        while not done and steps < max_steps:
            action = np.random.choice(range(len(actions)), p=policy[cur_state])
            next_state, reward, done = step(cur_state, action)
            episode.append((cur_state, action, reward))
            cur_state = next_state
            steps += 1

        # Monte Carlo return calculation
        G = 0
        visited = set()
        for s, a, r in reversed(episode):
            G = gamma * G + r
            if (s, a) not in visited:
                returns_count[s, a] += 1
                Q[s, a] += (G - Q[s, a]) / returns_count[s, a]
                visited.add((s, a))

                # Improve policy: ε-soft greedy
                best_action = np.argmax(Q[s])
                policy[s] = np.ones(len(actions)) * (epsilon / len(actions))
                policy[s, best_action] += 1 - epsilon

        if episode_num % 1000 == 0:
            print(f"[Epsilon-Soft] Episode {episode_num}")

    return policy, Q

def mc_off_policy_importance_sampling(num_episodes=10000, gamma=0.95):
    Q = np.zeros((n_states, len(actions)))
    C = np.zeros((n_states, len(actions)))  # Cumulative weights
    target_policy = np.zeros((n_states, len(actions)))

    # Initialize target policy arbitrarily (e.g., always 'up')
    for s in range(n_states):
        target_policy[s] = np.eye(len(actions))[0]

    for episode_num in range(num_episodes):
        # Start from a random non-terminal state
        while True:
            state = random.randint(0, n_states - 1)
            if state not in terminal_states:
                break

        episode = []
        cur_state = state
        done = False
        max_steps = 100
        steps = 0

        # Generate episode using random behavior policy (uniform)
        while not done and steps < max_steps:
            action = random.choice(range(len(actions)))
            next_state, reward, done = step(cur_state, action)
            episode.append((cur_state, action, reward))
            cur_state = next_state
            steps += 1

        # Importance sampling update
        G = 0
        W = 1.0
        for s, a, r in reversed(episode):
            G = gamma * G + r
            C[s, a] += W
            Q[s, a] += (W / C[s, a]) * (G - Q[s, a])

            best_action = np.argmax(Q[s])
            target_policy[s] = np.eye(len(actions))[best_action]

            # Stop updating if the taken action doesn't match target policy
            if a != best_action:
                break
            W *= 1 / 0.25  # Behavior policy is uniform (prob = 0.25)

    return target_policy, Q

action_names = ['↑', '↓', '←', '→']

def plot_policy(policy, title):
    grid = np.zeros((n_rows, n_cols), dtype='<U2')

    for s in range(n_states):
        r, c = state_to_coords(s)
        if s in terminal_states:
            grid[r, c] = '■'
        elif s == blue:
            grid[r, c] = 'B'
        elif s == green:
            grid[r, c] = 'G'
        elif s == red:
            grid[r, c] = 'R'
        elif s == yellow:
            grid[r, c] = 'Y'
        else:
            a = np.argmax(policy[s])
            grid[r, c] = action_names[a]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(title)
    ax.axis('off')

    for i in range(n_rows):
        for j in range(n_cols):
            ax.text(j, i, grid[i, j], ha='center', va='center', fontsize=14)

    ax.imshow(np.zeros((n_rows, n_cols)), cmap='gray', alpha=0)
    plt.show()
    
# Monte Carlo: Exploring Starts
policy_es, Q_es = mc_exploring_starts()
plot_policy(policy_es, "Policy from MC with Exploring Starts")

# Monte Carlo: ε-soft
policy_eps, Q_eps = mc_e_soft()
plot_policy(policy_eps, "Policy from MC with Epsilon-Soft")

# Monte Carlo: Off-policy with Importance Sampling
policy_is, Q_is = mc_off_policy_importance_sampling()
plot_policy(policy_is, "Policy from Off-Policy MC with Importance Sampling")
