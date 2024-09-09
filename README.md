# Reinforcement Learning for Self-Driving with Carla
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white)

> A Reinforcement Learning project for self-driving cars

## About the Project

A project to train a reinforcement learning model for lane-following driving.
This serves as a foundation for high-level autonomous driving, and through various attempts, we expect to achieve a more advanced implementation of autonomous driving.

## Preview

> Preview of driving.</br>

<div align="center">
  <table>
    <tr align="center">
      <th>Top View</th>
      <th>1st View</th>
      <th>3rd View</th>
    </tr>
    <tr align="center">
      <td><video src="https://github.com/kuper0201/RL_Self_Driving/assets/17348056/aec4244f-75d3-4ccb-b98f-77c416c95398"/></td>
      <td><video src="https://github.com/kuper0201/RL_Self_Driving/assets/17348056/47ed6aa7-58a9-4555-9fd9-533e07aa1019"/></td>
      <td><video src="https://github.com/kuper0201/RL_Self_Driving/assets/17348056/47ed6aa7-58a9-4555-9fd9-533e07aa1019"/></td>
    </tr>
  </table>
</div>

## Requirements

- PyTorch
- Stable Baselines3
- Gymnasium
- Pillow
- Carla(v0.9.15)
- Carla Additional Maps(optional)

## Features

- Train a new RL model for self-driving car.
- Test lane-follow driving with pre-trained model.

## To do

- Troubleshooting a car that won't drive straight and oscillates side to side.
- Interact with other cars, traffic lights.
- Route planning.