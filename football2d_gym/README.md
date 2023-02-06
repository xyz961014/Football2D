# Documentation for Football2D Env

### SelfTraining

#### SelfTraining-v0

##### States

States include ball position, ball speed, player position, player speed. Each of them is a 2-D numpy array. The center of the field is position (0, 0). The size of the field is 1050 $\times$ 680. 

```python
states = {
    "ball_position": np.array([ball.position.x, ball.position.y]),
    "ball_speed": np.array([ball.speed.x, ball.speed.y]),
    "player_position": np.array([player.position.x, player.position.y]),
    "player_speed": np.array([player.speed.x, player.speed.y])
}
```



##### Actions

Continuous ($2  \times 2$) shape action. The first vector is player acceleration. The second vector is ball kicking momentum. The ball will only be kicked when it is within the kicking range (3 uniformly in all directions in this env) of the player. 

| Action              | Form                             |
| ------------------- | -------------------------------- |
| Player acceleration | [acceleration.x, acceleration.y] |
| Kicking momentum    | [momentum.x, momentum.y]         |



##### Rewards

The reward is always zero unless a goal is scored. If a goal scored in the right goal, the reward is 1. If a goal scored in the left goal, the reward is -1. 

| Situation              | Reward |
| ---------------------- | ------ |
| Goal in the right goal | +1     |
| Goal in the left goal  | -1     |
| Otherwise              | 0      |

##### Features

- The ball will be bounced back if it touches the border. The bounce factor is 0.5.

- The ball is decelerated by two resistance components:

  - fixed deceleration 15
  - force proportional to $v^2$

- The player cannot go out of the field, and will stop if reach the border

- The player is decelerated by a fixed deceleration.

- The player has a max acceleration and a max speed.

- The player can run up to kick the ball to get an extra momentum (proportional to projection of player speed on momentum direction).

- The ball is only kicked when it is in the kicking range of the player (distance between ball and player < 5).

- If a goal is scored, the game terminated.

- Time limit is 120s. If time limit is reached, the game is truncated.

- No collisions, the ball can go through the player if the player doesn't kick it.

- All actions are accurate (we consider add noise to actions in next version).

- The env will return an info to help training:

  ```python
  info = {
      "distance_to_ball": Distance(player, ball),
      "distance_to_goal": Distance(ball, right_goal)
  }
  ```

#### SelfTraining-v1

##### New Features
