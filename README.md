# __Snake-Game with Reinforcement Learning__


## __Step by Step__
### Step 1: Create a basic snake-game:
---
- Using Pygame package.
- Loop game through `play_step()`.
- Using arrows to navigate.
- Return score when game over.

### Step 2: Implement software agent:

<img src="img/readme/step2.png" alt="Step 2" width="350"/>

#### __Variables for training:__
- Reward:

    | | |
    | --- | --- |
    | Eat food | +10 |
    | Game over | -10 |
    | Else | 0 |

- Action:

    | | |
    | --- | --- |
    | [1, 0, 0] | straight |
    | [0, 1, 0] | right turn |
    | [0, 0, 1] | left turn ||

- State (11 values):
    ```javascript
    [
        danger straight, danger left, danger right,

        direction left, direction right,
        direction up, direction down,

        food left, food right,
        food up, food down
    ]
    ```
    - Example:

        <img src="img/readme/state-example.png" alt="Example" width="200"/>

        ```javascript
        [
            0, 0, 0
            0, 1, 0, 0
            0, 1, 0, 1
        ]
        ```

### Step 3: Implement model with PyTorch
<img src="img/readme/step3.png" alt="Step 3" width="200"/>

1. Model:
    - Using neural network.
    - Input get 11 different boolean values (0 or 1) in State.
    - Output return 3 raw numbers, choose the maximum value. 
    
    <img src="img/readme/model.png" alt="Step 3" width="400"/>

2. Training model
    - Using 2 types of training:
        1. __Temporal Difference Learning__: Estimate the rewards at each step.
        - ```train_short_memory``` uses for each play_step().
        2. __Monte Carlo Learning__: Collecting the rewards at the end of each game and then calculating the reward.
        - ```train_long_memory``` uses for each game over.
    - Using Deep Q Learning
    - __Q Value__: Quality of action
    
        0. Init Q value (= init model).
        1. Choose action (model.predict(state)) __OR__ random move.
        2. Perform action.
        3. Measure reward value.
        4. Update Q value + train model.
        5. Repeat step 1.

    - Bellman Equation:
    ```
        NewQ(s, a) = Q(s, a) + α[R(s, a) + γmaxQ'(s', a') - Q(s, a)]
    ```

    - Q update rules simplified:
        - Q = model.predict(state_0)
        - Q_new = R + γ * max(Q(state_1))

    - Loss function:
        - Mean squared error.
        - loss = (Q_new - Q)²
        - Using for optimization.

### Step 4: Learning strategies and Improvement:
---
1. Exploration/Exploitation tradeoff:
    - Exploration is finding more information about the environment.
    - Exploitation is exploiting known information to maximize the reward.
