# Mini Project 2 — Mini Chess AI

- **Due:** 6/20
- **Demo:** 6/21 online

> **Note on conflicts:** Where the original slides disagree with the *Instructor Clarifications* section at the end of this document, the clarifications take precedence. Conflicting points are flagged inline with a ⚠️ marker.

---

## Outline

1. Introduction
2. Chess and Mini Chess
3. State Value Function
4. Minimax
5. Alpha-Beta Pruning
6. How To Design Your AI
7. Package
8. Requirements
9. Grading
10. Submission

---

## Introduction

Design and implement an AI which can play MiniChess.

- Read the current board and output the next move.
- Design a state value function to evaluate the score of the board.
- Determine the next move with a tree search algorithm.

---

## Chess and Mini Chess

### Chess

(Standard chess — shown as reference.)

### Mini Chess

- Has a lot of variance (many variants exist).
- We use **"MinitChess"** in this project.

### Basic rules

- We have 2 players: **white** and **black**. White plays first.
- If the target place of your piece has an opponent's piece, you can take it out (catch the piece). You **cannot** take your own piece.
- If a player can catch the opponent's king on its turn, it wins.
  - So a player can only win on their turn or lose on the opponent's turn.
- If a player makes an illegal move, he/she **loses**.
- Our rule is simplified; any rules that differ from standard chess are marked in red on the slides:
  - There is **no castling**.
  - A pawn can only promote to **Queen** (details in the Piece Movement section).

### Basic rules — draw / loop prevention

If two players have almost the same ability, it is very likely to get a draw and fall into an infinite loop, so these rules are added:

- If the game lasts over **50 steps (25 turns)**, we count the piece value.
- Piece values: **Queen = 20, Bishop = 8, Knight = 7, Rook = 6, Pawn = 2.**
- The player with the higher piece value after 50 steps wins.
- If both sides have the same value, it is a **draw**.
- A Queen promoted from a Pawn counts as a Queen.

### Piece Movement

- **King and Queen** — standard movement.
- **Rook and Bishop** — standard movement.
- **Knight**
  - The knight can "jump over pieces."
  - It cannot be blocked.
- **Pawn**
  - Every turn, a pawn can move forward one step — even on that pawn's first move.
  - Because of the above rule, there is **no En passant**.
  - If the left/right forward square holds an opponent's piece, you can catch it.
  - When a pawn moves to the last row (row 6 for white, row 1 for black), it becomes a Queen (**promotion**).
  - Promotion and moving to the last row happen simultaneously.

---

## State Value Function

- The program should decide which move is better.
- We can pick the move that leads to the board with the highest score.
- Thus, we need a function to evaluate the score of the board — this is the **state value function**.

Breaking it down:

- **State** → the board
- **Value** → how "good" the board is
- **Function** → given a board, output the value

### Super simple example

- Give every piece a score (king = ∞, queen = 100, …).
- `Your pieces – Opponent's pieces = value of the state.`

Some upgrades:

- A piece in a different place can have a different value.

### Keywords for more complicated algorithms

- KP (King-Piece), PP (Piece-Piece), KPPT (King-Piece-Piece-Turn)
- KKPT (King-King-Piece-Turn, with King-Piece-Piece)
- MCTS (Monte Carlo Tree Search)
- AlphaZero
- Leela Chess Zero
- DLShogi
- **NNUE** (Efficiently Updatable Neural Network) — this is the SOTA for Chess and Shogi.
- Stockfish (2022 TCEC 1st place, 2022 CCC 1st place)
- 水匠 (第3回世界将棋AI電竜戦優勝)

### Picking a move with the value function (worked example)

Suppose we have three valid moves: A, B, and C. From the current board, each move leads to a resulting board (After Move A / B / C). We use the value function to pick the next move.

- After evaluating the state values, we get: **A = 20, B = -15, C = 30.**
- We pick **move C** to be our next step, since it leads to the highest value.

---

## Minimax

- In the previous example, we only looked forward one step.
- However, the opponent will try its best to defeat you — a greedy choice is not always the best.
- We should look forward more steps and simulate how the opponent thinks, to make the best choice with the least risk.

How it works:

- The player tries its best to win → picks the move with the **highest** score.
- The opponent tries its best to defeat the player → picks the move with the **lowest** "player's value function" score (i.e., the opponent tends to give the player the worst board).
- The Minimax algorithm is based on this player–opponent interaction.

### Minimax Pseudocode

*Source: https://en.wikipedia.org/wiki/Minimax*

```
function minimax(node, depth, maximizingPlayer) is
    if depth = 0 or node is a terminal node then
        return the heuristic value of node
    if maximizingPlayer then
        value := −∞
        for each child of node do
            value := max(value, minimax(child, depth − 1, FALSE))
        return value
    else (* minimizing player *)
        value := +∞
        for each child of node do
            value := min(value, minimax(child, depth − 1, TRUE))
        return value
```

### Worked example (Minimax tree)

A three-level tree with the current node ("Now") at the top, player nodes, opponent nodes, and leaf nodes labeled A–K. The procedure:

1. Evaluate scores at the leaves (leaf values revealed one at a time: 5, 6, 5, 7, 4, 5, 3, …).
2. At **player** nodes, pick the **largest** child score.
3. At **opponent** nodes, pick the **smallest** child score.
4. Propagating values up the tree, **Move A has the largest score (6)**.
5. The player picks **move A** to be the next move.

---

## Alpha-Beta Pruning

- With Minimax, we can simulate the opponent's moves and pick a move with minimum risk and maximum value.
- Looking forward more steps may improve the policy.
- However, the size of the search tree can drastically increase as search depth increases.
- Since we only have limited time, to increase search depth we must optimize the search process.
- Many branches in the Minimax process are not related to the result; we can "prune" these branches to improve efficiency.
- **Alpha-Beta Pruning** is the improved version of Minimax that eliminates some unnecessary branches.

Definitions:

- **Alpha (α):** the maximum score that the player is assured of in the current search process.
- **Beta (β):** the minimum score that the opponent is assured of in the current search process.

Pruning conditions:

- If **α ≥ β** on a player node, we can stop searching this branch.
  - The player will return a value ≥ β on this branch, but the opponent already has a better choice (β).
  - No matter what value is later discovered on this branch, the opponent will not pick it, so we can prune it without affecting the result.
- Likewise, we can stop searching if **β ≤ α** on an opponent node.

### Alpha-Beta Pruning Pseudocode

*Source: https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning*

```
function alphabeta(node, depth, α, β, maximizingPlayer) is
    if depth = 0 or node is a terminal node then
        return the heuristic value of node
    if maximizingPlayer then
        value := −∞
        for each child of node do
            value := max(value, alphabeta(child, depth − 1, α, β, FALSE))
            α := max(α, value)
            if α ≥ β then
                break (* β cutoff *)
        return value
    else
        value := +∞
        for each child of node do
            value := min(value, alphabeta(child, depth − 1, α, β, TRUE))
            β := min(β, value)
            if β ≤ α then
                break (* α cutoff *)
        return value
```

### Worked example (Alpha-Beta tree)

Same search tree as the Minimax example, with each node tracking `α` and `β` (all initialized to α: −INF, β: INF). Walking through:

- Evaluate leaf scores; **update alpha** at player nodes and **update beta** at opponent nodes as values come in.
- Alpha/beta values are propagated down to child nodes.
- At one player node, after updates we reach **α: 7, β: 6**, i.e. **α ≥ β** in a player node → **stop searching** that branch.
- The opponent then picks the smallest score.

Result:

- We use the same search tree as Minimax.
- By pruning, we eliminate branches **I** and **J**.
- We still get the same result on branch **A**.
- Alpha-Beta Pruning effectively speeds up the process while maintaining the same result as Minimax.

---

## How To Design Your AI

- You can refer to `random.cpp` / `random.hpp` in the `src/policy` folder.
- Design your state value function in `state.cpp` to evaluate the board.
- Implement the Minimax method and use your value function in the search process.
- Run Minimax and decide which move to output.

---

## Package

You don't need to implement everything yourself. Some useful utilities are provided so you can focus on the state value function and the tree search algorithm.

You will get:

- A **game runner**.
- A **State class**, including:
  - A native method to get all legal actions.
  - A method to generate a new state based on an action and a state.
- An **example player** (with a random-choice policy).

You need to implement the **state value function** yourself.

### Package — How to run it

Besides the source files, you also get some additional files:

- **Makefile** — helps you compile the project.
- **.gitignore** — if you push your code to GitHub, this helps you ignore some files.

You can modify your Makefile, but you must make sure it can compile in the TAs' environment. With the Makefile and `make` utilities (more details in the environment-setup document), you can compile your code with `make`.

After running `make all` and compiling successfully, you can run the game with:

```bash
$ python gui/main.py
pygame 2.6.1 (SDL 2.28.4, Python 3.13.13)
Hello from the pygame community. https://www.pygame.org/contribute.html
[   0.525] INFO  [Main] Main loop started
```

And it will start running! (A Minichess GUI window opens — a 5-column board with White-to-move, a "Free Play" panel, step/time indicators, and New / Pause / Reset / Analyze / Undo / Settings buttons.)

### Package — State

Now you should start your project. You can check the State class first:

- `next_state` can generate a new state based on a move.
- `get_legal_actions` will generate all legal actions of this state and store them in `legal_actions`.
- `evaluate` is the state value function you need to implement.

```cpp
class State : public BaseState {
public:
    Board board;
    int score = 0;
    mutable uint64_t zobrist_hash = 0;
    mutable bool zobrist_valid = false;

    State(){}
    State(int player){
        this->player = player;
    }
    State(Board board): board(board){}
    State(Board board, int player): board(board){
        this->player = player;
    }

    int evaluate(
        bool use_kp_eval = true,
        bool use_mobility = true,
        const GameHistory* history = nullptr
    ) override;
    State* next_state(const Move& move) override;
    void get_legal_actions() override;
    void get_legal_actions_naive();
    void get_legal_actions_bitboard();
    std::string encode_output() const override;
    std::string encode_state();
```

### Package — Policy

In this project, you should implement your own policies. You will get a random policy as an example:

```cpp
SearchResult Random::search(
    State *state,
    int depth,
    GameHistory& history,
    SearchContext& ctx
){
    (void)history;
    ctx.reset();
    SearchResult result;
    result.depth = 1;

    if(!state->legal_actions.size()){
        state->get_legal_actions();
    }

    auto actions = state->legal_actions;
    if(actions.empty()){
        result.best_move = Move();
        return result;
    }

    int idx = (rand() + depth) % actions.size();

    result.best_move = actions[idx];
    result.score = 0;
    result.nodes = 1;
    result.pv = {result.best_move};
    return result;
}
```

---

## Requirements

### Requirements — code

In this project you should do these things:

1. Design your own state value function.
2. Implement **Minimax, Alpha-Beta, PVS, and quiescence**.
3. Utilize your algorithm and state value function to make a strong AI.

If you are not satisfied with the basic algorithms, you can try more advanced methods (MCTS, NNUE). However, make sure you can explain how your algorithms work during the demo. If you cannot complete the harder algorithms, implementing the basic ones still earns you some score.

Constraints:

- You will lose immediately if your program outputs an invalid move.
- ⚠️ The slide states a per-move time limit of **10 seconds** and a memory limit of **4 GB**. **Per the Instructor Clarifications, the enforced per-move time limit during matches is 2 seconds** (see below). The 4 GB memory limit is unchanged.
- You can keep outputting moves within the time limit; only the **last** move is used by the game runner.

You **cannot** use these in your code:

- Third-party libraries (only the standard library is acceptable).
- Inline ASM.
- Multi-thread / multi-process.
- Vectorized operations (e.g., AVX).

### Requirements — structure

- Please use **C++** and write your program with the structure in the provided folder.
- Make sure to make a copy of the policy you want to submit and rename it to **`submission.cpp`**.

Your program will be compiled in a GNU/Linux environment by:

```bash
make
g++ --std=c++2a -Wall -Wextra -Wpedantic -g -O3 -march=native -Isrc/games/minichess -Isrc/state -Isrc -o build/minichess-ubgi src/games/minichess/state.cpp src/policy/minimax.cpp src/policy/random.cpp src/ubgi/ubgi.cpp
g++ --std=c++2a -Wall -Wextra -Wpedantic -g -O3 -march=native -Isrc/games/minichess -Isrc/state -Isrc -o build/minichess-benchmark src/games/minichess/state.cpp src/policy/minimax.cpp src/policy/random.cpp src/benchmark.cpp
g++ --std=c++2a -Wall -Wextra -Wpedantic -g -O3 -march=native -Isrc/games/minichess -Isrc/state -Isrc -o unittest/build/state_test src/games/minichess/state.cpp src/policy/minimax.cpp src/policy/random.cpp unittest/state_test.cpp
```

Please make sure your program compiles by the command above with no errors. (If you don't change the makefile, just use `make` and check for errors.)

### Requirements — Report and Demo

- You should write a report explaining how you designed your AI.
- If you implemented NNUE or MCTS, briefly describe in the report how it was implemented, with code explanations if possible.
- The report is **not directly graded**, but is your **only** available reference during the TA demo (you cannot refer to your code in the demo).
- You **must** attend the demo and answer the TA's questions.
- The demo date and method will be announced soon.

---

## Grading

The project accounts for **15 points** of your total grade:

- Design of your state value function → **+1 point**
- Implement tree search (Minimax) → **+2 points**
- Implement Alpha-Beta Pruning → **+1 point**
- Beat every baseline → **+8 points**
  - 1.5 points for each of the 4 baselines; +2 points if you beat all baselines with both white and black.
- Implement PVS → **+2 points**
- Quiescence → **+1 point**

*(Slide note: "TODOs in hackathon.")*

### Grading — baselines

There are 4 baselines:

1. Weak MiniMax
2. Strong MiniMax
3. AlphaBeta
4. PVS

Your program will play each baseline as both white and black.

- If you get **1 win + 1 draw (or 2 wins)**, you advance to the next baseline and earn the score.
- If you beat all baselines with **8 wins**, you earn the final +2 points.

### Grading — Bonus

- **(Bonus) Use version control software → +1 point.** You must push your code to GitHub as the submission anyway; but if you also use GitHub + git as version control with at least **3 commits**, you earn this point.
- **(Bonus) Class ranking → at most +2 points.** You can join the class ranking if you beat all baselines (8 wins). Your AI plays against classmates' AIs and gains bonus score according to your ranking.
- **(Bonus) Beat the visible boss → +1 point.**
- **(Bonus) Beat the secret boss → +2 points.**

### Grading — Plagiarism

- Code will be compared across both classes and previous years to detect plagiarism.
- If plagiarism is found, **all** involved submissions receive a score of 0, regardless of who copied from whom.

---

## Submission

- Submit the report to **Mini Project 2 繳交區** on eeclass.
  - File name: `<student_id>_project2.pdf`
  - Example: `114000000_project2.pdf`
- **Deadline: 6/20** (both report and Google form).
- Late submissions within 2 hours receive a **40% deduction** (六折).
- Late submissions more than 2 hours receive a score of **0**.

---

## Happy Coding!

---

# Instructor Clarifications (Unified Response to Common Questions)

> The following clarifications were issued in response to many students asking similar questions. **Where these conflict with the slides above, these clarifications take precedence.**

### What should the depth be set to?

All submitted code will uniformly be evaluated with **unlimited depth** and a **two-second time limit per move** when competing against the Baseline, Boss, or other students' algorithms.

- If you are testing using the **GUI**, you can set the depth to **0**; when the depth is 0, it will automatically calculate using unlimited depth.
- If you are testing using the **CLI**, there is no need to manually set the depth; the program will automatically default to calculating with unlimited depth.

### Which files can be modified?

Any files inside the **`src/policy`** folder can be modified, but please be careful not to break the program's compilation. Please do **not** make any modifications to files outside of `src/policy`.

### My code wins in the GUI but loses in the CLI. Which one should be the standard?

The **CLI is the standard**.

### Match Rules and Time Limits

During matches, a time limit of **2 seconds per move** is enforced with **no depth limit** (although the default maximum depth in the code is 100, in practice it is highly unlikely to reach this depth within two seconds). Additionally, if your program fails to generate a legal move within the two-second limit, it automatically results in a **loss**.

### How Timeouts are Evaluated

The program running the match (GUI or CLI) will request the engine to output what it currently considers the best move exactly when the time limit ends. If the engine cannot return a valid, rule-compliant move at that moment, it is declared a **loss by timeout**.

For the relevant implementation details, refer to the Python files for the GUI or CLI, as well as `ubgi.c`, `ubgi.h`, and other files responsible for the engine's communication protocol.

### Winning Criteria

"Winning" is defined as achieving **at least one win and one draw**.

### Visible Boss

The "visible boss" is indeed the **`boss-ubgi.exe`** file that was provided to you.

### Testing
You can use the following code to test your program.
```
python -m cli.cli --black Student --white TA --time 2000 --games 2
```