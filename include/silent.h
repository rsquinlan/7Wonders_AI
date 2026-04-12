#pragma once

// Set true during MCTS tree expansion / rollouts to suppress game-engine cout.
// Single-threaded — safe because MCTS runs on the same thread as the game loop.
inline bool g_silent = false;

struct SilenceGuard {
    SilenceGuard()  { g_silent = true;  }
    ~SilenceGuard() { g_silent = false; }
};
