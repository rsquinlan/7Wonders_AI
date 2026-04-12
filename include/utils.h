#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include "mcts.h"

using json = nlohmann::json;

// === Embedding layout constants ===
// Card IDs run 1-75 (none=0 excluded); index = GetId()-1
static constexpr int N_CARDS       = 75;
// Wonder IDs run 0-13
static constexpr int N_WONDERS     = 14;
// Resources: wood,ore,clay,stone,loom,glass,papyrus,gear,compass,tablet,coins,shields
static constexpr int N_RESOURCES   = 12;
static constexpr int N_SCORE_COMPS = 6;  // civilian,commercial,guild,military,scientific,wonder

// Per-opponent block: wonder_id(14) + wonder_stage(4) + played(75) + resources(12) + known_hand(75) = 180
static constexpr int OPP_EMBED_SIZE  = N_WONDERS + 4 + N_CARDS + N_RESOURCES + N_CARDS;

// Self block: wonder_id(14)+stage(4)+hand(75)+played(75)+playable(75)+resources(12)+score(6) = 261
static constexpr int SELF_EMBED_SIZE = N_WONDERS + 4 + N_CARDS + N_CARDS + N_CARDS + N_RESOURCES + N_SCORE_COMPS;

// Total for an N-player game: 3 + 7 + SELF_EMBED_SIZE + (N-1)*OPP_EMBED_SIZE
// For 5 players: 3 + 7 + 261 + 4*180 = 991
inline int stateEmbedSize(int numPlayers) {
    return 3 + 7 + SELF_EMBED_SIZE + (numPlayers - 1) * OPP_EMBED_SIZE;
}

// Policy vector: 75 BUILD_STRUCTURE slots + 1 BUILD_WONDER + 1 DISCARD = 77 slots
static constexpr int POLICY_SIZE = N_CARDS + 2;

struct Transition {
    std::string state;           // human-readable JSON
    std::string action;          // card name chosen
    int reward;                  // immediate score delta
    std::string next_state;      // human-readable JSON after action
    int final_score;
    bool win;
    int player_index;
    std::vector<float> state_embedding; // stateEmbedSize(num_players) floats
    std::vector<float> mcts_policy;     // POLICY_SIZE floats, normalized visit counts
};

// Open (or re-open) the log file for a new game. Must be called before logTransition.
void initLogFile(const std::string& path);

// Log a transition tuple to the currently open log file
void logTransition(const Transition& t);

// Append a result summary line ({"type":"result",...}) to the current log file
void logResult(const std::vector<int>& final_scores, int max_score);

// Serialize the game state for a given player as a human-readable JSON string
std::string serializeGameState(const DMAG::Game& game, int playerIndex);

// Encode the game state as a fixed-size float embedding vector
std::vector<float> encodeGameState(const DMAG::Game& game, int playerIndex);

// Returns true if playerIndex can see opponentIndex's hand (drafting visibility rule)
bool isHandKnown(const DMAG::Game& game, int playerIndex, int opponentIndex);

// Get the known hand of an opponent as seen by the current player (card names)
std::vector<std::string> getKnownHand(const DMAG::Game& game, int playerIndex, int opponentIndex);

// Print detailed stats for a player
void PrintPlayerStats(DMAG::Player* player);
