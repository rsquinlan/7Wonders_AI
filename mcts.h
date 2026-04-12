#ifndef MCTS_H
#define MCTS_H

#include <memory>
#include <vector>
#include <functional>
#include <cmath>
#include <limits>
#include <algorithm>
#include <stdexcept>

#include "game.h"
#include "card.h"
#include "node.h"

// Rollout policy: given a game state and player index, return the card to play.
using RolloutPolicy = std::function<DMAG::Card(const std::shared_ptr<DMAG::Game>&, int)>;

// AlphaZero eval function: returns (policy vector, value) for a state.
// policy: POLICY_SIZE floats (one per card, softmax). value: win probability [0,1].
using EvalFn = std::function<std::pair<std::vector<float>, float>(const DMAG::Game&, int)>;

class MCTS {
private:
    std::shared_ptr<Node> root;  // Root node of the tree
    int totalPlayers;            // Total number of players in the game
    int currentPlayer;           // Current player for whom MCTS is being executed
    double explorationConstant;  // Exploration constant for UCB1 formula
    RolloutPolicy rolloutPolicy; // Fallback rollout policy (used only when evalFn is null)
    EvalFn evalFn;               // AlphaZero: NN policy+value evaluation. When set, replaces rollout.

    // Selection step: Select the most promising child node based on UCB1 value
    std::shared_ptr<Node> select(std::shared_ptr<Node> node);

    // Expansion step: Expand a node by generating all possible joint actions
    std::shared_ptr<Node> expand(std::shared_ptr<Node> node);

    // Backpropagation step: Backpropagate the simulation result up the tree
    void backpropagate(std::shared_ptr<Node> node, double reward);

    // Helper function to print the tree recursively
    void printTreeRecursive(const Node* node, int depth) const;

    double simulate(std::shared_ptr<DMAG::Game> game, double explorationChance);
    void setPriorsFromPolicy(std::shared_ptr<Node> node);

    double evaluateMoveHeuristic(const std::shared_ptr<DMAG::Game>& game, int playerIndex, const DMAG::Card& card);

public:
    // Constructor for initializing MCTS with the game state and player information.
    // Pass a RolloutPolicy to replace the heuristic in simulation (e.g. NN-based policy).
    MCTS(const DMAG::Game& initialState, int totalPlayers, int currentPlayer,
         double explorationConstant = std::sqrt(2),
         RolloutPolicy rolloutPolicy = nullptr,
         EvalFn evalFn = nullptr);

    // Destructor to clean up resources
    ~MCTS();

    // Perform MCTS search and return the best move for the current player
    std::shared_ptr<Node> search(int iterations, double explorationChance);

    // Print the entire MCTS tree structure (for debugging or analysis)
    void printTree() const;
    
    // Get the root node of the MCTS tree
    std::shared_ptr<Node> getRoot();

    // Set the root node of the MCTS tree (useful for tree reuse across iterations)
    void setRoot(std::shared_ptr<Node> newRoot);

    // Sync the tree's root state with an updated game state
    void syncTreeWithGameState(std::shared_ptr<DMAG::Game> updatedState);

    DMAG::Card getBestMove(const std::shared_ptr<DMAG::Game>& game, int playerIndex);

    // Single select+expand step for batched external evaluation.
    // Returns the expanded leaf; caller must call finishStep() with results.
    std::shared_ptr<Node> selectAndExpand();

    // Complete a batched step: set priors on leaf's children and backpropagate.
    void finishStep(std::shared_ptr<Node> leaf,
                    const std::vector<float>& policy, float value);

    // Return the most-visited child of root (best move after search).
    std::shared_ptr<Node> getBestChild() const;

    void hideGameStateForPlayer(std::shared_ptr<DMAG::Game>& game, int playerIndex);

    // Returns a POLICY_SIZE (75) vector of normalized MCTS visit counts over
    // root's children after search(). Index = card.GetId()-1. Zeros for
    // unexplored or out-of-hand cards.
    std::vector<float> getVisitDistribution() const;
};

#endif // MCTS_H
