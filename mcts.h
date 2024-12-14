#ifndef MCTS_H
#define MCTS_H

#include <memory>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <stdexcept>

#include "game.h"
#include "card.h"
#include "node.h"

class MCTS {
private:
    std::shared_ptr<Node> root;  // Root node of the tree
    int totalPlayers;            // Total number of players in the game
    int currentPlayer;           // Current player for whom MCTS is being executed
    double explorationConstant;  // Exploration constant for UCB1 formula

    // Selection step: Select the most promising child node based on UCB1 value
    std::shared_ptr<Node> select(std::shared_ptr<Node> node);

    // Expansion step: Expand a node by generating all possible joint actions
    std::shared_ptr<Node> expand(std::shared_ptr<Node> node);

    // Backpropagation step: Backpropagate the simulation result up the tree
    void backpropagate(std::shared_ptr<Node> node, double reward);

    // Helper function to print the tree recursively
    void printTreeRecursive(const Node* node, int depth) const;

    double simulate(std::shared_ptr<DMAG::Game> game, double explorationChance);

    double evaluateMoveHeuristic(const std::shared_ptr<DMAG::Game>& game, int playerIndex, const DMAG::Card& card);

public:
    // Constructor for initializing MCTS with the game state and player information
    MCTS(const DMAG::Game& initialState, int totalPlayers, int currentPlayer, double explorationConstant = std::sqrt(2));

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

    void hideGameStateForPlayer(std::shared_ptr<DMAG::Game>& game, int playerIndex);
};

#endif // MCTS_H
