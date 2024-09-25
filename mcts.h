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
    void expand(std::shared_ptr<Node> node);

    // Simulation step: Simulate a random playout from a given node
    double simulate(std::shared_ptr<Node> node);

    // Backpropagation step: Backpropagate the simulation result up the tree
    void backpropagate(std::shared_ptr<Node> node, double reward);

    void printTreeRecursive(const Node* node, int depth) const;


public:
    MCTS(const DMAG::Game& initialState, int totalPlayers, int currentPlayer, double explorationConstant = std::sqrt(2));

    // Perform MCTS search and return the best move for the current player
    std::shared_ptr<Node> search(int iterations, int currentPlayer, std::shared_ptr<Node> currentNode);

    void printTree() const;
    
    std::shared_ptr<Node> getRoot();
};

#endif // MCTS_H
