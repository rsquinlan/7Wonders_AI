#include "mcts.h"
#include <random>

// Selection step: Select the most promising child node based on UCB1 value
std::shared_ptr<Node> MCTS::select(std::shared_ptr<Node> node) {
    while (node->isFullyExpanded()) {
        node = node->selectBestChild();
    }
    return node;
}

// Expansion step: Expand a node by generating all possible joint actions
std::shared_ptr<Node> MCTS::expand(std::shared_ptr<Node> node) {
    return node->expand();
}

// Simulation step: Simulate a random playout from a given node
double MCTS::simulate(std::shared_ptr<Node> node) {
    DMAG::Game state = node->getState();
    while (state.InGame()) {
        std::vector<DMAG::Card> action;
        for (int i = 0; i < totalPlayers; ++i) {
            std::vector<DMAG::Card> possibleCards = state.getPossibleCardsForPlayer(i);
            if (!possibleCards.empty()) {
                int randomIndex = rand() % possibleCards.size();
                action.push_back(possibleCards[randomIndex]);
            }
        }
        for (size_t i = 0; i < action.size(); ++i) {
            std::vector<DMAG::Card> possibleCards = state.getAllCardsForPlayer(i);
            for(auto card : possibleCards){
                std::cout << card.GetName() << ", ";
            }
            std::cout << std::endl << action[i].GetName() << std::endl;
            state.playCard(i, action[i]);
        }
        state.endTurn();
    }
    return state.getPlayerScore(currentPlayer);
}

// Backpropagation step: Backpropagate the simulation result up the tree
void MCTS::backpropagate(std::shared_ptr<Node> node, double reward) {
    while (node != nullptr) {
        node->update(reward);
        node = node->getParent();
    }
}

MCTS::MCTS(const DMAG::Game& initialState, int totalPlayers, int currentPlayer, double explorationConstant)
    : totalPlayers(totalPlayers), currentPlayer(currentPlayer), explorationConstant(explorationConstant) {
    DMAG::Game* initialStatePtr = new DMAG::Game(initialState); // Create a new game instance
    root = std::make_shared<Node>(initialStatePtr, totalPlayers, currentPlayer, nullptr);
}

// Perform MCTS search and return the best move for the current player
std::shared_ptr<Node> MCTS::search(int iterations) {
    for (int i = 0; i < iterations; ++i) {
        std::shared_ptr<Node> selectedNode = select(root);
        std::shared_ptr<Node> expandedNode = expand(selectedNode);
        // Create a new Game instance based on the selectedNode's state
        DMAG::Game* initialStatePtr = new DMAG::Game(expandedNode->getState());
        auto clonedNode = std::make_shared<Node>(initialStatePtr, totalPlayers, currentPlayer, nullptr);
        double reward = simulate(clonedNode);
        backpropagate(selectedNode, reward);
        // std::cout << i << std::endl;
        // std::cin.ignore();
    }

    // Find the best child node based on the average value per visit
    auto bestChildIt = std::max_element(
        root->getChildren().begin(),
        root->getChildren().end(),
        [](const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b) {
            return a->getValue() / a->getVisitCount() < b->getValue() / b->getVisitCount();
        }
    );

    // Return the move corresponding to the current player
    return *bestChildIt;
}


void MCTS::printTree() const {
        printTreeRecursive(root.get(), 0);
}

void MCTS::printTreeRecursive(const Node* node, int depth) const {
    if (!node) {
        return;
    }

    // Print the current node's information with indentation based on depth
    for (int i = 0; i < depth; ++i) {
        std::cout << "  ";
    }
    std::cout << "Node: " << node << ", Value: " << node->getValue() 
              << ", Visits: " << node->getVisitCount() << ", Children: " << node->getChildren().size() << std::endl;

    // Recurse for each child node
    for (const auto& child : node->getChildren()) {
        printTreeRecursive(child.get(), depth + 1);  // Use .get() to get the raw pointer
    }
}

std::shared_ptr<Node> MCTS::getRoot() {
    if (!root) {
        throw std::runtime_error("Root node is null.");
    }
    return root;  // Dereference the shared_ptr to get the Node reference
}

void MCTS::setRoot(std::shared_ptr<Node> newRoot) {
    root = newRoot;  // Simply update the root to the selected node
}

void MCTS::syncTreeWithGameState(DMAG::Game& updatedState) {
    root->setState(updatedState);  // Set the new root state for synchronization
}