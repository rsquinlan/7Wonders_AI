#include "mcts.h"
#include <random>

// Selection step: Select the most promising child node based on UCB1 value
std::shared_ptr<Node> MCTS::select(std::shared_ptr<Node> node) {
    while (!node->isLeaf()) {
        node = node->selectBestChild();
    }

    return node;
}

// Expansion step: Expand a node by generating all possible joint actions
void MCTS::expand(std::shared_ptr<Node> node) {
    if (node->isLeaf()) {
        node->expand();
    }
}

// Simulation step: Simulate a random playout from a given node
double MCTS::simulate(std::shared_ptr<Node> node) {
    DMAG::Game state = node->getState();
    while (state.InGame()) {
        std::vector<DMAG::Card> jointAction;
        for (int i = 0; i < totalPlayers; ++i) {
            std::vector<DMAG::Card> possibleCards = state.getPossibleCardsForPlayer(i);
            if (!possibleCards.empty()) {
                int randomIndex = rand() % possibleCards.size();
                jointAction.push_back(possibleCards[randomIndex]);
            }
        }
        for (size_t i = 0; i < jointAction.size(); ++i) {
            std::vector<DMAG::Card> possibleCards = state.getAllCardsForPlayer(i);
            for(auto card : possibleCards){
                std::cout << card.GetName() << ", ";
            }
            std::cout << std::endl << jointAction[i].GetName() << std::endl;
            state.playCard(i, jointAction[i]);
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
    root = std::make_shared<Node>(initialStatePtr, totalPlayers, nullptr);
}

// Perform MCTS search and return the best move for the current player
std::shared_ptr<Node> MCTS::search(int iterations, int currentPlayer, std::shared_ptr<Node> currentNode) {
    for (int i = 0; i < iterations; ++i) {
        std::shared_ptr<Node> selectedNode = select(currentNode);
        expand(selectedNode);

        // Create a new Game instance based on the selectedNode's state
        DMAG::Game* initialStatePtr = new DMAG::Game(selectedNode->getState());
        auto clonedNode = std::make_shared<Node>(initialStatePtr, totalPlayers, nullptr);
        
        double reward = simulate(clonedNode);
        backpropagate(selectedNode, reward);
    }

    if(currentNode->getChildren().size() == 0){
        std::cin.ignore();
    }

    // Find the best child node based on the average value per visit
    auto bestChildIt = std::max_element(
        currentNode->getChildren().begin(),
        currentNode->getChildren().end(),
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
