#include "node.h"
#include <limits>
#include <cmath>
#include <random> 
#include <stdexcept>

// Constructor: Initializes a node with a given state and an optional parent.
Node::Node(DMAG::Game* state, int totalPlayers, std::shared_ptr<Node> parent)
    : state(state), parent(parent), totalPlayers(totalPlayers),
      visitCount(0), value(0.0) {}

// Adds a child node to this node.
void Node::addChild(std::shared_ptr<Node> child) {
    children.push_back(child);
}

// Selects the best child based on some strategy (e.g., Upper Confidence Bound).
std::shared_ptr<Node> Node::selectBestChild() const {
    if (children.empty()) {
        return nullptr; // No children available.
    }

    std::shared_ptr<Node> bestChild = nullptr;
    double bestValue = -std::numeric_limits<double>::infinity();

    for (const auto& child : children) {
         double ucbValue;
        if (child->visitCount == 0) {
            // Prioritize unvisited nodes with a very high UCB value
            return child;
        }  
        
        ucbValue = (child->value / child->visitCount) + 
                          sqrt(2 * log(visitCount) / child->visitCount);

        if (ucbValue > bestValue) {
            bestValue = ucbValue;
            bestChild = child;  // Directly assign the shared_ptr
        }
    }

    return bestChild;  // Return the selected child
}



// Updates the node's value and visit count after a simulation.
void Node::update(double value) {
    this->value += value;
    visitCount++;
}

// Getter for the game state of this node.
DMAG::Game& Node::getState() {
    return *state;
}

// Getter for the parent node.
std::shared_ptr<Node> Node::getParent() const {
    return parent;
}

// Getter for the children of this node.
const std::vector<std::shared_ptr<Node>>& Node::getChildren() const {
    return children;
}

// Returns the number of times the node has been visited.
int Node::getVisitCount() const {
    return visitCount;
}

// Returns the accumulated value of the node.
double Node::getValue() const {
    return value;
}

// Checks if the node is a leaf (i.e., has no children).
bool Node::isLeaf() const {
    return children.empty();
}

// Returns the joint action for this node.
const std::vector<DMAG::Card>& Node::getJointAction() const {
    return jointAction;
}

void Node::setJointAction(const std::vector<DMAG::Card>& jointAction) {
    this->jointAction = jointAction;
}

void Node::expand() {
    // Step 1: Copy parent's state and execute joint action
    if(parent){
        DMAG::Game newState(parent->getState());  // Copy the parent's state

        // Step 2: Apply the joint action to the copied state
        applyJointAction(newState, jointAction);  // Apply joint action to the new state

        // Step 3: End the turn for the current state after joint action
        newState.endTurn();

        state = new DMAG::Game(newState);
    }

    // Step 2: Get possible cards to play for each player
    std::vector<std::vector<DMAG::Card>> allPlayersCards;
    for (int playerIndex = 0; playerIndex < totalPlayers; playerIndex++) {
        std::vector<DMAG::Card> possibleCards = state->getPossibleCardsForPlayer(playerIndex);
        if (possibleCards.empty()) {
            possibleCards.push_back(state->getAllCardsForPlayer(playerIndex)[0]);
        }
        allPlayersCards.push_back(possibleCards);
    }

    // Step 3: Generate all combinations of cards (joint actions)
    std::vector<std::vector<DMAG::Card>> jointActions;
    generateCombinations(allPlayersCards, jointActions);

    // Step 4: Create child, which stores an empty gameState until it is expanded (to save on memory)
    for (const auto& jointAction : jointActions) {
        // Use std::make_shared instead of std::make_unique
        auto childNode = std::make_shared<Node>(nullptr, totalPlayers, shared_from_this());
        childNode->setJointAction(jointAction);  // Set joint action for the child node
        addChild(childNode);
    }
}

void Node::generateCombinations(const std::vector<std::vector<DMAG::Card>>& allPlayersCards, 
                                std::vector<std::vector<DMAG::Card>>& jointActions,
                                std::vector<DMAG::Card> currentCombination, 
                                int depth) const {
    // Debugging: Print the current depth and combination being processed
    std::cout << "Depth: " << depth << std::endl;
    std::cout << "Current Combination: ";
    for (const auto& card : currentCombination) {
        std::cout << card.GetName() << " ";  // Assuming Card has a GetName() method
    }
    std::cout << std::endl;

    // Base case: all players have been processed
    if (depth == allPlayersCards.size()) {
        jointActions.push_back(currentCombination);

        // Debugging: Print the combination being added to jointActions
        std::cout << "Added Combination: ";
        for (const auto& card : currentCombination) {
            std::cout << card.GetName() << " ";  // Assuming Card has a GetName() method
        }
        std::cout << std::endl;

        return;
    }

    // Recursive case: process each card for the current player
    for (const DMAG::Card& card : allPlayersCards[depth]) {
        currentCombination.push_back(card);

        // Debugging: Print the card being added for the current player
        std::cout << "Adding Card: " << card.GetName() << " for player " << depth << std::endl;

        generateCombinations(allPlayersCards, jointActions, currentCombination, depth + 1);

        // Backtrack: remove the last added card
        currentCombination.pop_back();

        // Debugging: Print the card being removed
        std::cout << "Removing Card: " << card.GetName() << " for player " << depth << std::endl;
    }
}

void Node::applyJointAction(DMAG::Game& state, const std::vector<DMAG::Card>& jointAction) const {
    for (size_t i = 0; i < jointAction.size(); ++i) {
        const DMAG::Card& card = jointAction[i];
        if(!state.playCard(i, card)){
            state.discardCard(i, card);
        }
    }
}
