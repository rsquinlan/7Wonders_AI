#include "node.h"
#include <limits>
#include <cmath>
#include <random> 
#include <stdexcept>

// Constructor: Initializes a node with a given state and an optional parent.
Node::Node(DMAG::Game* state, int totalPlayers, int activePlayer, std::shared_ptr<Node> parent)
    : state(state), parent(parent), totalPlayers(totalPlayers), activePlayer(activePlayer),
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
const DMAG::Card Node::getAction() const {
    return action;
}

void Node::setAction(const DMAG::Card action) {
    this->action = action;
}

void Node::expand() {
    // Step 1: Copy the parent's state and execute the action
    if (parent) {
        DMAG::Game newState(parent->getState());  // Copy the parent's state

        // Step 2: Apply the action taken by the parent (for the active player) to the copied state
        applyAction(newState, action);  // Apply the action to the new state

        // Step 3: End the turn for the current state after the action
        newState.endTurn();

        // Update this node's state to reflect the new state after the action
        state = new DMAG::Game(newState);
    }

    // Step 2: Get the possible actions (cards) for the active player
    std::vector<DMAG::Card> possibleCards = state->getPossibleCardsForPlayer(activePlayer);

    // Ensure there is at least one card (fallback if no cards are available)
    if (possibleCards.empty()) {
        possibleCards.push_back(state->getAllCardsForPlayer(activePlayer)[0]);
    }

    // Step 3: Create a child node for each possible action (single card) the active player can take
    for (const DMAG::Card& cardAction : possibleCards) {
        // Create a child node with an empty game state (to save memory until it's expanded)
        auto childNode = std::make_shared<Node>(nullptr, totalPlayers, shared_from_this());

        // Set the action (card) for the child node
        childNode->setAction(cardAction);  // This action represents the active player's move

        // Add the child node to the current node's children
        addChild(childNode);
    }
}
