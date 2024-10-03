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

bool Node::isFullyExpanded() const {
    if (children.empty()){
        return false;
    }
    return children.size() == state->getPossibleCardsForPlayer(activePlayer).size();
}

// Returns the joint action for this node.
DMAG::Card Node::getAction() const {
    return action;
}

void Node::setAction(const DMAG::Card action) {
    this->action = action;
}

void Node::setState(DMAG::Game& newState) {
    // Assign the new state to the node, creating a deep copy of the passed-in game state
    if (state) {
        delete state;  // Free the previous state to prevent memory leaks
    }
    state = new DMAG::Game(newState);  // Allocate new memory for the game state
}


#include <algorithm> // For std::find_if

std::shared_ptr<Node> Node::expand() {
    // Step 2: Get the possible actions (cards) for the active player
    std::vector<DMAG::Card> possibleActions = state->getPossibleCardsForPlayer(activePlayer);

    // Ensure there is at least one card (fallback if no cards are available)
    if (possibleActions.empty()) {
        possibleActions.push_back(state->getAllCardsForPlayer(activePlayer)[0]);
    }

    // Step 3: Select a random action until we find one that hasn't been expanded yet
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, possibleActions.size() - 1);

    DMAG::Card selectedAction;
    bool actionExists = true;

    // Loop until we find a non-expanded action
    while (actionExists) {
        selectedAction = possibleActions[dist(gen)];  // Pick a random action

        // Check if this action has already been expanded as a child
        actionExists = std::any_of(children.begin(), children.end(),
            [&](const std::shared_ptr<Node>& child) {
                return child->getAction().GetId() == selectedAction.GetId();
            });

        // If action doesn't exist as a child, break the loop
        if (!actionExists) {
            break;
        }
    }

    DMAG::Game* newState(state);  // Copy the parent's state

    // Apply the active player's action
    newState->playCard(activePlayer, selectedAction);  // Apply action to the new state for the active player

    // Simulate actions for the other players to bring the game state up to date
    for (int playerIndex = 0; playerIndex < totalPlayers; ++playerIndex) {
        if (playerIndex != activePlayer) {
            std::vector<DMAG::Card> possibleCards = newState->getPossibleCardsForPlayer(playerIndex);
            
            if (!possibleCards.empty()) {
                // Use random device to generate a random card index
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_int_distribution<> dist(0, possibleCards.size() - 1);

                // Pick a random card from the possible cards
                int randomIndex = dist(gen);
                DMAG::Card randomAction = possibleCards[randomIndex];

                // Apply the randomly selected card
                newState->playCard(playerIndex, randomAction);
            }
        }
    }
    newState->endTurn();

    // Step 4: Create a child node with the selected action
    auto childNode = std::make_shared<Node>(newState, totalPlayers, activePlayer, shared_from_this());

    // Set the action (card) for the child node
    childNode->setAction(selectedAction);  // This action represents the active player's move

    // Add the child node to the current node's children
    addChild(childNode);

    return childNode;
}
