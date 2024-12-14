#include "node.h"
#include <limits>
#include <cmath>
#include <random> 
#include <stdexcept>

// Constructor: Initializes a node with a given state and an optional parent.
Node::Node(std::shared_ptr<DMAG::Game> state, int totalPlayers, int activePlayer, std::shared_ptr<Node> parent)
    : state(state), parent(parent), totalPlayers(totalPlayers), activePlayer(activePlayer),
      visitCount(0), value(0.0), terminalChildren(0) {}

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
        if (child->visitCount == 0 && !child->isFullyTerminal()) {
            // Prioritize unvisited nodes with a very high UCB value
            return child;
        }  
        
        ucbValue = (child->value / child->visitCount) + 
                          sqrt(1 * log(visitCount) / child->visitCount);

        if (ucbValue > bestValue && !child->isFullyTerminal()) {
            bestValue = ucbValue;
            bestChild = child;  // Directly assign the shared_ptr
        }
    }

    if(!bestChild){
        return(children[0]);
    }

    return bestChild;  // Return the selected child
}

// Updates the node's value and visit count after a simulation.
void Node::update(double value) {
    this->value += value;
    visitCount++;
}

// Getter for the game state of this node.
std::shared_ptr<DMAG::Game> Node::getState() const {
    return state;  // Return shared_ptr to the state
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
    return children.size() >= state->getPossibleCardsForPlayer(activePlayer).size() && state->InGame();
}

// Returns the joint action for this node.
DMAG::Card Node::getAction() const {
    return action;
}

void Node::setAction(const DMAG::Card action) {
    this->action = action;
}

void Node::setState(std::shared_ptr<DMAG::Game> newState) {
    state = newState;  // Simply set the new state; shared_ptr manages memory automatically
}

std::shared_ptr<Node> Node::expand() {
    // Step 2: Get the possible actions (cards) for the active player
    std::vector<DMAG::Card> possibleActions = state->getPossibleCardsForPlayer(activePlayer);

    // Ensure there is at least one card (fallback if no cards are available)
    if (possibleActions.empty()) {
        possibleActions.push_back(state->getAllCardsForPlayer(activePlayer)[0]);
    }

    // Step 3: Loop through all possible actions and create child nodes for each
    for (const DMAG::Card& selectedAction : possibleActions) {
        // Create a new state to avoid modifying the current state
        auto newState = std::make_shared<DMAG::Game>(*state);

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

        // If the game state is terminal for the child node, mark it as terminal
        if (!newState->InGame()) {
            childNode->markChildAsTerminal();  // Mark this child node as terminal
        }
    }

    // Step 5: Select and return a random child from the newly created children
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, children.size() - 1);

    return children[dist(gen)];
}

bool Node::isFullyTerminal() {
    return terminalChildren >= state->getPossibleCardsForPlayer(activePlayer).size();
}

void Node::markChildAsTerminal() {
    // Increment the terminal count
    terminalChildren++;
    
    // If all children are terminal, mark this node as terminal
    if (isFullyTerminal()) {
        // Recursively propagate terminal status to the parent if applicable
        if (parent) {
            parent->markChildAsTerminal();  // Parent might also become terminal
        }
    }
}

bool Node::isLeaf() {
    return children.empty();
}