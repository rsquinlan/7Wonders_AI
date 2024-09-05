#include <node.h>

// Constructor: Initializes a node with a given state and an optional parent.
Node::Node(DMAG::Game state, std::shared_ptr<Node> parent)
    : state(), parent(parent), visitCount(0), value(0) {}

// Adds a child node to this node.
void Node::addChild(std::unique_ptr<Node> child) {
    children.push_back(std::move(child));
}

// Selects the best child based on some strategy (e.g., Upper Confidence Bound).
std::shared_ptr<Node> Node::selectBestChild() const {
    // Example using UCB1 (Upper Confidence Bound 1) formula:
    // UCB1 = (child->value / child->visitCount) + sqrt(2 * log(visitCount) / child->visitCount)
    
    if (children.empty()) {
        return nullptr; // No children available.
    }

    auto bestChild = children[0].get();
    double bestValue = -std::numeric_limits<double>::infinity();

    for (const auto& child : children) {
        if (child->visitCount == 0) continue;

        double ucbValue = (child->value / child->visitCount) + 
                          sqrt(2 * log(visitCount) / child->visitCount);

        if (ucbValue > bestValue) {
            bestValue = ucbValue;
            bestChild = child.get();
        }
    }

    return std::make_shared<Node>(*bestChild);
}

// Updates the node's value and visit count after a simulation.
void Node::update(double value) {
    this->value += value;
    visitCount++;
}

// Getter for the game state of this node.
DMAG::Game Node::getState() const {
    return state;
}

// Getter for the parent node.
std::shared_ptr<Node> Node::getParent() const {
    return parent;
}

// Getter for the children of this node.
const std::vector<std::unique_ptr<Node>>& Node::getChildren() const {
    return children;
}

// Returns the number of times the node has been visited.
int Node::getVisitCount() const {
    return visitCount;
}

// Returns the accumulated value of the node.
int Node::getValue() const {
    return value;
}

// Checks if the node is a leaf (i.e., has no children).
bool Node::isLeaf() const {
    return children.empty();
}

// Expands the node by generating possible children from the current state.
void Node::expand() {
    // Assuming DMAG::Game has a method getPossibleMoves() that returns a list of possible game states.
    auto possibleMoves = state.getPossibleMoves();

    for (auto& move : possibleMoves) {
        auto childNode = std::make_shared<Node>(move, shared_from_this());
        addChild(childNode);
    }
}
