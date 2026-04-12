#include "node.h"
#include "game.h"
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

// Selects the best child using PUCT (AlphaZero) when priors are set,
// falling back to UCB1 for unprimed nodes.
std::shared_ptr<Node> Node::selectBestChild(double cPuct) const {
    if (children.empty()) return nullptr;

    double sqrtTotal = std::sqrt(static_cast<double>(visitCount));

    std::shared_ptr<Node> bestChild = nullptr;
    double bestValue = -std::numeric_limits<double>::infinity();

    for (const auto& child : children) {
        if (child->isFullyTerminal()) continue;

        double q = child->visitCount > 0
            ? child->value / child->visitCount
            : 0.0;

        // PUCT: Q(s,a) + c * P(s,a) * sqrt(N) / (1 + n)
        double u = cPuct * child->prior * sqrtTotal / (1.0 + child->visitCount);
        double puct = q + u;

        if (puct > bestValue) {
            bestValue = puct;
            bestChild = child;
        }
    }

    return bestChild ? bestChild : children[0];
}

void Node::setPrior(float p) {
    prior = p;
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
    return parent.lock();
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
    if (children.empty()) return false;
    return children.size() >= canonicalMoves().size() && state->InGame();
}

// Returns the joint action for this node.
DMAG::Card Node::getAction() const {
    return action;
}

void Node::setAction(const DMAG::Card action, DMAG::MoveType mt) {
    this->action = action;
    this->moveType = mt;
}

DMAG::MoveType Node::getMoveType() const {
    return moveType;
}

void Node::setState(std::shared_ptr<DMAG::Game> newState) {
    state = newState;  // Simply set the new state; shared_ptr manages memory automatically
}

// Build a deduplicated list of canonical moves: one per BUILD_STRUCTURE card,
// plus at most one BUILD_WONDER and one DISCARD (random card chosen at expansion time).
const std::vector<DMAG::Move>& Node::canonicalMoves() const {
    if (!moves_cache.empty()) return moves_cache;

    auto allMoves = state->getAllMovesForPlayer(activePlayer);
    std::vector<DMAG::Move> out;
    static std::mt19937 rng(std::random_device{}());

    std::vector<DMAG::Move> wonderCandidates, discardCandidates;
    for (const auto& m : allMoves) {
        if (m.type == DMAG::MoveType::BUILD_STRUCTURE) out.push_back(m);
        else if (m.type == DMAG::MoveType::BUILD_WONDER) wonderCandidates.push_back(m);
        else if (m.type == DMAG::MoveType::DISCARD)     discardCandidates.push_back(m);
    }
    if (!wonderCandidates.empty()) {
        std::uniform_int_distribution<int> d(0, wonderCandidates.size() - 1);
        moves_cache.push_back(wonderCandidates[d(rng)]);
    }
    if (!discardCandidates.empty()) {
        std::uniform_int_distribution<int> d(0, discardCandidates.size() - 1);
        moves_cache.push_back(discardCandidates[d(rng)]);
    }
    moves_cache.insert(moves_cache.end(), out.begin(), out.end());
    return moves_cache;
}

std::shared_ptr<Node> Node::expand() {
    auto moves = canonicalMoves();
    if (moves.empty()) return shared_from_this();

    // Lazy: one new child per call.
    if (children.size() >= moves.size()) return children[0];

    static std::mt19937 rng(std::random_device{}());

    const DMAG::Move& move = moves[children.size()];

    auto newState = std::make_shared<DMAG::Game>(*state);
    SilenceGuard _silence;
    newState->applyMove(activePlayer, move);

    for (int pi = 0; pi < totalPlayers; ++pi) {
        if (pi == activePlayer) continue;
        auto otherMoves = newState->getAllMovesForPlayer(pi);
        if (!otherMoves.empty()) {
            std::uniform_int_distribution<int> dist(0, otherMoves.size() - 1);
            newState->applyMove(pi, otherMoves[dist(rng)]);
        }
    }
    newState->endTurn();

    auto child = std::make_shared<Node>(newState, totalPlayers, activePlayer, shared_from_this());
    child->setAction(move.card, move.type);

    if (!policy_cache.empty()) {
        int idx = policyIndex(move.type, move.card.GetId());
        if (idx >= 0 && idx < (int)policy_cache.size())
            child->setPrior(policy_cache[idx]);
    } else {
        // Without NN: give wonder/discard a small baseline prior so PUCT explores them.
        if (move.type == DMAG::MoveType::BUILD_WONDER)
            child->setPrior(0.15f);
        else if (move.type == DMAG::MoveType::DISCARD)
            child->setPrior(0.05f);
    }

    addChild(child);
    if (!newState->InGame()) child->markChildAsTerminal();

    return child;
}

void Node::setPolicyCache(const std::vector<float>& policy) {
    policy_cache = policy;
}

bool Node::isFullyTerminal() {
    return terminalChildren >= (int)canonicalMoves().size();
}

void Node::markChildAsTerminal() {
    // Increment the terminal count
    terminalChildren++;
    
    // If all children are terminal, mark this node as terminal
    if (isFullyTerminal()) {
        if (auto p = parent.lock()) {
            p->markChildAsTerminal();
        }
    }
}

bool Node::isLeaf() {
    if (!state->InGame()) return true;
    return children.size() < canonicalMoves().size();
}