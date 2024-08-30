#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <cstdlib>
#include <ctime>

class GameState {
    public:
        std::vector<DMAG::Card> get_legal_action() {
            return 
        }
}

class Node {
public:
    GameState state;
    Node* parent;
    std::vector<Node*> children;
    int visits;
    double wins;

    Node(const GameState& state, Node* parent = nullptr)
        : state(state), parent(parent), visits(0), wins(0) {}

    bool is_fully_expanded() const {
        return children.size() == state.get_legal_actions().size();
    }

    bool is_terminal() const {
        return state.is_terminal();
    }

    Node* best_child(double exploration_weight = 1.0) const {
        Node* best = nullptr;
        double best_value = std::numeric_limits<double>::lowest();
        for (Node* child : children) {
            double uct_value = (child->wins / child->visits) +
                exploration_weight * std::sqrt(std::log(visits) / child->visits);
            if (uct_value > best_value) {
                best_value = uct_value;
                best = child;
            }
        }
        return best;
    }

    Node* expand() {
        std::vector<int> legal_actions = state.get_legal_actions();
        for (int action : legal_actions) {
            bool already_expanded = false;
            for (Node* child : children) {
                if (child->state == state.apply_action(action)) {
                    already_expanded = true;
                    break;
                }
            }
            if (!already_expanded) {
                GameState new_state = state.apply_action(action);
                Node* new_node = new Node(new_state, this);
                children.push_back(new_node);
                return new_node;
            }
        }
        return nullptr; // Should not reach here if not fully expanded
    }
};

class MCTS {
public:
    MCTS(int iterations, double exploration_weight = 1.0)
        : iterations(iterations), exploration_weight(exploration_weight) {}

    GameState search(const GameState& initial_state) {
        Node* root = new Node(initial_state);

        for (int i = 0; i < iterations; ++i) {
            Node* node = tree_policy(root);
            double reward = default_policy(node->state);
            backup(node, reward);
        }

        Node* best_node = root->best_child(0);  // No exploration in final decision
        GameState best_action_state = best_node->state;
        delete root;
        return best_action_state;
    }

private:
    int iterations;
    double exploration_weight;

    Node* tree_policy(Node* node) {
        while (!node->is_terminal()) {
            if (!node->is_fully_expanded()) {
                return node->expand();
            } else {
                node = node->best_child(exploration_weight);
            }
        }
        return node;
    }

    double default_policy(const GameState& state) {
        GameState current_state = state;
        while (!current_state.is_terminal()) {
            std::vector<int> legal_actions = current_state.get_legal_actions();
            int random_action = legal_actions[rand() % legal_actions.size()];
            current_state = current_state.apply_action(random_action);
        }
        return current_state.get_reward();
    }

    void backup(Node* node, double reward) {
        while (node != nullptr) {
            node->visits += 1;
            node->wins += reward;
            node = node->parent;
        }
    }
};

int main() {
    srand(static_cast<unsigned>(time(0)));  // Initialize random seed

    GameState initial_state;
    MCTS mcts(1000);  // 1000 iterations for MCTS

    GameState best_state = mcts.search(initial_state);

    // Output or use the best_state according to your application
    return 0;
}
