#include "mcts.h"
#include "utils.h"
#include <random>
#include <algorithm> // For std::max_element

enum PlayerMode { MCTS_MODE, HEURISTIC_MODE, RANDOM_MODE };


// Selection step: Select the most promising child node based on UCB1 value
std::shared_ptr<Node> MCTS::select(std::shared_ptr<Node> node) {
    while (!node->isLeaf()) {
        node = node->selectBestChild();
        if (!node->getState()->InGame()) {
            return node;
        }
    }
    return node;  // Return the leaf or expandable node
}

// Expansion step: Expand a node by generating all possible joint actions
std::shared_ptr<Node> MCTS::expand(std::shared_ptr<Node> node) {
    return node->expand();
}

#include <random>

double MCTS::simulate(std::shared_ptr<DMAG::Game> game, double explorationChance) {
    std::random_device rd;              // Seed for random number generator
    std::mt19937 gen(rd());             // Mersenne Twister RNG
    std::uniform_real_distribution<> probDist(0.0, 1.0); // For probability

    // Loop through all players and simulate their moves
    while (game->InGame()) {
        std::cout << "-------------------------" << std::endl;
        std::cout << "Turn: " << (int)game->turn << std::endl;
        for (int i = 0; i < game->player_list.size(); ++i) {
            std::cout << "Player " << i << " hand: ";
            auto cards = game->getAllCardsForPlayer(i);
            std::cout << cards.size() << " cards: ";
            for (const auto& card : cards) {
            std::cout << card.GetName() << ", ";
            }
            std::cout << std::endl;
        }
        std::cout << "-------------------------" << std::endl;
        for (size_t i = 0; i < game->player_list.size(); ++i) {
            if (probDist(gen) < explorationChance) {
                // Exploration: Choose a random playable card
                auto playableCards = game->player_list[i]->GetPlayableCards();
                if (!playableCards.empty()) {
                    std::uniform_int_distribution<> cardDist(0, playableCards.size() - 1);
                    std::cout << "here" << std::endl;
                    game->applyAction(i, playableCards[cardDist(gen)]);
                }
            } else {
                // Exploitation: use injected rollout policy if set, otherwise heuristic
                DMAG::Card bestMove = rolloutPolicy
                    ? rolloutPolicy(game, i)
                    : getBestMove(game, i);
                game->applyAction(i, bestMove);
            }
        }
        game->endTurn();
    }

    // Return the final score of the player who initiated the simulation
    return game->getPlayerScore(currentPlayer);
}


// Backpropagation step: Backpropagate the simulation result up the tree
void MCTS::backpropagate(std::shared_ptr<Node> node, double reward) {
    while (node != nullptr) {
        node->update(reward);
        node = node->getParent();
    }
}

MCTS::MCTS(const DMAG::Game& initialState, int totalPlayers, int currentPlayer,
           double explorationConstant, RolloutPolicy rolloutPolicy)
    : totalPlayers(totalPlayers), currentPlayer(currentPlayer),
      explorationConstant(explorationConstant), rolloutPolicy(std::move(rolloutPolicy)) {
    auto initialStatePtr = std::make_shared<DMAG::Game>(initialState);
    root = std::make_shared<Node>(initialStatePtr, totalPlayers, currentPlayer, nullptr);
}

// Perform MCTS search and return the best move for the current player
std::shared_ptr<Node> MCTS::search(int iterations, double explorationChance) {
    for (int i = 0; i < 5; ++i) {
        if (root->isFullyTerminal()) {
            break;
        }

        auto selectedNode = select(root);
        if (selectedNode->isFullyTerminal()) {
            backpropagate(selectedNode, selectedNode->getValue() / selectedNode->getVisitCount());
            continue;
        }

        auto expandedNode = expand(selectedNode);
        double reward = simulate(expandedNode->getState(), explorationChance);
        backpropagate(expandedNode, reward);
    }

    if (root->getChildren().empty()) {
        root->setAction(root->getState()->getAllCardsForPlayer(currentPlayer)[0]);
        return root;
    }

    auto bestChildIt = std::max_element(
        root->getChildren().begin(),
        root->getChildren().end(),
        [](const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b) {
            return a->getValue() / a->getVisitCount() < b->getValue() / b->getVisitCount();
        }
    );

    return *bestChildIt;
}

void MCTS::printTree() const {
    printTreeRecursive(root.get(), 0);
}

void MCTS::printTreeRecursive(const Node* node, int depth) const {
    if (!node) return;
    for (int i = 0; i < depth; ++i) std::cout << "  ";
    std::cout << "Node: " << node << ", Value: " << node->getValue()
              << ", Visits: " << node->getVisitCount() << ", Children: " << node->getChildren().size() << std::endl;

    for (const auto& child : node->getChildren()) {
        printTreeRecursive(child.get(), depth + 1);
    }
}

std::shared_ptr<Node> MCTS::getRoot() {
    if (!root) {
        throw std::runtime_error("Root node is null.");
    }
    return root;
}

void MCTS::setRoot(std::shared_ptr<Node> newRoot) {
    root = newRoot;
}

void MCTS::syncTreeWithGameState(std::shared_ptr<DMAG::Game> updatedState) {
    root->setState(updatedState);
}

MCTS::~MCTS() {
    root.reset();
}

std::vector<float> MCTS::getVisitDistribution() const {
    std::vector<float> dist(POLICY_SIZE, 0.0f);
    const auto& children = root->getChildren();
    if (children.empty()) return dist;

    double total = 0.0;
    for (const auto& child : children)
        total += child->getVisitCount();
    if (total == 0.0) return dist;

    for (const auto& child : children) {
        int idx = child->getAction().GetId() - 1;  // 0-based index into policy vector
        if (idx >= 0 && idx < POLICY_SIZE)
            dist[idx] = static_cast<float>(child->getVisitCount() / total);
    }
    return dist;
}

double MCTS::evaluateMoveHeuristic(const std::shared_ptr<DMAG::Game>& game, int playerIndex, const DMAG::Card& card) {
    // Create a deep copy of the game state
    auto gameCopy = std::make_shared<DMAG::Game>(*game);

    // Get initial score and resources
    int originalScore = gameCopy->getPlayerScore(playerIndex);
    auto originalResources = gameCopy->player_list[playerIndex]->GetResources();

    // Apply the move on the copied state
    gameCopy->applyAction(playerIndex, card);

    // Recalculate score and resources after the move
    int updatedScore = gameCopy->getPlayerScore(playerIndex);
    auto updatedResources = gameCopy->player_list[playerIndex]->GetResources();

    // Calculate deltas
    int deltaScore = updatedScore - originalScore;
    double deltaResources = 0.0;

    for (const auto& [resourceType, updatedQuantity] : updatedResources) {
        int originalQuantity = originalResources.at(resourceType);
        int delta = updatedQuantity - originalQuantity;

        // Ignore specific resources
        if (resourceType == RESOURCE::shields || 
            resourceType == RESOURCE::tablet || 
            resourceType == RESOURCE::compass || 
            resourceType == RESOURCE::gear) {
            continue;
        }

        // Weigh coins less than other resources
        if (resourceType == RESOURCE::coins) {
            deltaResources += 0.33 * delta;
        } else {
            deltaResources += delta;
        }
    }

    // **New: Turn-based resource weighting**
    int era = card.GetEra(); // Card's era
    double eraMultiplier = (era == 1) ? 2.0 : (era == 2) ? 1.5 : 0.5;
    deltaResources *= eraMultiplier;

    // **New: Neighbor strategy awareness**
    double neighborPenalty = 0.0;

    // Analyze left and right neighbors
    const auto& leftNeighbor = gameCopy->player_list[playerIndex]->GetWestNeighbor();
    const auto& rightNeighbor = gameCopy->player_list[playerIndex]->GetEastNeighbor();

    int leftMilitaryCards = leftNeighbor->countCardsOfType(CARD_TYPE::military);
    int rightMilitaryCards = rightNeighbor->countCardsOfType(CARD_TYPE::military);

    int leftScienceCards = leftNeighbor->countCardsOfType(CARD_TYPE::scientific);
    int rightScienceCards = rightNeighbor->countCardsOfType(CARD_TYPE::scientific);

    // Penalize selecting a card of the same type if neighbors are focused on it
    if (card.GetType() == CARD_TYPE::military && (leftMilitaryCards > 2 || rightMilitaryCards > 2)) {
        neighborPenalty -= 5.0; // Military becomes less attractive
    }

    if (card.GetType() == CARD_TYPE::scientific && (leftScienceCards > 2 || rightScienceCards > 2)) {
        neighborPenalty -= 3.0; // Science becomes less attractive
    }

    // **New: Synergy with card type and current player strategy**
    int playerScienceCards = gameCopy->player_list[playerIndex]->countCardsOfType(CARD_TYPE::scientific);
    int playerMilitaryCards = gameCopy->player_list[playerIndex]->countCardsOfType(CARD_TYPE::military);

    double synergyBonus = 0.0;
    if (card.GetType() == CARD_TYPE::scientific && playerScienceCards >= 3) {
        synergyBonus += 5.0; // Boost for continuing science strategy
    }

    if (card.GetType() == CARD_TYPE::military && playerMilitaryCards >= 2) {
        synergyBonus += 3.0; // Boost for continuing military strategy
    }

    // Combine heuristic factors
    return deltaScore + deltaResources + neighborPenalty + synergyBonus;
}


DMAG::Card MCTS::getBestMove(const std::shared_ptr<DMAG::Game>& game, int playerIndex) {
    auto gameCopy = std::make_shared<DMAG::Game>(*game);

    hideGameStateForPlayer(gameCopy, playerIndex);

    auto playableCards = game->player_list[playerIndex]->GetPlayableCards();

    if (playableCards.empty()) {
        const auto& playerHand = game->player_list[playerIndex]->GetHandCards();
        if (!playerHand.empty()) {
            return playerHand.front(); // Return the first card
        }
    }

    double bestHeuristicValue = -std::numeric_limits<double>::infinity();
    DMAG::Card bestCard;

    // Iterate through each card to evaluate heuristic
    for (const auto& card : playableCards) {
        double heuristicValue = evaluateMoveHeuristic(game, playerIndex, card);

        if (heuristicValue > bestHeuristicValue) {
            bestHeuristicValue = heuristicValue;
            bestCard = card;
        }
    }

    return bestCard;
}

void MCTS::hideGameStateForPlayer(std::shared_ptr<DMAG::Game>& game, int playerIndex) {
    auto currentPlayer = game->player_list[playerIndex];
    int turn = game->turn; // Assume game has a `turn` member
    bool clockwise = (game->era == 1 || game->era == 3);

    // Pool all cards from opponents' hands
    std::vector<DMAG::Card> cardPool;
    DMAG::Player* player = currentPlayer;
    DMAG::Player* neighbor;

    // Traverse neighbors and pool cards
    for (int i = 0; i < game->number_of_players; ++i) {
        player = clockwise ? player->GetWestNeighbor() : player->GetEastNeighbor();
        if (i < turn - 1) {
            // Skip pooling for neighbors whose hands the player already knows
            continue;
        }
        auto handCards = player->GetHandCards();
        cardPool.insert(cardPool.end(), handCards.begin(), handCards.end());
    }

    // Redistribute plausible hands
    std::shuffle(cardPool.begin(), cardPool.end(), std::mt19937{std::random_device{}()});
    player = currentPlayer;
    for (int i = 0; i < game->number_of_players; ++i) {
        player = clockwise ? player->GetWestNeighbor() : player->GetEastNeighbor();
        if (i < turn - 1) {
            // Skip redistribution for known neighbors
            continue;
        }

        // Create a plausible hand for this neighbor
        int handSize = player->GetHandCards().size();
        std::vector<DMAG::Card> plausibleHand(cardPool.begin(), cardPool.begin() + handSize);
        player->ReceiveCards(plausibleHand);

        // Remove the distributed cards from the pool
        cardPool.erase(cardPool.begin(), cardPool.begin() + handSize);
    }
}
