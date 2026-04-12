#include <optional>
#include <memory>
#include <vector>
#include <game.h>

class Node : public std::enable_shared_from_this<Node> {
public:
    Node(std::shared_ptr<DMAG::Game> state, int totalPlayers, int activePlayer, std::shared_ptr<Node> parent);

    void addChild(std::shared_ptr<Node> child);
    std::shared_ptr<Node> selectBestChild(double cPuct = 1.0) const;
    void update(double value);
    std::shared_ptr<DMAG::Game> getState() const;  // Return shared_ptr to manage the state
    std::shared_ptr<Node> getParent() const;  // returns nullptr if parent was freed
    const std::vector<std::shared_ptr<Node>>& getChildren() const;
    int getVisitCount() const;
    double getValue() const;
    bool isFullyExpanded() const;
    std::shared_ptr<Node> expand();
    void setAction(DMAG::Card action, DMAG::MoveType moveType = DMAG::MoveType::BUILD_STRUCTURE);
    DMAG::Card getAction() const;
    DMAG::MoveType getMoveType() const;
    void setState(std::shared_ptr<DMAG::Game> newState);
    void setPrior(float p);
    // Cache the NN policy so future lazily-created children get correct priors.
    void setPolicyCache(const std::vector<float>& policy);
    bool isFullyTerminal();
    bool isLeaf();

private:
    std::shared_ptr<DMAG::Game> state;  // Use shared_ptr for the state
    std::weak_ptr<Node> parent;
    int totalPlayers;
    int activePlayer;
    int visitCount;
    int terminalChildren;
    double value;
    std::vector<std::shared_ptr<Node>> children;
    DMAG::Card action;
    DMAG::MoveType moveType = DMAG::MoveType::BUILD_STRUCTURE;
    float prior = 0.0f;              // prior probability from policy head (PUCT)
    std::vector<float> policy_cache; // NN policy cached here; used when new children are lazily added
    mutable std::vector<DMAG::Move> moves_cache; // cached result of canonicalMoves()

    void applyAction(DMAG::Game& state, DMAG::Card action);
    void markChildAsTerminal();
    const std::vector<DMAG::Move>& canonicalMoves() const;
};
