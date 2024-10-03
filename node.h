// Header file: node.h
#include <optional>
#include <memory>
#include <vector>
#include <game.h>

class Node : public std::enable_shared_from_this<Node> {
public:
    Node(DMAG::Game* state, int totalPlayers, int activePlayer, std::shared_ptr<Node> parent);

    void addChild(std::shared_ptr<Node> child);
    std::shared_ptr<Node> selectBestChild() const;
    void update(double value);
    DMAG::Game& getState();
    std::shared_ptr<Node> getParent() const;
    const std::vector<std::shared_ptr<Node>>& getChildren() const;
    int getVisitCount() const;
    double getValue() const;
    bool isFullyExpanded() const;
    std::shared_ptr<Node> expand();
    void setAction(DMAG::Card action);  
    DMAG::Card getAction() const;  
    void setState(DMAG::Game& newState);

private:
    DMAG::Game *state;
    std::shared_ptr<Node> parent;
    int totalPlayers;
    int activePlayer;
    int visitCount;
    double value;
    std::vector<std::shared_ptr<Node>> children;
    DMAG::Card action;
	
    void applyAction(DMAG::Game& state, DMAG::Card action);
};
