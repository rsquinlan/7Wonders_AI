// Header file: node.h
#include <optional>
#include <memory>
#include <vector>
#include <game.h>

class Node : public std::enable_shared_from_this<Node> {
public:
    Node(DMAG::Game* state, int totalPlayers, std::shared_ptr<Node> parent);

    void addChild(std::shared_ptr<Node> child);
    std::shared_ptr<Node> selectBestChild() const;
    void update(double value);
    DMAG::Game& getState();
    std::shared_ptr<Node> getParent() const;
    const std::vector<std::shared_ptr<Node>>& getChildren() const;
    int getVisitCount() const;
    double getValue() const;
    bool isLeaf() const;
    void expand();
    void setJointAction(const std::vector<DMAG::Card>& jointAction);  
    const std::vector<DMAG::Card>& getJointAction() const;  
    void setState(const DMAG::Game& newState);

private:
    DMAG::Game *state;
    std::shared_ptr<Node> parent;
    int totalPlayers;
    int visitCount;
    double value;
    std::vector<std::shared_ptr<Node>> children;
    std::vector<DMAG::Card> jointAction;  // Store joint action
	
    void applyJointAction(DMAG::Game& state, const std::vector<DMAG::Card>& jointAction) const;

	// Declare the function to generate combinations
    void generateCombinations(const std::vector<std::vector<DMAG::Card>>& allPlayersCards, 
                              std::vector<std::vector<DMAG::Card>>& jointActions,
                              std::vector<DMAG::Card> currentCombination = {}, 
                              int depth = 0) const;
};
