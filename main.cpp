#include "mcts.h"

#define NUM_PLAYERS 3

/*
    * Args will be used to tell how to load
    * a new game.
    * We'll use -f filename to tell a file that
    * has information about a previously played game
    * (or just a specific card configuration)
    */
int main(int argc, char **argv)
{
    // Initialize the game state
    DMAG::Game gameState;
    gameState.Init();
    gameState.NewGame(NUM_PLAYERS);

    // Number of players
    int totalPlayers = NUM_PLAYERS;

    // Create an array of MCTS objects for each player
    std::vector<MCTS> mctsPlayers;
    for (int i = 0; i < totalPlayers; ++i) {
        mctsPlayers.emplace_back(gameState, totalPlayers, i);
    }

    // Play the game
    bool gameInProgress = gameState.InGame();
    std::vector<std::shared_ptr<Node>> currentNodes(totalPlayers, nullptr); // Keep track of current nodes for each player

    int count = 1;

    while (gameInProgress) {
        // Loop through each player
        for (int currentPlayer = 0; currentPlayer < totalPlayers; ++currentPlayer) {
            if (!gameState.InGame()) {
                break; // Exit if the game is over
            }

            // If the current node is null, initialize it to the root
            if (currentNodes[currentPlayer] == nullptr) {
                currentNodes[currentPlayer] = mctsPlayers[currentPlayer].getRoot(); // Initialize to the root node
            }

            if(count==20){
                std::cout << "1" << std::endl;
                std::cout << currentNodes[currentPlayer] << std::endl;
                std::cin.ignore();
            }

            // Perform MCTS search to get the best move for the current player
            std::shared_ptr<Node> bestChild = mctsPlayers[currentPlayer].search(5, currentPlayer, currentNodes[currentPlayer]);
            DMAG::Card bestMove = bestChild.get()->getJointAction()[currentPlayer];

            std::cout << "\nBest move: " << bestMove.GetName() << std::endl;
            
            // Apply the best move to the game state
            gameState.playCard(currentPlayer, bestMove);

            // Update the current node to the best child after the move
            currentNodes[currentPlayer] = bestChild; // Initialize to the root node

            // Select the best child node based on the criteria
            currentNodes[currentPlayer] = *std::max_element(
                currentNodes[currentPlayer]->getChildren().begin(),
                currentNodes[currentPlayer]->getChildren().end(),
                [](const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b) {
                    return a->getValue() / a->getVisitCount() < b->getValue() / b->getVisitCount();
                }
            );

        }

        count++;        
        gameState.endTurn();
        
        // Check if the game should continue
        gameInProgress = gameState.InGame();
        if (!gameInProgress) {
            std::cout << "Game over." << std::endl;
            break;
        }
    }

    // Output final game state or results
    std::cout << "Game Results:" << std::endl;
    // Implement a function to print or analyze the game results
    // e.g., print victory points for each player
    gameState.gameEnd();

    return 0;
}
