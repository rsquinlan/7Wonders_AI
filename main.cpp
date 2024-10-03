#include "mcts.h"

#define NUM_PLAYERS 3

/*
    * Args will be used to tell how to load
    * a new game.
    * We'll use -f filename to tell a file that
    * has information about a previously played game
    * (or just a specific card configuration)
    */
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
        for (int currentPlayer = 0; currentPlayer < totalPlayers; ++currentPlayer) {
            if (!gameState.InGame()) {
                break;  // Exit if the game is over
            }

            // Perform MCTS on the player's own tree starting from their current node
            std::shared_ptr<Node> selectedNode = mctsPlayers[currentPlayer].search(5);

            // Get the best move from the selected node
            DMAG::Card bestMove = selectedNode->getAction();
            std::cout << "\nPlayer " << currentPlayer << " best move: " << bestMove.GetName() << std::endl;

            // Apply the best move to the game state
            gameState.playCard(currentPlayer, bestMove);

            // After applying the move, promote the selected node to the root of the current player's tree
            mctsPlayers[currentPlayer].setRoot(selectedNode);  // Shift currentNode to be the new root

            // Update the current node for the player
            currentNodes[currentPlayer] = mctsPlayers[currentPlayer].getRoot();
        }

        gameState.NextTurn();

        for(auto tree : mctsPlayers){
            tree.syncTreeWithGameState(gameState);
        }

        gameInProgress = gameState.InGame();
        gameState.WriteGameStatus();
        for (int playerIndex = 0; playerIndex < totalPlayers; ++playerIndex) {
            // Get the root of the current player's tree
            std::shared_ptr<Node> rootNode = mctsPlayers[playerIndex].getRoot();

            // Access the game state from the root node
            DMAG::Game& treeState = rootNode->getState();
        }
    }


    // Output final game state or results
    std::cout << "Game Results:" << std::endl;
    // Implement a function to print or analyze the game results
    // e.g., print victory points for each player
    gameState.gameEnd();

    return 0;
}
