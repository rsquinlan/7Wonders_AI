#include "mcts.h"
#include <random>

#define NUM_PLAYERS 5

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

    // Play the game
    bool gameInProgress = gameState.InGame();
    std::vector<std::shared_ptr<Node>> currentNodes(totalPlayers, nullptr); // Keep track of current nodes for each player

    // Random number generator for random players
    std::random_device rd;
    std::mt19937 gen(rd());

    while (gameInProgress) {
        for (int currentPlayer = 0; currentPlayer < totalPlayers; ++currentPlayer) {
            if (!gameState.InGame()) {
                break;  // Exit if the game is over
            }

            if (currentPlayer == 0) {
                // Create a new MCTS instance with the current game state
                MCTS mctsPlayer0(gameState, totalPlayers, currentPlayer);

                std::cout << "here1" << std::endl;
                // Perform MCTS search and get the best move for player 0
                std::shared_ptr<Node> selectedNode = mctsPlayer0.search(1000);
                std::cout << "here2" << std::endl;

                // Get the best move from the selected node
                DMAG::Card bestMove = selectedNode->getAction();
                std::cout << "\nPlayer 0 (MCTS) best move: " << bestMove.GetName() << std::endl;

                // Apply the best move to the game state
                gameState.playCard(currentPlayer, bestMove);
            } else {
                // Random move for other players (players 1 and 2)
                std::vector<DMAG::Card> possibleActions = gameState.getPossibleCardsForPlayer(currentPlayer);

                if (!possibleActions.empty()) {
                    std::uniform_int_distribution<> dist(0, possibleActions.size() - 1);
                    DMAG::Card randomMove = possibleActions[dist(gen)];

                    // Apply the random move to the game state
                    gameState.playCard(currentPlayer, randomMove);
                    std::cout << "Player " << currentPlayer << " random move: " << randomMove.GetName() << std::endl;
                } else{
                    gameState.playCard(currentPlayer, gameState.getAllCardsForPlayer(currentPlayer)[0]);
                }
            }
        }

        // Move to the next turn
        gameState.NextTurn();

        // Check if the game is still in progress
        gameInProgress = gameState.InGame();

        // Write the game status to a file (if needed) and print the MCTS tree
        gameState.WriteGameStatus();
    }

    // Output final game state or results
    std::cout << "Game Results:" << std::endl;
    gameState.gameEnd(); // Implement this to output final game results

    return 0;
}
