#include "mcts.h"
#include <random>
#include <iostream>
#include <cstdlib>

void PrintPlayerStats(DMAG::Player* player) {
    std::cout << "\nPlayer " << player->GetId() << " Stats:\n";
    std::cout << "  Total Score: " << player->CalculateScore() << "\n";
    std::cout << "  Victory Points Breakdown:\n";
    std::cout << "    Civilian: " << player->CalculateCivilianScore() << "\n";
    std::cout << "    Commercial: " << player->CalculateCommercialScore() << "\n";
    std::cout << "    Guild: " << player->CalculateGuildScore() << "\n";
    std::cout << "    Military: " << player->CalculateMilitaryScore() << "\n";
    std::cout << "    Scientific: " << player->CalculateScientificScore() << "\n";
    std::cout << "    Wonder: " << player->CalculateWonderScore() << "\n";
    std::cout << "  Card Type Counts:\n";
    std::cout << "    Civilian Cards: " << player->CalculateAmountCivilianCards() << "\n";
    std::cout << "    Military Cards: " << player->CalculateAmountMilitaryCards() << "\n";
    std::cout << "    Scientific Cards: " << player->CalculateAmountScientificCards() << "\n";
    std::cout << "    Guild Cards: " << player->CalculateAmountGuildCards() << "\n";
    std::cout << "    Commercial Cards: " << player->CalculateAmountCommercialCards() << "\n";
    std::cout << "  Resource Production:\n";
    for (const auto& [resource, amount] : player->GetResources()) {
        std::cout << "    Resource " << resource << ": " << amount << "\n";
    }
    std::cout << "  Victory Tokens: " << player->GetShields() << "\n";
    std::cout << "  Defeat Tokens: " << player->GetDefeatTokens() << "\n";
    std::cout << "  Wonder Completion: "
              << (player->CanBuildWonder() ? "Incomplete" : "Complete") << "\n";
}

int main(int argc, char **argv){
    DMAG::Game g(3);

    g.Init();
    g.NewGame();

    //    g.NextTurn(p, 0); //this function is not completed
    g.Loop();
    g.Close();

    return 0;
}

// int main(int argc, char **argv) {
//     // Default parameters
//     int num_players = 3;
//     int mode = 1;
//     int search_depth = 500; // Default search depth
//     double exploration_constant = 0.2; // Default exploration constant

//     if (argc > 1) {
//         num_players = std::atoi(argv[1]);
//         if (num_players < 2 || num_players > 7) {
//             std::cerr << "Invalid number of players. Please enter a number between 2 and 7." << std::endl;
//             return 1;
//         }
//     }

//     if (argc > 2) {
//         mode = std::atoi(argv[2]);
//     }

//     if (argc > 3) {
//         search_depth = std::atoi(argv[3]);
//         if (search_depth <= 0) {
//             std::cerr << "Invalid search depth. Please enter a positive number." << std::endl;
//             return 1;
//         }
//     }

//     if (argc > 4) {
//         exploration_constant = std::atof(argv[4]);
//         if (exploration_constant <= 0 || exploration_constant > 1) {
//             std::cerr << "Invalid exploration constant. Please enter a value between 0 and 1." << std::endl;
//             return 1;
//         }
//     }

//     DMAG::Game gameState(num_players);
//     gameState.Init();
//     gameState.NewGame();

//     bool gameInProgress = gameState.InGame();
//     std::random_device rd;
//     std::mt19937 gen(rd());

//     while (gameInProgress) {
//         for (int currentPlayer = 0; currentPlayer < num_players; ++currentPlayer) {
//             if (!gameState.InGame()) break;

//             if (currentPlayer == 0) {
//                 MCTS mctsPlayer(gameState, num_players, currentPlayer);
//                 std::shared_ptr<Node> selectedNode = mctsPlayer.search(search_depth, exploration_constant);
//                 DMAG::Card bestMove = selectedNode->getAction();
//                 gameState.playCard(currentPlayer, bestMove);
//             } else {
//                 auto possibleActions = gameState.getPossibleCardsForPlayer(currentPlayer);
//                 if (!possibleActions.empty()) {
//                     std::uniform_int_distribution<> dist(0, possibleActions.size() - 1);
//                     DMAG::Card randomMove = possibleActions[dist(gen)];
//                     gameState.playCard(currentPlayer, randomMove);
//                 }
//             }
//         }

//         gameState.endTurn();
//         gameInProgress = gameState.InGame();
//     }

//     gameState.gameEnd();

//     PrintPlayerStats(gameState.player_list[0]);
//     return 0;
// }