#include "mcts.h"
#include "utils.h"
#include "filer.h"
#ifdef WITH_NN_PLAYER
#include "nn_player.h"
#endif
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cstdlib>

int main(int argc, char **argv) {
    int game_id                 = 0;
    int num_players             = 5;
    int search_depth            = 500;
    double exploration_constant = 0.2;

    if (argc > 1) game_id     = std::atoi(argv[1]);
    if (argc > 2) {
        num_players = std::atoi(argv[2]);
        if (num_players < 2 || num_players > 7) {
            std::cerr << "Invalid number of players (2-7)." << std::endl;
            return 1;
        }
    }
    if (argc > 3) search_depth        = std::atoi(argv[3]);
    if (argc > 4) exploration_constant = std::atof(argv[4]);
    std::string output_dir = "training";
    if (argc > 5) output_dir = argv[5];
    std::string model_path = "training/model.onnx";
    if (argc > 6) model_path = argv[6];
    // argv[7] = "1" → all players use NN rollout; default (0) → only player 0
    bool nn_all_players = (argc > 7 && std::string(argv[7]) == "1");

#ifdef WITH_NN_PLAYER
    std::unique_ptr<NNPlayer> nn_player;
    if (std::filesystem::exists(model_path)) {
        nn_player = std::make_unique<NNPlayer>(model_path, num_players);
        std::cout << "Loaded NN model: " << model_path
                  << (nn_all_players ? " — all players use NN rollout\n"
                                     : " — player 0 uses NN rollout\n");
    } else {
        std::cout << "No model at " << model_path << " — all players use heuristic\n";
    }
#endif

    std::filesystem::create_directories(output_dir);

    std::ostringstream fname;
    fname << output_dir << "/game_" << std::setw(3) << std::setfill('0') << game_id << ".jsonl";
    initLogFile(fname.str());

    std::cout << "========== GAME " << game_id << " ==========\n";

    DMAG::Game gameState(num_players);
    gameState.Init();
    gameState.NewGame();

    while (gameState.InGame()) {
        // Phase 1: all players decide from identical pre-turn state
        std::vector<std::string>        state_jsons(num_players);
        std::vector<std::vector<float>> state_embeddings(num_players);
        std::vector<int>                old_scores(num_players);
        std::vector<DMAG::Card>         best_moves(num_players);
        std::vector<std::vector<float>> mcts_policies(num_players);

        for (int p = 0; p < num_players; ++p) {
            if (!gameState.InGame()) break;
            state_jsons[p]      = serializeGameState(gameState, p);
            state_embeddings[p] = encodeGameState(gameState, p);
            old_scores[p]       = gameState.getPlayerScore(p);

            RolloutPolicy policy = nullptr;
#ifdef WITH_NN_PLAYER
            if (nn_player && (nn_all_players || p == 0)) {
                policy = [&nn_player](const std::shared_ptr<DMAG::Game>& g, int idx) {
                    return nn_player->getBestMoveNN(g, idx);
                };
            }
#endif
            MCTS mcts(gameState, num_players, p, exploration_constant, policy);
            auto selected     = mcts.search(search_depth, exploration_constant);
            best_moves[p]     = selected->getAction();
            mcts_policies[p]  = mcts.getVisitDistribution();
        }

        // Phase 2: apply moves and log immediately
        for (int p = 0; p < num_players; ++p) {
            if (!gameState.InGame()) break;
            gameState.playCard(p, best_moves[p]);

            Transition t;
            t.state           = std::move(state_jsons[p]);
            t.state_embedding = std::move(state_embeddings[p]);
            t.action          = best_moves[p].GetName();
            t.mcts_policy     = std::move(mcts_policies[p]);
            t.reward          = gameState.getPlayerScore(p) - old_scores[p];
            t.next_state      = serializeGameState(gameState, p);
            t.player_index    = p;
            t.final_score     = 0;
            t.win             = false;
            logTransition(t);
        }

        gameState.endTurn();
    }

    gameState.gameEnd();

    std::vector<int> final_scores(num_players);
    int max_score = -1;
    for (int p = 0; p < num_players; ++p) {
        final_scores[p] = gameState.getPlayerScore(p);
        if (final_scores[p] > max_score) max_score = final_scores[p];
    }
    logResult(final_scores, max_score);

    std::cout << "Game " << game_id << " done -> " << fname.str() << "\n";
    for (int p = 0; p < num_players; ++p)
        std::cout << "  Player " << p << ": " << final_scores[p] << " pts"
                  << (final_scores[p] == max_score ? " [winner]" : "") << "\n";

    return 0;
}
