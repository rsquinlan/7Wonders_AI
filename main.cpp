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
#include <random>

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
    bool nn_all_players = (argc > 7 && std::string(argv[7]) == "1");

#ifdef WITH_NN_PLAYER
    std::unique_ptr<NNPlayer> nn_player;
    if (std::filesystem::exists(model_path)) {
        nn_player = std::make_unique<NNPlayer>(model_path, num_players);
        std::cout << "Loaded NN model: " << model_path
                  << (nn_all_players ? " — all players use NN\n"
                                     : " — player 0 uses NN\n");
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

    std::mt19937 rng(std::random_device{}());

    // Persistent per-player MCTS trees for tree reuse across turns (NN mode only).
    std::vector<std::unique_ptr<MCTS>> trees(num_players);

    while (gameState.InGame()) {
        std::vector<std::string>        state_jsons(num_players);
        std::vector<std::vector<float>> state_embeddings(num_players);
        std::vector<int>                old_scores(num_players);
        std::vector<DMAG::Card>         best_moves(num_players);
        std::vector<DMAG::MoveType>     best_move_types(num_players, DMAG::MoveType::BUILD_STRUCTURE);
        std::vector<std::vector<float>> mcts_policies(num_players);

        // Snapshot state for all players before any moves are applied.
        for (int p = 0; p < num_players; ++p) {
            state_jsons[p]      = serializeGameState(gameState, p);
            state_embeddings[p] = encodeGameState(gameState, p);
            old_scores[p]       = gameState.getPlayerScore(p);
        }

        if (search_depth == 0) {
            // ── Random mode: one uniform sample per player, no MCTS ────────────
            for (int p = 0; p < num_players; ++p) {
                auto allMoves = gameState.getAllMovesForPlayer(p);
                std::uniform_int_distribution<int> dist(0, allMoves.size() - 1);
                const auto& chosen = allMoves[dist(rng)];
                best_moves[p]      = chosen.card;
                best_move_types[p] = chosen.type;
                // Uniform policy over canonical slots (BUILD_STRUCTURE per card, one WONDER, one DISCARD)
                std::vector<float> uniform_policy(POLICY_SIZE, 0.0f);
                bool hasWonder = false, hasDiscard = false;
                int nSlots = 0;
                for (const auto& m : allMoves) {
                    if (m.type == DMAG::MoveType::BUILD_STRUCTURE) nSlots++;
                    else if (m.type == DMAG::MoveType::BUILD_WONDER && !hasWonder) { hasWonder = true; nSlots++; }
                    else if (m.type == DMAG::MoveType::DISCARD      && !hasDiscard) { hasDiscard = true; nSlots++; }
                }
                float w = nSlots > 0 ? 1.0f / nSlots : 0.0f;
                hasWonder = hasDiscard = false;
                for (const auto& m : allMoves) {
                    if (m.type == DMAG::MoveType::BUILD_WONDER && !hasWonder) { hasWonder = true; uniform_policy[75] = w; }
                    else if (m.type == DMAG::MoveType::DISCARD && !hasDiscard) { hasDiscard = true; uniform_policy[76] = w; }
                    else if (m.type == DMAG::MoveType::BUILD_STRUCTURE) uniform_policy[policyIndex(m.type, m.card.GetId())] = w;
                }
                mcts_policies[p] = std::move(uniform_policy);
            }
        }
#ifdef WITH_NN_PLAYER
        else if (nn_player && nn_all_players) {
            // ── NN batched mode: interleave all players' iterations so every  ──
            // ── NN call processes num_players states at once. Trees are reused ──
            // ── across turns (root advances to the chosen child each turn).   ──

            // Fresh tree each turn — avoids accumulating stale game-state copies
            // across turns. Tree reuse can be re-enabled once memory budget allows.
            for (int p = 0; p < num_players; ++p) {
                trees[p] = std::make_unique<MCTS>(
                    gameState, num_players, p, exploration_constant, nullptr, nullptr);
            }

            // Run search_depth iterations with batched NN evaluation.
            for (int iter = 0; iter < search_depth; ++iter) {
                std::vector<std::shared_ptr<Node>> leaves;
                std::vector<std::vector<float>>    embeddings;
                std::vector<int>                   active;

                for (int p = 0; p < num_players; ++p) {
                    if (trees[p]->getRoot()->isFullyTerminal()) continue;
                    auto leaf = trees[p]->selectAndExpand();
                    leaves.push_back(leaf);
                    embeddings.push_back(encodeGameState(*leaf->getState(), p));
                    active.push_back(p);
                }

                if (embeddings.empty()) break;

                // Single batched NN call for all active players.
                std::vector<std::vector<float>> policies;
                std::vector<float>              values;
                nn_player->inferBatch(embeddings, policies, values);

                for (int i = 0; i < static_cast<int>(leaves.size()); ++i)
                    trees[active[i]]->finishStep(leaves[i], policies[i], values[i]);
            }

            // Extract best moves and advance roots for next turn (tree reuse).
            for (int p = 0; p < num_players; ++p) {
                auto best       = trees[p]->getBestChild();
                best_moves[p]      = best->getAction();
                best_move_types[p] = best->getMoveType();
                mcts_policies[p]   = trees[p]->getVisitDistribution();
            }
        }
#endif
        else {
            // ── Heuristic MCTS (or single-player NN): one tree per player ──────
            for (int p = 0; p < num_players; ++p) {
#ifdef WITH_NN_PLAYER
                EvalFn evalFn = nullptr;
                if (nn_player && p == 0) {
                    evalFn = [&nn_player](const DMAG::Game& g, int idx) {
                        std::vector<float> policy;
                        float value;
                        nn_player->infer(encodeGameState(g, idx), policy, value);
                        return std::make_pair(policy, value);
                    };
                }
                MCTS mcts(gameState, num_players, p, exploration_constant, nullptr, evalFn);
#else
                MCTS mcts(gameState, num_players, p, exploration_constant, nullptr, nullptr);
#endif
                auto selected      = mcts.search(search_depth, exploration_constant);
                best_moves[p]      = selected->getAction();
                best_move_types[p] = selected->getMoveType();
                mcts_policies[p]   = mcts.getVisitDistribution();
            }
        }

        // ── Apply moves and log transitions ────────────────────────────────────
        for (int p = 0; p < num_players; ++p) {
            if (!gameState.InGame()) break;
            gameState.applyMove(p, DMAG::Move(best_move_types[p], best_moves[p]));

            Transition t;
            t.state           = std::move(state_jsons[p]);
            t.state_embedding = std::move(state_embeddings[p]);
            t.action          = (best_move_types[p] == DMAG::MoveType::BUILD_WONDER ? "WONDER:" :
                                  best_move_types[p] == DMAG::MoveType::DISCARD     ? "DISCARD:" : "")
                                 + best_moves[p].GetName();
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
