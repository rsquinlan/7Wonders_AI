#include "mcts.h"
#include "utils.h"
#include "game.h"
#include <fstream>

using json = nlohmann::json;

// Ordered resource IDs matching RESOURCE enum (0-11)
static const int RESOURCE_ORDER[N_RESOURCES] = {
    RESOURCE::wood, RESOURCE::ore, RESOURCE::clay, RESOURCE::stone,
    RESOURCE::loom, RESOURCE::glass, RESOURCE::papyrus,
    RESOURCE::gear, RESOURCE::compass, RESOURCE::tablet,
    RESOURCE::coins, RESOURCE::shields
};

static const char* RESOURCE_NAMES[N_RESOURCES] = {
    "wood", "ore", "clay", "stone",
    "loom", "glass", "papyrus",
    "gear", "compass", "tablet",
    "coins", "shields"
};

// ─── Logging ──────────────────────────────────────────────────────────────────

static std::ofstream g_log_file;

void initLogFile(const std::string& path) {
    if (g_log_file.is_open()) g_log_file.close();
    g_log_file.open(path, std::ios::out);
    if (!g_log_file)
        std::cerr << "Failed to open log file: " << path << std::endl;
}

void logTransition(const Transition& t) {
    json j;
    j["state"]           = t.state;
    j["action"]          = t.action;
    j["reward"]          = t.reward;
    j["next_state"]      = t.next_state;
    j["final_score"]     = t.final_score;
    j["win"]             = t.win;
    j["player_index"]    = t.player_index;
    j["state_embedding"] = t.state_embedding;
    j["mcts_policy"]     = t.mcts_policy;
    g_log_file << j.dump() << std::endl;
}

void logResult(const std::vector<int>& final_scores, int max_score) {
    json j;
    j["type"]         = "result";
    j["final_scores"] = final_scores;
    j["max_score"]    = max_score;
    g_log_file << j.dump() << std::endl;
}

// ─── State serialization (human-readable JSON) ────────────────────────────────

bool isHandKnown(const DMAG::Game& game, int playerIndex, int opponentIndex) {
    int num_players = game.player_list.size();
    int era         = game.era;
    int turn        = game.turn;
    bool passLeft   = (era == 1 || era == 3);
    int knownDepth  = turn - 1;
    if (knownDepth <= 0) return false;

    for (int d = 1; d <= knownDepth; ++d) {
        int idx = passLeft ? (playerIndex + d) % num_players
                           : (playerIndex - d + num_players) % num_players;
        if (idx == opponentIndex) return true;
    }
    return false;
}

std::vector<std::string> getKnownHand(const DMAG::Game& game, int playerIndex, int opponentIndex) {
    if (!isHandKnown(game, playerIndex, opponentIndex)) return {};
    std::vector<std::string> hand;
    for (const auto& card : game.getAllCardsForPlayer(opponentIndex))
        hand.push_back(card.GetName());
    return hand;
}

std::string serializeGameState(const DMAG::Game& game, int playerIndex) {
    json j;
    j["era"]    = game.era;
    j["turn"]   = game.turn;
    j["player"] = playerIndex;
    j["score"]  = game.getPlayerScore(playerIndex);

    std::vector<std::string> hand;
    for (const auto& card : game.getAllCardsForPlayer(playerIndex))
        hand.push_back(card.GetName());
    j["hand"] = hand;

    std::map<int, int> resources = game.player_list[playerIndex]->GetResources();
    json res_j;
    for (int i = 0; i < N_RESOURCES; ++i)
        res_j[RESOURCE_NAMES[i]] = resources[RESOURCE_ORDER[i]];
    j["resources"] = res_j;

    json opponents = json::array();
    for (int i = 0; i < (int)game.player_list.size(); ++i) {
        if (i == playerIndex) continue;
        json opp;
        opp["wonder"] = game.player_list[i]->GetBoard()->GetName();

        std::vector<std::string> played;
        for (const auto& card : game.player_list[i]->GetPlayedCards())
            played.push_back(card.GetName());
        opp["played"] = played;
        opp["hand"]   = getKnownHand(game, playerIndex, i);
        opponents.push_back(opp);
    }
    j["opponents"] = opponents;

    return j.dump();
}

// ─── State embedding (fixed-size float vector) ───────────────────────────────
//
// Layout (for P players, total = 3 + 7 + SELF_EMBED_SIZE + (P-1)*OPP_EMBED_SIZE):
//   [0..2]   era one-hot (3)
//   [3..9]   turn-within-era one-hot (7)
//   Self block (261):
//     wonder_id one-hot (14)
//     wonder_stage one-hot (4)
//     hand one-hot (75)
//     played one-hot (75)
//     playable one-hot (75)
//     resources (12)
//     score components (6): civilian,commercial,guild,military,scientific,wonder
//   Per-opponent block (180 each):
//     wonder_id one-hot (14)
//     wonder_stage one-hot (4)
//     played one-hot (75)
//     resources (12)
//     known_hand one-hot (75)

static void setCardBits(std::vector<float>& v, int offset, const std::vector<DMAG::Card>& cards) {
    for (const auto& c : cards) {
        int idx = c.GetId() - 1;
        if (idx >= 0 && idx < N_CARDS)
            v[offset + idx] = 1.0f;
    }
}

std::vector<float> encodeGameState(const DMAG::Game& game, int playerIndex) {
    int num_players = game.player_list.size();
    int total       = stateEmbedSize(num_players);
    std::vector<float> v(total, 0.0f);
    int pos = 0;

    // Era one-hot (3)
    int era_idx = std::max(0, std::min(2, game.era - 1));
    v[pos + era_idx] = 1.0f;
    pos += 3;

    // Turn-within-era one-hot (7)
    int turn_idx = std::max(0, std::min(6, game.turn % 7));
    v[pos + turn_idx] = 1.0f;
    pos += 7;

    // ── Self block ────────────────────────────────────────────────────────────
    DMAG::Player* self = game.player_list[playerIndex];

    // Wonder id one-hot (14)
    int wid = self->GetBoard()->GetId();
    if (wid >= 0 && wid < N_WONDERS) v[pos + wid] = 1.0f;
    pos += N_WONDERS;

    // Wonder stage one-hot (4)
    int stage = std::max(0, std::min(3, self->GetBoard()->GetStage()));
    v[pos + stage] = 1.0f;
    pos += 4;

    // Hand (75)
    setCardBits(v, pos, game.getAllCardsForPlayer(playerIndex));
    pos += N_CARDS;

    // Played (75)
    setCardBits(v, pos, self->GetPlayedCards());
    pos += N_CARDS;

    // Playable (75)
    setCardBits(v, pos, self->GetPlayableCards());
    pos += N_CARDS;

    // Resources (12)
    auto res = self->GetResources();
    for (int i = 0; i < N_RESOURCES; ++i)
        v[pos + i] = static_cast<float>(res[RESOURCE_ORDER[i]]);
    pos += N_RESOURCES;

    // Score components (6)
    v[pos + 0] = static_cast<float>(self->CalculateCivilianScore());
    v[pos + 1] = static_cast<float>(self->CalculateCommercialScore());
    v[pos + 2] = static_cast<float>(self->CalculateGuildScore());
    v[pos + 3] = static_cast<float>(self->CalculateMilitaryScore());
    v[pos + 4] = static_cast<float>(self->CalculateScientificScore());
    v[pos + 5] = static_cast<float>(self->CalculateWonderScore());
    pos += N_SCORE_COMPS;

    // ── Opponent blocks ───────────────────────────────────────────────────────
    for (int i = 0; i < num_players; ++i) {
        if (i == playerIndex) continue;
        DMAG::Player* opp = game.player_list[i];

        // Wonder id one-hot (14)
        int owid = opp->GetBoard()->GetId();
        if (owid >= 0 && owid < N_WONDERS) v[pos + owid] = 1.0f;
        pos += N_WONDERS;

        // Wonder stage one-hot (4)
        int ostage = std::max(0, std::min(3, opp->GetBoard()->GetStage()));
        v[pos + ostage] = 1.0f;
        pos += 4;

        // Played (75)
        setCardBits(v, pos, opp->GetPlayedCards());
        pos += N_CARDS;

        // Resources (12)
        auto ores = opp->GetResources();
        for (int r = 0; r < N_RESOURCES; ++r)
            v[pos + r] = static_cast<float>(ores[RESOURCE_ORDER[r]]);
        pos += N_RESOURCES;

        // Known hand (75) — zeros if hand is not visible
        if (isHandKnown(game, playerIndex, i))
            setCardBits(v, pos, game.getAllCardsForPlayer(i));
        pos += N_CARDS;
    }

    return v;
}

// ─── Debug printing ───────────────────────────────────────────────────────────

void PrintPlayerStats(DMAG::Player* player) {
    std::cout << "\nPlayer " << player->GetId() << " Stats:\n";
    std::cout << "  Total Score: " << player->CalculateScore() << "\n";
    std::cout << "  Victory Points:\n";
    std::cout << "    Civilian:   " << player->CalculateCivilianScore()   << "\n";
    std::cout << "    Commercial: " << player->CalculateCommercialScore() << "\n";
    std::cout << "    Guild:      " << player->CalculateGuildScore()      << "\n";
    std::cout << "    Military:   " << player->CalculateMilitaryScore()   << "\n";
    std::cout << "    Scientific: " << player->CalculateScientificScore() << "\n";
    std::cout << "    Wonder:     " << player->CalculateWonderScore()     << "\n";
}
