#ifndef GAME_H
#define GAME_H

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <string>
#include <algorithm>
#include <iomanip>
#include <list>
#include <nlohmann/json.hpp>
#include <card.h>
#include <player.h>
#include <wonder.h>
#include <resources.h>
#include <filer.h>

using json = nlohmann::json;

namespace DMAG {

enum class MoveType { BUILD_STRUCTURE, BUILD_WONDER, DISCARD };

// Policy vector index for a move type.
// Layout: [0..74] BUILD_STRUCTURE (by card ID), [75] BUILD_WONDER, [76] DISCARD
inline int policyIndex(MoveType type, int cardId = 0) {
    if (type == MoveType::BUILD_WONDER) return 75;
    if (type == MoveType::DISCARD)      return 76;
    return cardId - 1;  // BUILD_STRUCTURE: 0-based card ID
}

struct Move {
    MoveType type;
    Card card;
    Move(MoveType t, Card c) : type(t), card(c) {}
};

    class Game {
    private:
        std::vector<Wonder*> wonders;
        std::vector<Card> deck[3]; // To be changed to Deck deck[3];
        std::vector<Card> discard_pile; // To be changed to Deck discard_pile;
        Filer fp;

    public:
        std::vector<Player*> player_list;
        unsigned char turn;
        short era;
        unsigned char number_of_players;
        
        Game(int num_players);
        Game(const Game& toCopy);
        ~Game();
        void Init();
        int NewGame();
        void Close();
        void Loop();
        bool InGame();
        void NextTurn();
        void GiveCards();
        void GiveWonders();
        void CreateWonders();
        void CreateDecks();

        Card GetCardByName(std::string name);
        int GetResourceByName(std::string name);
        int getNumberOfPlayers();
        std::vector<int> getScores();

        std::vector<Card> getPossibleCardsForPlayer(int playerIndex) const;
        std::vector<Card> getAllCardsForPlayer(int playerIndex) const;
        std::vector<Move> getAllMovesForPlayer(int playerIndex) const;
        void applyMove(int playerIndex, const Move& move);
        void applyAction(int playerIndex, Card card);  // legacy fallback chain
        bool playCard(int playerIndex, Card card);
        bool buildWonder(int playerIndex, Card card);
        void discardCard(int playerIndex, Card card);
        int getPlayerScore(int playerIndex) const;

        void gameEnd();
        void WriteGameStatus();
        void endTurn();
    };
}

#endif // GAME_H
