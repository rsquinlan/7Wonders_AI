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
        void applyAction(int playerIndex, Card card);
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
