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

#define NUM_PLAYERS 3

using json = nlohmann::json;

namespace DMAG {

    class Game {
    private:
        std::vector<Player*> player_list;
        unsigned char number_of_players;
        short era;
        unsigned char turn;
        std::vector<Wonder*> wonders;
        std::vector<Card> deck[3]; // To be changed to Deck deck[3];
        std::vector<Card> discard_pile; // To be changed to Deck discard_pile;
        Filer fp;

    public:
        Game();
        Game(const Game& toCopy);
        void Init();
        int NewGame(int _players);
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

        std::vector<Card> getPossibleCardsForPlayer(int playerIndex);
        std::vector<Card> getAllCardsForPlayer(int playerIndex);
        void applyAction(int playerIndex, Card card);
        bool playCard(int playerIndex, Card card);
        bool buildWonder(int playerIndex, Card card);
        void discardCard(int playerIndex, Card card);
        int getPlayerScore(int playerIndex);

        void gameEnd();
        void WriteGameStatus();
        void endTurn();
    };
}

#endif // GAME_H
