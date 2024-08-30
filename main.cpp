#include <game.h>

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
    DMAG::Game g;

    g.Init();
    g.NewGame(NUM_PLAYERS);

    //    g.NextTurn(p, 0); //this function is not completed
    g.Loop();
    g.Close();

    return 0;
}