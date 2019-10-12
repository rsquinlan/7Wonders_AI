#include "../include/player.h"

namespace DMAG {
Player::Player()
{
    this->score = 0;
    this->victory_points = 0;
    this->victory_tokens = 0;
    this->defeat_tokens = 0;
    this->coins = 3; // every player gets 3 coins at the start of each game
    this->wonder_built = false;
}

void Player::CalculateScore(){
    int _score = (this->victory_points + this->victory_tokens + this->coins) - this->defeat_tokens;
    if (wonder_built) {
        int wonder_points = 0; // need to figure out how we'll represent wonders to calculate this!
        _score += wonder_points;
    }
    //Card _c;
    this->score = _score;
}
    
// this method must be called every time a scientific card is played by the player to update
// their earned scientific points (also used to see which symbol is most advantageous in
// cases where the player has won a free symbol through the Babylon board or guild card)
int Player::CalculateScientificScore(int gear, int tablet, int compass)
{
    // the smallest value among the three is the amount of completed sets
    int completed_sets = 0;
    if (gear <= tablet && gear <= compass)    completed_sets = gear;
    if (tablet <= gear && tablet <= compass)  completed_sets = tablet;
    if (compass <= gear && compass <= tablet) completed_sets = compass;

    int points_per_set_completed = 7;
    return (gear * gear) + (tablet * tablet) + (compass * compass) + (completed_sets * points_per_set_completed);
}

int Player::GetScore(){
    return this->score;
}

void Player::Discard(Card c){
    // logic: the card taken at a certain round will be pushed back to the vector,
    //        effectively making it the last one. Therefore, if the player decides do
    //        discard the card he's just taken for three coins, it'll be removed (pop)
    //        from the vector.
    this->cards.pop_back();
    this->coins += 3;
}

void Player::Battle(Player p){
    int current_age_value = 0; // we'll need to get the actual value for this!
    int shields = 0; // we'll also need to calculate the total number of shields from the military cards
    int enemy_shields = 0;
    // win condition
    if (shields > enemy_shields){
        this->victory_tokens += current_age_value;
        // Age I   ->  +1 victory token
        // Age II  ->  +3 victory tokens
        // Age III ->  +5 victory tokens
    } else if (shields < enemy_shields){
        this->defeat_tokens++;
        p.victory_tokens += current_age_value;
    } // else (equal number of shields) none are taken
}

}
