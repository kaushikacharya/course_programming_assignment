#ifndef __AIHexPlayer_hpp__
#define __AIHexPlayer_hpp__

//
//Interface for the Hex Board Class
//
//
#include <vector>
#include <string>
#include "HexBoard.hpp"

class AIHexPlayer {

    public:
        AIHexPlayer(hexPlayer p);
        AIHexPlayer();
        ~AIHexPlayer();
        int computeMove(HexBoard h);

    private:

        hexPlayer player;
        //given the current board state and start move
        //for AI player, simulate the game and return
        // player who wins.
        hexPlayer simulateGame(int startMove, HexBoard h);
};

#endif
