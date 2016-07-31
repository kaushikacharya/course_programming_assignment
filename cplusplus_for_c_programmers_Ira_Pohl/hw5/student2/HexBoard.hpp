#ifndef __HexBoard_hpp__
#define __HexBoard_hpp__

//
//Interface for the Hex Board Class
//
//
#include <vector>
#include <string>
#include "graph.hpp"

typedef enum {
    PLAYER_INVALID = 0,
    PLAYER_1,
    PLAYER_2,
}hexPlayer;


class HexBoard: public Graph {

    public:
        HexBoard(int r, int c);
        HexBoard();
        HexBoard(const HexBoard &obj);
        ~HexBoard();
        bool move(hexPlayer p, string node);
        bool move(hexPlayer p, int node);
        vector<int> legalMoves();
        friend ostream& operator<< (ostream &out, const HexBoard& h);
        hexPlayer winner();


    private:
        int p1Head, p1Goal;
        int p2Head, p2Goal;
        vector<int> parent;
        int numRows;
        int numColumns;
        int getVertexFromLocation(string location);
        int findParent(int v);
        void join(int v1, int v2);
};

#endif
