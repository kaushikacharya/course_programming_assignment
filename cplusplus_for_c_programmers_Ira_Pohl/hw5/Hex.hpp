#ifndef Hex_HPP
#define Hex_HPP

#include "Board.hpp"
#include "Graph.hpp"
#include "PriorityQueue.hpp"
#include <limits>
#include <algorithm>
#include <cassert>

const int NSimulation = 1000;
enum MoveType {Valid, AlreadyOccupied, Impossible};

template<typename T>
class Hex
{
public:
	Hex(T size);
	~Hex();
public:
	MoveType add_move(T row, col_t col, player_t player);
	void delete_move(T row, col_t col);
	player_t position_player(T row, col_t col);
	void display_board();
	bool is_winning_move(player_t player);
	T getNextBestMove(player_t player);
private:
	// Player #1: (Blue): west to east
	// Player #2: (Red): north to south
	T pos_virtual_start_blue();
	T pos_virtual_end_blue();
	T pos_virtual_start_red();
	T pos_virtual_end_red();

	bool shortest_path(T source, T target, player_t player);
	void create_graph();
	bool is_pos_on_board(T pos);
	void populate_legal_move_list(player_t player, std::vector<T>& vecLegalMove);
private:
	T size_; // Board dimension: size_ * size_
	Board<T> board_;
	// Graph will have (size_*size_ + 4) vertices. 
	// Extra 2 for virtual start and virtual goal for each player.
	Graph<T> graph_;
	T nMove_; // keeps count of player #1
private:
	// traversal related
	std::vector<bool> vecVisited_;
	std::vector<T> vecCost_;
};

template<typename T>
Hex<T>::Hex(T size)
: size_(size)
, board_(size)
, graph_(size*size+4)
, nMove_(0)
{
	create_graph();

	vecVisited_.reserve(graph_.nVertex());
	vecCost_.reserve(graph_.nVertex());

	//initialize
	for (T vtx_i = 0; vtx_i != graph_.nVertex(); ++vtx_i)
	{
		vecVisited_.push_back(false);
		vecCost_.push_back(std::numeric_limits<T>::max());
	}
}

template<typename T>
Hex<T>::~Hex()
{
}

template<typename T>
void Hex<T>::create_graph()
{
	// vertices [0,1,....,(size*size-1)] belong to the on board positions.
	// Player #1: (Blue) (West to East)
	//			vertex [size*size] : virtual start
	//			vertex [size*size+1] : virtual end
	// Player #2: (Red) (North to South)
	//			vertex [size*size+2] : virtual start
	//			vertex [size*size+3] : virtual end

	// edges from virtual start(Blue) to the first column positions
	// edges from last column positions to virtual end(Blue)
	for (T row_i = 0; row_i != size_; ++row_i)
	{
		graph_.add_edge(pos_virtual_start_blue(), row_i*size_);
		graph_.add_edge((row_i+1)*size_-1, pos_virtual_end_blue());
	}
	
	//T pos_virtual_start_red = size_*size_ + 2;
	//T pos_virtual_end_red = size_*size_ + 3;
	for (T col_i = 0; col_i != size_; ++col_i)
	{
		graph_.add_edge(pos_virtual_start_red(), col_i);
		graph_.add_edge((size_-1)*size_ + col_i, pos_virtual_end_red());
	}

	//Now assigning the edges inside the board
	for (T row_i = 0; row_i != size_; ++row_i)
	{
		for (T col_i = 0; col_i != size_; ++col_i)
		{
			T pos_source = row_i*size_ + col_i;
			T pos_target;
			
			// edges to pos at same row
			if (col_i > 0)
			{
				pos_target = row_i*size_ + col_i - 1;
				graph_.add_edge(pos_source, pos_target);
			}
			if (col_i < (size_-1))
			{
				pos_target = row_i*size_ + col_i + 1;
				graph_.add_edge(pos_source, pos_target);
			}

			//edges to pos at previous row
			if (row_i > 0)
			{
				pos_target = (row_i-1)*size_ + col_i;
				graph_.add_edge(pos_source, pos_target);
				if (col_i < (size_-1))
				{
					pos_target = (row_i-1)*size_ + col_i+1;
					graph_.add_edge(pos_source, pos_target);
				}
			}

			//edges to pos at next row
			if (row_i < (size_-1))
			{
				if (col_i > 0)
				{
					pos_target = (row_i+1)*size_ + col_i - 1;
					graph_.add_edge(pos_source, pos_target);
				}
				pos_target = (row_i+1)*size_ + col_i;
				graph_.add_edge(pos_source, pos_target);
			}
		}
	}

}

template<typename T>
bool Hex<T>::shortest_path(T source, T target, player_t player)
{
	PriorityQueue<T,T> priorityQueue;
	// Apart from board positions, we also add source and target in the priority queue.
	// source & target: These are virtual home and goal.

	//Initialize cost of vertices as infinite except the source vertex
	for (T vtx_i = 0; vtx_i != graph_.nVertex(); ++vtx_i)
	{
		vecVisited_[vtx_i] = false;
		if (vtx_i == source)
		{
			vecCost_[vtx_i] = 0;
			priorityQueue.insert_element(vtx_i,0);
		} 
		else
		{
			vecCost_[vtx_i] = std::numeric_limits<T>::max();

			if ( ( is_pos_on_board(vtx_i) && (board_.position_player(vtx_i) == player) ) ||
				(vtx_i == target) )
			{
				priorityQueue.insert_element(vtx_i,std::numeric_limits<T>::max());
			}
		}
	}

	bool reachedTarget = false;
	while (!priorityQueue.empty())
	{
		std::pair<T,T> topElem = priorityQueue.top();
		T vtxTop = topElem.first;
		T costTop = topElem.second;
		vecVisited_[vtxTop] = true;
		priorityQueue.remove_element(vtxTop,costTop);

		if (costTop == std::numeric_limits<T>::max())
		{
			break; // reaching here means path from source to target is not present
		}
		if (vtxTop == target)
		{
			reachedTarget = true;
			break;
		}

		std::vector<T> neighbors = graph_.neighbors(vtxTop);
		for (std::vector<T>::iterator it = neighbors.begin(); it != neighbors.end(); ++it)
		{
			if (vecVisited_[*it])
			{
				continue;
			}
			// For a board position consider this neighbor only if the current player has placed
			// his/her hex on this position.
			
			if (is_pos_on_board(*it))
			{
				if (board_.position_player(*it) != player)
				{
					continue;
				}
			}
			else if ((*it) != target)
			{
				continue;
			}

			T newCost = vecCost_[vtxTop] + 1;
			if (newCost < vecCost_[*it])
			{
				priorityQueue.change_priority(*it,vecCost_[*it],newCost);
				vecCost_[*it] = newCost;
			}
		}
	}

	return reachedTarget;
}

template<typename T>
bool Hex<T>::is_pos_on_board(T pos)
{
	return pos < graph_.nVertex()-4; //i.e. pos < size_*size_
}

template<typename T>
T Hex<T>::pos_virtual_start_blue()
{
	return size_*size_;
}

template<typename T>
T Hex<T>::pos_virtual_end_blue()
{
	return size_*size_ + 1;
}

template<typename T>
T Hex<T>::pos_virtual_start_red()
{
	return size_*size_ + 2;
}

template<typename T>
T Hex<T>::pos_virtual_end_red()
{
	return size_*size_ + 3;
}

template<typename T>
MoveType Hex<T>::add_move(T row, col_t col, player_t player)
{
	if ( (row > size_) || (col > ('A'+size_-1)) )
	{
		return Impossible;
	}
	bool flag = board_.add_move(row, col, player);
	
	if (flag)
	{
		if (player == 1)
		{
			++nMove_;
		}
		return Valid;
	}
	else
	{
		return AlreadyOccupied;
	}
}

template<typename T>
player_t Hex<T>::position_player(T row, col_t col)
{
	T pos = (row-1)*size_ + (col - 'A');
	return board_.position_player(pos);
}

template<typename T>
void Hex<T>::delete_move(T row, col_t col)
{
	player_t player = position_player(row,col);
	if (player == 1)
	{
		--nMove_;
	}
	board_.delete_move(row,col);
}

template<typename T>
void Hex<T>::display_board()
{
	board_.display();
}

template<typename T>
bool Hex<T>::is_winning_move(player_t player)
{
	//check for winning move only after getting atleast size_ moves
	if (nMove_ < size_)
	{
		return false;
	}
	if (player == 1)
	{
		return shortest_path(pos_virtual_start_blue(), pos_virtual_end_blue(), player);
	}
	else
	{
		return shortest_path(pos_virtual_start_red(), pos_virtual_end_red(), player);
	}
}

template<typename T>
T Hex<T>::getNextBestMove(player_t player)
{
	//First list out all the possible legal moves.
	//For each move:
	//			Simulate N times to calculate the wins if this move gets selected.
	std::vector<T> vec_legal_move;
	populate_legal_move_list(player,vec_legal_move);

	//dummy initializations
	T best_move = vec_legal_move[0];
	T best_win_count = 0;

	for (T best_move_i = 0; best_move_i != vec_legal_move.size(); ++best_move_i)
	{
		{
			T row = (vec_legal_move[best_move_i]/size_)+1; //Hex board row is 1-indexed
			col_t col = vec_legal_move[best_move_i]%size_ + 'A';
			MoveType moveType = add_move(row,col,player);
			assert((moveType == MoveType::Valid) && "move tagged as non valid. check");
		}
		//Now we will run Monte-Carlo assuming we have selected vec_legal_move[best_move_i] for the current player.
		std::vector<T> vecShuffledMove;
		vecShuffledMove.reserve(vec_legal_move.size()-1);
		for (T move_i = 0; move_i != vec_legal_move.size(); ++move_i)
		{
			if (move_i != best_move_i)
			{
				vecShuffledMove.push_back(vec_legal_move[move_i]);
			}
		}

		//After every simulation we also need to reset the board.
		//Also reset the vec_legal_move[best_move_i] after completion of monte-carlo simulation for this move.

		T winCount = 0;
		player_t otherPlayer = (player == 1) ? 2 : 1;
		for (T simulation_i = 0; simulation_i != NSimulation; ++simulation_i)
		{
			std::random_shuffle(vecShuffledMove.begin(),vecShuffledMove.end());
			//iterate over vecShuffledMove and keep assigning the position to the two players alternatively.
			for (std::vector<T>::iterator it = vecShuffledMove.begin(); it != vecShuffledMove.end(); ++it)
			{
				T pos = (*it);
				T row = (pos/size_)+1; //Hex board row is 1-indexed
				col_t col = pos%size_ + 'A';
				player_t curMovePlayer = (std::distance(vecShuffledMove.begin(),it)%2 == 0) ? otherPlayer : player;
				MoveType moveType = add_move(row,col,curMovePlayer);
				assert((moveType == MoveType::Valid) && "move tagged as non valid. check");
			}
			//Now board has been filled up.
			//check the winner of this simulation.
			bool flagSuccessCurPlayer = is_winning_move(player);
			if (flagSuccessCurPlayer)
			{
				++winCount;
			}
			//checking individually for both players is for debugging only
			bool flagSuccessOtherPlayer = is_winning_move(otherPlayer);
			assert((flagSuccessCurPlayer != flagSuccessOtherPlayer) && "One and only one must win");

			//Now empty all the positions of the hex board that were filled up in this simulation.
			for (std::vector<T>::iterator it = vecShuffledMove.begin(); it != vecShuffledMove.end(); ++it)
			{
				T pos = (*it);
				T row = (pos/size_)+1; //Hex board row is 1-indexed
				col_t col = pos%size_ + 'A';
				delete_move(row,col);
			}
		}

		if (best_win_count < winCount)
		{
			best_win_count = winCount;
			best_move = vec_legal_move[best_move_i];
		}

		//Now remove the move which we considered as possible best move in this iteration.
		{
			T row = vec_legal_move[best_move_i]/size_ + 1;
			col_t col = vec_legal_move[best_move_i]%size_ + 'A';
			delete_move(row,col);
		}
	}

	return best_move;
}

template<typename T>
void Hex<T>::populate_legal_move_list(player_t player, std::vector<T>& vecLegalMove)
{
	if (player == 1)
	{
		vecLegalMove.reserve(size_*size_ - 2*nMove_);
	}
	else
	{
		vecLegalMove.reserve(size_*size_ - 2*nMove_ + 1);
	}

	for (T row_i = 0; row_i != size_; ++row_i)
	{
		for (T col_i = 0; col_i != size_; ++col_i)
		{
			if (board_.is_legal_move(row_i+1,'A'+col_i))
			{
				vecLegalMove.push_back(row_i*size_+col_i);
			}
		}
	}
}

#endif