#ifndef Board_HPP
#define Board_HPP

#include <iostream>
#include <vector>

typedef unsigned char player_t;
typedef unsigned char col_t;

template<typename T>
class Board
{
public:
	Board(T size);
	~Board();
public:
	bool is_legal_move(T row, col_t col);
	bool add_move(T row, col_t col, player_t player);
	void delete_move(T row, col_t col);
	player_t position_player(T pos);
	void display();
private:
	// Board dimension: size_ * size_
	T size_;
	// 2d array has been implemented using 1d array
	// Notation for value:
	// a) 0 - none of the players have picked the position yet
	// b) 1 - player #1 has picked the position
	// c) 2 - player #2 has picked the position
	std::vector<player_t> vecPos_;
};

template<typename T>
Board<T>::Board(T size)
: size_(size)
{
	vecPos_.reserve(size_*size_);
	//initialize empty board
	for (T pos_i = 0; pos_i != size_*size_; ++pos_i)
	{
		vecPos_.push_back(0);
	}
}

template<typename T>
Board<T>::~Board()
{
}

//row and col starts from 1,A
template<typename T>
bool Board<T>::is_legal_move(T row, col_t col)
{
	T pos = (row-1)*size_+(col-'A');
	return vecPos_[pos] == 0;
}

// True: successful in adding move
// False: unsuccessful as the move is illegal
template<typename T>
bool Board<T>::add_move(T row, col_t col, player_t player)
{
	if (is_legal_move(row, col))
	{
		T pos = (row-1)*size_+(col-'A');
		vecPos_[pos] = player;
		return true;
	}
	else
	{
		return false;
	}
}

template<typename T>
void Board<T>::delete_move(T row, col_t col)
{
	T pos = (row-1)*size_+(col-'A');
	vecPos_[pos] = 0;
}

// return the player who has taken the given position
template<typename T>
player_t Board<T>::position_player(T pos)
{
	return vecPos_[pos];
}

template<typename T>
void Board<T>::display()
{
	std::cout << " ";
	for (T col_i = 1; col_i != (size_+1); ++col_i)
	{
		std::cout << static_cast<char>('A'+(col_i-1));
	}
	std::cout << std::endl;
	for (T row_i = 0; row_i != size_; ++row_i)
	{
		for (T col_i = 0; col_i != row_i; ++col_i)
		{
			std::cout << " ";
		}
		std::cout << row_i+1;
		for (T col_i = 0; col_i != size_; ++col_i)
		{
			T pos = row_i*size_ + col_i;
			if (vecPos_[pos])
			{
				if (vecPos_[pos] == 1)
				{
					std::cout << 'X';
				}
				else
				{
					std::cout << 'O';
				}
			}
			else
			{
				std::cout << ".";
			}
		}
		std::cout << row_i+1 << std::endl;
	}
	for (T col_i = 0; col_i != (size_+1); ++col_i)
	{
		std::cout << " ";
	}
	for (T col_i = 0; col_i != size_; ++col_i)
	{
		std::cout << static_cast<char>('A'+col_i);
	}
	std::cout << std::endl;
}

#endif
/*
https://class.coursera.org/cplusplus4c-002/forum/thread?thread_id=676
https://class.coursera.org/cplusplus4c-002/forum/thread?thread_id=677

suggestion for traversal using virtual home and virtual goal
https://class.coursera.org/cplusplus4c-002/forum/thread?thread_id=805#post-3568
*/