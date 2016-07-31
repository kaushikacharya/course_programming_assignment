#ifndef PriorityQueue_HPP
#define PriorityQueue_HPP

#include <set>
// Priority Queue is implemented using set
template<typename T1, typename T2>
class PriorityQueue
{
public:
	PriorityQueue();
	~PriorityQueue();
public:
	void insert_element(T1 vertex, T2 cost);
	void remove_element(T1 vertex, T2 cost);
	void change_priority(T1 vertex, T2 oldCost, T2 newCost);
	std::pair<T1,T2> top();
	bool empty();
private:
	// comparison function
	struct Comp
	{
		//first - index of vertex
		//second - cost
		bool operator()(const std::pair<T1,T2>& elem1, const std::pair<T1,T2>& elem2)
		{
			if (elem1.second == elem2.second)
			{
				// tie break based on vertex index
				return elem1.first < elem2.first;
			}
			else
			{
				return elem1.second < elem2.second;
			}
		}
	};
private:
	std::set<std::pair<T1,T2>, Comp> setPQ_;
};

template<typename T1, typename T2>
PriorityQueue<T1,T2>::PriorityQueue()
{}

template<typename T1, typename T2>
PriorityQueue<T1,T2>::~PriorityQueue()
{}

template<typename T1, typename T2>
void PriorityQueue<T1,T2>::insert_element(T1 vertex, T2 cost)
{
	std::pair<T1,T2> elem = std::make_pair(vertex,cost);
	setPQ_.insert(elem);
}

template<typename T1, typename T2>
void PriorityQueue<T1,T2>::remove_element(T1 vertex, T2 cost)
{
	std::set<std::pair<T1,T2>, Comp>::iterator it = setPQ_.find(std::make_pair(vertex,cost));
	assert((it != setPQ_.end()) && "Error: element absent");
	setPQ_.erase(it);
}

template<typename T1, typename T2>
void PriorityQueue<T1,T2>::change_priority(T1 vertex, T2 oldCost, T2 newCost)
{
	//changing priority is implemented by two steps:
	// 1. remove the existing element with old cost
	// 2. then re-inserting element with new cost
	remove_element(vertex,oldCost);
	insert_element(vertex,newCost);
}

template<typename T1, typename T2>
std::pair<T1,T2> PriorityQueue<T1,T2>::top()
{
	return *setPQ_.begin();
}

template<typename T1, typename T2>
bool PriorityQueue<T1,T2>::empty()
{
	return setPQ_.empty();
}

#endif