#ifndef MST_HPP
#define MST_HPP

#include "PriorityQueue.hpp"
#include "Graph.hpp"
#include <iostream>
#include <limits>
#include <vector>

// Prim's algorithm

template<typename T1, typename T2>
class MST
{
public:
	MST(Graph<T1,T2> graph);
	~MST();
public:
	T2 compute_minimum_spanning_tree(T1 source);
	void display_minimum_spanning_tree();
private:
	void initialize(T1 source);
private:
	Graph<T1,T2> graph_;
	PriorityQueue<T1,T2> priorityQueue_;
	std::vector<bool> vecVisited_; //True means vertex in closed set
	std::vector<T2> vecCost_;
	std::vector<T1> vecParent_; //stores parent of a vertex in the MST
};

template<typename T1, typename T2>
MST<T1,T2>::MST(Graph<T1,T2> graph)
: graph_(graph)
{
}

template<typename T1, typename T2>
MST<T1,T2>::~MST()
{
}

template<typename T1, typename T2>
T2 MST<T1,T2>::compute_minimum_spanning_tree(T1 source)
{
	initialize(source);

	while (!priorityQueue_.empty())
	{
		std::pair<T1,T2> topElemPQ = priorityQueue_.top();
		vecVisited_[topElemPQ.first] = true;
		priorityQueue_.remove_element(topElemPQ.first,topElemPQ.second);

		std::vector<T1> neighbors = graph_.neighbors(topElemPQ.first);
		for (typename std::vector<T1>::iterator it = neighbors.begin(); it != neighbors.end(); ++it)
		{
			if (!vecVisited_[*it])
			{
				T2 cost = graph_.getEdgeWeight(topElemPQ.first,(*it));

				if (cost < vecCost_[*it])
				{
					priorityQueue_.change_priority(*it,vecCost_[*it],cost);
					vecCost_[*it] = cost;
					vecParent_[*it] = topElemPQ.first;
				}
			}
		}
	}

	T2 cost_of_mst = 0;
	for (T1 vtx_i = 0; vtx_i != graph_.nVertex(); ++vtx_i)
	{
		cost_of_mst += vecCost_[vtx_i];
	}

	return cost_of_mst;
}

template<typename T1, typename T2>
void MST<T1,T2>::initialize(T1 source)
{
	vecVisited_.reserve(graph_.nVertex());
	vecCost_.reserve(graph_.nVertex());
	vecParent_.reserve(graph_.nVertex());

	//Initialize cost of vertices as infinite except the source vertex
	for (T1 vtx_i = 0; vtx_i != graph_.nVertex(); ++vtx_i)
	{
		vecVisited_.push_back(false);
		vecParent_.push_back(vtx_i); // dummy initialization
		if (vtx_i == source)
		{
			vecCost_.push_back(0);
			priorityQueue_.insert_element(vtx_i,0);
		} 
		else
		{
			vecCost_.push_back(std::numeric_limits<T2>::max());
			priorityQueue_.insert_element(vtx_i,std::numeric_limits<T2>::max());
		}
	}
}

template<typename T1, typename T2>
void MST<T1,T2>::display_minimum_spanning_tree()
{
	std::cout << "Minimum Spanning Tree" << std::endl;
	std::cout << "vertex1 -> verxtex2 (weight)" << std::endl;
	for (T1 vtx_i = 0; vtx_i != graph_.nVertex(); ++vtx_i)
	{
		if (vecParent_[vtx_i] == vtx_i)
		{
			continue; //root of the tree
		}
		std::cout << vecParent_[vtx_i] << " -> " << vtx_i << " : " << graph_.getEdgeWeight(vecParent_[vtx_i],vtx_i) <<  std::endl;
	}
}

#endif