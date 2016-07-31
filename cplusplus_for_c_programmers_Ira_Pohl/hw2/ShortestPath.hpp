#ifndef ShortestPath_HPP
#define ShortestPath_HPP

#include "Graph.hpp"
#include "PriorityQueue.hpp"
#include <vector>

template<typename T1, typename T2>
class ShortestPath
{
public:
	ShortestPath(Graph<T1,T2>& graph);
	~ShortestPath();
public:
	void compute_shortest_paths(T1 source);
	T2 average_path_length(T1 source);
private:
	Graph<T1,T2> graph_;
	PriorityQueue<T1,T2> priorityQueue_;
private:
	std::vector<bool> vecVisited_;
	std::vector<T2> vecCost_;
};

template<typename T1, typename T2>
ShortestPath<T1,T2>::ShortestPath(Graph<T1,T2> &graph)
: graph_(graph)
{}

template<typename T1, typename T2>
ShortestPath<T1,T2>::~ShortestPath()
{}

template<typename T1, typename T2>
void ShortestPath<T1,T2>::compute_shortest_paths(T1 source)
{
	//Dijkstra's Algorithm

	vecVisited_.reserve(graph_.nVertex());
	vecCost_.reserve(graph_.nVertex());

	//Initialize cost of vertices as infinite except the source vertex
	for (T1 vtx_i = 0; vtx_i != graph_.nVertex(); ++vtx_i)
	{
		vecVisited_.push_back(false);
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
				T2 cost = vecCost_[topElemPQ.first] + graph_.getEdgeWeight(topElemPQ.first,(*it));

				if (cost < vecCost_[*it])
				{
					priorityQueue_.change_priority(*it,vecCost_[*it],cost);
					vecCost_[*it] = cost;
				}
			}
		}
	}
}

template<typename T1, typename T2>
T2 ShortestPath<T1,T2>::average_path_length(T1 source)
{
	T1 count_vertices_visited_from_source = 0;
	T2 average_path_length = 0;
	for (T1 vtx_i = 0; vtx_i != graph_.nVertex(); ++vtx_i)
	{
		if (vtx_i == source)
		{
			continue;
		}
		if (vecVisited_[vtx_i])
		{
			++count_vertices_visited_from_source;
			average_path_length += vecCost_[vtx_i];
		}
	}

	average_path_length /= count_vertices_visited_from_source;
	return average_path_length;
}

#endif