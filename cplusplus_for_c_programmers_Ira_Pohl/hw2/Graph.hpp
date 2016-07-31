#ifndef Graph_HPP
#define Graph_HPP

#include "Edge.hpp"
#include <cstdlib> //rand
#include <ctime>
#include <limits>
#include <vector>
#include <cassert>

template<typename T>
inline T prob()
{
	return (static_cast<T>(rand()%100))/100;
}

//undirected graph
template<typename T1, typename T2>
class Graph
{
public:
	Graph(T1 nVertex);
	~Graph();
public:
	void random_generation(T2 edge_density, T1 weight_lower_limit, T1 weight_upper_limit);
	void add_edge(T1 vtx1, T1 vtx2, T2 weight);
	void delete_edge(T1 vtx1, T1 vtx2);
public:
	typedef std::vector<Edge<T1,T2> > VecEdge;
public:
	//properties of the graph
	T1 nVertex();
	T1 nEdge();
	bool isAdjacent(T1 vtx1, T1 vtx2);
	typename VecEdge::iterator adjacent(T1 vtx1, T1 vtx2);
	T2 getEdgeWeight(T1 vtx1, T1 vtx2);
	std::vector<T1> neighbors(T1 vtx1);
private:
	T1 nVertex_;
	T1 nEdge_;
	std::vector<std::vector<Edge<T1,T2> > > adjList_;
};

template<typename T1, typename T2>
Graph<T1,T2>::Graph(T1 nVertex)
: nVertex_(nVertex)
{
	adjList_.reserve(nVertex_);

	for (T1 vtx_i=0; vtx_i != nVertex_; ++vtx_i)
	{
		adjList_.push_back(std::vector<Edge<T1,T2> >()); // assigning empty vector
	}

	nEdge_ = 0;
}
template<typename T1, typename T2>
Graph<T1,T2>::~Graph()
{
	// STL Vector has its own destructor.
	// Since we are creating adjacency list using vector, deallocation is handled by vector itself.
}
template<typename T1, typename T2>
void Graph<T1,T2>::random_generation(T2 edge_density, T1 weight_lower_limit, T1 weight_upper_limit)
{
	T1 weight_range_span = weight_upper_limit - weight_lower_limit;
	srand(std::time(NULL));
	for (T1 vtx1_i = 0; vtx1_i != nVertex_; ++vtx1_i)
	{
		for (T1 vtx2_i = vtx1_i+1; vtx2_i < nVertex_; ++vtx2_i)
		{
			T2 prob_val = prob<T2>();
			if (prob_val < edge_density)
			{
				//we create a edge between nodes: vtx1_i,vtx2_i
				T1 rand_val = rand()%(weight_range_span*10);
				T2 weight = (rand_val)/static_cast<T2>(10) + weight_lower_limit;
				Edge<T1,T2> edge(vtx1_i,vtx2_i,weight);
				adjList_[vtx1_i].push_back(edge);
				adjList_[vtx2_i].push_back(edge);
				++nEdge_;
			}
		}
	}
}

template<typename T1, typename T2>
T1 Graph<T1,T2>::nVertex()
{
	return nVertex_;
}

template<typename T1, typename T2>
T1 Graph<T1,T2>::nEdge()
{
	return nEdge_;
}
template<typename T1, typename T2>
bool Graph<T1,T2>::isAdjacent(T1 vtx1, T1 vtx2)
{
	VecEdge::iterator edgeIt = adjacent(vtx1,vtx2);
	return (edgeIt != adjList_[vtx1].end());
	/*
	bool adjacent_flag = false;
	for (VecEdge::iterator adjIt = adjList_[vtx1].begin(); adjIt != adjList_[vtx1].end(); ++adjIt)
	{
	Edge<T1,T2> edge = (*adjIt);

	if (edge.vertex1() == vtx1)
	{
	if (edge.vertex2() == vtx2)
	{
	adjacent_flag = true;
	}
	}
	else
	{
	if (edge.vertex1() == vtx2)
	{
	adjacent_flag = true;
	}
	}

	if (adjacent_flag)
	{
	break;
	}
	}

	return adjacent_flag;
	*/
}

template<typename T1, typename T2>
typename Graph<T1,T2>::VecEdge::iterator Graph<T1,T2>::adjacent(T1 vtx1, T1 vtx2)
{
	VecEdge::iterator adjIt;
	for (adjIt = adjList_[vtx1].begin(); adjIt != adjList_[vtx1].end(); ++adjIt)
	{
		Edge<T1,T2> edge = (*adjIt);

		if (edge.vertex1() == vtx1)
		{
			if (edge.vertex2() == vtx2)
			{
				break;
			}
		}
		else
		{
			if (edge.vertex1() == vtx2)
			{
				break;
			}
		}
	}

	return adjIt;
}

template<typename T1, typename T2>
T2 Graph<T1,T2>::getEdgeWeight(T1 vtx1, T1 vtx2)
{
	VecEdge::iterator it = adjacent(vtx1,vtx2);
	return (*it).weight();
}

template<typename T1, typename T2>
std::vector<T1> Graph<T1,T2>::neighbors(T1 vtx1)
{
	std::vector<T1> vecNeighbor;

	for (VecEdge::iterator adjIt = adjList_[vtx1].begin(); adjIt != adjList_[vtx1].end(); ++adjIt)
	{
		Edge<T1,T2> edge = (*adjIt);

		if (edge.vertex1() == vtx1)
		{
			vecNeighbor.push_back(edge.vertex2());
		}
		else
		{
			vecNeighbor.push_back(edge.vertex1());
		}
	}

	return vecNeighbor;
}

template<typename T1, typename T2>
void Graph<T1,T2>::add_edge(T1 vtx1, T1 vtx2, T2 weight)
{
	if (!isAdjacent(vtx1,vtx2))
	{
		Edge<T1,T2> edge(vtx1,vtx2,weight);
		adjList_[vtx1].push_back(edge);
	}
}

template<typename T1, typename T2>
void Graph<T1,T2>::delete_edge(T1 vtx1, T1 vtx2)
{
	VecEdge::iterator edgeIt = adjacent(vtx1,vtx2);
	if (edgeIt != adjList_[vtx1].end())
	{
		adjList_[vtx1].erase(edgeIt);
	}
}

#endif