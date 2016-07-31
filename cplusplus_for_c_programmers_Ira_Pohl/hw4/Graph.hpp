#ifndef Graph_HPP
#define Graph_HPP

#include <vector>
#include <cassert>

// Adjacency list based implementation of undirected graph
template<typename T1>
class Graph
{
public:
	Graph(T1 nVertex);
	~Graph();
public:
	void add_edge(T1 vtx1, T1 vtx2);
	void delete_edge(T1 vtx1, T1 vtx2);
private:
	typename std::vector<T1>::iterator edge_iterator(T1 vtx1, T1 vtx2);
public:
	//properties of the graph
	T1 nVertex();
	T1 nEdge();
	std::vector<T1> neighbors(T1 vtx);
private:
	T1 nVertex_;
	T1 nEdge_;
	//adjacency list of neighbor vertices
	std::vector<std::vector<T1> > adjList_;
};

template<typename T1>
Graph<T1>::Graph(T1 nVertex)
: nVertex_(nVertex)
{
	adjList_.reserve(nVertex_);
	for (T1 vtx_i = 0; vtx_i != nVertex_; ++vtx_i)
	{
		adjList_.push_back(std::vector<T1>());
	}
}

template<typename T1>
Graph<T1>::~Graph()
{
	// STL Vector has its own destructor.
	// Since we are creating adjacency list using vector, deallocation is handled by vector itself.
}

template<typename T1>
typename std::vector<T1>::iterator Graph<T1>::edge_iterator(T1 vtx1, T1 vtx2)
{
	std::vector<T1>::iterator it;
	for (it = adjList_[vtx1].begin(); it != adjList_[vtx1].end(); ++it)
	{
		if (*it == vtx2)
		{
			break;
		}
	}
	return it;
}

template<typename T1>
void Graph<T1>::add_edge(T1 vtx1, T1 vtx2)
{
	// add edge only if not present
	std::vector<T1>::iterator it = edge_iterator(vtx1,vtx2);
	
	if (it == adjList_[vtx1].end())
	{
		adjList_[vtx1].push_back(vtx2);
		++nEdge_;
	}
}

template<typename T1>
void Graph<T1>::delete_edge(T1 vtx1, T1 vtx2)
{
	// first check for the existence of the edge
	std::vector<T1>::iterator it = edge_iterator(vtx1,vtx2);
	assert((it != adjList_[vtx1].end()) && "Non-existent edge");
	adjList_[vtx1].erase(it);
	--nEdge_;
}

template<typename T1>
T1 Graph<T1>::nVertex()
{
	return nVertex_;
}

template<typename T1>
T1 Graph<T1>::nEdge()
{
	return nEdge_;
}

template<typename T1>
std::vector<T1> Graph<T1>::neighbors(T1 vtx)
{
	return adjList_[vtx];
}

#endif