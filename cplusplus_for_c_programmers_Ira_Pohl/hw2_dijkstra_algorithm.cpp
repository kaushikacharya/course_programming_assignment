#include <iostream>
#include <cstdlib> //rand
#include <ctime>
#include <vector>
#include <limits>
#include <set>
#include <cassert>

template<typename T>
inline T prob()
{
	return (static_cast<T>(rand()%100))/100;
}
template<typename T1, typename T2>
class Edge
{
public:
	Edge(T1 vertex1, T1 vertex2, T2 weight);
	~Edge();
public:
	T1 vertex1();
	T1 vertex2();
	T2 weight();
private:
	T1 vertex1_;
	T1 vertex2_;
	T2 weight_;
};

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
		bool operator()(const std::pair<T1,T2>& elem1, const std::pair<T1,T2>& elem2)
		{
			if (elem1.second == elem2.second)
			{
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

//------------
template<typename T1, typename T2>
Edge<T1,T2>::Edge(T1 vertex1, T1 vertex2, T2 weight)
: vertex1_(vertex1)
, vertex2_(vertex2)
, weight_(weight)
{}
template<typename T1, typename T2>
Edge<T1,T2>::~Edge()
{}
template<typename T1, typename T2>
T1 Edge<T1,T2>::vertex1()
{
	return vertex1_;
}
template<typename T1, typename T2>
T1 Edge<T1,T2>::vertex2()
{
	return vertex2_;
}
template<typename T1, typename T2>
T2 Edge<T1,T2>::weight()
{
	return weight_;
}
//------------

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
//------------

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
//------------

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
//------------
int main(int argc, char* argv[])
{
	unsigned int nVertex = 50;
	Graph<unsigned int, float> graph(nVertex);
	unsigned int weight_lower_limit = 1;
	unsigned int weight_upper_limit = 10;
	float edge_density = 0.1;
	graph.random_generation(edge_density,weight_lower_limit,weight_upper_limit);
	ShortestPath<unsigned int, float> shortestPath(graph);
	unsigned int source = 0;
	shortestPath.compute_shortest_paths(source);
	std::cout << "average path distance: " << shortestPath.average_path_length(source) << std::endl;

	return 0;
}