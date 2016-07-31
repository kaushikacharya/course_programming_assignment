#ifndef EDGE_HPP
#define EDGE_HPP

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

#endif