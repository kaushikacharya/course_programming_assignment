//
//Interface for the Vertex, Edge and Graph Classes
//

#ifndef __Graph_hpp__
#define __Graph_hpp__
#include <vector>

using namespace std;

class Edge;

// represents the vertex of a graph
// Each vertex has 
//  - vertex Id
//  - a value
//  - a list of weighted edges.
// The graph class includes a list of vertices
class Vertex {
    int value;
    int id;

    public:
    vector<Edge*> edges;
    Vertex();
    ~Vertex();
    
    Vertex(int ident, double v);

    void setId(int i) ;
    int getId() ;
    void setValue(int v) ;
    int getValue() ;
    
    void addEdge(Edge *e);
    Edge* getEdge(int v2);
    bool deleteEdge(Edge *e) ;
    
    vector<Edge *> neighbors();
    
    bool isNeighbor(int v2);
    void showNeighbors() ;
};

// represents the Edge of a graph
// the edge is included in the list of edges for the src and dest
// vertices
// The edge carries 
//   - the weight
//   - a reference to the dest vertex for this edge.
class Edge {
    double weight;
    
    public:
    Vertex  *destNode;
    Edge (Vertex *v, double w) ;
    double getWeight() ;
    void setWeight(double w) ;
    Vertex *getDestNode() ;
    int getDestNodeId() ;
};

// represents a graph
// The graph class has
//   - a list of vertices.
//   - a method to generate a random graph
class Graph {
    private:
        int numVertices;
        vector<Vertex *> vertices;

    public:     
        void addVertex(int id, int value);
        //default constructor
        Graph();
        //default destructor
        ~Graph();
        // constructor for graph with no edges
        Graph(int nVertices);
        Graph(const char *filename);
        Graph(const Graph& obj);

        //number of vertices in the graph
        int V(); 
        // number of edges in graph
        int E(); 

        // get the neighbors of node with id v1
        vector<Edge*> neighbors(int v1); 
        
        // return true if there is a direct arc from v1 to v2
        bool adjacent(int v1, int v2);

        bool addEdge(int v1, int v2, double weight) ;

        bool deleteEdge(int v1, int v2) ;

        // generate a random graph. 
        // for a graph of n nodes, the density parameter determines the 
        // number of edges,
        // #edges = (n(n-1)/2) * density
        // distMin and distMax defines the range of weights assigned to each
        // edge
        void generateRandomGraph(double density, double distMin, double distMax);

        Vertex *getVertex(int v1) ;

        //get the value associated with node id v1
        int get_node_value(int v1);

        bool set_node_value(int v1, int val) ;

        double get_edge_value(int v1, int v2) ;

        bool set_edge_value(int v1, int v2, double w);

        vector<Vertex *> getVertices();
};

#endif
