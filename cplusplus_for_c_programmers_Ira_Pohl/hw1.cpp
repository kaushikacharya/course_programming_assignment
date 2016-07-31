#include <iostream>
#include <vector>
const int N = 40;

// computes sum of the elements of the vector
void sum(int& p, const std::vector<int>& d)
{ 
	p = 0;
	for(int i = 0; i < d.size(); ++i)
		p += d[i];
}

int main()
{
	std::vector<int> data;
	for(int i = 0; i < N; ++i)
	{
		data.push_back(i);
	}

	int accum = 0; 
	sum(accum, data);
	std::cout << "sum is " << accum << std::endl;

	return 0;
}