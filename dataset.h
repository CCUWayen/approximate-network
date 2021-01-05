#include <vector>
#include <fstream>
#include "iostream"
#ifndef _NEURON_H_
#include"neuron.h"
#define _NEURON_H_
#endif

using namespace std;
class dataset {
	public:
		dataset(int,int,int,int,string,unsigned int,unsigned int);
		vector<vector<vector<neuron>>> getImg(int);
		int getLabel(int);
		vector<vector<vector<neuron>>> readData(int,int,int,string,int);
		int readLabel(string);
	private:
		vector<vector<vector<vector<neuron>>>> imgs;
		vector<int> label;
		int number;
		int channel;
		int width;
		int height;
		unsigned int tb;
		unsigned int fb;
	
};
