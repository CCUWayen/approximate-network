#ifndef _VECTOR_H_
#define _VECTOR_H_
#include <vector>
#endif
#ifndef _MATH_H_
#define _MATH_H_
#include <math.h>
#endif
#include "iostream"
#ifndef _NEURON_H_
#include"neuron.h"
#define _NEURON_H_
#endif
using namespace std;
class Layer{
	private:
		int out_channel;
		int kernel_size;
		int in_channel;
		int padding;
		int stride;
		int fully_relu;
		int mul[256][256];
		unsigned int fb;
		unsigned int tb;
		string mul_file;
		string name;
		vector<vector<vector<vector<neuron>>>> weight;
		vector<vector<vector<vector<neuron>>>> bias;
	public:
		Layer();
		Layer(int,int,int,int,int,string,unsigned int,unsigned int,string);
		Layer(int,int,int,int,int,string,int,unsigned int,unsigned int,string);
		Layer(int,int,int,int,int,string,vector<vector<vector<vector<neuron>>>>,vector<vector<vector<vector<neuron>>>>,unsigned int,unsigned int,string);
		Layer(int,int,int,int,int,string,vector<vector<vector<vector<neuron>>>>,vector<vector<vector<vector<neuron>>>>,int,unsigned int,unsigned int,string);
		vector<vector<vector<neuron>>> cal(vector<vector<vector<neuron>>>,int);
		vector<vector<vector<neuron>>> conv(vector<vector<vector<neuron>>>,int);
		vector<vector<vector<neuron>>> maxpool(vector<vector<vector<neuron>>>,int);
		vector<vector<vector<neuron>>> relu(vector<vector<vector<neuron>>>,int);
		vector<vector<vector<neuron>>> batch_normal(vector<vector<vector<neuron>>>,int);
		vector<neuron> fully_connect(vector<neuron>,int);
		vector<double> fully_connect_last(vector<neuron>,int);
		vector<vector<vector<vector<neuron>>>> getWeight();
		vector<vector<vector<vector<neuron>>>> getBias();
		void print();
		void printImg(vector<vector<vector<neuron>>> );
		string getName();
		void readmul();
		double muli(int ,int );
};

