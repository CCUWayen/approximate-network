#ifndef _FSTREAM_
#define _FSTREAM_
#include<fstream>
#endif
#ifndef _LAYER_H_
#define _LAYER_H_
#include "Layer.h"
#endif
class LeNet{
	private:
		vector<Layer> layers;
		vector<Layer> classify;
		vector<Layer> last_layer;
		int layer_size;
		int in_channel;
		unsigned int fb;
		unsigned int tb;
		string root;
		string mul_file;
	public:
		LeNet(string [],int,int,unsigned int,unsigned int,string,string);
		vector<Layer> MakeLayer(string [],int);
		vector<Layer> Makeclassify();
		vector<vector<vector<vector<neuron>>>> getNeuron(int,int,int,int,string);
		vector<neuron> forward(vector<vector<vector<neuron>>>,int);
		void printResult(vector<vector<vector<neuron>>>,int);
		void printFully(vector<neuron>,int);
		Layer getFullyConnect(int,int,int,int);
};
