#ifndef _FSTREAM_
#define _FSTREAM_
#include<fstream>
#endif
#ifndef _LAYER_H_
#define _LAYER_H_
#include "Layer.h"
#endif
class VGG{
	private:
		vector<Layer> layers;
		vector<Layer> classify;
		int layer_size;
		int in_channel;
		unsigned int tb;
		unsigned int fb;
	public:
		VGG(string [],int,int,unsigned int,unsigned int);
		vector<Layer> MakeLayer(string [],int);
		vector<Layer> Makeclassify();
		vector<vector<vector<vector<neuron>>>> getNeuron(int,int,int,int,string);
		vector<neuron> forward(vector<vector<vector<neuron>>>,int);
		void printNetwork();
		Layer getFullyConnect(int,int,int,int);
};
