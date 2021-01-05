#include "dataset.h"
#include <math.h>

dataset::dataset(int number,int depth,int height,int width,string filepath,unsigned int tb,unsigned fb){
	this->number = number;
	this->channel = depth;
	this->height =height;
	this->width = width;
	this->tb = tb;
	this->fb = fb;
	for(int i=0;i<this->number;i++){
		int verbose = i == 1;
		string filename = filepath +"/data/data" + to_string(i);
		string filelabel = filepath+"/label/label" +to_string(i);
		vector<vector<vector<neuron>>> tmp = readData(depth,height,width,filename,verbose);
		this->imgs.push_back(tmp);
		int label = readLabel(filelabel);
		this->label.push_back(label);
	}
}

vector<vector<vector<neuron>>> dataset::getImg(int index){
	return this->imgs.at(index);
}

int dataset::getLabel(int index){
	return this->label.at(index);
}

int dataset::readLabel(string filename){
	fstream file;
	int x;
	file.open(filename,ios::in);
	if(!file){
		cout << filename<<" can't open" <<endl;
	}else {
		char buffer[100];
		file.getline(buffer,sizeof(buffer));
		x = atoi(buffer);
		file.close();
	}
	return x;
}


vector<vector<vector<neuron>>> dataset::readData(int in_channel,int height,int width,string filename,int verbose){
	fstream file;
	file.open(filename,ios::in);
	vector<neuron> row(width,neuron(this->tb,this->fb,0));
	vector<vector<neuron>> img(height,row);
	vector<vector<vector<neuron>>> imgs(in_channel,img);
	if(!file){
		
		cout << filename<<" can't open" <<endl;
	}else {
		for (int in=0;in <in_channel;in++){
			for(int i=0;i<height;i++){
				for(int j=0;j<width;j++){
					char buffer[100];
					file.getline(buffer,sizeof(buffer));
					double weight = atof(buffer);
					int x =round (weight * (1 << this->fb));
					imgs.at(in).at(i).at(j).setNumber(x);
				}
				
			}
		}
		file.close();
	}
	return imgs;
}

