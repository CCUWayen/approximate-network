#include "VGG.h"

VGG::VGG(string cfg[],int layer_count,int in_channel,unsigned int fb,unsigned int tb){	
	this->layer_size = 0;
	this->tb = tb;
	this->fb = fb;
	this->in_channel = in_channel;
	this->layers = MakeLayer(cfg,layer_count);
	this->classify = Makeclassify();
}

vector<Layer> VGG::Makeclassify(){
	vector <Layer> tmp;
	tmp.push_back(getFullyConnect(0,512,256,0));
	tmp.push_back(getFullyConnect(1,256,256,0));
	tmp.push_back(getFullyConnect(2,256,10,0));
	return tmp;
}
Layer VGG::getFullyConnect(int idx,int in_channel,int out_channel,int relu){
	string filename_w = "./weight/classifier."+to_string(idx)+".weight.txt";	
	string filename_b = "./weight/classifier."+to_string(idx)+".bias.txt";	
	vector<vector<vector<vector<neuron>>>> weight=getNeuron(out_channel,in_channel,1,1,filename_w);
	vector<vector<vector<vector<neuron>>>> bias=getNeuron(out_channel,1,1,1,filename_b);
	Layer s = Layer(out_channel,in_channel,0,0,0,"fully_connect",weight,bias,relu,this->tb,this->fb);
	return s;
}

vector<Layer> VGG::MakeLayer(string cfg[],int layer_count){
	vector <Layer> tmp;
	for (int i =0;i<layer_count;i++){
		if(cfg[i] == "M"){
			Layer t = Layer(this->in_channel,this->in_channel,0,2,2,"MaxPool",this->tb,this->fb);
			tmp.push_back(t);	
			this->layer_size =this->layer_size + 1;
		}
		else {
			string filename_w ="./weight/feature." + to_string(this->layer_size)+".weight.txt" ; 
			string filename_b ="./weight/feature." + to_string(this->layer_size)+".bias.txt" ; 
			int out_channel = stoi(cfg[i]);
			vector<vector<vector<vector<neuron>>>> weight = getNeuron(out_channel,in_channel,3,3,filename_w);
			vector<vector<vector<vector<neuron>>>> bias= getNeuron(out_channel,1,1,1,filename_b);
			Layer t = Layer(out_channel,this->in_channel,1,3,1,"Conv",weight,bias,this->tb,this->fb);
			tmp.push_back(t);
			this->layer_size = this->layer_size + 1;
			Layer s = Layer(0,0,0,0,0,"ReLU",this->tb,this->fb);
			tmp.push_back(s);
			this->layer_size = this->layer_size + 1 ;
			this->in_channel = out_channel;
		}
		
	}
	return tmp;
}

vector<neuron> VGG::forward(vector<vector<vector<neuron>>> img,int verbose){
	int layer_count = 0;
	for(auto i :this->layers){
		img = i.cal(img,verbose);
		layer_count ++;
	}
	int depth = img.size();
	int height =img.at(0).size();
	int width = img.at(0).at(0).size();
	int total = depth * height * width;
	vector<neuron> point(total,neuron(this->tb,this->fb,0));
	int count =0;
	for(int k=0;k<depth;k++){
		for(int i =0;i<height;i++){
			for(int j=0;j<width;j++){
				point.at(count) = img.at(k).at(i).at(j);
				count ++ ;		
			}
		}
	}
//	ans = this->classify.at(0).fuuly_connect(point);
	for(auto i : this->classify){
		point = i.fully_connect(point,verbose);
	}
	return point;
}
vector<vector<vector<vector<neuron>>>> VGG::getNeuron(int out_channel,int in_channel,int height,int width,string filename){
	fstream file;
	file.open(filename,ios::in);
	vector<neuron> row(width,neuron(this->tb,this->fb,0));
	vector<vector<neuron>> kernel(height,row);
	vector<vector<vector<neuron>>> kernel_map(in_channel,kernel);
	vector<vector<vector<vector<neuron>>>> weight(out_channel,kernel_map);
	if(!file){
		
		cout << filename<<" can't open" <<endl;
	}else {
		for(int out=0;out<out_channel;out++){
			for (int in=0;in <in_channel;in++){
				for(int i=0;i<height;i++){
					for(int j=0;j<width;j++){
						char buffer[100];
						file.getline(buffer,sizeof(buffer));
						double x = atof(buffer);
						weight.at(out).at(in).at(i).at(j).setNumber(x);
						
					}
					
				}
			}
		}
		file.close();
	}
	return weight;
}

void VGG::printNetwork(){
	for(auto i : this->layers){
		i.print();
		vector<vector<vector<vector<neuron>>>> weight = i.getWeight();
		vector<vector<vector<vector<neuron>>>> bias   = i.getBias();
		if(i.getName() == "Conv"){
			cout << "weight:(" << weight.size() << ","<<weight.at(0).size() << "," << weight.at(0).at(0).size() << "," << weight.at(0).at(0).at(0).size() << ")"<<endl;
		}
	}
}

