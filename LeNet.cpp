#include "LeNet.h"

LeNet::LeNet(string cfg[],int layer_count,int in_channel,unsigned int tb,unsigned int fb,string root,string path){	
	this->layer_size = 0;
	this->in_channel = in_channel;
	this->tb = tb;
	this->fb = fb;
	this->root = root;
	this->mul_file = path;
	this->layers = MakeLayer(cfg,layer_count);
	this->classify = Makeclassify();
	//this->last_layer.push_back(getFullyConnect(2,120,84,1));
	//this->last_layer.push_back(getFullyConnect(3,84,10,0));
}


Layer LeNet::getFullyConnect(int idx,int in_channel,int out_channel,int relu){
	string filename_w = this->root+"/weight/fc"+to_string(idx)+".0.weight";	
	string filename_b = this->root+"/weight/fc"+to_string(idx)+".0.bias";	
	vector<vector<vector<vector<neuron>>>> weight=getNeuron(out_channel,in_channel,1,1,filename_w);
	vector<vector<vector<vector<neuron>>>> bias=getNeuron(out_channel,1,1,1,filename_b);
	Layer s = Layer(out_channel,in_channel,0,0,0,"fully_connect",weight,bias,relu,this->tb,this->fb,this->mul_file);
	return s;
}

vector<Layer> LeNet::Makeclassify(){
	vector <Layer> tmp;
	tmp.push_back(getFullyConnect(1,400,120,1));
	tmp.push_back(getFullyConnect(2,120,84,1));
	tmp.push_back(getFullyConnect(3,84,10,0));
	return tmp;
}

vector<Layer> LeNet::MakeLayer(string cfg[],int layer_count){
	vector <Layer> tmp;
	for (int i =0;i<layer_count;i++){
		string filename_w =this->root+"/weight/conv" + to_string((i+1))+".0.weight" ; 
		string filename_b =this->root+"/weight/conv" + to_string((i+1))+".0.bias" ; 
		int out_channel = stoi(cfg[i]);
		vector<vector<vector<vector<neuron>>>> weight = getNeuron(out_channel,in_channel,5,5,filename_w);
		vector<vector<vector<vector<neuron>>>> bias= getNeuron(out_channel,1,1,1,filename_b);
		int padding = 0;
		if(i==0) padding = 2;
		else padding = 0;
		Layer t = Layer(out_channel,this->in_channel,padding,5,1,"Conv",weight,bias,this->tb,this->fb,this->mul_file);
		tmp.push_back(t);
		Layer k = Layer(this->in_channel,this->in_channel,0,2,2,"MaxPool",this->tb,this->fb,this->mul_file);
		tmp.push_back(k);	
		this->layer_size =this->layer_size + 1;
		Layer s = Layer(0,0,0,0,0,"ReLu",this->tb,this->fb,this->mul_file);
		tmp.push_back(s);
		this->in_channel = out_channel;
	}
	return tmp;
}

vector<neuron> LeNet::forward(vector<vector<vector<neuron>>> img,int verbose){
	int layer_count = 0;
	for(auto i :this->layers){
		i.print();
		img = i.cal(img,0);
		if(verbose) printResult(img,layer_count);
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
	for(auto i : this->classify){
		i.print();
		point = i.fully_connect(point,0);
		if(verbose) printFully(point,layer_count);
		layer_count ++;
	}

	//
	//vector<double> ans;
	//for(auto i : this->last_layer){
	//	ans = i.fully_connect_last(point,0);
	//	if(verbose) printFully(point,layer_count);
	//	layer_count ++;
	//}
	return point;
}
vector<vector<vector<vector<neuron>>>> LeNet::getNeuron(int out_channel,int in_channel,int height,int width,string filename){
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
						double w = atof(buffer);
						int x = round (w * (1 << this->fb));
						weight.at(out).at(in).at(i).at(j).setNumber(x);
						
					}
					
				}
			}
		}
		file.close();
	}
	return weight;
}

void LeNet::printResult(vector<vector<vector<neuron>>> img,int layer_count){
	fstream file;
	string filename = "./Result/layer"+to_string(layer_count);
	file.open(filename,ios::out);
	if(!file)
		cout << filename<<" can't open!" << endl;
	else {
		for(auto k : img){
			for(auto i : k){
				for(auto j : i){
					file << j.getFixed()<<endl;
				}
			}
		}
	}

}
void LeNet::printFully(vector<neuron> img,int layer_count){
	fstream file;
	string filename = "./Result/layer"+to_string(layer_count);
	file.open(filename,ios::out);
	if(!file)
		cout << filename<<" can't open!" << endl;
	else {
		for(auto k : img){
			file << k.getFixed()<<endl;
		}
	}

}

