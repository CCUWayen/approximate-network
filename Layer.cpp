#include "Layer.h"

Layer::Layer(){
}

Layer::Layer(int out_channel,int in_channel,int padding,int kernel_size,int stride,string name,vector<vector<vector<vector<neuron>>>> weight,vector<vector<vector<vector<neuron>>>> bias,unsigned int tb,unsigned int fb,string path){
	this->out_channel = out_channel;
	this->in_channel = in_channel;
	this->padding = padding;
	this->kernel_size = kernel_size;
	this->stride = stride;
	this->name = name;
	this->weight = weight;
	this->bias = bias;
	this->fully_relu= 0;
	this->tb = tb;
	this->fb = fb;	
	this->mul_file = path;
	readmul();
}
Layer::Layer(int out_channel,int in_channel,int padding,int kernel_size,int stride,string name,vector<vector<vector<vector<neuron>>>> weight,vector<vector<vector<vector<neuron>>>> bias,int relu,unsigned int tb,unsigned int fb,string path){
	this->out_channel = out_channel;
	this->in_channel = in_channel;
	this->padding = padding;
	this->kernel_size = kernel_size;
	this->stride = stride;
	this->name = name;
	this->weight = weight;
	this->bias = bias;
	this->fully_relu= relu;	
	this->tb = tb;
	this->fb = fb;	
	this->mul_file = path;
	readmul();
}

Layer::Layer(int out_channel,int in_channel,int padding,int kernel_size,int stride,string name,unsigned int tb,unsigned int fb,string path){
	this->out_channel = out_channel;
	this->in_channel = in_channel;
	this->padding = padding;
	this->kernel_size = kernel_size;
	this->stride = stride;
	this->name = name;
	this->fully_relu= 0;	
	this->tb = tb;
	this->fb = fb;	
	this->mul_file = path;
	readmul();
}

double Layer::muli(int i,int j){
	int flag1 = i < 0 ;
	int flag2 = j < 0 ;
	int x = i < 0 ? -i : i;
	int y = j < 0 ? -j : j;
	int out = mul[x][y];
	if(flag1 ^ flag2) out = -out;
	double ans = (double) out / (1 << (2*this->fb));
	return ans;
}

Layer::Layer(int out_channel,int in_channel,int padding,int kernel_size,int stride,string name,int fully_relu,unsigned int tb,unsigned int fb,string path){
	this->out_channel = out_channel;
	this->in_channel = in_channel;
	this->padding = padding;
	this->kernel_size = kernel_size;
	this->stride = stride;
	this->name = name;
	this->fully_relu = fully_relu;
	this->tb = tb;
	this->fb = fb;	
	this->mul_file = path;
	readmul();
}
vector<vector<vector<neuron>>> Layer::cal(vector<vector<vector<neuron>>> img,int verbose){
	vector<vector<vector<neuron>>> out;
	if (this->name == "Conv"){
		out = this->conv(img,verbose);
	}
	else if (this->name == "MaxPool"){
		out = this->maxpool(img,verbose);
	}
	else if(this->name == "ReLu"){
		out = this->relu(img,verbose);
	}
	else if(this->name == "BatchNorm"){
		out = this->batch_normal(img,verbose);
	}
	return out;
}
vector<neuron> Layer::fully_connect (vector<neuron> img,int verbose){
	vector<neuron> out(this->out_channel,neuron(this->tb,this->fb,0));
	for(int i=0;i<this->out_channel;i++){
		double ans = 0;
		for(int j=0;j<this->in_channel;j++){
			//ans = ans + img.at(j).FixedMult(this->weight.at(i).at(j).at(0).at(0));
			ans = ans + muli(img.at(j).getData(),this->weight.at(i).at(j).at(0).at(0).getData());
		}
		ans = ans + this->bias.at(i).at(0).at(0).at(0).getFixed();
		if(this->fully_relu) {
			if(ans <0) ans = 0;
		}
		int x = round(ans * (1 << this->fb));
		neuron tmp = neuron(this->tb,this->fb,x);
		out.at(i) = tmp;
	}	
	return out;
}

void Layer::readmul(){
	const char *file = this->mul_file.c_str();
	freopen(file,"r",stdin);
	int i,j,r;
	while(scanf("%d %d %d",&i,&j,&r)!=EOF){
		this->mul[i][j] = r;
	}
	fclose(stdin);
}

vector<double> Layer::fully_connect_last (vector<neuron> img,int verbose){
	vector<double> out(this->out_channel);
	for(int i=0;i<this->out_channel;i++){
		double ans = 0;
		for(int j=0;j<this->in_channel;j++){
			ans = ans + img.at(j).FixedMult(this->weight.at(i).at(j).at(0).at(0));
			//ans = ans + img.at(j).getFixed() * this->weight.at(i).at(j).at(0).at(0).getFixed();//img.at(j).FixedMult(this->weight.at(i).at(j).at(0).at(0));
		}
		ans = ans + this->bias.at(i).at(0).at(0).at(0).getFixed();
		if(this->fully_relu) {
			if(ans <0) ans = 0;
		}
		int data = round(ans * (1 << this->fb));
		if(data >= 127) data = 127;
		else if (data <= -128) data = -128;
		int flag = (data >= 0) ? 1 : -1;
		int tmp = (flag == -1 ) ? ~data + 1 : data;
		ans = flag * ((double)tmp /(double) ( 1 << 7));
		
		//neuron tmp = neuron(this->tb,this->fb,x);
		//out.at(i) = tmp;
		out.at(i) = ans;
	}	
	return out;
}
vector<vector<vector<neuron>>> Layer::conv(vector<vector<vector<neuron>>> img,int verbose=0){
	int depth = img.size();
	int height = img.at(0).size();
	int width = img.at(0).at(0).size();
	int final_width = ((width-this->kernel_size + 2*this->padding) / this->stride)+1;
	int final_height = ((height-this->kernel_size + 2*this->padding) / this->stride)+1;
	vector<neuron> img_col(final_width,neuron(this->tb,this->fb,0));
	vector<vector<neuron>> image(final_height,img_col);
	vector<vector<vector<neuron>>> out_feature (this->out_channel,image);
	for(int k =0;k<this->out_channel;k++){ 
		for(int j = -this->padding ; j <= height + this->padding-this->kernel_size ; j = j +this->stride){
			for(int i = -this->padding ; i <= width + this->padding - this->kernel_size; i = i + this->stride) {
				//neuron conv = neuron(this->tb,this->fb,0);
				double result = 0;
				for(int s =0 ;s < this->in_channel;s++){
					for(int y=0;y<this->kernel_size;y++){
						for(int x=0;x<this->kernel_size;x++){
							bool flag = ((j+y) < 0 ) || ((j+y) >= height) || ((i+x) < 0) || ((i+x) >= width);
							if(flag == false) {
								//conv = conv + this->weight.at(k).at(s).at(y).at(x) * img.at(s).at(j+y).at(i+x);
								//double conv = this->weight.at(k).at(s).at(y).at(x).FixedMult(img.at(s).at(j+y).at(i+x));	
								double conv = this->muli(this->weight.at(k).at(s).at(y).at(x).getData(),img.at(s).at(j+y).at(i+x).getData());	
								result = result + conv;
							}
							if(((j+this->padding)==1) && ((i+this->padding)==19) && (depth==1) && (s==0) && (k==0) & verbose){
								if(flag){
									neuron n0 = this->weight.at(k).at(s).at(y).at(x);
									printf("0 * %6.3lf = 0\n",n0.getFixed());
								}
								else {
									neuron n0 = this->weight.at(k).at(s).at(y).at(x);
									neuron n1 = img.at(s).at(j+y).at(i+x);
									double n2 = n0.FixedMult(n1);
									printf("%lf * %lf = %lf\n",n1.getFixed(),n0.getFixed(),n2);
								}
							}
						}
					}
				}
				if(((j+this->padding)==1) && ((i+this->padding)==19) && depth==1 && (k==0) & verbose) printf("pixed(%d,%d):%lf\n",j+this->padding,i+this->padding,result);
				result = result + this->bias.at(k).at(0).at(0).at(0).getFixed();
				int ans = (int)(result * (1<<this->fb));
				neuron sum = neuron(this->tb,this->fb,ans);
				out_feature.at(k).at(j+this->padding).at(i+this->padding) = sum;
				//conv  = conv + this->bias.at(k).at(0).at(0).at(0);
				if(((j+this->padding)==1) && ((i+this->padding)==19) && depth==1 && (k==0) & verbose) printf("pixed(%d,%d):%lf(%lf)\n",j+this->padding,i+this->padding,sum.getFixed(),result);
				//out_feature.at(k).at(j+this->padding).at(i+this->padding) = conv;
			}
		}
	}
	//printImg(out_feature);
	return out_feature ;
}
vector<vector<vector<neuron>>> Layer::maxpool(vector<vector<vector<neuron>>> img,int verbose){
	int depth = img.size();
	int height = img.at(0).size();
	int width = img.at(0).at(0).size();
	int final_height = ceil(height/2);
	int final_width = ceil(width/2);
	vector<neuron> row(final_width,neuron(this->tb,this->fb,0));
	vector<vector<neuron>> imgs(final_height,row);
	vector<vector<vector<neuron>>> out_feature(depth,imgs);
	for(int k=0;k<depth;k++){
		for(int j=0;j<height;j=j+this->stride){
			for(int i=0;i<width;i=i+this->stride){
				neuron tmp = img.at(k).at(j).at(i);
				for(int y = 0;y<this->stride;y++){
					for(int x=0;x<this->stride;x++){
						bool flag = ((j+y) < height) && ((i+x)<width);
						if(flag) {
							if(img.at(k).at(j+y).at(i+x) > tmp)
								tmp = img.at(k).at(j+y).at(i+x);
						}
					}
				}
				out_feature.at(k).at(j/2).at(i/2) = tmp;
			}
		}
	}
	return out_feature;	
	
}
vector<vector<vector<neuron>>> Layer::batch_normal(vector<vector<vector<neuron>>> img,int verbose){
	cout << "BatchNorm"<<endl;
	int depth = img.size();
	int height = img.at(0).size();
	int width = img.at(0).at(0).size();
	for(int k=0;k<depth;k++){
		for(int i=0;i<height;i++){
			for(int j=0;j<width;j++){
				img.at(k).at(i).at(j) = this->weight.at(k).at(0).at(0).at(0) * img.at(k).at(i).at(j) + this->bias.at(k).at(0).at(0).at(0);
			}
		}
	}
	return img;

};
vector<vector<vector<neuron>>> Layer::relu(vector<vector<vector<neuron>>> img,int verbose){
	int depth = img.size();
	int height = img.at(0).size();
	int width = img.at(0).at(0).size();
	if (verbose){
		printf("ReLU\n");
	}
	for(int k=0;k<depth;k++){
		for(int i=0;i<height;i++){
			for(int j=0;j<width;j++)
				if(img.at(k).at(i).at(j).getFixed() <=0){
					img.at(k).at(i).at(j).setNumber(0);
				}
		}
	}
	return img;
}


void Layer::print(){
	if (this->name == "MaxPool") 
		cout << "Layer Name :" << this->name << endl;
	else if (this->name == "Conv")
		cout << "Layer Name :" << this->name <<"(out_channel:" << this->out_channel <<" ,in_channel:" << this->in_channel << " ,kernel_size :" << this->kernel_size << "*" << this->kernel_size << " ,padding:"<< this->padding <<" ,stride:"<<this->stride <<")"<<endl;
	else if (this->name == "ReLu")
		cout <<"Layer Name:" << this->name << endl;
	else if (this->name =="fully_connect")
		cout << "Layer Name :" << this->name <<"(out_channel:" << this->out_channel <<" ,in_channel:" << this->in_channel <<")"<<endl;
}
vector<vector<vector<vector<neuron>>>> Layer::getWeight(){
	return this->weight;
}
vector<vector<vector<vector<neuron>>>> Layer::getBias(){
	return this->bias;
}
string Layer::getName(){
	return this->name;
}
void Layer::printImg(vector<vector<vector<neuron>>> x){
	int Depth = x.size();
	int Height = x.at(0).size();
	int Width = x.at(0).at(0).size();
	for(int k=0;k<Depth;k++){
		for(int i =0;i<Height;i++){
			for(int j=0;j<Width;j++){
				printf("%5.2lf ",x.at(k).at(i).at(j).getFixed());
			}
			printf("\n");
		}
		printf("\n\n\n");
	}
}

