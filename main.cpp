#include "LeNet.h"
#include "dataset.h"
vector<vector<vector<vector<neuron>>>> getWeight(int out_channel,int in_channel,int height,int width,string filename,int tb,int fb){
	fstream file;
	file.open(filename,ios::in);
	vector<neuron> row(width,neuron(tb,fb,0));
	vector<vector<neuron>> kernel(height,row);
	vector<vector<vector<neuron>>> kernel_map(in_channel,kernel);
	vector<vector<vector<vector<neuron>>>> weight(out_channel,kernel_map);
	if(!file){
		cout << filename<<" can't open" <<endl;
		exit(0);
	}else {
		for(int out=0;out<out_channel;out++){
			for (int in=0;in <in_channel;in++){
				for(int i=0;i<height;i++){
					for(int j=0;j<width;j++){
						char buffer[100];
						file.getline(buffer,sizeof(buffer));
						double w = atof(buffer);
						int x = round (w * (1 << fb));
						weight.at(out).at(in).at(i).at(j).setNumber(x);
						
					}
					
				}
			}
		}
		file.close();
	}
	return weight;
}

void printImg(vector<vector<vector<neuron>>> x){
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

int vectorMax(vector<neuron> x){
	int index=0;
	double tmp = x.at(0).getFixed();
	for(int i=0;i<x.size();i++){
		cout << x.at(i).getFixed()<< " "; 
		if(tmp < x.at(i).getFixed()){
			tmp = x.at(i).getFixed();
			index = i;
		}
	}
	cout <<endl;
	return index;
}

void InnLeNet(){
	dataset img = dataset(10000,1,28,28,"./Network/LeNet/image",8,4);
	string str[18] = {"6","16"};
	LeNet net = LeNet(str,2,1,8,4,"./Network/LeNet","./mul/exact.txt");
	int correct = 0;
	for(int i=0;i<10000;i=i+1){
		vector<neuron> out = net.forward(img.getImg(i),0);
		int pre = vectorMax(out);
		int label = img.getLabel(i);
		if(pre == label) correct ++;
		printf("Finish %d:pre=%d correct=%d\n",i,pre,label);
		if(i==0) break;
	}
	printf("correct = %d\n",correct);
}

void test(string NetworkDIR,string MulName,int tb,int fb){
	int imgNum,imgWidth,imgHeight,imgChannel;
	int outChannel,inChannel,padding,stride,kernelSize,relu,idx_conv=1,idx_fully=1,layerCount;
	string ParaFile = NetworkDIR+"/parameter";
	const char* parameter =  ParaFile.c_str();
	char layerName[100]; 
	string imgFileDir = NetworkDIR+"/image";//layerName;
	freopen(parameter,"r",stdin);
	scanf("%d",&layerCount);
	scanf("%d %d %d %d",&imgNum,&imgChannel,&imgHeight,&imgWidth);
	dataset imgs = dataset(imgNum,imgChannel,imgHeight,imgWidth,imgFileDir,tb,fb);
	vector<Layer> conv;
	vector<Layer> fully;
	string a[layerCount];
	int outChannels[layerCount],inChannels[layerCount],paddings[layerCount],kernelSizes[layerCount],strides[layerCount],relus[layerCount],idx = 0;
	while(scanf("%s %d %d %d %d %d %d",layerName,&outChannel,&inChannel,&padding,&kernelSize,&stride,&relu)!=EOF){
		string name(layerName);
		outChannels[idx] = outChannel;
		inChannels[idx] = inChannel;
		paddings[idx] = padding;
		kernelSizes[idx] = kernelSize;
		strides[idx] = stride;
		relus[idx] = relu;
		a[idx] = name;
		idx ++ ;
	}
	for(int i =0 ; i < layerCount ;i++){
		if(a[i] == "fully_connect"){
			string filename_w = NetworkDIR+"/weight/fc"+to_string(idx_fully)+".0.weight";	
			string filename_b = NetworkDIR+"/weight/fc"+to_string(idx_fully)+".0.bias";	
			vector<vector<vector<vector<neuron>>>> weight = getWeight(outChannels[i],inChannels[i],1,1,filename_w,tb,fb);
			vector<vector<vector<vector<neuron>>>> bias = getWeight(outChannels[i],1,1,1,filename_b,tb,fb);
			Layer tmp = Layer(outChannels[i],inChannels[i],paddings[i],kernelSizes[i],strides[i],a[i],weight,bias,relus[i],tb,fb,MulName);
			fully.push_back(tmp);
			idx_fully++;
		}
		else if (a[i] == "Conv"){
			string filename_w = NetworkDIR+"/weight/conv"+to_string(idx_conv)+".0.weight";	
			string filename_b = NetworkDIR+"/weight/conv"+to_string(idx_conv)+".0.bias";	
			vector<vector<vector<vector<neuron>>>> weight = getWeight(outChannels[i],inChannels[i],kernelSizes[i],kernelSizes[i],filename_w,tb,fb);
			vector<vector<vector<vector<neuron>>>> bias = getWeight(outChannels[i],1,1,1,filename_b,tb,fb);
			Layer tmp = Layer(outChannels[i],inChannels[i],paddings[i],kernelSizes[i],strides[i],a[i],weight,bias,relus[i],tb,fb,MulName);
			conv.push_back(tmp);
			idx_conv++;
		}
		else {
			Layer tmp = Layer(outChannels[i],inChannels[i],paddings[i],kernelSizes[i],strides[i],a[i],tb,fb,MulName);
			conv.push_back(tmp);
		}
	}
	fclose(stdin);
	cout << "Start Innference" << endl;
	int correct = 0;
	for(int i=0;i<imgNum;i++){
		vector<vector<vector<neuron>>> img = imgs.getImg(i);
		int label = imgs.getLabel(i);
		for(auto x : conv){
			img = x.cal(img,0);
		}
		int depth = img.size();
		int height =img.at(0).size();
		int width = img.at(0).at(0).size();
		int total = depth * height * width;
		vector<neuron> point(total,neuron(tb,fb,0));
		for(int k=0,count=0;k<depth;k++){
			for(int s =0;s<height;s++){
				for(int j=0;j<width;j++){
					point.at(count) = img.at(k).at(s).at(j);
					count ++ ;		
				}
			}
		}
		for(auto x : fully) {
			point = x.fully_connect(point,0);
		}
		int pre = vectorMax(point);
		if(pre == label) correct ++;
		printf("Finish %d:pre=%d correct=%d\n",i,pre,label);
	} 
	cout << "correct=" << correct<<endl; 
}

int main(int argc,char *argv[]){
	test("./Network/LeNet","./mul/exact.txt",8,4);	
	return 0;
}

