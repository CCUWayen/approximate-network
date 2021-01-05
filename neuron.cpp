#include "neuron.h"


int neuron::_max(){
	return (1 << (tb-1)) -1;
}

int neuron::_min(){
	return -(1 << (tb-1));
}

void neuron::_set(long x){
		if(x > _max())
			data = _max();
		else if(x < _min())
			data = _min();
		else 
			data = x;
}

void neuron::setNumber(int x) {
	long tmp(x);
	_set(tmp);
}

neuron::neuron(unsigned int TB,unsigned int FB,int DATA){
	tb = TB;
	fb = FB;
	long tmp(DATA);
	_set(tmp);
}

neuron::neuron(unsigned int TB,unsigned int FB,long DATA){
	tb = TB;
	fb = FB;
	_set(DATA);
}


int neuron::getData(){
	return data;
}
unsigned int neuron::gettb(){
	return tb;
}
unsigned int neuron::getfb(){
	return fb;
}

neuron& neuron::operator=(neuron n){
	tb = n.gettb();
	fb = n.getfb();
	data = n.getData();
	return *this;	
}

neuron& neuron::operator-=(neuron n){
	*this = *this - n;
	return *this;
}


neuron& neuron::operator*=(neuron n){
	*this = *this * n;
	return *this;
}
neuron& neuron::operator+=(neuron n){
	*this = *this + n;
	return *this;
}

neuron neuron::operator+(neuron n){
	long tmp(data);
	tmp += n.getData();
	neuron result(tb,fb,tmp);
	return result;
}
neuron neuron::operator-(neuron n){
	long tmp(data);
	tmp -= n.getData();
	neuron result(tb,fb,tmp);
	return result;
}
neuron neuron::operator*(neuron n){
	long tmp(data);
	tmp *= n.getData();
	int mask = (1 << tb) -1;
	tmp = (tmp >> (fb-1));
	tmp = (tmp >> 1) + (tmp & 1);
	neuron result(tb,fb,tmp);
	return result;
}

double neuron::FixedMult(neuron n){
	double ans = n.getFixed() * getFixed();
	//int x = round(ans * (1 << fb));
	//ans = (double)x / (1 << fb);
	return ans;
}
double neuron::getFixed(){
	int flag = (data >= 0) ? 1 : -1;
	int tmp = (flag == -1 ) ? ~data + 1 : data;
	double x = flag * ((double)tmp /(double) ( 1 << fb));
	return x;
}

bool neuron::operator<(neuron n){
	return getFixed() < n.getFixed();	
}

bool neuron::operator>(neuron n){
	return getFixed() > n.getFixed();	
}
bool neuron::operator>=(neuron n){
	return getFixed() >= n.getFixed();	
}
bool neuron::operator<=(neuron n){
	return getFixed() <= n.getFixed();	
}
bool neuron::operator==(neuron n){
	return getFixed() == n.getFixed();	
}
bool neuron::operator!=(neuron n){
	return getFixed() != n.getFixed();	
}

