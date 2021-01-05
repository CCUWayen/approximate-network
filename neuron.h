#define TOTAL_BITS
#define FIXED_POINT
#ifndef _MATH_H_
#define _MATH_H_
#include<math.h>
#endif

class neuron{
	unsigned int tb; //total bit
	unsigned int fb; //fixed point 
	int data;
	int _max();
	int _min();
	void _set(long);
	public:
		neuron(unsigned int,unsigned int,int);
		neuron(unsigned int,unsigned int,long);
		void setNumber(int);
		int getData();
		double getFixed();
		unsigned int gettb();
		unsigned int getfb();
		neuron& operator=(neuron);
		neuron operator*(neuron);
		neuron operator-(neuron);
		neuron operator+(neuron);
		neuron& operator*=(neuron);
		neuron& operator-=(neuron);
		neuron& operator+=(neuron);
		bool operator<(neuron);
		bool operator>(neuron);
		bool operator<=(neuron);
		bool operator>=(neuron);
		bool operator==(neuron);
		bool operator!=(neuron);
		double FixedMult(neuron);
};
