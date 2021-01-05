cc=g++
object=neuron.o  Layer.o main.o dataset.o LeNet.o
flag = -O3

NN:$(object)
	$(cc)  $(flag) -o CNN $(object)
main.o:main.cpp
	$(cc) -w $(flag) -c -o $@ $^
neuron.o:neuron.cpp
	$(cc) $(flag) -c -o $@ $^
Layer.o:Layer.cpp
	$(cc) $(flag) -c -o $@ $^
#VGG.o:VGG.cpp
#	$(cc) $(flag) -c -o $@ $^
dataset.o:dataset.cpp
	$(cc) $(flag) -c -o $@ $^
LeNet.o :LeNet.cpp
	$(cc) $(flag) -c -o $@ $^

	
RUN:$(NN)
	./CNN 0  0 499     >> log/log0.txt & 
	./CNN 1  500 999   >> log/log1.txt &
	./CNN 2  1000 1499 >> log/log2.txt &
	./CNN 3  1500 1999 >> log/log3.txt &
	./CNN 4  2000 2499 >> log/log4.txt &
	./CNN 5  2500 2999 >> log/log5.txt &
	./CNN 6  3000 3499 >> log/log6.txt &
	./CNN 7  3500 3999 >> log/log7.txt &
	./CNN 8  4000 4499 >> log/log8.txt &
	./CNN 9  4500 4999 >> log/log9.txt &
	#./CNN 10 5000 5499 >> log/log10.txt &
	#./CNN 11 5500 5999 >> log/log11.txt &
	#./CNN 12 6000 6499 >> log/log12.txt &
	#./CNN 13 6500 6999 >> log/log13.txt &
	#./CNN 14 7000 7499 >> log/log14.txt &
	#./CNN 15 7500 7999 >> log/log15.txt &
	#./CNN 16 8000 8499 >> log/log16.txt &
	#./CNN 17 8500 8999 >> log/log17.txt &
	#./CNN 18 9000 9499 >> log/log18.txt &
	#./CNN 19 9500 9999 >> log/log19.txt &
	

clean:
	rm *.o CNN
