all:	cgp

cgp:	cgp.cpp cgp.h data/*.h
		g++ cgp.cpp -O3 -o cgp

clean:
		rm cgp

cleanlog:
		rm log_*.*

run:	cgp
		./cgp