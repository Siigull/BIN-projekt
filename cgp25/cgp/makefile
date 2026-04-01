all:	cgp

cgp:	cgp.cpp cgp.h data/*.h
		g++ cgp.cpp -O3 -mavx2 -o cgp

old:    cgpold.cpp cgp.h data/*.h
		g++ cgpold.cpp -O3 -mavx2 -o cgpold

clean:
		rm cgp

cleanlog:
		rm log_*.*

run:	cgp
		./cgp