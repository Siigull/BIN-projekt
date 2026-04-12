all:	cgp

cgp:	cgp.cpp cgp.h data/*.h
		clang++ cgp.cpp -std=c++20 -O3 -fsanitize=address -mcpu=apple-m1 -o cgp

old:    cgpold.cpp cgp.h data/*.h
		g++ cgpold.cpp -O3 -mavx2 -o cgpold

clean:
		rm cgp

cleanlog:
		rm log_*.*

run:	cgp
		./cgp