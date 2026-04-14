# Check the architecture and set the specific SIMD flag
ARCH := $(shell uname -m)
ifeq ($(ARCH),x86_64)
    SIMD_FLAG = -mavx2
else ifeq ($(ARCH),arm64)
    SIMD_FLAG = -mcpu=apple-m1
endif

all:    cgp

cgp:    cgp.cpp cgp.h data/*.h
	clang++ cgp.cpp -std=c++20 -O3 -fsanitize=address $(SIMD_FLAG) -o cgp

old:    cgpold.cpp cgp.h data/*.h
	g++ cgpold.cpp -O3 -mavx2 -o cgpold

clean:
	rm cgp

cleanlog:
	rm log_*.*

run:    cgp
	./cgp
