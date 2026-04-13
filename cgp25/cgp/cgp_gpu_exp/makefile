CXX := clang++
OBJCXX := clang++
CXXFLAGS := -std=c++20 -O3 -mcpu=apple-m1 -I..
OBJCXXFLAGS := $(CXXFLAGS) -fobjc-arc
LDFLAGS := -framework Metal -framework Foundation

all: cgp_gpu

cgp_gpu: cgp_gpu.o metal_eval_bridge.o
	$(CXX) $(CXXFLAGS) -DUSE_METAL -o $@ cgp_gpu.o metal_eval_bridge.o $(LDFLAGS)

cgp_gpu.o: cgp_gpu.cpp SharedDefinitions.h metal_eval_bridge.h ../cgp.h ../data/*.h
	$(CXX) $(CXXFLAGS) -DUSE_METAL -c cgp_gpu.cpp -o $@

metal_eval_bridge.o: metal_eval_bridge.mm metal_eval_bridge.h SharedDefinitions.h
	$(OBJCXX) $(OBJCXXFLAGS) -c metal_eval_bridge.mm -o $@

clean:
	rm -f cgp_gpu *.o
