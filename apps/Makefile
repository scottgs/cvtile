all: process_task

CUDA_INCLUDE=-I/opt/cuda/include/ -I/opt/cuda/sdk/C/common/inc/ -I/opt/cuda/sdk/shared/inc/ -I/opt/cuda/sdk/common/inc/
CUDA_LIB=-L/opt/cuda/lib64/

process_task: main.o
	g++ -std=c++11 main.o -lboost_program_options -lcvtile -lcudart $(CUDA_LIB) $(CUDA_INCLUDE) -o process_task

main.o:
	g++ -std=c++11 process_task.cpp -c


