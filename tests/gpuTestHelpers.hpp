#ifndef GPU_TEST_HELPERS
#define GPU_TEST_HELPERS

#define TEST_ALL_TYPES(a) \
	do { \
		a<signed char>(); \
		a<unsigned char>(); \
		a<short>(); \
		a<unsigned short>(); \
		a<int>(); \
		a<float>(); \
	} while(0) 

#endif
