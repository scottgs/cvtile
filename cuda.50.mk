.cu.o:
	$(NVCC) -gencode=arch=compute_50,code=sm_50 -o $@ -c $<

.cu.lo:
	$(top_srcdir)/cudalt.py $@ $(NVCC) -gencode=arch=compute_50,code=sm_50 --compiler-options=\" $(CFLAGS) $(DEFAULT_INCLUDES) $(INCLUDES) $(AM_CPPFLAGS) $(CPPFLAGS) \" -c $<

