.cu.o:
	$(NVCC) -gencode=arch=compute_20,code=sm_20 -o $@ -c $<

.cu.lo:
	$(top_srcdir)/cudalt.py $@ $(NVCC) -gencode=arch=compute_20,code=sm_20 --compiler-options=\" $(CFLAGS) $(DEFAULT_INCLUDES) $(INCLUDES) $(AM_CPPFLAGS) $(CPPFLAGS) \" -c $<

