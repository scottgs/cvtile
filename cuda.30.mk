.cu.o:
	$(NVCC) -gencode=arch=compute_30,code=sm_30 -o $@ -c $<

.cu.lo:
	$(top_srcdir)/cudalt.py $@ $(NVCC) -gencode=arch=compute_30,code=sm_30 --compiler-options=\" $(CFLAGS) $(DEFAULT_INCLUDES) $(INCLUDES) $(AM_CPPFLAGS) $(CPPFLAGS) \" -c $<

