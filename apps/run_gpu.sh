#!/bin/bash

# SINGLE
echo "SINGLE\n"
time ./scene_processor -a GpuWHS -c img128.cfg -o /workfaster/diff_stuff/MattsTimingStuff/gpu1thread.tif 
time ./scene_processor -a GpuWHS -c img256.cfg -o /workfaster/diff_stuff/MattsTimingStuff/gpu1thread.tif 
time ./scene_processor -a GpuWHS -c img384.cfg -o /workfaster/diff_stuff/MattsTimingStuff/gpu1thread.tif 
time ./scene_processor -a GpuWHS -c img512.cfg -o /workfaster/diff_stuff/MattsTimingStuff/gpu1thread.tif 
time ./scene_processor -a GpuWHS -c img640.cfg -o /workfaster/diff_stuff/MattsTimingStuff/gpu1thread.tif 
time ./scene_processor -a GpuWHS -c img768.cfg -o /workfaster/diff_stuff/MattsTimingStuff/gpu1thread.tif 
time ./scene_processor -a GpuWHS -c img896.cfg -o /workfaster/diff_stuff/MattsTimingStuff/gpu1thread.tif 
time ./scene_processor -a GpuWHS -c img1024.cfg -o /workfaster/diff_stuff/MattsTimingStuff/gpu1thread.tif 

# DOUBLE
echo "DOUBLE\n"
time ./scene_processor -a GpuWHS -c img128.cfg -o /workfaster/diff_stuff/MattsTimingStuff/gpu1thread.tif -t 2 
time ./scene_processor -a GpuWHS -c img256.cfg -o /workfaster/diff_stuff/MattsTimingStuff/gpu1thread.tif -t 2
time ./scene_processor -a GpuWHS -c img384.cfg -o /workfaster/diff_stuff/MattsTimingStuff/gpu1thread.tif -t 2
time ./scene_processor -a GpuWHS -c img512.cfg -o /workfaster/diff_stuff/MattsTimingStuff/gpu1thread.tif -t 2
time ./scene_processor -a GpuWHS -c img640.cfg -o /workfaster/diff_stuff/MattsTimingStuff/gpu1thread.tif -t 2
time ./scene_processor -a GpuWHS -c img768.cfg -o /workfaster/diff_stuff/MattsTimingStuff/gpu1thread.tif -t 2
time ./scene_processor -a GpuWHS -c img896.cfg -o /workfaster/diff_stuff/MattsTimingStuff/gpu1thread.tif -t 2
time ./scene_processor -a GpuWHS -c img1024.cfg -o /workfaster/diff_stuff/MattsTimingStuff/gpu1thread.tif -t 2
