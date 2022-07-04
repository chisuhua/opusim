#ifndef STAT_WRAPER_H
#define STAT_WRAPER_H

namespace opu {

class Stats_gpgpu* StatCreate (const char * name, double bin_size, int num_bins) ;
void StatClear(void * st);
void StatAddSample (void * st, int val);
double StatAverage(void * st) ;
double StatMax(void * st) ;
double StatMin(void * st) ;
void StatDisp (void * st);
}

#endif
