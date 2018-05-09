#include "miner.h"

#include "cuda_helper.h"

static uint32_t *d_hash[MAX_GPUS];

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_progpow(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	const int dev_id = device_map[thr_id];

	uint32_t throughput =  cuda_default_throughput(thr_id, 1U << 17); 
	
	if (opt_benchmark)
		((uint32_t*)ptarget)[7] = 0x00ff;

	if (!init[thr_id])
	{
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1)
		{
			cudaDeviceReset();
			// reduce cpu usage
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
		}
		
		gpulog(LOG_INFO, thr_id, "Generating DAG for GPU #%d...", thr_id);
		
		ethash_generate_dag(dag, dagBytes, light, lightWords, s_gridSize, s_blockSize, m_streams[0], m_device_num);
		
		gpulog(LOG_INFO, thr_id, "DAG generation completed.\nIntensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);
