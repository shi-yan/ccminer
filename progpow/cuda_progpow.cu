#ifndef SEARCH_RESULTS
#define SEARCH_RESULTS 4
#endif

typedef struct {
    uint32_t count;
    struct {
        // One word for gid and 4 for mix hash
        uint32_t gid;
        uint32_t mix[4];
    } result[SEARCH_RESULTS];
} search_results;

typedef struct
{
    uint32_t uint32s[32 / sizeof(uint32_t)];
} hash32_t;

__device__ __constant__ const uint64_t keccakf_1600_rndc[24] = {
	0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808AULL,
	0x8000000080008000ULL, 0x000000000000808BULL, 0x0000000080000001ULL,
	0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008AULL,
	0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000AULL,
	0x000000008000808BULL, 0x800000000000008BULL, 0x8000000000008089ULL,
	0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
	0x000000000000800AULL, 0x800000008000000AULL, 0x8000000080008081ULL,
	0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

__device__ __constant__ const uint32_t keccakf_800_rndc[24] = {
    0x00000001, 0x00008082, 0x0000808a, 0x80008000, 0x0000808b, 0x80000001,
    0x80008081, 0x00008009, 0x0000008a, 0x00000088, 0x80008009, 0x8000000a,
    0x8000808b, 0x0000008b, 0x00008089, 0x00008003, 0x00008002, 0x00000080,
    0x0000800a, 0x8000000a, 0x80008081, 0x00008080, 0x80000001, 0x80008008
};

__device__ __forceinline__ void keccakf_1600_round(uint64_t st[25], const int r)
{

	const uint32_t keccakf_rotc[24] = {
		1,  3,  6,  10, 15, 21, 28, 36, 45, 55, 2,  14,
		27, 41, 56, 8,  25, 43, 62, 18, 39, 61, 20, 44
	};
	const uint32_t keccakf_piln[24] = {
		10, 7,  11, 17, 18, 3, 5,  16, 8,  21, 24, 4,
		15, 23, 19, 13, 12, 2, 20, 14, 22, 9,  6,  1
	};

	uint64_t t, bc[5];
	// Theta
	for (int i = 0; i < 5; i++)
		bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];

	for (int i = 0; i < 5; i++) {
		t = bc[(i + 4) % 5] ^ ROTL32(bc[(i + 1) % 5], 1);
		for (uint32_t j = 0; j < 25; j += 5)
			st[j + i] ^= t;
	}

	// Rho Pi
	t = st[1];
	for (int i = 0; i < 24; i++) {
		uint32_t j = keccakf_piln[i];
		bc[0] = st[j];
		st[j] = ROTL32(t, keccakf_rotc[i]);
		t = bc[0];
	}

	//  Chi
	for (uint32_t j = 0; j < 25; j += 5) {
		for (int i = 0; i < 5; i++)
			bc[i] = st[j + i];
		for (int i = 0; i < 5; i++)
			st[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
	}

	//  Iota
	st[0] ^= keccakf_1600_rndc[r];
}

__device__ __forceinline__ void keccakf_800_round(uint32_t st[25], const int r)
{

    const uint32_t keccakf_rotc[24] = {
        1,  3,  6,  10, 15, 21, 28, 36, 45, 55, 2,  14,
        27, 41, 56, 8,  25, 43, 62, 18, 39, 61, 20, 44
    };
    const uint32_t keccakf_piln[24] = {
        10, 7,  11, 17, 18, 3, 5,  16, 8,  21, 24, 4,
        15, 23, 19, 13, 12, 2, 20, 14, 22, 9,  6,  1
    };

    uint32_t t, bc[5];
    // Theta
    for (int i = 0; i < 5; i++)
        bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];

    for (int i = 0; i < 5; i++) {
        t = bc[(i + 4) % 5] ^ ROTL32(bc[(i + 1) % 5], 1);
        for (uint32_t j = 0; j < 25; j += 5)
            st[j + i] ^= t;
    }

    // Rho Pi
    t = st[1];
    for (int i = 0; i < 24; i++) {
        uint32_t j = keccakf_piln[i];
        bc[0] = st[j];
        st[j] = ROTL32(t, keccakf_rotc[i]);
        t = bc[0];
    }

    //  Chi
    for (uint32_t j = 0; j < 25; j += 5) {
        for (int i = 0; i < 5; i++)
            bc[i] = st[j + i];
        for (int i = 0; i < 5; i++)
            st[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
    }

    //  Iota
    st[0] ^= keccakf_800_rndc[r];
}

__device__ __forceinline__ void keccak_f1600(uint64_t st[25])
{
	for (int i = 8; i < 25; i++)
	{
		st[i] = 0;
	}
	st[8] = 0x8000000000000001;

	for (int r = 0; r < 24; r++) {
		keccakf_1600_round(st, r);
	}
}

__device__ __noinline__ uint64_t keccak_f800(hash32_t header, uint64_t seed, uint4 result)
{
    uint32_t st[25];

    for (int i = 0; i < 25; i++)
        st[i] = 0;
    for (int i = 0; i < 8; i++)
        st[i] = header.uint32s[i];
    st[8] = seed;
    st[9] = seed >> 32;
    st[10] = result.x;
    st[11] = result.y;
    st[12] = result.z;
    st[13] = result.w;

    for (int r = 0; r < 21; r++) {
        keccakf_800_round(st, r);
    }
    // last round can be simplified due to partial output
    keccak_f800_round(st, 21);

    return (uint64_t)st[1] << 32 | st[0];
}

#define fnv1a(h, d) (h = (h ^ d) * 0x1000193)

typedef struct {
    uint32_t z, w, jsr, jcong;
} kiss99_t;

// KISS99 is simple, fast, and passes the TestU01 suite
// https://en.wikipedia.org/wiki/KISS_(algorithm)
// http://www.cse.yorku.ca/~oz/marsaglia-rng.html
__device__ __forceinline__ uint32_t kiss99(kiss99_t &st)
{
    uint32_t znew = (st.z = 36969 * (st.z & 65535) + (st.z >> 16));
    uint32_t wnew = (st.w = 18000 * (st.w & 65535) + (st.w >> 16));
    uint32_t MWC = ((znew << 16) + wnew);
    uint32_t SHR3 = (st.jsr ^= (st.jsr << 17), st.jsr ^= (st.jsr >> 13), st.jsr ^= (st.jsr << 5));
    uint32_t CONG = (st.jcong = 69069 * st.jcong + 1234567);
    return ((MWC^CONG) + SHR3);
}

__device__ __forceinline__ void fill_mix(uint64_t seed, uint32_t lane_id, uint32_t mix[PROGPOW_REGS])
{
    // Use FNV to expand the per-warp seed to per-lane
    // Use KISS to expand the per-lane seed to fill mix
    uint32_t fnv_hash = 0x811c9dc5;
    kiss99_t st;
    st.z = fnv1a(fnv_hash, seed);
    st.w = fnv1a(fnv_hash, seed >> 32);
    st.jsr = fnv1a(fnv_hash, lane_id);
    st.jcong = fnv1a(fnv_hash, lane_id);
    #pragma unroll
    for (int i = 0; i < PROGPOW_REGS; i++)
        mix[i] = kiss99(st);
}

__global__ void progpow_gpu_hash(uint64_t start_nonce, const hash32_t header, const uint64_t target, const uint64_t *g_dag, volatile search_results *g_output)
{
	__shared__ uint32_t c_dag[PROGPOW_CACHE_WORDS];
    uint32_t const gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t const nonce = start_nonce + gid;

    const uint32_t lane_id = threadIdx.x & (PROGPOW_LANES - 1);

    // Load random data into the cache
    // TODO: should be a new blob of data, not existing DAG data
    for (uint32_t word = threadIdx.x*2; word < PROGPOW_CACHE_WORDS; word += blockDim.x*2)
    {
        uint64_t data = g_dag[word];
        c_dag[word + 0] = data;
        c_dag[word + 1] = data >> 32;
    }

    uint4 result;
    result.x = result.y = result.z = result.w = 0;
    // keccak(header..nonce)
    uint64_t seed = keccak_f800(header, nonce, result);

    __syncthreads();

    #pragma unroll 1
    for (uint32_t h = 0; h < PROGPOW_LANES; h++)
    {
        uint32_t mix[PROGPOW_REGS];

        // share the hash's seed across all lanes
        uint64_t hash_seed = __shfl_sync(0xFFFFFFFF, seed, h, PROGPOW_LANES);
        // initialize mix for all lanes
        fill_mix(hash_seed, lane_id, mix);

        #pragma unroll 1
        for (uint32_t l = 0; l < PROGPOW_CNT_MEM; l++)
            progPowLoop(l, mix, g_dag, c_dag);


        // Reduce mix data to a single per-lane result
        uint32_t mix_hash = 0x811c9dc5;
        #pragma unroll
        for (int i = 0; i < PROGPOW_REGS; i++)
            fnv1a(mix_hash, mix[i]);

        // Reduce all lanes to a single 128-bit result
        uint4 result_hash;
        result_hash.x = result_hash.y = result_hash.z = result_hash.w = 0x811c9dc5;
        #pragma unroll
        for (int i = 0; i < PROGPOW_LANES; i += 4)
        {
            fnv1a(result_hash.x, __shfl_sync(0xFFFFFFFF, mix_hash, i + 0, PROGPOW_LANES));
            fnv1a(result_hash.y, __shfl_sync(0xFFFFFFFF, mix_hash, i + 1, PROGPOW_LANES));
            fnv1a(result_hash.z, __shfl_sync(0xFFFFFFFF, mix_hash, i + 2, PROGPOW_LANES));
            fnv1a(result_hash.w, __shfl_sync(0xFFFFFFFF, mix_hash, i + 3, PROGPOW_LANES));
        }
        if (h == lane_id)
            result = result_hash;
    }

    // keccak(header .. keccak(header..nonce) .. result);
    if (keccak_f800(header, seed, result) > target)
        return;

    uint32_t index = atomicInc((uint32_t *)&g_output->count, 0xffffffff);
    if (index >= SEARCH_RESULTS)
        return;

    g_output->result[index].gid = gid;
    g_output->result[index].mix[0] = result.x;
    g_output->result[index].mix[1] = result.y;
    g_output->result[index].mix[2] = result.z;
    g_output->result[index].mix[3] = result.w;
}

#define FNV_PRIME	0x01000193
#define fnv(x,y) ((x) * FNV_PRIME ^(y))
__device__ uint4 fnv4(uint4 a, uint4 b)
{
	uint4 c;
	c.x = a.x * FNV_PRIME ^ b.x;
	c.y = a.y * FNV_PRIME ^ b.y;
	c.z = a.z * FNV_PRIME ^ b.z;
	c.w = a.w * FNV_PRIME ^ b.w;
	return c;
}

#define NODE_WORDS (ETHASH_HASH_BYTES/sizeof(uint32_t))

__global__ void
ethash_calculate_dag_item(uint32_t start, hash64_t *g_dag, uint64_t dag_bytes, hash64_t* g_light, uint32_t light_words)
{
	uint64_t const node_index = start + blockIdx.x * blockDim.x + threadIdx.x;
	if (node_index * sizeof(hash64_t) >= dag_bytes ) return;

	hash200_t dag_node;
	for(int i=0; i<4; i++)
		dag_node.uint4s[i] = g_light[node_index % light_words].uint4s[i];
	dag_node.words[0] ^= node_index;
	keccakf_1600(dag_node.uint64s);

	const int thread_id = threadIdx.x & 3;

	#pragma unroll
	for (uint32_t i = 0; i < ETHASH_DATASET_PARENTS; ++i) {
		uint32_t parent_index = fnv(node_index ^ i, dag_node.words[i % NODE_WORDS]) % light_words;
		for (uint32_t t = 0; t < 4; t++) {

			uint32_t shuffle_index = __shfl_sync(0xFFFFFFFF,parent_index, t, 4);

			uint4 p4 = g_light[shuffle_index].uint4s[thread_id];

			#pragma unroll
			for (int w = 0; w < 4; w++) {

				uint4 s4 = make_uint4(__shfl_sync(0xFFFFFFFF,p4.x, w, 4),
									  __shfl_sync(0xFFFFFFFF,p4.y, w, 4),
									  __shfl_sync(0xFFFFFFFF,p4.z, w, 4),
									  __shfl_sync(0xFFFFFFFF,p4.w, w, 4));
				if (t == thread_id) {
					dag_node.uint4s[w] = fnv4(dag_node.uint4s[w], s4);
				}
			}
		}
	}
	keccakf_1600(dag_node.uint64s);

	for (uint32_t t = 0; t < 4; t++) {
		uint32_t shuffle_index = __shfl_sync(0xFFFFFFFF,node_index, t, 4);
		uint4 s[4];
		for (uint32_t w = 0; w < 4; w++) {
			s[w] = make_uint4(__shfl_sync(0xFFFFFFFF,dag_node.uint4s[w].x, t, 4),
						      __shfl_sync(0xFFFFFFFF,dag_node.uint4s[w].y, t, 4),
							  __shfl_sync(0xFFFFFFFF,dag_node.uint4s[w].z, t, 4),
							  __shfl_sync(0xFFFFFFFF,dag_node.uint4s[w].w, t, 4));
		}
		g_dag[shuffle_index].uint4s[thread_id] = s[thread_id];
	}
}

void ethash_generate_dag(
	hash64_t* dag,
	uint64_t dag_bytes,
	hash64_t * light,
	uint32_t light_words,
	uint32_t blocks,
	uint32_t threads,
	int device
	)
{
	uint64_t const work = dag_bytes / sizeof(hash64_t);

	uint32_t fullRuns = (uint32_t)(work / (blocks * threads));
	uint32_t const restWork = (uint32_t)(work % (blocks * threads));
	if (restWork > 0) fullRuns++;
	for (uint32_t i = 0; i < fullRuns; i++)
	{
		ethash_calculate_dag_item <<<blocks, threads, 0 >>>(i * blocks * threads, dag, dag_bytes, light, light_words);
		CUDA_SAFE_CALL(cudaDeviceSynchronize());
	}
	CUDA_SAFE_CALL(cudaGetLastError());
}
