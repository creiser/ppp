/* Hey, Emacs, this file contains -*- c -*- code. */

/*
 * Define some types so we can use the same types as
 * in the host code.
 */
typedef uint  uint32_t;
typedef short int16_t;
typedef ushort uint16_t;
typedef uchar uint8_t;
typedef char int8_t;

/*
 * We access individual bytes, so we need the
 * byte addressable storage extension
 */
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

/*
 * Return the id of the current thread within the
 * thread's macro block, i.e., each thread gets a number
 * between 0 and 63 (x coordinate plus 8 times y coordinate
 * within the macro block).
 */
inline size_t SELF64() {
    return get_local_id(0)%8 + 8*(get_local_id(1)%8);
}

/*
 * Coefficients of the matrix A to be used in DCT computation
 */
__attribute__((aligned(16)))
constant float dct_coeffs[64] = {
    0.35355338f ,  0.35355338f ,  0.35355338f ,  0.35355338f  ,
                   0.35355338f ,  0.35355338f ,  0.35355338f  ,  0.35355338f, 
    0.49039263f ,  0.4157348f  ,  0.2777851f  ,  9.754512e-2f ,
                  -9.754516e-2f, -0.27778518f , -0.41573483f  , -0.49039266f,
    0.46193978f ,  0.19134171f , -0.19134176f , -0.46193978f  ,
                  -0.46193978f , -0.19134156f ,  0.1913418f   ,  0.46193978f,
    0.4157348f  , -9.754516e-2f, -0.49039266f , -0.277785f    ,  
                   0.2777852f  ,  0.49039263f ,  9.7545035e-2f, -0.4157349f,   
    0.35355338f , -0.35355338f , -0.35355332f ,  0.3535535f   , 
                   0.35355338f , -0.35355362f , -0.35355327f  ,  0.3535534f,
    0.2777851f  , -0.49039266f ,  9.754521e-2f,  0.41573468f  ,
                  -0.4157349f  , -9.754511e-2f,  0.49039266f  , -0.27778542f, 
    0.19134171f , -0.46193978f ,  0.46193978f , -0.19134195f  ,
                  -0.19134149f ,  0.46193966f , -0.46193987f  ,  0.19134195f,
    9.754512e-2f, -0.277785f   ,  0.41573468f , -0.4903926f   ,
                   0.4903927f  , -0.4157348f  ,  0.27778557f  , -9.754577e-2f
};

/*
 * Coefficients of the matrix A^tr to be used in DCT computation
 */
__attribute__((aligned(16)))
constant float dct_coeffs_tr[64] = {
    0.35355338f,  0.49039263f ,  0.46193978f ,  0.4157348f   ,
                  0.35355338f ,  0.2777851f  ,  0.19134171f  , 9.754512e-2f,
    0.35355338f,  0.4157348f  ,  0.19134171f , -9.754516e-2f ,
                 -0.35355338f , -0.49039266f , -0.46193978f  , -0.277785f,
    0.35355338f,  0.2777851f  , -0.19134176f , -0.49039266f  ,
                 -0.35355332f ,  9.754521e-2f, 0.46193978f   , 0.41573468f,
    0.35355338f,  9.754512e-2f, -0.46193978f , -0.277785f    ,
                  0.3535535f  ,  0.41573468f , -0.19134195f  , -0.4903926f,
    0.35355338f, -9.754516e-2f, -0.46193978f ,  0.2777852f   ,
                  0.35355338f , -0.4157349f  , -0.19134149f  , 0.4903927f,
    0.35355338f, -0.27778518f , -0.19134156f ,  0.49039263f  ,
                 -0.35355362f , -9.754511e-2f,  0.46193966f  , -0.4157348f,
    0.35355338f, -0.41573483f ,  0.1913418f  ,  9.7545035e-2f,
                 -0.35355327f ,  0.49039266f , -0.46193987f  , 0.27778557f,
    0.35355338f, -0.49039266f ,  0.46193978f , -0.4157349f   ,
                  0.3535534f ,  -0.27778542f ,  0.19134195f  , -9.754577e-2f
};

/*
 * Permutation of 64 values in a macro block.
 * Used in qdct_block() and iqdct_block().
 */
constant int permut[64] = {
     0,  1,  5,  6, 14, 15, 27, 28,
     2,  4,  7, 13, 16, 26, 29, 42,
     3,  8, 12, 17, 25, 30, 41, 43,
     9, 11, 18, 24, 31, 40, 44, 53,
    10, 19, 23, 32, 39, 45, 52, 54,
    20, 22, 33, 38, 46, 51, 55, 60,
    21, 34, 37, 47, 50, 56, 59, 61,
    35, 36, 48, 49, 57, 58, 62, 63
};

/*
 * Quantization factors for the results of the DCT
 * of a macro block. Used in qdct_block() and iqdct_block().
 */
constant int quantization_factors[64] = {
    16,  11,  10,  16,  24,  40,  51,  61,
    12,  12,  14,  19,  26,  58,  60,  55,
    14,  13,  16,  24,  40,  57,  69,  56,
    14,  17,  22,  29,  51,  87,  80,  62,
    18,  22,  37,  56,  68, 109, 103,  77,
    24,  35,  55,  64,  81, 104, 113,  92,
    49,  64,  78,  87, 103, 121, 120, 101,
    72,  92,  95,  98, 112, 100, 103,  99
};


/*
 * Return the offset (in bytes) where the output data for our
 * block (which has 'len' bytes) with block number 'block_nr'
 * has to be stored. 'compr' indicates whether compression is
 * enabled. 'size' and 'offsets_and_sizes' correspond to the
 * kernel arguments of the same names.
 */
void get_result_offset(int len, uint block_nr, bool compr,
                       global uint *size, global uint *offsets_and_sizes,
                       local size_t *off) {
    const size_t self64 = SELF64();
    if (self64 == 0) {
        /* Atomically increment 'size' by 'len'. The return value
         * is the old value of 'size'.
         */
        size_t old_size = atomic_add(size, len);

        /* When compression is not enabled, our data has to be put
         * at 64*block_nr. With compression, we store the output data
         * contiguously, but we cannot guarantee the order (the CPU
         * will reorder the data later).
         */
        *off = compr ? old_size : 64*block_nr;

        /* Write out the offset and the length of our data so
         * the CPU knows to find the data for our block.
	 * We do it in the strange way (using atomic_xchg)
	 * instead of simple assignments solely for the
	 * the purpose of avoiding the (spurious)
	 * "kernel has register spilling. Lower performance is expected."
	 * warning a certain vendor's OpenCL implementation
	 * generates.
         */
	// offsets_and_sizes[2*block_nr  ] = off;
	// offsets_and_sizes[2*block_nr+1] = len;
	atomic_xchg(&offsets_and_sizes[2*block_nr  ], *off);
	atomic_xchg(&offsets_and_sizes[2*block_nr+1], len);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

uint8_t first_nibble(int8_t val) {
    if (val == 1)
        return 0x8;
    else if (val == 2)
        return 0x9;
    else if (val == -1)
        return 0xA;
    else if (val == -2)
        return 0xB;
    else if (val >= 19)
        return 0xE;
    else if (val <= -19)
        return 0xF;
    else if (val <= -3)
        return 0xD;
    else
        return 0xC;
}

int8_t get_block(global uint8_t *current, int columns, const uint offsetX, const uint offsetY)
{
	return (int)current[columns * offsetY + offsetX] - 128;
}

void dct(local float* floatBuffer, local int8_t* sresult, const uint self64, const uint offsetX, const uint offsetY)
{
	barrier(CLK_LOCAL_MEM_FENCE);
	// A * B (A := dct_coeffs, B := floatBuffer, A^tr := dct_coeffs_tr)
	float matrixProduct = 0;
	for (int8_t i = 0; i < 8; i++)
		matrixProduct += dct_coeffs[offsetY * 8 + i] * floatBuffer[i * 8 + offsetX];
	barrier(CLK_LOCAL_MEM_FENCE);
	floatBuffer[self64] = matrixProduct;
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// B * A^tr
	matrixProduct = 0;
	for (int8_t i = 0; i < 8; i++)
		matrixProduct += floatBuffer[offsetY * 8 + i] * dct_coeffs_tr[i * 8 + offsetX];
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// Quantization amd zig zag permutation
	sresult[permut[self64]] = convert_char_rte(matrixProduct / quantization_factors[self64]);
}

/*
 * Encode 4 macro blocks with 8x8 pixels in each work group.
 * Each thread computes its macro block's x and y coordinate
 * in blockX and blockY. self64 is the (linearized row-wise)
 * index of each thread within its macro block (in the range 0..63).
 * block_nr is the global number of the macro block (numbering the
 * macro blocks row-wise from left to right).
 * 'image', 'rows' and 'columns' describe the input image.
 * 'format' tell whether to reoder only, apply DCT or
 * compute DCT+compression.
 * The result is stored in 'frame'.
 * 'size' and 'offsets_and_sizes' are used when compression is enabled
 * to tell the host where to find the data for each macro block
 * (see below).
 */
__attribute__((reqd_work_group_size(16, 16, 1)))
kernel void encode_frame(global uint8_t *image, int rows, int columns,
                         int format, global uint *size,
                         global uint *offsets_and_sizes,
                         global uint8_t *frame) {
    const uint self64 = SELF64();
    const uint blockX = get_global_id(0)/8;
    const uint blockY = get_global_id(1)/8;
    const uint block_nr = blockY*(columns/8) + blockX;

    // lb is the local block number, i.e., which of the 4
    // macro blocks in a work group this thread is in
    // (0 = upper left, 1 = upper right, 2 = lower left, 3 = lower right).
    const uint lb = get_local_id(0)/8 + 2*(get_local_id(1)/8);
	
	// For each macro block a multipurpose array with size 256 bytes 
	// is allocated, since we need enough space for the DCT phase, i.e. 64 floats.
	local uint8_t result_[4][256] __attribute__((aligned(16))); // 
    local uint8_t *result = result_[lb]; // result max 64 nibbles, i.e. 96 bytes < 256 bytes
    local int8_t *sresult = (local int8_t *)result;
	local float *floatBuffer = (local float *)result; // 64 floats * 4 byte/float = 256 bytes
	
	// By letting "codeOffset" point somewhere else than "result" we can avoid a barrier.
	local uint8_t *codeOffsetA = &result[96];
	local uint8_t *codeOffsetB = &result[160];
	local uint8_t *zeroEndsA = result;
	local uint8_t *zeroEndsB = &result[64];

	// Save one length per macroblock, instead of using a private variable.
	local size_t lens[4] __attribute__((aligned(16)));
	
    const bool compr  =  format == 2;  // Is compression (-c) requested?

    // 'current' points to the upper left corner of the macro block
    // assigned to this thread.
    global uint8_t *current = image + 8*blockY*columns + 8*blockX;
	
	if (get_global_id(0) == 0 && get_global_id(1) == 0)
		printf("v15\n");
	barrier(0);
	
	const uint offsetX = self64 % 8;
	const uint offsetY = self64 / 8;
	
    switch(format) {
    case 0: {
        // Reorder block directly to the output location
		sresult[self64] = get_block(current, columns, offsetX, offsetY);
        lens[lb] = 64;
        break;
    }
    case 1: {
		floatBuffer[self64] = get_block(current, columns, offsetX, offsetY);
        dct(floatBuffer, sresult, self64, offsetX, offsetY);
		lens[lb] = 64;
        break;
    }
    case 2: {  // Exercise (a/b)
		floatBuffer[self64] = get_block(current, columns, offsetX, offsetY);
        dct(floatBuffer, sresult, self64, offsetX, offsetY);
		barrier(CLK_LOCAL_MEM_FENCE);
		
		// Sheet 6: parallel entropy coding
		// Every thread calculates up to three nibbles. Exception: If
		// a thread is assigned the first item of a zero sequence, it 
		// will encode the whole zero sequence. If there are more than
		// 21 zeros, it will therefore encode more than 3 nibbles.
		uint8_t nibbles[3];
		
		// Save a few values to private memory, because we will reuse the local
		// buffer again to save the ends of zero sequences.
		int8_t val = sresult[self64];
		int8_t isFirstZero = 0, isLastZero = 0;
		if (val == 0) {
			isFirstZero = self64 ? sresult[self64 - 1] : 1;
			isLastZero = self64 != 63 ? sresult[self64 + 1] : 1;
		}
		barrier(0);
		zeroEndsA[self64] = isLastZero ? self64 : 64;
		barrier(CLK_LOCAL_MEM_FENCE);
		
		// Reversed min reduce to get the end of zero sequences.
		for (uint8_t d = 1; d < 64; d = d << 1) {
			if (self64 < 64 - d) {
				zeroEndsB[self64] = min(zeroEndsA[self64 + d], zeroEndsA[self64]);
			} else {
				zeroEndsB[self64] = zeroEndsA[self64];
			}
			local uint8_t *temp = zeroEndsA;
			zeroEndsA = zeroEndsB;
			zeroEndsB = temp;
			barrier(CLK_LOCAL_MEM_FENCE);
		}
	
		uint8_t myNumNibbles;
		uint8_t zerosToWrite = 0;
		if (val == 0) {
			if (isFirstZero) {
				if (zeroEndsA[self64] != 63) {
					// Instead of writing to the private nibbles array, we just save
					// the length of the zero sequence that starts at this thread and
					// convert that information later to nibbles.
					zerosToWrite = zeroEndsA[self64] - self64 + 1;
					myNumNibbles = codeOffsetA[self64] = (zerosToWrite + 6) / 7;
				} else {
					// Write a terminating zero nibble.
					nibbles[0] = 0;
					myNumNibbles = codeOffsetA[self64] = 1;
				}
			} else {
				// Only the thread corresponding to the first item of a zero sequence
				// encodes the number of zeros.
				myNumNibbles = codeOffsetA[self64] = 0;
			}
		} else {
			uint8_t absval = val < 0 ? -val : val;
			nibbles[0] = first_nibble(val);
			if (absval >= 19) {
				uint8_t code = absval - 19;
				nibbles[1] = code >> 4;
				nibbles[2] = code & 0xF;
				myNumNibbles = codeOffsetA[self64] = 3;
			} else if (absval >= 3) {
				nibbles[1] = absval - 3;
				myNumNibbles = codeOffsetA[self64] = 2;
			} else {
				myNumNibbles = codeOffsetA[self64] = 1;
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		
		// Calculate the prefix sums in parallel with the naive algorithm
		for (uint8_t d = 1; d < 64; d = d << 1) {
			if (self64 >= d) {
				codeOffsetB[self64] = codeOffsetA[self64 - d] + codeOffsetA[self64];
			} else {
				codeOffsetB[self64] = codeOffsetA[self64];
			}
			local uint8_t *temp = codeOffsetA;
			codeOffsetA = codeOffsetB;
			codeOffsetB = temp;
			barrier(CLK_LOCAL_MEM_FENCE);
		}
		
		// Store the length of the compressed macro block
		if (!self64) {
			lens[lb] = (codeOffsetA[63] + 1) / 2;
		}
		// A barrier would have to be placed here, if we didn't let "codeOffset"
		// point somewhere else in memory than "result".
		
		// Initialize all 96 values with 0s.
		for (uint8_t i = self64; i < 96; i += 64)
			result[i] = 0;
		barrier(CLK_LOCAL_MEM_FENCE);
		
		// Pack nibbles into bytes.
		// Define shifts for little endianess.
		const uint8_t shifts[8] = {4, 0, 12, 8, 20, 16, 28, 24};
		uint8_t currentNibble = 0;
		uint8_t myCodeOffset = self64 ? codeOffsetA[self64 - 1] : 0;
		for (uint8_t i = myCodeOffset; i < myCodeOffset + myNumNibbles; i++) {
	
			// Divide by 8, since there are 8 nibbles in a a 32-bit int.
			local uint32_t *p = ((local uint32_t *)result) + (i >> 3);
			uint8_t nibble;
			if (zerosToWrite) {
				// We can encode only 7 zeros in a single nibble.
				nibble = min(zerosToWrite - 7 * currentNibble, 7); 
			} else {
				nibble = nibbles[currentNibble];
			}
			uint8_t shift = shifts[i % 8];
			uint32_t shifted_nibble = ((uint32_t)nibble) << shift;		
			atomic_or(p, shifted_nibble);
			currentNibble++;
		}
        break;
    }
    default:
        lens[lb] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
	
    // Compute at which offset to put our data.
    // When compression is not enabled, the position of the output
    // data is simply 64*block_nr. When compression is enabled,
    // this is more difficult as we do not know how many bytes the
    // preceding macro blocks occupy (and we cannot synchronize
    // accross work groups). Therefore, the data for each macro block
    // is appended to the output as soon as the block is ready.
    // get_result_offset atomically increments *size by len
    // and stores the old value of *size together with len
    // in offsets_and_sizes, so the host knows later where
    // we stored our data. The host reorders the results to
    // be in the correct order. The value of *size before
    // the increment by len is stored in off[lb], so we can
    // put our data at frame[off[lb]..off[lb]+len-1].
    local size_t off[4];
    get_result_offset(lens[lb], block_nr, compr,
                      size, offsets_and_sizes, &off[lb]);
    for (int i=self64; i<lens[lb]; i+=64)
        frame[off[lb]+i] = result[i];
    barrier(CLK_GLOBAL_MEM_FENCE);
}
