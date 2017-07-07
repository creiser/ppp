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


#define BYTE_TO_BINARY_PATTERN "%c%c%c%c%c%c%c%c"
#define BYTE_TO_BINARY(byte)  \
  (byte & 0x80 ? '1' : '0'), \
  (byte & 0x40 ? '1' : '0'), \
  (byte & 0x20 ? '1' : '0'), \
  (byte & 0x10 ? '1' : '0'), \
  (byte & 0x08 ? '1' : '0'), \
  (byte & 0x04 ? '1' : '0'), \
  (byte & 0x02 ? '1' : '0'), \
  (byte & 0x01 ? '1' : '0') 
  

#define INT_TO_BINARY_PATTERN "%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c"
#define INT_TO_BINARY(byte)  \
	(byte & 0x80000000 ? '1' : '0'), \
	(byte & 0x40000000 ? '1' : '0'), \
	(byte & 0x20000000 ? '1' : '0'), \
	(byte & 0x10000000 ? '1' : '0'), \
	(byte & 0x8000000 ? '1' : '0'), \
	(byte & 0x4000000 ? '1' : '0'), \
	(byte & 0x2000000 ? '1' : '0'), \
	(byte & 0x1000000 ? '1' : '0'), \
	(byte & 0x800000 ? '1' : '0'), \
	(byte & 0x400000 ? '1' : '0'), \
	(byte & 0x200000 ? '1' : '0'), \
	(byte & 0x100000 ? '1' : '0'), \
	(byte & 0x80000 ? '1' : '0'), \
	(byte & 0x40000 ? '1' : '0'), \
	(byte & 0x20000 ? '1' : '0'), \
	(byte & 0x10000 ? '1' : '0'), \
	(byte & 0x8000 ? '1' : '0'), \
	(byte & 0x4000 ? '1' : '0'), \
	(byte & 0x2000 ? '1' : '0'), \
	(byte & 0x1000 ? '1' : '0'), \
	(byte & 0x800 ? '1' : '0'), \
	(byte & 0x400 ? '1' : '0'), \
	(byte & 0x200 ? '1' : '0'), \
	(byte & 0x100 ? '1' : '0'), \
	(byte & 0x80 ? '1' : '0'), \
	(byte & 0x40 ? '1' : '0'), \
	(byte & 0x20 ? '1' : '0'), \
	(byte & 0x10 ? '1' : '0'), \
	(byte & 0x8 ? '1' : '0'), \
	(byte & 0x4 ? '1' : '0'), \
	(byte & 0x2 ? '1' : '0'), \
	(byte & 0x1 ? '1' : '0')


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
	
	// For each macro block a multipurpose array with size 288 bytes 
	// is allocated. We need enough space for the nibbles when they
	// are not packed yet (i.e. one nibble per byte), which sums
	// to 2*96=192 bytes. At the same time we will need space for the
	// packed result, which equals to 96 bytes. So in total 96+192=288
	// bytes are needed. We use the same space to save intermediate
	// floating point results. Since each float needs 4 byte of memory,
	// we need 64*4=256 bytes for a macro block. We do not need this
	// memory at the same as the other results and therefore the
	// array suffices, since 256 < 288.
	local uint8_t result_[4][288] __attribute__((aligned(16))); // result (96 bytes) + nibbles (192 bytes) = 288 bytes
    local uint8_t *result = result_[lb];
    local int8_t *sresult = (local int8_t *)result;
	local float *floatBuffer = (local float *)result; // 64 floats * 4 byte/float = 256 bytes < 288 bytes
	local uint8_t *nibbles = &result[96]; // first 96 bytes are reserved for "packed" nibbles, while the remaining 192 bytes are for "unpacked" nibbles 

	// TODO: memory efficieny
	local uint8_t codeOffset_[4][2][64] __attribute__((aligned(16)));
	local uint8_t *codeOffsetA = codeOffset_[lb][0];
	local uint8_t *codeOffsetB = codeOffset_[lb][1];


	// Save one length per macroblock, instead of using a private variable.
	local size_t lens[4] __attribute__((aligned(16)));
	
    const bool compr  =  format == 2;  // Is compression (-c) requested?

    // 'current' points to the upper left corner of the macro block
    // assigned to this thread.
    global uint8_t *current = image + 8*blockY*columns + 8*blockX;
	
	if (get_global_id(0) == 0 && get_global_id(1) == 0)
		printf("v13\n");
	barrier(0);
	
	const uint offsetX = self64 % 8;
	const uint offsetY = self64 / 8;
	
    switch(format) {
    case 0: {  // Exercise (a)
        // Reorder block directly to the output location
		sresult[self64] = get_block(current, columns, offsetX, offsetY);
        lens[lb] = 64;
        break;
    }
    case 1: {  // Exercise (b)
		floatBuffer[self64] = get_block(current, columns, offsetX, offsetY);
        dct(floatBuffer, sresult, self64, offsetX, offsetY);
		lens[lb] = 64;
        break;
    }
    case 2: {  // Exercise (c)
		floatBuffer[self64] = get_block(current, columns, offsetX, offsetY);
        dct(floatBuffer, sresult, self64, offsetX, offsetY);
		barrier(CLK_LOCAL_MEM_FENCE);
		
		// sheet 6 (a)
		// Every thread calculates up to three nibbles.
		uint8_t nibbles[3];
		uint8_t myNumNibbles;
		//nibbles[0] = nibbles[1] = nibbles[2] = 0;
		
		
		codeOffsetA[self64] = 42;
		
		int8_t val = sresult[self64];
		if (val == 0) {
			nibbles[0] = 1;
			myNumNibbles = codeOffsetA[self64] = 1;
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
		
		const int DEBUG_BLOCK = 4000;
		/*if (!self64 && block_nr == DEBUG_BLOCK) {
			printf("before\n");
			for (int i = 0; i < 64; i++)
				printf("i: %d, codeOffsetA[i]: %u, sresult[i]: %d\n", i, codeOffsetA[i], sresult[i]);
		}*/
		barrier(0);
		
		// Calculate the prefix sums in parallel with the naive algorithm
		// TODO: Double buffer
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
			
			/*if (!self64 && block_nr == 0) {
			printf("after %u\n", d);
			for (int i = 0; i < 64; i++)
				printf("i: %d, codeOffsetA[i]: %u\n", i, codeOffsetA[i]);
			}*/
		}
		
		/*if (!self64 && block_nr == DEBUG_BLOCK) {
			printf("after\n");
			for (int i = 0; i < 64; i++)
				printf("i: %d, codeOffsetA[i]: %u\n", i, codeOffsetA[i]);
		}*/
		
		// Initialize all 96 values with 0s.
		for (uint8_t i = self64; i < 96; i+=64)
			result[i] = 0;
		barrier(CLK_LOCAL_MEM_FENCE);
		
		if (block_nr == DEBUG_BLOCK) {
			for (int i = 0; i < myNumNibbles; i++) {
				printf("self64: %u, nibbles: %x\n", self64, nibbles[i]);
			}
		}
		barrier(0);
		
		const uint8_t shifts[8] = {4, 0, 12, 8, 20, 16, 28, 24};
		
		uint8_t currentNibble = 0;
		uint8_t myCodeOffset = self64 ? codeOffsetA[self64 - 1] : 0;
		for (uint8_t i = myCodeOffset; i < myCodeOffset + /*3*/myNumNibbles; i++) {
	
			// Divide by 8, since there are 8 nibbles in a a 32-bit int.
			local uint32_t *p = ((local uint32_t *)result) + (i >> 3);
			
			uint8_t shift = shifts[i % 8];
			uint32_t shifted_nibble = ((uint32_t)nibbles[currentNibble]) << shift;
			
			/*if (block_nr == DEBUG_BLOCK)
				printf("self64: %u, i: %u offset: %u shift: %u\n", self64, i, i >> 3, shift);*/
		
			// 0: shifted 28 bits, 1: shifted 24 bits, 2: shifted: 20 bits, ..
			atomic_or(p, shifted_nibble);
			
			
			/*if (block_nr == DEBUG_BLOCK) {
				printf("p: %08x, shifted: %08x, orig: %x, shift: %u\n", *p, shifted_nibble, nibbles[currentNibble], shift);
			}*/
			
			currentNibble++;
			
			//barrier(CLK_LOCAL_MEM_FENCE);
		}
		if (!self64) {
			lens[lb] = (codeOffsetA[63] + 1) / 2;
			//printf("codeOffsetA[63]: %u, lens[lb]: %u\n", codeOffsetA[63], lens[lb]);
		}
		
		barrier(CLK_LOCAL_MEM_FENCE);
		if (!self64 && block_nr == DEBUG_BLOCK) {
			for (int i = 0; i < 96; i++) {
				printf("i: %u, result: %02x\n", i, result[i]);
			}
		}
		
		
		
		
		//lens[lb] = 0; // TODO: correctly calculate a length
		// TODO: How to barrier this loop? Probably not at all because atomic operation immediately write their changes..
		
		// TODO: Add a nibble if the number of nibbles is odd.
		
		
        /*if (self64 == 0) {
			// Walk through the values.
			int zeros = 0, pos = 0;
			for (int i=0; i<64; i++) {
				int8_t val = sresult[i];
				if (val == 0)
					zeros++;
				else {
					// When we meet a non-zero value, we output an appropriate
					// number of codes for the zeros preceding the current
					// non-zero value.
					uint8_t absval = val < 0 ? -val : val;
					while (zeros > 0) {
						int z = zeros > 7 ? 7 : zeros;
						nibbles[pos++] = z;
						zeros -= z;
					}
					nibbles[pos++] = first_nibble(val);
					if (absval >= 19) {
						uint8_t code = absval - 19;
						nibbles[pos++] = code >> 4;
						nibbles[pos++] = code & 0xF;
					} else if (absval >= 3) {
						nibbles[pos++] = absval - 3;
					}
				}
			}

			// When the sequence ends with zeros, terminate the
			// sequence with 0x0.
			if (zeros > 0)
				nibbles[pos++] = 0;
			
			// Add a nibble 0x0 if the number of nibbles in the code
			// is odd.
			if ((pos & 1) != 0)
				nibbles[pos++] = 0;
			
			// Save the number of bytes used.
			lens[lb] = pos/2;
		}
		//
		// Pack the nibbles into bytes. The first nibble of each
		// pair of nibbles goes into the upper half of the byte.
		// We copy from the back part (unpacked nibbles) to the
		// the front part (packed nibbles).
		barrier(CLK_LOCAL_MEM_FENCE);
		for (int i=self64; i<lens[lb]; i+=64)
			result[i] = (nibbles[2*i] << 4) | nibbles[2*i+1];*/
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
