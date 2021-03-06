#include <assert.h>
#include <getopt.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <CL/cl.h>

#include "compression_stats.h"
#include "frame_encoding.h"
#include "ocl_init.h"
#include "ppp_image.h"
#include "ppp_pnm.h"


enum impl_type {
    IMPL_SEQ, IMPL_CPU, IMPL_GPU, IMPL_ACC
};

double seconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + ((double)tv.tv_usec)/1000000.0;
}  

/*
 * Encode a frame using the CPU implementation.
 */
static ppp_frame *encode(uint8_t *image, const ppp_image_info *info) {
    ppp_frame *frame;
    const int rows = info->rows;
    const int columns = info->columns;
    const enum ppp_image_format format = info->format;

    /*
     * max_enc_bytes is the maximal number of bytes needed 
     * in any of the supported formats.
     */
    const int max_enc_bytes = max_encoded_length(rows*columns);

    /*
     * Allocate a frame for the result.
     */
    frame = ppp_frame_alloc(max_enc_bytes);
    if (frame == NULL)
        return NULL;

    double t = seconds();

    /*
     * Encode the image according to 'format'.
     */
    encode_frame(image, rows, columns, format, frame);
    
    t = seconds() - t;
    printf("Size: %u\n", (unsigned int)frame->length);
    printf("Duration: %.3f ms\n", t*1000);

    return frame;
}


/*
 * Encode a frame using the OpenCL implementation.
 */
static ppp_frame *encode_opencl(uint8_t *image, const ppp_image_info *info,
                                enum ocl_device_type devtype) {
    const cl_uint rows = info->rows;
    const cl_uint columns = info->columns;
    const cl_uint n_blocks = (rows/8) * (columns/8);

    const cl_uint format = info->format;
    const bool compr  =  format == PPP_IMGFMT_COMPRESSED_DCT;

    /*
     * max_enc_bytes is the maximal number of bytes needed 
     * in any of the supported formats.
     */
    const int max_enc_bytes = max_encoded_length(rows*columns);

    cl_context context;
    cl_device_id devid;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernelEncode;
    cl_mem imageGPU, offsets_and_sizesGPU, sizeGPU, frameGPU;
    cl_int res;

    /*
     * Number of dimensions for the workgroups.
     */
    size_t work_dims = 2;

    /*
     * Set the work group size and global number of work items.
     */
    size_t global_work_size[] = { columns, rows };
    size_t local_work_size[]  = {      16,   16 };

    /*
     * Kernel function name
     */
    const char *kernelName = "encode_frame";

    ppp_frame *frame;
    cl_uint size;

    context = create_cl_context(devtype, &devid);
    
    /* Create a command queue (which allows timing) */
    queue = clCreateCommandQueue(context, devid, CL_QUEUE_PROFILING_ENABLE,
                                 &res);
    if (queue == NULL)
        error_and_abort("Could not create command queue", res);
    
    /* Allocate the buffer memory object for the input image.
     * We request that the image (from the host) is copied to the device.
     */
    imageGPU = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              sizeof(*image)*rows*columns, (void *)image, &res);
   if (res != CL_SUCCESS)
        error_and_abort("Could not allocate imageGPU", res);
   offsets_and_sizesGPU =
        clCreateBuffer(context, CL_MEM_READ_WRITE,
                       sizeof(cl_uint)*2*n_blocks, NULL, &res);
   if (res != CL_SUCCESS)
        error_and_abort("Could not allocate offsets_and_sizesGPU", res);

   /* Initialize the result frame size to 0. */
   size = 0;
   sizeGPU  = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                              sizeof(size), (void*)&size, &res);
   if (res != CL_SUCCESS)
        error_and_abort("Could not allocate sizesGPU", res);
    /* Allocate the buffer memory object for the result.
     * We need at most 'max_enc_bytes' to represent the result.
     */
    frameGPU = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                              max_enc_bytes, NULL, &res);
   if (res != CL_SUCCESS)
        error_and_abort("Could not allocate frameGPU", res);

    /*
     * Load the OpenCL program from file "image_encoder_kernels.cl".
     */
    program = build_program("image_encoder_kernels.cl", context, devid);
    
    /* Find kernel "encode_frame" in the compiled program. */
    kernelEncode = clCreateKernel(program, kernelName, &res);
    if (res != CL_SUCCESS)
        error_and_abort("Could not create kernel", res);

    /* Set the arguments for the kernel invocation.
     * We pass the pointers to the input image, the number of rows
     * and columns, the format and the pointer to the result.
     */
    clSetKernelArg(kernelEncode, 0, sizeof(cl_mem), &imageGPU);
    clSetKernelArg(kernelEncode, 1, sizeof(cl_int), &rows);
    clSetKernelArg(kernelEncode, 2, sizeof(cl_int), &columns);
    clSetKernelArg(kernelEncode, 3, sizeof(cl_int), &format);
    clSetKernelArg(kernelEncode, 4, sizeof(cl_mem), &sizeGPU);
    clSetKernelArg(kernelEncode, 5, sizeof(cl_mem), &offsets_and_sizesGPU);
    clSetKernelArg(kernelEncode, 6, sizeof(cl_mem), &frameGPU);

    cl_event encodeEvent;
    res = clEnqueueNDRangeKernel(queue, kernelEncode, work_dims, NULL,
                                 global_work_size, local_work_size,
                                 0, NULL, &encodeEvent);
    if (res != CL_SUCCESS)
        error_and_abort("Could not enqueue kernel invocation", res);

    res = clWaitForEvents(1, &encodeEvent);
    if (res == CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST) {
        cl_int kernelStatus;
        res = clGetEventInfo(encodeEvent, CL_EVENT_COMMAND_EXECUTION_STATUS,
                             sizeof(kernelStatus), &kernelStatus, NULL);
        if (res != CL_SUCCESS)
            error_and_abort("Could not query kernel execution status", res);
        if (kernelStatus != CL_COMPLETE)
            error_and_abort("Kernel did not complete correctly", kernelStatus);
    } else if (res != CL_SUCCESS)
        error_and_abort("Waiting for kernel completion failed", res);

    res = clEnqueueReadBuffer(queue, sizeGPU, CL_TRUE, 0,
                              sizeof(size), &size, 0, NULL, NULL);
    if (res != CL_SUCCESS)
        error_and_abort("Could not enqueue buffer read for 'size'", res);

    if (size > 96*n_blocks)
        error_and_abort("Frame size is greather than 96*n_blocks", size);

    /* Allocate space for the result. */
    frame = ppp_frame_alloc(size);
    if (frame == NULL)
        error_and_abort("Could not allocate frame", 0);
    frame->length = size;

    double reorderTime = 0;
    if (compr) {
        reorderTime = seconds();
        /* Map 'offsets_and_sizesGPU' and 'frameGPU' into host address space
         * and walk through the blocks copying them to 'frame' in the
         * correct position.
         */
        cl_uint *offsets_and_sizes = (cl_uint *)
            clEnqueueMapBuffer(queue, offsets_and_sizesGPU, CL_TRUE,
                               CL_MAP_READ, 0, sizeof(cl_uint)*2*n_blocks,
                               0, NULL, NULL, &res);
        if (res != CL_SUCCESS)
            error_and_abort("Could not enqueue buffer mapping for 'offsets_and_sizes'", res);

        uint8_t *framePtr = (uint8_t *)
            (frame->length > 0 ?
               clEnqueueMapBuffer(queue, frameGPU, CL_TRUE, CL_MAP_READ,
                                  0, frame->length, 0, NULL, NULL, &res)
             : NULL);
        if (res != CL_SUCCESS)
            error_and_abort("Could not map 'frameGPU'", res);

        uint8_t *frameData = (uint8_t *)frame->data;
        for (int blk=0; blk<n_blocks; blk++) {
            cl_uint off = offsets_and_sizes[2*blk];
            cl_uint len = offsets_and_sizes[2*blk+1];
            if (len <= 96 && off+len <= size) {
                memcpy(frameData, framePtr+off, len);
                frameData += len;
            } else
                fprintf(stderr, "Error in block %d: len=%u, offset=%u\n",
                        blk, (unsigned)len, (unsigned)off);
        }

        res = clEnqueueUnmapMemObject(queue, offsets_and_sizesGPU, offsets_and_sizes,
                                      0, NULL, NULL);
            if (res != CL_SUCCESS)
                error_and_abort("Could not unmap 'offsets_and_sizesGPU'", res);

        if (framePtr != NULL) {
            res = clEnqueueUnmapMemObject(queue, frameGPU, framePtr, 0, NULL, NULL);
            if (res != CL_SUCCESS)
                error_and_abort("Could not unmap 'frameGPU'", res);
        }
        reorderTime = seconds() - reorderTime;
    } else {
        /* Copy the result from the device to the host. */
        res = clEnqueueReadBuffer(queue, frameGPU, CL_TRUE, 0,
                                  frame->length, frame->data, 0, NULL, NULL);
        if (res != CL_SUCCESS)
            error_and_abort("Could not enqueue buffer read for 'frame'", res);
    }
    printf("Size: %u\n", (unsigned int)frame->length);

    res = clFinish(queue);
    if (res != CL_SUCCESS)
        error_and_abort("Error at clFinish", res);

    cl_ulong nanos = get_event_end_nanos(encodeEvent) -
       get_event_start_nanos(encodeEvent);
    printf("Encoding: %.3f ms\n", nanos/1.0e6);
    if (compr) {
        printf("Reordering: %.3f ms\n", reorderTime*1.0e3);
    }

    clReleaseMemObject(imageGPU);
    clReleaseMemObject(sizeGPU);
    clReleaseMemObject(offsets_and_sizesGPU);
    clReleaseMemObject(frameGPU);
    clReleaseKernel(kernelEncode);
    clReleaseContext(context);

    return frame;
}


static void usage(const char *progname) {
  fprintf(stderr, "USAGE: %s -i IN -o OUT [-d | -c] [-z IMPL]\n",
          progname);
  fprintf(stderr,
	  "  -d: with DCT\n"
	  "  -c: with DCT and compression\n"
          "  -z: select implementation:\n"
          "        seq: CPU sequential (default)\n"
          "        cpu: CPU via OpenCL\n"
          "        gpu: GPU via OpenCL\n"
          "        acc: Accelerator via OpenCL\n"
          "  -s: show encoder statistics (only with -c)\n"
          "  -h: show this help\n"
          "\n"
	  );
}

int main(int argc, char *argv[]) {
    enum pnm_kind kind;
    enum ppp_image_format format;
    int rows, columns, maxcolor;
    uint8_t *image;
    int option;

    char *infile, *outfile;
    bool dct, compress, show_stats;
    enum impl_type implementation;

    init_qdct();

    infile = NULL;
    outfile = NULL;
    dct = false;
    compress = false;
    show_stats = false;
    implementation = IMPL_SEQ;
    while ((option = getopt(argc,argv,"cdhi:o:sz:")) != -1) {
        switch(option) {
        case 'd': dct = true; break;
        case 'c': compress = true; break;
        case 'i': infile = strdup(optarg); break;
        case 'o': outfile = strdup(optarg); break;
        case 's': show_stats = true; break;
        case 'z':
            if (strcmp(optarg, "seq") == 0) {
                implementation = IMPL_SEQ;
                break;
            } else if (strcmp(optarg, "cpu") == 0) {
                implementation = IMPL_CPU;
                break;
            } else if (strcmp(optarg, "gpu") == 0) {
                implementation = IMPL_GPU;
                break;
            } else if (strcmp(optarg, "acc") == 0) {
                implementation = IMPL_ACC;
                break;
            }
            /* fall through */
        case 'h':
        default:
            usage(argv[0]);
            return 1;
        }
    }
    
    if (infile == NULL || outfile == NULL || (dct && compress)) {
        usage(argv[0]);
        return 1;
    }
    
    if (dct)
        format = PPP_IMGFMT_UNCOMPRESSED_DCT;
    else if (compress)
        format = PPP_IMGFMT_COMPRESSED_DCT;
    else
        format = PPP_IMGFMT_UNCOMPRESSED_BLOCKS;
    
    image = ppp_pnm_read(infile, &kind, &rows, &columns, &maxcolor);
    
    if (image != NULL) {
        if (rows%8 != 0 || columns%8 != 0) {
            fprintf(stderr, "Error: number of rows and columns must be "
                    "multiples of 8\n");
        } else if (kind == PNM_KIND_PGM) {
            ppp_image_info info;
            ppp_frame *frame;
            info.rows = rows;
            info.columns = columns;
            info.format = format;

            if (implementation == IMPL_SEQ)
                frame = encode(image, &info);
            else {
                enum ocl_device_type devtype;
                switch(implementation) {
                case IMPL_CPU: devtype = ODT_CPU; break;
                case IMPL_GPU: devtype = ODT_GPU; break;
                case IMPL_ACC: devtype = ODT_ACC; break;
                default:
                    fprintf(stderr, "Internal error: implementation type "
                            "%d not handled\n", implementation);
                    devtype = ODT_CPU;
                }
                frame = encode_opencl(image, &info, devtype);
				
				// Compare to reference implementation (good for debugging UNCOMPRESSED_ options)
				/*ppp_frame *ref_frame = encode(image, &info);
				int numDifferent = 0;
				for (int i = 0; i < ref_frame->length; i++)
				{
					if (ref_frame->data[i] != frame->data[i])
					{
						printf("Difference at (%d, %d): %d vs. %d\n", i % columns, i / columns, (int8_t)ref_frame->data[i], (int8_t)frame->data[i]);
						numDifferent++;
					}
				}
				printf("Number of different values: %d\n", numDifferent);
				free(ref_frame);*/
            }
            if (frame != NULL) {
                if (show_stats && format == PPP_IMGFMT_COMPRESSED_DCT) {
                    encoder_stats_init();
                    encoder_stats(frame->data, frame->length);
                    encoder_stats_print(stderr);
                }

                if (ppp_image_write(outfile, &info, frame) != 0)
                    fprintf(stderr, "could not write image\n");
                free(frame);
            } else
                fprintf(stderr, "error while encoding\n");
        } else
            fprintf(stderr, "not a PGM image\n");
        free(image);
    } else
        fprintf(stderr, "could not load image\n");
    return 0;
}

