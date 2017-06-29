#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "ppp_video.h"

struct _video {
    AVFormatContext *pFormatCtx;
    AVCodecContext  *pCodecCtx;
    AVCodec         *pCodec;
    AVFrame         *pFrame; 
    AVFrame         *pFrameGray;
    struct SwsContext *imgConvertCtx;
};

video *video_open(const char *filename) {
    AVFormatContext *pFormatCtx;
    AVCodecContext  *pCodecCtx;
    AVCodec         *pCodec;
    AVFrame         *pFrame; 
    AVFrame         *pFrameGray;
    struct SwsContext *imgConvertCtx;
    video *v;
    int videoStream;
    int i;
    
    av_register_all();
    
    pFormatCtx = NULL;
    if (avformat_open_input(&pFormatCtx, filename, NULL, NULL) != 0)
        return NULL;
    
    if (avformat_find_stream_info(pFormatCtx, NULL) < 0)
        return NULL;
    
    videoStream = -1;
    for (i=0; i<pFormatCtx->nb_streams; i++) {
        if (pFormatCtx->streams[i]->codec->codec_type == AVMEDIA_TYPE_VIDEO) {
            videoStream = i;
            break;
        }
    }
    if (videoStream == -1)
        return NULL;
    
    pCodecCtx = pFormatCtx->streams[videoStream]->codec;
    
    pCodec = avcodec_find_decoder(pCodecCtx->codec_id);
    if(pCodec == NULL)
        return NULL;
    
    if (avcodec_open2(pCodecCtx, pCodec, NULL) < 0)
        return NULL;

    /* Allocate video frame */
    pFrame = av_frame_alloc();
    if (pFrame == NULL)
        return NULL;
    
    /* Allocate a frame for the gray version of the frame */
    pFrameGray = av_frame_alloc();
    if (pFrameGray == NULL)
        return NULL;
    
    /* Get a conversion to GRAY8 format */
    imgConvertCtx = sws_getContext(pCodecCtx->width, pCodecCtx->height, 
                                   pCodecCtx->pix_fmt, 
                                   pCodecCtx->width, pCodecCtx->height,
                                   AV_PIX_FMT_GRAY8, SWS_BICUBIC,
                                   NULL, NULL, NULL);
    if (imgConvertCtx == NULL)
        return NULL;

    v = (video *) malloc(sizeof(video));
    if (v == NULL)
        return NULL;
    
    v->pFormatCtx = pFormatCtx;
    v->pCodecCtx = pCodecCtx;
    v->imgConvertCtx = imgConvertCtx;
    v->pFrameGray = pFrameGray;
    v->pFrame = pFrame;
    
    return v;
}

void video_close(video *v) {
    av_free(v->pFrame);
    av_free(v->pFrameGray);
    avformat_close_input(&(v->pFormatCtx));
}

ppp_frame *video_alloc_frame(const video *v) {
    int numBytes;
    
    /* Determine required buffer size */
    numBytes = avpicture_get_size(AV_PIX_FMT_GRAY8, v->pCodecCtx->width,
                                  v->pCodecCtx->height);
    
    /* Allocate the buffer */
    return ppp_frame_alloc(numBytes);
}

int video_get_width(const video *v) {
    return v->pCodecCtx->width;
}

int video_get_height(const video *v) {
    return v->pCodecCtx->height;
}

float video_get_fps(const video *v) {
    AVRational time_base = v->pCodecCtx->time_base;
    int ticks_per_frame = v->pCodecCtx->ticks_per_frame;
    float fps = (float)time_base.den / ticks_per_frame / time_base.num;
    return fps;
}

int video_get_next_frame(video *v, ppp_frame *frame) {
    AVPacket packet;
    int frameFinished;
    int res;
    
    do {
        res = av_read_frame(v->pFormatCtx, &packet);
        if (res < 0)
            return res;
        
        avcodec_decode_video2(v->pCodecCtx, v->pFrame, &frameFinished, 
                              &packet);
    } while (frameFinished == 0);
    
    /* Use frame->data as buffer for the gray frame */
    avpicture_fill((AVPicture *)v->pFrameGray, frame->data, AV_PIX_FMT_GRAY8,
                   v->pCodecCtx->width, v->pCodecCtx->height);

    /* Convert frame to gray scale */
    const uint8_t *const *p = (const uint8_t *const *)v->pFrame->data;
    res = sws_scale(v->imgConvertCtx, p, v->pFrame->linesize, 0, 
                    v->pCodecCtx->height, v->pFrameGray->data,
                    v->pFrameGray->linesize);
    frame->length = v->pCodecCtx->height * v->pCodecCtx->width;
    return 0;
}
