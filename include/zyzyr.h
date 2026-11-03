#ifndef ZYZYR_H
#define ZYZYR_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ZyzyrDecoder ZyzyrDecoder;
typedef struct ZyzyrDict    ZyzyrDict;

typedef struct ZyzyrStatus {
    int32_t code;
} ZyzyrStatus;

#define ZYZYR_OK                  0
#define ZYZYR_UNKNOWN_SIZE        1
#define ZYZYR_NULL               -1
#define ZYZYR_SHORT_BUFFER       -2
#define ZYZYR_LENGTH             -3
#define ZYZYR_ERR_TRUNCATED     -10
#define ZYZYR_ERR_BAD_MAGIC     -11
#define ZYZYR_ERR_RESERVED_BIT  -12
#define ZYZYR_ERR_DICTIONARY    -13
#define ZYZYR_ERR_DICT_FORMAT   -14
#define ZYZYR_ERR_BLOCK_TYPE    -15
#define ZYZYR_ERR_BLOCK_SIZE    -16
#define ZYZYR_ERR_LITERALS      -17
#define ZYZYR_ERR_HUFFMAN       -18
#define ZYZYR_ERR_FSE           -19
#define ZYZYR_ERR_BITSTREAM     -20
#define ZYZYR_ERR_SEQUENCES     -21
#define ZYZYR_ERR_OFFSET        -22
#define ZYZYR_ERR_CONTENT_SIZE  -23
#define ZYZYR_ERR_CHECKSUM      -24

ZyzyrStatus zyzyr_decompress(
    const uint8_t *data,
    size_t         len,
    uint8_t       *out,
    size_t         out_cap,
    size_t        *out_len);

ZyzyrStatus zyzyr_decompress_with_dict(
    const uint8_t       *data,
    size_t               len,
    const ZyzyrDict *dict,
    uint8_t             *out,
    size_t               out_cap,
    size_t              *out_len);

ZyzyrStatus zyzyr_content_size(
    const uint8_t *data,
    size_t         len,
    uint64_t      *size);

ZyzyrStatus zyzyr_dictionary_id(
    const uint8_t *data,
    size_t         len,
    uint32_t      *id);

ZyzyrStatus zyzyr_dict_new(
    const uint8_t  *data,
    size_t          len,
    ZyzyrDict **out);
void     zyzyr_dict_free(ZyzyrDict *dict);
uint32_t zyzyr_dict_id(const ZyzyrDict *dict);

ZyzyrDecoder *zyzyr_decoder_new(void);
void              zyzyr_decoder_free(ZyzyrDecoder *dec);

ZyzyrStatus zyzyr_decoder_decompress(
    ZyzyrDecoder    *dec,
    const uint8_t       *data,
    size_t               len,
    const ZyzyrDict *dict,
    const uint8_t      **out,
    size_t              *out_len);

ZyzyrStatus zyzyr_decoder_decompress_into(
    ZyzyrDecoder    *dec,
    const uint8_t       *data,
    size_t               len,
    const ZyzyrDict *dict,
    uint8_t             *out,
    size_t               out_cap,
    size_t              *out_len);

typedef struct ZyzyrStream ZyzyrStream;

ZyzyrStream *zyzyr_stream_new(const ZyzyrDict *dict);
void         zyzyr_stream_free(ZyzyrStream *stream);

ZyzyrStatus zyzyr_stream_feed(
    ZyzyrStream   *stream,
    const uint8_t *data,
    size_t         len,
    size_t        *consumed);

ZyzyrStatus zyzyr_stream_drain(
    ZyzyrStream *stream,
    uint8_t     *out,
    size_t       out_cap,
    size_t      *written);

size_t      zyzyr_stream_pending(const ZyzyrStream *stream);
ZyzyrStatus zyzyr_stream_finish(const ZyzyrStream *stream);
ZyzyrStatus zyzyr_stream_reset(ZyzyrStream *stream);

const char *zyzyr_status_message(int32_t code);
const char *zyzyr_version(void);

#ifdef __cplusplus
}
#endif

#endif
