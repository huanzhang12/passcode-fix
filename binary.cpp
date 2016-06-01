#include <vector>
#include <cstdio>
#include <cstdlib>
#include <assert.h>
#include "linear.h"
#include "zlib/zlib.h"
#include "binary.h"
#include "cpu_affinity.h"

using namespace std;

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define Realloc(ptr, type, n) (type *)realloc((ptr), (n)*sizeof(type))
#define INF HUGE_VAL


void myfread(void *ptr, size_t size, size_t nmemb, FILE * stream) {
	size_t ret = fread(ptr, size, nmemb, stream);
	if(ret != nmemb) {
		fprintf(stderr, "Read Error! Bye Bye %ld %ld\n", ret, size*nmemb);
		exit(-1);
	}
}

enum { SINGLE, BINARY, COMPRESSION}; // data format
const char *data_format_table[]=
{
	"SINGLE", "BINARY", "COMPRESSION", NULL
};

#define CHUNKSIZE 2048UL
int myuncompress(void *dest, size_t *destlen, const void *source, size_t sourcelen) {
    int ret;
	z_stream strm;

    strm.zalloc = Z_NULL;
    strm.zfree = Z_NULL;
    strm.opaque = Z_NULL;
    strm.avail_in = 0;
    strm.next_in = Z_NULL;
    ret = inflateInit(&strm);
    if (ret != Z_OK) {
		(void)inflateEnd(&strm);
        return ret;
	}

	unsigned char *in = (unsigned char *)source;
	unsigned char *out = (unsigned char *)dest;
	unsigned long bytesread = 0, byteswritten = 0;

    /* decompress until deflate stream ends or end of file */
    do {
		strm.avail_in = (uInt) min(CHUNKSIZE, sourcelen - bytesread);
		//finish all input
        if (strm.avail_in == 0)
            break;
        strm.next_in = in + bytesread;
		bytesread += strm.avail_in;

        /* run inflate() on input until output buffer not full */
        do {
            strm.avail_out = (uInt)CHUNKSIZE;
            strm.next_out = out + byteswritten;
            ret = inflate(&strm, Z_NO_FLUSH);
            assert(ret != Z_STREAM_ERROR);  /* state not clobbered */
            switch (ret) {
            case Z_NEED_DICT:
                ret = Z_DATA_ERROR;     /* and fall through */
            case Z_DATA_ERROR:
            case Z_MEM_ERROR:
                (void)inflateEnd(&strm);
                return ret;
            }
            byteswritten += CHUNKSIZE - strm.avail_out;
        } while (strm.avail_out == 0);

        /* done when inflate() says it's done */
    } while (ret != Z_STREAM_END);

	if(byteswritten != *destlen)
		fprintf(stderr,"something wrong!!!\n");
	*destlen = byteswritten;
	(void)inflateEnd(&strm);
	return 0;

}


class BinaryProblem{
	public:
		int l , n;
		BinaryProblem(){
			l = n = 0;
			buflen = 0;
			buf = NULL;
			bias = -1;
			bias_idx = -1;
		}
		~BinaryProblem(){
			if(buflen>0) free(buf);
		}

		void setBias(int idx, double val){
			if(bias >= 0) 
				fprintf(stderr, "Warning: the bias have been set to %lf\n.", bias);
			bias_idx = idx;
			bias = val;
		}

		unsigned char* get_bufptr() {return buf;}

		void load_problem(const char* filename, int datafmt) {
			FILE *fp = fopen(filename, "rb");
			load_header(fp);
			load_body(fp, datafmt);
			parseBinary();
			l = prob.l;
			n = prob.n;
			fclose(fp);
		}

		struct problem* get_problem() {
			retprob = prob;
			if(bias >= 0 && prob.bias != bias) {
				struct feature_node node;
				prob.n = retprob.n = bias_idx;
				prob.bias = retprob.bias = bias;
				node.index = bias_idx;
				node.value = bias;

				for(int i=1;i<retprob.l;i++) 
					*(retprob.x[i]-2) = node; 
				x_space[n_x_space-2] = node;
			} 

			return &retprob;
		}

		struct problem* get_subproblem(int start, int subl){
			get_problem();
			retprob.x = retprob.x + start;
			retprob.y = retprob.y + start;
			retprob.l = subl;
			return &retprob;
		}

		void gen_subproblem(BinaryProblem& ret, vector<int> &mask){
			ret = *this;
			ret.l = (int) mask.size();
			ret.buflen = (sizeof(struct node*) + sizeof(double)) * ret.l;
			ret.buf = Malloc(unsigned char, ret.buflen);
			ret.prob.y = (double*)ret.buf;
			ret.prob.x = (struct feature_node**)(ret.buf + sizeof(double) * ret.l);
			for(int i = 0; i < ret.l; i++) {
				ret.prob.y[i] = prob.y[mask[i]];
				ret.prob.x[i] = prob.x[mask[i]];
			}
		}

	private:
		int bias_idx;
		double bias;
		unsigned char* buf;
		unsigned long buflen, n_x_space, filelen;
		struct feature_node* x_space;
		struct problem prob, retprob;


		void load_header(FILE *fp) {
			myfread(&prob.l, sizeof(int), 1, fp);
			myfread(&prob.n, sizeof(int), 1, fp);
			myfread(&n_x_space, sizeof(unsigned long), 1, fp);
			myfread(&filelen, sizeof(unsigned long), 1, fp);
			prob.bias = -1;
			buflen = n_x_space * sizeof(struct feature_node) + prob.l * (sizeof(double)+sizeof(unsigned long));
		}

		void load_body(FILE *fp, int datafmt) {
			if(buf) free(buf);
			cpu_affinity_t cpu_info;
			//buf = (unsigned char*)cpu_info.interleaved_malloc(sizeof(unsigned char)*buflen);
			buf = Realloc(buf, unsigned char, buflen);
			if (buf == NULL)
				fprintf(stderr,"Memory Error!\n");
			if(datafmt == BINARY) {
				if (buflen != filelen) {
					fprintf(stderr,"l = %d n_x_space = %ld buflen%ld filelen = %ld\n",prob.l, n_x_space, buflen, filelen);
				}
				myfread(buf, sizeof(unsigned char), buflen, fp);
			} else if(datafmt == COMPRESSION) {
				unsigned char *compressedbuf;
				compressedbuf = Malloc(unsigned char, filelen);
				myfread(compressedbuf, sizeof(unsigned char), filelen, fp);
				int retcode = myuncompress(buf, &buflen, compressedbuf, filelen);
				if(retcode != Z_OK) {
					printf("OK %d MEM %d BUF %d DATA %d g %d %p %ld\n", Z_OK, Z_MEM_ERROR, Z_BUF_ERROR, Z_DATA_ERROR, retcode, buf, buflen);
					fflush(stdout);
				}
				free(compressedbuf);
			}
		}

		void parseBinary(){
			unsigned long offset = 0;
			x_space = (struct feature_node*) (buf + offset); 
			offset += sizeof(struct feature_node) * n_x_space;

			prob.y = (double*) (buf + offset); 
			offset += sizeof(double) * prob.l;

			prob.x = (struct feature_node**) (buf + offset); 
			for(int i = 0; i < prob.l; i++) 
				prob.x[i] = x_space + (unsigned long)prob.x[i];
			//prob.x[prob.l] = (struct feature_node*) prob.y;
		}

};

struct binary_problem read_binary(const char *file_name, double bias) {
	struct binary_problem ret;
	BinaryProblem *binprob = new BinaryProblem();
	binprob->load_problem(file_name, COMPRESSION);
	if(bias >= 0) 
		binprob->setBias(binprob->n+1, bias);

	ret.prob = binprob->get_problem();
	ret.buf = binprob;
	return ret;
}

void destroy_binary_problem(struct binary_problem *binprob) {
	delete (BinaryProblem*)binprob->buf;
}
