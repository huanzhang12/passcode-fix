#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>
#include <set>
#include <errno.h>
#include <assert.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include "block.h"
#include "linear.h"
#include "zlib/zlib.h"
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL

using namespace std;

static char *line;
static int max_line_len;
static char* readline(FILE *input)
{
	int len;

	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}

void exit_with_help()
{
	printf(
			"Usage: convert2binary training_set_file [training_binary]\n"
			//"options:\n"
			//"-c fmt: if fmt = 1 compressed format, if fmt = 0 binary format (default 1)\n"
		  );
	exit(1);
}

int datatype = COMPRESSION;
int nsplits;

#define CHUNKSIZE 2048
#define UNITSIZE 16

int zinit(z_stream *strm, int level = Z_DEFAULT_COMPRESSION)
{
	strm->zalloc = Z_NULL;
	strm->zfree = Z_NULL;
	strm->opaque = Z_NULL;
	if(deflateInit(strm, level) != Z_OK)
	{
		deflateEnd(strm);
		fprintf(stderr,"z_stream initial fails\n");
		exit(-1);
	}
	return 0;
}

long zwrite(const void *ptr, size_t size, size_t nmemb, z_stream *strm, FILE* fp, int flush = Z_NO_FLUSH)
{
	unsigned char out[CHUNKSIZE * UNITSIZE];
	unsigned int have;
	int state;
	long byteswritten = 0;

	strm->avail_in = (uInt)size*nmemb;
	strm->next_in = (unsigned char*)ptr;
	do {
		strm->avail_out = CHUNKSIZE * UNITSIZE;
		strm->next_out = out;
		state = deflate(strm, flush);    /* no bad return value */
		assert(state != Z_STREAM_ERROR);  /* state not clobbered */
		have = CHUNKSIZE * UNITSIZE - strm->avail_out;
		if (fwrite(out, 1, have, fp) != have || ferror(fp))
		{
			deflateEnd(strm);
			fprintf(stderr,"Compression Error\n");
			exit(-1);
		}
		byteswritten += have;
	} while (strm->avail_out == 0);
	assert(strm->avail_in == 0);     /* all input will be used */
	return byteswritten;
}


void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name);

class BlockInfo{
	public:
		int l, n;
		unsigned long n_x_space;
		FILE* fp;
		z_stream strm;
		vector<unsigned long> offsets;
		vector<double> labels;
		// Add comment 
		struct feature_node fnbuf[CHUNKSIZE];
		unsigned long n_fnbuf;
		unsigned long n_output;

		const static int HeaderSize = 
			sizeof(int)+sizeof(int)+sizeof(long)+sizeof(long);
		BlockInfo()
		{
			n_x_space = l = n = 0;
			n_output = n_fnbuf = 0;
		}
		void open_file(const char *filename)
		{
			fp = fopen(filename, "wb+");
			fseek(fp, HeaderSize, SEEK_SET);
			if(datatype == COMPRESSION) 
				zinit(&strm);
		}
		void close_file()
		{
			deflateEnd(&strm);
			fclose(fp);
		}
		void add_feature_node(struct feature_node& fn, int flag = 0)
		{
			// flag = 0 -> normal feature node
			// flag < 0 -> end of an instance
			n_x_space++;
			if(flag == 0) 
				n = max(n, fn.index);
			else if(flag < 0) 
				fn.index = -1;
			fnbuf[n_fnbuf++] = fn;
			if(n_fnbuf == CHUNKSIZE || flag < 0)
			{
				if(datatype == BINARY)
				{
					fwrite(fnbuf, sizeof(struct feature_node), n_fnbuf, fp);
					n_output += n_fnbuf * sizeof(struct feature_node);
				} 
				else if (datatype == COMPRESSION)
				{
					n_output += zwrite(fnbuf, sizeof(struct feature_node), n_fnbuf, &strm, fp);
				}
				n_fnbuf = 0;
			}
		}
		bool add_instance(char* buf, int* y=NULL)
		{
			char *label_p, *idx, *val, *endptr;
			struct feature_node node;
			int label;
			l++;
			label_p = strtok(buf," \t");
			label = (int) strtol(label_p, &endptr, 10);
			if(y) *y = label;
			if(endptr == label_p) 
				return false;
			labels.push_back(label);
			offsets.push_back(n_x_space);

			while(1)
			{
				idx = strtok(NULL,":");
				val = strtok(NULL," \t");

				if(val == NULL)
					break;

				errno = 0;
				node.index = (int) strtol(idx,&endptr,10);
				if(endptr == idx || errno != 0 || *endptr != '\0' || node.index <= 0)
					return false;

				errno = 0;
				node.value = strtod(val,&endptr);
				if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
					return false;
				add_feature_node(node);
			}

			node.value = 1;
			add_feature_node(node, -1); // For adding bias
			add_feature_node(node, -1); // For end of instance
			return true;
		}

		// footer contains the label yi and the offset of xi
		void emit_footer()
		{
			if(datatype == BINARY)
			{
				fwrite(&labels.front(), sizeof(double), l, fp);
				fwrite(&offsets.front(), sizeof(unsigned long), l, fp);
				n_output += l * (sizeof(double) + sizeof(unsigned long));
			} 
			else if (datatype == COMPRESSION)
			{
				n_output += zwrite(&labels.front(), sizeof(double), l, &strm, fp);
				n_output += zwrite(&offsets.front(), sizeof(unsigned long), l, &strm, fp,Z_FINISH);
			}
		}
		void emit_header()
		{
			rewind(fp);
			fwrite(&l, sizeof(int), 1,  fp);
			fwrite(&n, sizeof(int), 1, fp);
			fwrite(&n_x_space, sizeof(unsigned long), 1, fp);
			fwrite(&n_output, sizeof(unsigned long), 1, fp);
		}
};

int main(int argc, char* argv[])
{
	char input_file_name[1024];
	char binary_dir_name[1024];
	char buf[1024];
	time_t start_t = time(NULL);
	parse_command_line(argc, argv, input_file_name, binary_dir_name);

	srand(1);
	BlockInfo block_info;
	max_line_len = 1024;
	nsplits = 1;
	line = Malloc(char,max_line_len);

	FILE *input = fopen(input_file_name, "r");
	
	if(input == NULL)
	{
		fprintf(stderr, "can't open input file %s\n", input_file_name);
		exit(1);
	}
	
	block_info.open_file(binary_dir_name);

	int l = 0, y;
		
	while(1)
	{
		int i;
		for(i = 0; i < nsplits && readline(input) != NULL; ++i)
		{
			l++;
			if(block_info.add_instance(line, &y) == false)
			{
				fprintf(stderr,"Wrong Input Formt at line  %d.\n", l);
				return -1;
			}
			if(l%10000 == 0)
			{
				printf(".");
				fflush(stdout);
			}
		}
		
		if(i < nsplits) break;
	}
	if(l>10000) printf("\n");

	block_info.emit_footer();
	block_info.emit_header();
	block_info.close_file();

	fclose(input);
	free(line);
	printf("time : %.5g\n", difftime(time(NULL), start_t));

	return 0;
}

void parse_command_line(int argc, char **argv, char *input_file_name, char *binary_dir_name)
{
	int i;
	// default values
	datatype = COMPRESSION;
	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		if(++i>=argc) exit_with_help();
		switch(argv[i-1][1])
		{
			case 'c':
				datatype = atoi(argv[i]);
				break;
			default:
				fprintf(stderr,"unknown option: -%c\n", argv[i-1][1]);
				exit_with_help();
				break;
		}
	}

	// determine filenames
	if(i>=argc) exit_with_help();

	strcpy(input_file_name, argv[i]);

	if(i<argc-1) strcpy(binary_dir_name,argv[i+1]);
	else 
	{
		char *p = strrchr(argv[i],'/');
		if(p==NULL) p = argv[i];
		else ++p;
		sprintf(binary_dir_name,"%s.cbin",p);
	}
}
