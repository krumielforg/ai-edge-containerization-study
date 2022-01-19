/* Copyright 2021, Scailable, All rights reserved. */

#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <string.h>

#include "sclbl_util.h"
#include "sclbl_runtime_core.h"

int main(int argc, char *argv[]) {

#ifdef __ARM_NEON__
	printf("NEON enabled\n");
#endif

	// 0. Retrieve WASM and input directory paths
	char *wasm_path = NULL;
	char *dir_path = NULL;
	char *pb_path = NULL;
	int acceleration = 0;

	DIR *d;
  	struct dirent *dir;
	char full_path[1000];
	
	if (argc == 4) {
		wasm_path = argv[1];
		dir_path = argv[2];
		acceleration = (int) strtol(argv[3], NULL, 10);
	} else {
		fprintf(stderr, "usage: sclbl-bin wasm_path dir_path acceleration\n");
		exit(1);
	}

	// 1. Read WASM into runtime.
	sclbl_core_wasm_read(wasm_path);

	// 2. Initialize runtime.
	sclbl_core_wasm_init();

	// Loop through all files and perform inference
  	d = opendir(dir_path);
  	if (d) {
		printf("entering input directory\n");
        while ((dir = readdir(d)) != NULL) {
            
			//Condition to check regular file.
            if(dir->d_type == DT_REG) {
                full_path[0] = '\0';
                strcat(full_path, dir_path);
                //strcat(full_path, "/");
                strcat(full_path, dir->d_name);
                printf("input: %s\n", full_path);

				// 3. Generate JSON formatted runtime input string.
				char *input_json = sclbl_util_pb_to_json(full_path);

				// 3a. Store JSON for inspection
				//sclbl_write_data("model.json",input_json);

				// 4. Run Sclbl inference.
				// warmup...
				char *output_json = sclbl_core_exec(input_json, acceleration);
				//output_json = sclbl_core_exec(input_json, acceleration);
				printf("%s\n", output_json);
            }
        }
        closedir(d);
    }
    //return(0); 

	// 6. Clean up.
	sclbl_core_free();
	sclbl_core_finalize();

	return 0;
}


