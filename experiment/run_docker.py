import sys
import helpers

def run(depl, model, row_id, run_id, lpath):
    helpers.run_docker(f'docker_run_{run_id}', depl, model, row_id, run_id, lpath)

if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
