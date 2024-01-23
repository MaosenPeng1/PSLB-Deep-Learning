import os
import csv

def create_lsf_content(seed, queue_name):
    content = f"""#BSUB -J KStrue_{seed}
#BSUB -W 2:00
#BSUB -o /rsrch6/home/biostatistics/mpeng1/log/KStrue_{seed}.out
#BSUB -e /rsrch6/home/biostatistics/mpeng1/log/KStrue_{seed}.err
#BSUB -cwd /rsrch6/home/biostatistics/mpeng1/PSLB/python/KS/5k
#BSUB -q {queue_name}
#BSUB -u mpeng1@mdanderson.org
#BSUB -n 1
#BSUB -M 6
#BSUB -R rusage[mem=6]

module load python/3.9.7-anaconda

python /rsrch6/home/biostatistics/mpeng1/PSLB/python/KS/5k/Kang and Schafer (PSLB Correctly Specified Model) Code.py --seed {seed}
"""
    return content

seed_values = []
with open("sim_seed.csv", "r") as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        seed_values.append(int(row[0]))


queue_names = ["e40short", "short", "e80short"]
output_dir = '/rsrch6/home/biostatistics/mpeng1/PSLB/python/KS/5k/'

for seed in seed_values:
    lsf_content = create_lsf_content(seed, queue_names[seed % 3])
    with open(os.path.join(output_dir, f'KStrue{seed}.lsf'), 'w') as lsf_file:
        lsf_file.write(lsf_content)

## for file in *.lsf; do 
## > bsub < $file 
## > done


