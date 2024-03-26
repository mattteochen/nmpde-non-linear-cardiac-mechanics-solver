import os
import glob

def list_files_with_extension(directory, extension):
    search_pattern = os.path.join(directory, f'*.{extension}')
    files = glob.glob(search_pattern)
    return files

problem_name = "slab_cubic" 
#modify this based on your machine
process_num = 12
directory_path = f'../tests/{problem_name}/test_parameters/'
extension = 'prm'
parameters_files = list_files_with_extension(directory_path, extension)
build_dir = f"../tests/{problem_name}/build/"
mesh_file_name = f'../../../../lifex_fiber_generation_examples/mesh/{problem_name}.msh'
executable = "main"

print("Compiling...")
try:
    os.system(f'cd {build_dir} && rm *')
except:
    pass
try:
    os.system(f'cd {build_dir} && rm -r *')
except:
    pass

if (os.system(f'cd {build_dir} && cmake .. && make -j {process_num}') != 0):
    print("Compilation failed, aborting")
else:
    print("Found the following parameters test:")
    for i, f in enumerate(parameters_files):
        f = f.replace(f'tests/{problem_name}/', '')
        print(f)
        parameters_files[i] = f
    print('')

    for p in parameters_files:
        print(f'############################################## Launching test for {p} ##############################################')
        cmd = f'cd {build_dir} && mpirun -n {process_num} {executable} {mesh_file_name} {p}'
        print("command: ", cmd)
        if os.system(cmd) != 0:
            print(f'Process failed for {p}')
        else:
            file_name = p.split('/')[-1]
            dir = file_name.split('.')[0]
            os.system(f'cd {build_dir} && mkdir {dir} && mv {problem_name}* {dir}')
        print(f'####################################################################################################################\n')
