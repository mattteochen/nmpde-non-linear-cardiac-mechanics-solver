import os
import glob

def list_files_with_extension(directory, extension):
    search_pattern = os.path.join(directory, f'*.{extension}')
    files = glob.glob(search_pattern)
    return files

#modify this based on your machine
process_num = 6
directory_path = '../tests/slab_cubic/test_parameters/'
extension = 'prm'
parameters_files = list_files_with_extension(directory_path, extension)
build_dir = "../tests/slab_cubic/build/"
mesh_file_name = '../../../../lifex_fiber_generation_examples/mesh/slab_cubic.msh'
executable = "main"

print("Compiling...")
if (os.system(f'cd {build_dir} && cmake .. && make -j {process_num}') != 0):
    print("Compilation failed, aborting")
else:
    print("Found the following parameters test:")
    for i, f in enumerate(parameters_files):
        f = f.replace('tests/slab_cubic/', '')
        print(f)
        parameters_files[i] = f
    print('')

    for p in parameters_files:
        print(f'============================================== Launching test for {p} ==============================================')
        cmd = f'cd {build_dir} && mpirun -n {process_num} {executable} {mesh_file_name} {p}'
        print("command: ", cmd)
        if os.system(cmd) != 0:
            print(f'Process failed for {p}')
        print(f'====================================================================================================================\n')
