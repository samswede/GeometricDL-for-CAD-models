import os
import trimesh

class DataFileConversion:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir

    def convert_file(self, dir_path, filename, output_format):
        input_path = os.path.join(dir_path, filename)
        output_path = os.path.join(self.output_dir, filename.replace('.off', '.' + output_format))

        # Load the mesh from the input file
        mesh = trimesh.load_mesh(input_path)

        # Export to the output file
        mesh.export(output_path)

    def convert_files(self, input_format='off', output_format='stl'):
        for class_dir in os.listdir(self.input_dir):
            class_path = os.path.join(self.input_dir, class_dir)
            if os.path.isdir(class_path):
                for split_dir in ['train', 'test']:
                    split_path = os.path.join(class_path, split_dir)
                    if os.path.isdir(split_path):
                        for filename in os.listdir(split_path):
                            if filename.endswith('.' + input_format):
                                self.convert_file(split_path, filename, output_format)
