from subprocess import Popen
import argparse
import signal
import shutil
import sys
import os


class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('Error: %s\n' % message)
        self.print_help()
        sys.exit(2)

parser = MyParser()

parser.add_argument('--video', type=str,
                    help='path to the input video', required=True)
parser.add_argument('--input_textgrid', type=str,
                    help='path to the base textgrid', required=True)
parser.add_argument('--output_dir_name', type=str,
                    help='name of the directory where the generated textgrid files should be saved', required=True)
parser.add_argument('--verbose',
                    help='a boolean indicating the mode for logging, '
                         'in case of True, prints will also be visible in the terminal; '
                         'otherwise the logs will be kept only in the log file', required=False, default=False, action='store_true')

parser.add_argument('--debug',
                    help='', required=False, default=False, action='store_true')

args = parser.parse_args()

output_dir_name = args.output_dir_name
directory_path = os.path.join("./", output_dir_name)
if not os.path.exists(directory_path):
    os.makedirs(directory_path)


base_command = ["python", "fp.py", "#type",  "--video", "#video", "--input_textgrid", "#input_textgrid", "--output_dir_name", "#output_dir_name"]

if args.debug:
    base_command.append("--debug")

types = ["--gaze", "--body", "--expressions", "--emotions" ]

#spawn processes
processes = []
for t in types:
    temp_tex_grid = args.input_textgrid.split(".TextGrid")[0] + t.strip("--") + ".TextGrid"
    shutil.copyfile(args.input_textgrid, temp_tex_grid)
    new_command = base_command[:]
    new_command[new_command.index("#type")] = t
    new_command[new_command.index("#video")] = args.video
    new_command[new_command.index("#input_textgrid")] = temp_tex_grid
    new_command[new_command.index("#output_dir_name")] = args.output_dir_name
    
    if args.verbose:
        new_command.append("--verbose")

    p = Popen(new_command)
    processes.append(p)


try:
    for p in processes:
        p.wait()
except KeyboardInterrupt:
    for p in processes:
        p.send_signal(signal.SIGINT)
        p.wait()

