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
parser.add_argument('--verbose', type=bool,
                    help='a boolean indicating the mode for logging, '
                         'in case of True, prints will also be visible in the terminal; '
                         'otherwise the logs will be kept only in the log file', required=True, default=False)


parser.add_argument('--emotions', default=False,
                    help='emotions only', required=False, action='store_true')
parser.add_argument('--expressions', default=False,
                    help='emotions only', required=False, action='store_true')
parser.add_argument('--body', default=False,
                    help='emotions only', required=False, action='store_true')
parser.add_argument('--gaze', default=False,
                    help='emotions only', required=False, action='store_true')

args = parser.parse_args()

output_dir_name = args.output_dir_name
directory_path = os.path.join("./", output_dir_name)
if not os.path.exists(directory_path):
    os.makedirs(directory_path)


base_command = ["python", "fp.py", "#type",  "--video", "#video", "--input_textgrid", "#input_textgrid", "--output_dir_name", "#output_dir_name", "--verbose", "#boolean"]

types = ["--gaze", "--body", "--expressions", "--emotions" ]

print(args.verbose)
#spawn processes
processes = []
for t in types:
    temp_tex_grid = t.strip("--") + args.input_textgrid
    shutil.copyfile(args.input_textgrid, temp_tex_grid)
    new_command = base_command[:]
    new_command[new_command.index("#type")] = t
    new_command[new_command.index("#video")] = args.video
    new_command[new_command.index("#input_textgrid")] = temp_tex_grid
    new_command[new_command.index("#output_dir_name")] = args.output_dir_name
    new_command[new_command.index("#boolean")] = str(args.verbose)

    p = Popen(new_command)
    processes.append(p)


try:
    for p in processes:
        # print("#####")
        p.wait()
except KeyboardInterrupt:
    for p in processes:
        # print("PPPPPPPPPP", p)
        # print("signal", signal.SIGINT)
        p.send_signal(signal.SIGINT)
        p.wait()

# for t in types:
#     temp_tex_grid = t.strip("--") + args.input_textgrid
    # os.remove(temp_tex_grid)
