"""
inputs:
    interval_idx - (it can be after passing several frames),
    text - (pipeline output)
    model_name - gaze, mediapipe_holistic, or face_exprs
    duration - video duration in seconds
    len_of_one_interval - in seconds
    x_min - starting time in seconds of an interval
    x_max - ending time in seconds of an interval
"""


with open('output.Textgrid','r') as f:
  data = f.readlines()

# print(data, type(data))
new_data = []
x_max_count = 0
x_min_count = 0
text_count = 0

for idx, line in enumerate(data):
    data[idx] = line.strip()
    if "xmax" in line:
        if x_max_count == 0 or x_max_count == 1:
            new_data.append(duration)
            x_max_count += 1
    if "xmin" in line:
        if x_min_count == 0 or x_min_count == 1:
            new_data.append("0")
            x_min_count += 1
    if f"intervals [{interval_idx}]" in line:
        new_data.append(line)
        new_data.append(x_min)
        new_data.append(x_max)
        new_data.append(text)
    if "name" in line:
        new_data.append(f"name = {model_name}")
    if "intervals: size" in line:
        new_data.append(f"intervals: size = {duration / len_of_one_interval}")

with open('stats.txt', 'w') as file:
    file.writelines(data)

# txttext = ''
# for line in data[9:]:  #informations needed begin on the 9th lines
#     # print(line)
#     line = re.sub('\n','',line) #as there's \n at the end of every sentence.
#     line = re.sub ('^ *','',line) #To remove any special characters
#     linepair = line.split('=')
#     if len(linepair) == 2:
#         if linepair[0] == 'xmin':
#             x_min == linepair[1]
#         if linepair[0] == 'xmax':
#             x_max == linepair[1]
#         if linepair[0] == 'text':
#             if linepair[1].strip().startswith('"') and linepair[1].strip().endswith('"'):
#                 text = linepair[1].strip()[1:-1]
#                 txttext += text + '\n'