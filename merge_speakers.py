import operator

from praatio import tgio
from configparser import ConfigParser
import os

config_object = ConfigParser()
config_object.read("./static/config.ini")
IGNORE_EXPRS = config_object["IGNORE_EXPRS"]


def get_combined_json(tg_file):
    tier_name_list = tg_file.tierNameList
    # print(tier_name_list)
    num = len(tier_name_list)
    for i in range(num):
        globals()[f"speaker_{str(i+1)}"] = []

    speakers_json = {}
    for tier_name in tier_name_list:
        tier = tg_file.tierDict[tier_name]
        entryList = tier.entryList

        intervals = []

        for entry in entryList:
            interval = []
            start = entry.start
            end = entry.end
            label = entry.label
            #
            # starts.append(start)
            # ends.append(end)
            # labels.append(label)

            interval.append(start)
            interval.append(end)
            interval.append(label)
            intervals.append(interval)

        speakers_json[tier_name] = intervals

    speaker_1 = speakers_json[list(speakers_json.keys())[0]]
    speaker_2 = speakers_json[list(speakers_json.keys())[1]]
    speaker_3 = speakers_json[list(speakers_json.keys())[2]]
    speaker_4 = speakers_json[list(speakers_json.keys())[3]]
    speaker_5 = speakers_json[list(speakers_json.keys())[4]]
    # print(speaker_5)
    # print(speaker_2)
    # print(speaker_3)
    # print(speaker_4)
    # print(speaker_1)

    return speaker_1, speaker_2, speaker_3, speaker_4, speaker_5


def merge_speakerss(combined_speakers_json):
    speaker_1 = combined_speakers_json[0]
    speaker_2 = combined_speakers_json[1]
    speaker_3 = combined_speakers_json[2]
    speaker_4 = combined_speakers_json[3]
    speaker_5 = combined_speakers_json[4]

    speaker1 = [interval for interval in speaker_1 if interval[2] != "0"]
    speaker2 = [interval for interval in speaker_2 if interval[2] != "0"]
    speaker3 = [interval for interval in speaker_3 if interval[2] != "0"]
    speaker4 = [interval for interval in speaker_4 if interval[2] != "0"]
    speaker5 = [interval for interval in speaker_5 if interval[2] != "0"]

    all_speakers = speaker1 + speaker2 + speaker3 + speaker4 + speaker5
    print(len(speaker1), len(speaker2), len(speaker3), len(speaker4), len(speaker5))
    # print(speaker1[:5])
    # print(speaker2[:5])
    # print(speaker3[:5])
    # print(speaker4[:5])
    # print(speaker5[:5])
    # all_speakers=[]
    # all_speakers.extend(speaker1)
    # all_speakers.extend(["2"])
    # all_speakers.extend(speaker2)
    # all_speakers.extend(["3"])
    # all_speakers.extend(speaker3)
    # all_speakers.extend(["4"])
    # all_speakers.extend(speaker4)
    # all_speakers.extend(["5"])
    # all_speakers.extend(speaker5)
    # print(all_speakers)
    xmins1 = [interval[0] for interval in speaker1]
    xmins2 = [interval[0] for interval in speaker2]
    xmins3 = [interval[0] for interval in speaker3]
    xmins4 = [interval[0] for interval in speaker4]
    xmins5 = [interval[0] for interval in speaker5]
    xmins = xmins1 + xmins2 + xmins3 + xmins4 + xmins5

    xmaxs1 = [interval[1] for interval in speaker1]
    xmaxs2 = [interval[1] for interval in speaker2]
    xmaxs3 = [interval[1] for interval in speaker3]
    xmaxs4 = [interval[1] for interval in speaker4]
    xmaxs5 = [interval[1] for interval in speaker5]
    xmaxs = xmaxs1 + xmaxs2 + xmaxs3 + xmaxs4 + xmaxs5

    labels1 = [interval[2] for interval in speaker1]
    labels2 = [interval[2] for interval in speaker2]
    labels3 = [interval[2] for interval in speaker3]
    labels4 = [interval[2] for interval in speaker4]
    labels5 = [interval[2] for interval in speaker5]
    labels = labels1 + labels2 + labels3 + labels4 + labels5

    xmaxs_sorted = sorted(xmaxs)
    xmins_sorted = sorted(xmins)

    print(z)



"""
[[581.35, 581.68, 'sorry'], [581.68, 582.17, 'good'], [582.17, 582.43, 'and'], [582.43, 582.54, 'then'], [582.54, 582.68, 'when']]
[[25.86, 26.14, 'okay'], [27.81, 28.38, 'um'], [28.83, 29.4, 'so'], [29.77, 30.06, 'i'], [30.06, 30.64, 'wanna']]
[[0.299, 0.639, 'please'], [0.639, 0.889, 'stay'], [0.889, 1.149, 'on'], [1.149, 1.419, 'mute'], [1.419, 1.549, 'so']]
[[5180.55, 5180.72, 'yeah'], [5180.75, 5180.98, 'hi'], [5188.44, 5188.52, 'yeah'], [5188.52, 5188.68, 'i'], [5188.68, 5188.85, 'did']]
[[2062.41, 2062.71, "what's"], [2062.71, 2062.8, 'the'], [2062.8, 2063.17, 'hardest'], [2064.3, 2064.65, 'figured'], [2064.65, 2064.99, 'out']]
"""
    # starts = combined_speakers_json[0]
    # ends = combined_speakers_json[1]
    # labels = combined_speakers_json[2]

    # all_speakers = [speaker_1, speaker_2, speaker_3, speaker_4, speaker_5]
    # for speaker in all_speakers:
    #     for interval in speaker:
    #         print(interval)
    #         if interval[2] == "0":
    #             continue
    #         continue

    # xmins = []
    # xmaxs = []
    # labels = []
    # all_speakers=[]
    # all_speakers.append(speaker_1)
    # all_speakers.append(speaker_2)
    # all_speakers.append(speaker_3)
    # all_speakers.append(speaker_4)
    # all_speakers.append(speaker_5)
    # print(all_speakers)

    # print(labels)



    # for s1, s2, s3, s4, s5 in itertools.zip_longest(speaker_1, speaker_2, speaker_3, speaker_4, speaker_5):
    #     print(s1, s2, s3, s4, s5)


def gen_new_tg(tg_list, output_dir_name):
    tg_combined = tgio.Textgrid()
    combined_tier = tier.new(entryList=tg_list) # get tier TODO
    tg_combined.addTier(combined_tier)

    directory_path = os.path.join("./", output_dir_name)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    output_file_path = os.path.join(directory_path, 'combined_trott.TextGrid')
    # change renmaing TODO
    new_tier_name = tier.split("words")[0] + ""
    tg_gaze.renameTier(tier_name, new_tier_name_gaze)

    tg_combined.save(output_file_path, useShortForm=False)

    return output_file_path

def main(txtg_path, output_dir_name):
    tg = tgio.openTextgrid(txtg_path)
    combined_tg_json = get_combined_json(tg)
    merged_tg_list = merge_speakerss(combined_tg_json)
    # gen_new_tg(merged_tg_list, output_dir_name)


if __name__  == "__main__":
    path = "trott.TextGrid"
    output_dir_name = "outputs"
    main(path, output_dir_name)