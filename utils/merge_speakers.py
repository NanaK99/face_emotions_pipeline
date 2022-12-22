from praatio import textgrid
import logging
import os


def get_speaker_entries(tg_file):
    tier_name_list = tg_file.tierNameList

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

    return speaker_1, speaker_2, speaker_3, speaker_4, speaker_5


def merge_speakerss(individual_entries):
    speaker_1 = individual_entries[0]
    speaker_2 = individual_entries[1]
    speaker_3 = individual_entries[2]
    speaker_4 = individual_entries[3]
    speaker_5 = individual_entries[4]

    speaker1 = [interval for interval in speaker_1 if interval[2] != "0"]
    speaker2 = [interval for interval in speaker_2 if interval[2] != "0"]
    speaker3 = [interval for interval in speaker_3 if interval[2] != "0"]
    speaker4 = [interval for interval in speaker_4 if interval[2] != "0"]
    speaker5 = [interval for interval in speaker_5 if interval[2] != "0"]

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

    labels_sorted = []
    for m, l in sorted(zip(xmaxs, labels)):
        labels_sorted.append(l)

    new_entrylist = []
    for xmin, xmax, label in zip(xmins_sorted, xmaxs_sorted, labels_sorted):
        new_entrylist.append((xmin, xmax, label))

    return new_entrylist


def gen_new_tg(tg_list, output_txtg_name, input_txtg_path):

    tg_combined = textgrid.Textgrid()
    tg_original = textgrid.openTextgrid(input_txtg_path, includeEmptyIntervals=True)

    tier_name_list = tg_original.tierNameList
    tier_name_list = [tier_name_list[0]]
    for tier_name in tier_name_list:
        tier = tg_original.tierDict[tier_name]

        tg_tier = tier.new(entryList=tg_list)
        tg_combined.addTier(tg_tier)

        output_file_path = os.path.join(output_txtg_name)

        new_tier_name = "Speakers-combined - words"
        tg_combined.renameTier(tier_name, new_tier_name)

        tg_combined.save(output_file_path, format="long_textgrid", includeBlankSpaces=True)
        logging.info(f"Merged TextGrid saved at {output_file_path}.")

        return output_file_path


def main(txtg_path, output_txtg_name):
    logging.info(f"STARTED merging speakers of {txtg_path}.")
    tg = textgrid.openTextgrid(txtg_path, includeEmptyIntervals=True)
    individual_entries = get_speaker_entries(tg)
    merged_tg_list = merge_speakerss(individual_entries)
    output_path = gen_new_tg(merged_tg_list, output_txtg_name, txtg_path)
    logging.info(f"FINISHED merging.")

    return output_path


if __name__  == "__main__":
    input_txtg_path = "trott.TextGrid"
    output_txtg_name = "merged_trott"
    main(input_txtg_path, output_txtg_name)