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

    ll = [speakers_json[sp] for sp in speakers_json.keys()]
     
    return (*ll,)

   # return [speakers_json[ind] for ind in speakers_json[list(speakers_json.keys())] ]


def merge_speakerss(individual_entries):
    xmins = []
    xmaxs = []
    labels = []

    for i in range(len(individual_entries)):
        for interval in individual_entries[i]:
            if interval[2] != "0":
                xmins.append(interval[0])
                xmaxs.append(interval[1])
                labels.append(interval[2])

    
    xmaxs_sorted = sorted(xmaxs)
    xmins_sorted = sorted(xmins)

    labels_sorted = []
    for m, l in sorted(zip(xmaxs, labels)):
        labels_sorted.append(l)

    new_entrylist = []
    for xmin, xmax, label in zip(xmins_sorted, xmaxs_sorted, labels_sorted):
        new_entrylist.append((xmin, xmax, label))

    return new_entrylist


def gen_new_tg(tg_list, output_txtg_name, input_txtg_path, verbose):

    tg_combined = textgrid.Textgrid()
    tg_original = textgrid.openTextgrid(input_txtg_path, includeEmptyIntervals=False)
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
        if verbose:
            print(f"Merged TextGrid saved at {output_file_path.split('/')[-1]}.")
        logging.info(f"Merged TextGrid saved at {output_file_path.split('/')[-1]}.")

        return output_file_path


def main(txtg_path, output_txtg_name, verbose):
    if verbose:
        print(f"STARTED merging speakers of {txtg_path}.")
    logging.info(f"STARTED merging speakers of {txtg_path}.")
    tg = textgrid.openTextgrid(txtg_path, includeEmptyIntervals=False)
    individual_entries = get_speaker_entries(tg)
    merged_tg_list = merge_speakerss(individual_entries)
    output_path = gen_new_tg(merged_tg_list, output_txtg_name, txtg_path, verbose)

    if verbose:
        print(f"FINISHED merging.")
    logging.info(f"FINISHED merging.")

    return output_path


if __name__  == "__main__":
    verbose = False
    input_txtg_path = "trott.TextGrid"
    output_txtg_name = "merged_trott"
    main(input_txtg_path, output_txtg_name)