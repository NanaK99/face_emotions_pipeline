from praatio import tgio


def replace_with_zeros(tg_file, tier_name):
    tier = tg_file.tierDict[tier_name]
    entryList = tier.entryList

    intervals = []

    for entry in entryList:
        interval = []
        start = entry.start
        end = entry.end
        if entry.label == "":
            label = "0"
        else:
            label = entry.label

        interval.append(start)
        interval.append(end)
        interval.append(label)
        intervals.append(interval)

    return intervals


def save_new_tg(tg_final, tg, tier_name, tg_list, output_txtg_name, input_txtg_path):
    tier = tg.tierDict[tier_name]
    tg_tier = tier.new(entryList=tg_list)
    tg_final.addTier(tg_tier)

    tg_final.save(output_txtg_name, useShortForm=False)
    print("Final-Preprocessed textgrid saved")

    return output_txtg_name


def main(txtg_path, output_txtg_name):
    print("Preprocessing")
    tg = tgio.openTextgrid(txtg_path)
    tg_final = tgio.Textgrid()

    tier_name_list = tg.tierNameList
    for tier_name in tier_name_list:
        new_entrylist = replace_with_zeros(tg, tier_name)
        output_file_path = save_new_tg(tg_final, tg, tier_name, new_entrylist, output_txtg_name, txtg_path)
    print("DONE with preprocessing")
    return output_file_path


if __name__  == "__main__":
    input_txtg_path = "./trot_22.TextGrid"
    output_txtg_name = "./final_tg.TextGrid"
    main(input_txtg_path, output_txtg_name)