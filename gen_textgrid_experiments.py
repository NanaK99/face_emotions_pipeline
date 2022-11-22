from praatio import tgio

# new_lines = []
# with open("Trott_Garner_Miranda_LG_2022_-_Audio2_new.TextGrid") as file:
#     for line in file.readlines():
#         if '""' in line:
#             new_lines.append('            text = "0" \n')
#         else:
#             new_lines.append(line)
#
# new_lines_str = "".join(new_lines)
# with open('trott.TextGrid', 'w') as fp:
#     fp.write(new_lines_str)


tg = tgio.openTextgrid("trott.TextGrid")

tier_name_list = ["mgarner - words", "TERRENCE - words", "Nicole-Dressel - words", "Jeannet-Trott - words", "Diane-Fetterman---Gap-International - words" ]

for tier_name in tier_name_list:
    print(f"Working on {tier_name} tier.")
    new_entrylist = []
    tier = tg.tierDict[tier_name]
    entryList = tier.entryList

    for entry in entryList:
        label = entry.label + "gago"
        start = entry.start
        end = entry.end
        new_entrylist.append((start, end, label))
    new_tier = tier.new(entryList=new_entrylist)
    tg_new = tgio.Textgrid()
    tg_new.addTier(new_tier)
    tg_new.save('new gago', useShortForm=False)

    # entryList[0].label = "Gago"
    # print (entryList[0])

    # new_entrylist = [(start, stop, label) for start, stop, label in entryList]
    # print(new_entrylist[0])
    # tier.new(entryList=new_entrylist)
    # print(tier.entryList)
    #print(f"Done with {tier_name} tier.")

    #print(tg.tierDict["mgarner - words"].entryList)

    # tg.save("/Users/nanakarapetyan/Desktop/face_emotions_pipeline/experimentsss.TextGrid", useShortForm=False)



# for tier_name in tier_name_list:
#     print(f"Working on {tier_name} tier.")
#     new_text = tier_name.split("-")[0]
#     new_entrylist = []
#     tier = tg.tierDict[tier_name]
#     entryList = tier.entryList
#
#     for entry in entryList:
#         label = entry.label
#         start = entry.start
#         end = entry.end
#
#         if label != "0":
#             entry = entry + (new_text,)
#             new_entrylist.append(entry)
#         else:
#             new_entrylist.append((start, end, label, ""))
#
#     tier.new(entryList=new_entrylist)
#
#     # tier.entryList = new_entrylist
#     print(tier.entryList)
#     print(f"Done with {tier_name} tier.")
#     break

# tg.save("/Users/nanakarapetyan/Desktop/face_emotions_pipeline/experiments.TextGrid")

#######TO CHANGE BACK the 0-s#########W
# new_lines = []
# with open("trorr.TextGrid") as file:
#     for line in file.readlines():
#         if '"0"' in line:
#             new_lines.append('            text = "" \n')
#         else:
#             new_lines.append(line)
#
# new_lines_str = "".join(new_lines)
# with open('trott.TextGrid', 'w') as fp:
#     fp.write(new_lines_str)