from praatio import tgio

tg_new = tgio.Textgrid()

tg = tgio.openTextgrid("trott_1.TextGrid")
print(tg.tierNameList)
tg.tierNameList = ['Diane-Fetterman---Gap-International - emotions', 'Jeannet-Trott - words', 'Nicole-Dressel - words', 'TERRENCE - words', 'mgarner - words']

tg.renameTier(tg.tierNameList[0], 'Diane-Fetterman---Gap-International - emotions')


# for idx, tier in enuemrate(tg.tierNameList):
#     tier.new(name=tier+str(idx))

# emotion_output_file_path = 'wow.TextGrid'
# print("########")
# tg_new.save(emotion_output_file_path, useShortForm=False)
# print("$$$$$$")

tgg = tgio.openTextgrid("wow.TextGrid")
for tier in tgg.tierNameList:
    print(tier)