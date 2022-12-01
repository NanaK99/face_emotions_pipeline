import os


def save_textgrids(tier, gaze_entrylist, expr_entrylist, body_entrylist, emotion_entrylist,
                   output_dir_name, tg_gaze, tg_expr, tg_body, tg_emotion):
    gaze_tier = tier.new(entryList=gaze_entrylist)
    expr_tier = tier.new(entryList=expr_entrylist)
    body_tier = tier.new(entryList=body_entrylist)
    emotion_tier = tier.new(entryList=emotion_entrylist)

    tg_gaze.addTier(gaze_tier)
    tg_expr.addTier(expr_tier)
    tg_body.addTier(body_tier)
    tg_emotion.addTier(emotion_tier)

    directory_path = os.path.join("./", output_dir_name)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    gaze_output_file_path = os.path.join(directory_path, 'gaze_output.TextGrid')
    expr_output_file_path = os.path.join(directory_path, 'expr_output.TextGrid')
    body_output_file_path = os.path.join(directory_path, 'body_output.TextGrid')
    emotion_output_file_path = os.path.join(directory_path, 'emotion_output.TextGrid')

    tg_gaze.save(gaze_output_file_path, useShortForm=False)
    tg_expr.save(expr_output_file_path, useShortForm=False)
    tg_body.save(body_output_file_path, useShortForm=False)
    tg_emotion.save(emotion_output_file_path, useShortForm=False)

    return gaze_output_file_path, expr_output_file_path, body_output_file_path, emotion_output_file_path