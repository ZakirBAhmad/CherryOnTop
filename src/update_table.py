import pandas as pd
import src.display_table_helpers as helpers


def pred_history(df, week_x):
    start_week = max(0,week_x - 5)
    end_week = min(start_week + 10,55)
    sliced_df = df.iloc[:,start_week*5:end_week*5]
    return (
        sliced_df.style
        .apply(helpers.highlight_and_bold_actuals_row_gray, axis=1)
        .apply(helpers.highlight_week_x, week_x=week_x, axis=None)
        .format(helpers.comma_format)
    )