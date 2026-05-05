import pandas as pd

def highlight_non_nan_gray(val):
    if val != ' ':
        return 'background-color: #b0b0b0'  # slightly darker gray
    return ''

def highlight_and_bold_actuals_row_gray(row):
    # Bold non-nan values in the 'actuals' row
    if row.name == 'actuals':
        return ['font-weight: bold; background-color: #8d8d8d' if pd.notna(v) else '' for v in row]  # use a slightly darker gray for this row
    else:
        # Highlight non-nan everywhere else with gray
        return [highlight_non_nan_gray(v) for v in row]

def comma_format(val):
    # Only try formatting if val is a number (either int or float not as string)
    if val != ' ':
        return f"{int(val):,}"
    else:
        return val

def highlight_week_x(df, week_x):
    # Returns DataFrame of same shape as df with red background color in the week_x column
    return pd.DataFrame(
        [
            ['background-color: red' if c[0] == week_x else '' for c in df.columns]
            for _ in range(len(df))
        ],
        index=df.index,
        columns=df.columns
    )