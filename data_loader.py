import pandas as pd
import os

def load_and_prepare_data(filepath):
    df_raw = pd.read_excel(filepath, sheet_name="LL Identified", skiprows=1)
    columns = {
        'NA.2': 'Lesson Title',
        'Description of Impact / Context / What happened?': 'Lesson Description',
        'Why did it happen?': 'Root Cause',
        'Key takeaways and recommendations for future projects.': 'Lesson Learned',
        'NA.13': 'Recommended Action',
        'NA.15': 'Action Remarks'
    }
    df = df_raw[list(columns.keys())].rename(columns=columns).dropna(subset=['Lesson Description'])

    def combine_fields(row):
        return f"Title: {row['Lesson Title']}\nContext: {row['Lesson Description']}\nCause: {row['Root Cause']}\nLesson: {row['Lesson Learned']}\nAction: {row['Recommended Action']}\nRemarks: {row['Action Remarks']}"

    return df.apply(combine_fields, axis=1).tolist()
