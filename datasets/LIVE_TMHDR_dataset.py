import glob
import os

import pandas as pd
from videolib import standards

dataset_name = 'LIVE_TMHDR'
dis_standard = standards.sRGB
width = 1920
height = 1080

df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'live_tmhdr_scores.csv'))
df.sort_values(by=['Content', 'TMO', 'Spatial', 'Temporal', 'CRF'], inplace=True, ascending=True)
df.reset_index(drop=True, inplace=True)

base_dir = '[path to dataset]'

framerates_dict = {
    'football3': 50,
    'golf2': 50,
}

ref_videos = {}
content_dict = {}
pq_files = sorted(glob.glob(os.path.join(base_dir, 'PQ', 'Ref', '*.mp4')))
hlg_files = sorted(glob.glob(os.path.join(base_dir, 'HLG', 'Ref', '*.mp4')))

for ind, file in enumerate(pq_files):
    content_id = ind
    vid_dict = {}
    vid_dict['path'] = file
    vid_dict['content_name'] = os.path.splitext(os.path.basename(vid_dict['path']))[0]
    vid_dict['standard'] = standards.rec_2100_pq
    ref_videos[content_id] = vid_dict
    content_dict[vid_dict['content_name']] = ind

for ind, file in enumerate(hlg_files):
    content_id = ind + len(pq_files)
    vid_dict = {}
    vid_dict['path'] = file
    vid_dict['content_name'] = os.path.splitext(os.path.basename(vid_dict['path']))[0]
    vid_dict['standard'] = standards.rec_2100_hlg
    ref_videos[content_id] = vid_dict
    content_dict[vid_dict['content_name']] = content_id

dis_videos = {}
for ind, row in df.iterrows():
    vid_dict = {}
    if row['TMO'] == 'ExpertTM':
        vid_dict['path'] = os.path.join(base_dir, row['Encoding'], f'CRF{row["CRF"]}', row['TMO'], row['Content'] + '.mp4')
    elif row['TMO'] in ['DolbyVisionTMO', 'ColorTransformTMO']:
        vid_dict['path'] = os.path.join(base_dir, row['Encoding'], f'CRF{row["CRF"]}', row['TMO'], row['Temporal'], row['Content'] + '.mp4')
    else:
        vid_dict['path'] = os.path.join(base_dir, row['Encoding'], f'CRF{row["CRF"]}', row['TMO'], row['Spatial'], row['Temporal'], row['Content'] + '.mp4')
    vid_dict['content_id'] = content_dict[row['Content']]
    vid_dict['fps'] = 30 if row['Encoding'] == 'HLG' else framerates_dict.get(ref_videos[vid_dict['content_id']]['content_name'], 60)
    vid_dict['score'] = row['SurealMOS']
    dis_videos[ind] = vid_dict
