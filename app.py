"""Streamlit dashboard for ReachMap hand and object observations."""
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

from analysis import (heatmap, interaction_log, load_session, object_summary,
                      overlap_durations, session_insights, summarize)

ROOT = Path(__file__).resolve().parent
SESSIONS = ROOT / 'data' / 'sessions'

st.set_page_config(page_title='ReachMap AI', layout='wide')
st.title('🧠 ReachMap AI — Smart Workspace Movement Analyzer')
st.write('Overhead-camera hand tracking, object detections, and possible interactions.')
st.info('Record a session with `python capture.py --camera 1`, then refresh this page.')

files = sorted(SESSIONS.glob('session_*.csv'), reverse=True) if SESSIONS.exists() else []
if not files:
    st.warning('No sessions found. Record a session using capture.py first.')
    st.stop()

selected = st.selectbox('Choose session', files, format_func=lambda path: path.name)
data = load_session(selected)
if data.empty:
    st.warning('This session contains no recorded frames.')
    st.stop()

zones = summarize(data)
objects = object_summary(data)
events = interaction_log(data)
hand_samples = int(data['zone'].notna().sum())

first, second, third, fourth = st.columns(4)
first.metric('Hand samples', f'{hand_samples:,}')
second.metric('Confirmed zone visits', int(zones['visits'].sum()))
third.metric('Object classes detected', len(objects))
fourth.metric('Possible interaction events', len(events))

st.subheader('Session insights')
for insight in session_insights(zones, objects, events):
    if insight.startswith('Far zone:'):
        st.error(insight)
    elif insight.startswith('Next action:'):
        st.warning(insight)
    else:
        st.info(insight)

st.subheader('Hand movement')
left, right = st.columns(2)
with left:
    st.write('**Hand-position observation density**')
    if hand_samples:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(heatmap(data), origin='upper', extent=[0, 1, 1, 0], cmap='YlOrRd')
        ax.axhline(1 / 3, color='black', linestyle='--', alpha=0.6)
        ax.axhline(2 / 3, color='black', linestyle='--', alpha=0.6)
        ax.set_xlabel('Horizontal desk position')
        ax.set_ylabel('Far edge → Near edge')
        ax.set_title('Hand Position Density Across Workspace')
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info('No hand positions were recorded within the desk.')
with right:
    st.write('**Confirmed visits by reach zone**')
    st.bar_chart(zones.set_index('zone')['visits'])
    st.write('**Zone summary**')
    st.dataframe(zones, hide_index=True, use_container_width=True)

st.subheader('Detected workspace objects')
if 'objects' not in data.columns:
    st.info('This is an older hand-only session. Record a new session to see YOLO data.')
elif objects.empty:
    st.info('YOLO did not detect any recognized objects inside the selected desk.')
else:
    st.caption('Counts refer to YOLO object classes, not individually tracked physical items.')

    # Show useful interaction measures rather than object detection frame counts.
    activity = objects[['object_class', 'possible_interactions']].copy()
    if 'overlap_objects' in data.columns:
        durations = overlap_durations(data)
        activity = activity.merge(durations, on='object_class', how='left')
        activity['overlap_seconds'] = activity['overlap_seconds'].fillna(0)
        activity = activity.sort_values('overlap_seconds', ascending=False)
        st.caption('Overlap duration is approximate fingertip-to-box overlap, not physical contact.')
    else:
        st.caption('Record a new session to include overlap durations.')
    st.dataframe(activity, hide_index=True, use_container_width=True)

    top_objects = objects.nlargest(3, 'possible_interactions')
    top_objects = top_objects[top_objects['possible_interactions'] > 0]

    if not top_objects.empty:
        left, center, right = st.columns([1, 2, 1])

        with center:
            st.subheader(f'**⬆️ Top {len(top_objects)} objects by possible interactions**')
            st.bar_chart(
                top_objects.set_index('object_class')['possible_interactions'],
                height=280
            )
    else:
        st.info('No possible interactions recorded.')

    st.write('---')

    st.write('**Possible interaction event log**')
    if events.empty:
        st.info('No events confirmed: an index fingertip must overlap a box long enough.')
    else:
        st.dataframe(events, hide_index=True, use_container_width=True)

st.caption('A detection frame is not a visit. A possible interaction is sustained index-fingertip '
           'overlap with a detected bounding box, not confirmed physical touch. '
           'Observed durations are approximate; no ergonomic risk is inferred.')
st.download_button('Download session CSV', selected.read_bytes(),
                   file_name=selected.name, mime='text/csv')
