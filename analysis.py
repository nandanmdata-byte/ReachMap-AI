"""Simple movement and object statistics from a ReachMap session."""
import numpy as np
import pandas as pd

ZONES = ['Near', 'Middle', 'Far']


def load_session(path):
    data = pd.read_csv(path)
    required = {'seconds', 'x', 'y', 'zone', 'visit'}
    if not required.issubset(data.columns):
        raise ValueError('Not a valid ReachMap session CSV.')
    return data.sort_values('seconds').reset_index(drop=True)


def summarize(data):
    visits = data.loc[data['visit'] == 1, 'zone'].value_counts()

    # Count time only when the current and next frames show the same zone.
    elapsed = data['seconds'].shift(-1) - data['seconds']
    same_zone = data['zone'].notna() & (data['zone'] == data['zone'].shift(-1))
    elapsed = elapsed.where(same_zone, 0).clip(lower=0, upper=0.25).fillna(0)
    durations = elapsed.groupby(data['zone']).sum()

    summary = pd.DataFrame({'zone': ZONES})
    summary['visits'] = summary['zone'].map(visits).fillna(0).astype(int)
    summary['observed_seconds'] = summary['zone'].map(durations).fillna(0).round(1)
    return summary


def heatmap(data, bins=25):
    hand = data.dropna(subset=['x', 'y'])
    counts, _, _ = np.histogram2d(hand['y'], hand['x'], bins=bins,
                                   range=[[0, 1], [0, 1]])
    return counts


def object_summary(data):
    """Count detection frames and confirmed overlap events per object CLASS."""
    columns = ['object_class', 'detected_frames', 'possible_interactions']
    if 'objects' not in data.columns:
        return pd.DataFrame(columns=columns)

    seen = data['objects'].fillna('').str.split('|').explode()
    seen = seen[seen != '']
    if 'interaction_events' in data.columns:
        events = data['interaction_events'].fillna('').str.split('|').explode()
        events = events[events != '']
    else:
        events = pd.Series(dtype='object')

    names = sorted(set(seen) | set(events))
    summary = pd.DataFrame({'object_class': names})
    summary['detected_frames'] = summary['object_class'].map(seen.value_counts()).fillna(0).astype(int)
    summary['possible_interactions'] = summary['object_class'].map(events.value_counts()).fillna(0).astype(int)
    return summary.sort_values('detected_frames', ascending=False).reset_index(drop=True)


def interaction_log(data):
    """One row per confirmed possible-interaction event."""
    columns = ['time_seconds', 'object_class', 'palm_zone']
    if 'interaction_events' not in data.columns:
        return pd.DataFrame(columns=columns)

    rows = data.loc[data['interaction_events'].notna(),
                    ['seconds', 'zone', 'interaction_events']].copy()
    rows['object_class'] = rows['interaction_events'].str.split('|')
    rows = rows.explode('object_class')
    rows = rows[rows['object_class'] != '']
    return rows.rename(columns={'seconds': 'time_seconds', 'zone': 'palm_zone'})[
        columns].reset_index(drop=True)


def overlap_durations(data):
    """Estimate observed fingertip overlap time by object class."""
    if 'overlap_objects' not in data.columns:
        return pd.DataFrame(columns=['object_class', 'overlap_seconds'])

    elapsed = data['seconds'].diff().fillna(0)
    overlaps = data['overlap_objects'].fillna('').str.split('|')

    totals = {}

    for i in range(1, len(data)):
        seconds = elapsed.iloc[i]

        # Ignore long gaps between frames.
        if seconds <= 0 or seconds > 0.25:
            continue

        # Count time only when an object overlaps in both frames.
        common = set(overlaps.iloc[i - 1]) & set(overlaps.iloc[i])

        for name in common:
            if name:
                totals[name] = totals.get(name, 0) + seconds

    summary = pd.DataFrame(
        [(name, round(seconds, 1)) for name, seconds in totals.items()],
        columns=['object_class', 'overlap_seconds']
    )

    return summary.sort_values('overlap_seconds', ascending=False)

def session_insights(summary, objects=None, events=None):
    total_visits = int(summary['visits'].sum())
    total_seconds = float(summary['observed_seconds'].sum())
    far_visits = int(summary.loc[summary['zone'] == 'Far', 'visits'].iloc[0])

    insights = []
    if total_visits:
        most_visits = summary['visits'].max()
        main_zones = summary.loc[summary['visits'] == most_visits, 'zone']
        insights.append(f'Most frequent zone entries: {", ".join(main_zones)} ({most_visits} visits).')
        insights.append(f'Far zone: {far_visits} of {total_visits} visits '
                        f'({100 * far_visits / total_visits:.0f}%).')
        if far_visits:
            insights.append('Next action: Identify what task required far-zone reaches. '
                            'A later test could move frequently needed items closer.')
    else:
        insights.append('No confirmed zone visits in this session.')

    if objects is not None and not objects.empty:
        total_events = int(objects['possible_interactions'].sum())
        if total_events:
            top = objects.sort_values('possible_interactions', ascending=False).iloc[0]
            insights.append(f'Most possible interactions: {top.object_class} '
                            f'({int(top.possible_interactions)} events). '
                            'Check the footage: box overlap does not prove touching.')
        else:
            insights.append('Objects were detected, but no possible-interaction events were confirmed.')

    if events is not None and not events.empty:
        far_events = events.loc[events['palm_zone'] == 'Far', 'object_class']
        if not far_events.empty:
            names = ', '.join(sorted(far_events.unique()))
            insights.append(f'Palm in Far zone during {len(far_events)} possible interaction '
                            f'events involving {names}. Review the footage before moving items.')

    if total_seconds < 60 or total_visits < 15:
        insights.append('Limited evidence: Repeat a representative task before changing the workspace.')
    insights.append('These measurements are image-based observations, not ergonomic risk or productivity scores.')
    return insights
