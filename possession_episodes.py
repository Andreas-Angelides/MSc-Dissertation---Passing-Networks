import pandas as pd

ACTION_EVENTS = {'Pass', 'Shot', 'Free Kick', 'Others on the ball', 'Goalkeeper leaving line'}


def segment_match_into_episodes(match_events: pd.DataFrame) -> pd.DataFrame:
    match_events = match_events.sort_values(['matchPeriod', 'eventSec']).reset_index(drop=True)

    action_mask = match_events['eventName'].isin(ACTION_EVENTS)
    action_events = match_events[action_mask].copy()

    action_events['period_change'] = action_events['matchPeriod'] != action_events['matchPeriod'].shift()
    action_events['team_change'] = action_events['teamId'] != action_events['teamId'].shift()
    action_events['new_segment'] = action_events['period_change'] | action_events['team_change']
    action_events['segment_id'] = action_events['new_segment'].cumsum()

    action_segments = action_events.groupby('segment_id').agg(
        team_id=('teamId', 'first'),
        period=('matchPeriod', 'first'),
        start=('eventSec', 'first'),
        end=('eventSec', 'last'),
        n_actions=('eventId', 'count'),
        n_passes=('eventName', lambda x: (x == 'Pass').sum()),
    ).reset_index()

    match_events = match_events.copy()
    match_events['segment_id'] = action_events['segment_id'].reindex(match_events.index).ffill()
    match_events = match_events.dropna(subset=['segment_id'])

    n_events_total = match_events.groupby('segment_id').size().rename('n_events_total').reset_index()
    segments = action_segments.merge(n_events_total, on='segment_id')
    segments['duration'] = segments['end'] - segments['start']
    segments['matchId'] = match_events['matchId'].iloc[0]

    return segments


def segment_season(events_df: pd.DataFrame) -> pd.DataFrame:
    all_segments = []
    for match_id, match_events in events_df.groupby('matchId'):
        segs = segment_match_into_episodes(match_events)
        all_segments.append(segs)
    return pd.concat(all_segments, ignore_index=True)


def apply_episode_threshold(
    segments: pd.DataFrame, min_passes: int = 2, min_duration: float = 3.0
) -> pd.DataFrame:
    mask = (segments['n_passes'] >= min_passes) & (segments['duration'] >= min_duration)
    return segments[mask].reset_index(drop=True)

def merge_tolerant_episodes(segs, team_id, min_passes=2, min_duration=3.0):
    """
    Walks through a match's full chronological segment sequence (both teams) and merges the team's own segments across brief interruptions, per supervisor guidance: an interrupting (opposing-team) segment is bridged over - treated as noise, not a genuine turnover - if it would ITSELF fail the standard valid-episode bar (min_passes, min_duration). A genuine opponent possession (one that would pass that bar) remains a real episode boundary. Merging never crosses a period boundary.
    """
    merged = []
    current = None

    def flush():
        nonlocal current
        if current is not None:
            merged.append(current)
        current = None

    for _, seg in segs.iterrows():
        if seg['team_id'] == team_id:
            if current is None:
                current = {
                    'matchId': seg['matchId'], 'period': seg['period'],
                    'start': seg['start'], 'end': seg['end'],
                    'n_passes': seg['n_passes'], 'n_actions': seg['n_actions'],
                    'n_sub_segments': 1,
                }
            elif current['period'] == seg['period']:
                current['end'] = seg['end']
                current['n_passes'] += seg['n_passes']
                current['n_actions'] += seg['n_actions']
                current['n_sub_segments'] += 1
            else:
                flush()
                current = {
                    'matchId': seg['matchId'], 'period': seg['period'],
                    'start': seg['start'], 'end': seg['end'],
                    'n_passes': seg['n_passes'], 'n_actions': seg['n_actions'],
                    'n_sub_segments': 1,
                }
        else:
            is_genuine_interruption = (seg['n_passes'] >= min_passes) and (seg['duration'] >= min_duration)
            crosses_period = current is not None and current['period'] != seg['period']
            if is_genuine_interruption or crosses_period:
                flush()
    flush()

    result = pd.DataFrame(merged)
    if len(result):
        result['duration'] = result['end'] - result['start']
    return result

def build_merged_episodes_season(events_df, team_id, min_passes=2, min_duration=3.0):
    """Apply merge_tolerant_episodes() across every match, returning the full season's set of valid (thresholded) merged episodes."""
    all_merged = []
    for match_id, match_events in events_df.groupby('matchId'):
        segs = segment_match_into_episodes(match_events)
        merged = merge_tolerant_episodes(segs, team_id, min_passes, min_duration)
        if len(merged):
            all_merged.append(merged)
    all_merged = pd.concat(all_merged, ignore_index=True)
    valid = all_merged[(all_merged['n_passes'] >= min_passes) & (all_merged['duration'] >= min_duration)].reset_index(drop=True)
    return valid