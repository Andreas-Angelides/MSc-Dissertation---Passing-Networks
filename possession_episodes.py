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