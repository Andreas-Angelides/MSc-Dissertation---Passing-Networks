import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

#For every match, this module splits play into time segments at each
#substitution. Each segment's roster of 10 outfield players is built
#directly from the authoritative lineup/substitution log (not inferred
#from which players happen to have events in that time window - this
#was found necessary to avoid a rare but real bug where a substitute's
#event, logged a few seconds before their officially recorded entry time,
#could silently displace a legitimate starter). Missing position data for
#a roster player in a short segment is filled in from the nearest other
#segment where they do have data. Each segment's 10 outfield players are
#then matched to 10 fixed reference role slots using the Hungarian
#algorithm for optimal one-to-one assignment (guaranteeing no two players
#share a role), while the goalkeeper is identified directly from the
#player reference table rather than by position.

REFERENCE_TEMPLATE = {
    'Defence-Left': (0.10, 0.10),
    'Defence-CentreLeft': (0.05, 0.40),
    'Defence-CentreRight': (0.05, 0.60),
    'Defence-Right': (0.10, 0.90),
    'Midfield-Left': (0.45, 0.15),
    'Midfield-Centre': (0.45, 0.50),
    'Midfield-Right': (0.45, 0.85),
    'Attack-Left': (0.85, 0.15),
    'Attack-Centre': (0.85, 0.50),
    'Attack-Right': (0.85, 0.85),
}
TEMPLATE_NAMES = list(REFERENCE_TEMPLATE.keys())
TEMPLATE_COORDS = np.array(list(REFERENCE_TEMPLATE.values()))

def get_segment_rosters(raw_match, team_id):
    """
    Returns a list of (seg_start, seg_end, roster) tuples, where roster is the definitive SET of playerIds on the pitch during that segment, built directly from the starting lineup and substitution log - not inferred from which players happen to have events in that time window.
    """
    team_data = raw_match['teamsData'][str(team_id)]
    formation = team_data.get('formation', {})
    lineup_ids = {p['playerId'] for p in formation.get('lineup', [])}
    subs = sorted(formation.get('substitutions', []), key=lambda s: s['minute'])

    boundaries = [0] + [s['minute'] * 60 for s in subs] + [float('inf')]
    current_roster = set(lineup_ids)

    segments = []
    for i in range(len(boundaries) - 1):
        segments.append((boundaries[i], boundaries[i + 1], set(current_roster)))
        if i < len(subs):
            current_roster.discard(subs[i]['playerOut'])
            current_roster.add(subs[i]['playerIn'])
    return segments


def compute_avg_positions(segment_events, team_id, roster):
    """
    Average (x, y) position per player in this segment, using only events from players who are actually in the authoritative roster for this segment. Roster players with zero events here are simply absent from the result and get filled in later.
    """
    ev = segment_events[
    (segment_events['teamId'] == team_id) & (segment_events['playerId'].isin(roster))
    ].copy()
    ev['start_x'] = ev['positions'].apply(lambda p: p[0]['x'] if isinstance(p, list) and len(p) > 0 else None)
    ev['start_y'] = ev['positions'].apply(lambda p: p[0]['y'] if isinstance(p, list) and len(p) > 0 else None)
    ev = ev.dropna(subset=['start_x', 'start_y'])
    return ev.groupby('playerId')[['start_x', 'start_y']].mean()
    
def assign_roles_for_match(match_events, raw_match, players_df, team_id):
    """
    Assigns roles per segment using the authoritative roster, filling in
    any roster player missing position data (e.g. very short segments
    where they had no events) by carrying forward their position from the
    nearest other segment in the same match where data exists.
    """
    match_events = match_events.copy()
    match_events['match_seconds'] = np.where(
        match_events['matchPeriod'] == '1H', match_events['eventSec'], 2700 + match_events['eventSec']
    )
    segments = get_segment_rosters(raw_match, team_id)
    gk_ids = set(players_df[players_df['role.name'] == 'Goalkeeper']['wyId'])

    # Pass 1: raw average positions per segment (some roster players may be missing)
    segment_positions = []
    for seg_id, (seg_start, seg_end, roster) in enumerate(segments):
        seg_events = match_events[
            (match_events['match_seconds'] >= seg_start) & (match_events['match_seconds'] < seg_end)
        ]
        avg_pos = compute_avg_positions(seg_events, team_id, roster)
        segment_positions.append({
            'seg_id': seg_id, 'seg_start': seg_start, 'seg_end': seg_end,
            'roster': roster, 'avg_pos': avg_pos,
        })

    # Pass 2: fill gaps by carrying forward/backward from the nearest segment
    # in this match where the missing player does have data
    for i, seg in enumerate(segment_positions):
        missing = seg['roster'] - set(seg['avg_pos'].index)
        for pid in missing:
            fallback = None
            for j in list(range(i - 1, -1, -1)) + list(range(i + 1, len(segment_positions))):
                if pid in segment_positions[j]['avg_pos'].index:
                    fallback = segment_positions[j]['avg_pos'].loc[pid]
                    break
            if fallback is not None:
                seg['avg_pos'].loc[pid] = fallback

    # Pass 3: assign roles per segment via Hungarian matching
    all_roles = []
    for seg in segment_positions:
        avg_pos = seg['avg_pos'].reset_index()
        outfield = avg_pos[~avg_pos['playerId'].isin(gk_ids)].copy()
        gk_rows = avg_pos[avg_pos['playerId'].isin(gk_ids)].copy()
        gk_rows['role'] = 'Goalkeeper'

        if len(outfield) == 0:
            roles = gk_rows
        else:
            x_min, x_max = outfield['start_x'].min(), outfield['start_x'].max()
            y_min, y_max = outfield['start_y'].min(), outfield['start_y'].max()
            outfield['norm_x'] = (outfield['start_x'] - x_min) / (x_max - x_min + 1e-9)
            outfield['norm_y'] = (outfield['start_y'] - y_min) / (y_max - y_min + 1e-9)
            coords = outfield[['norm_x', 'norm_y']].to_numpy()
            cost = np.linalg.norm(coords[:, None, :] - TEMPLATE_COORDS[None, :, :], axis=2)
            row_ind, col_ind = linear_sum_assignment(cost)
            outfield = outfield.iloc[row_ind].copy()
            outfield['role'] = [TEMPLATE_NAMES[c] for c in col_ind]
            roles = pd.concat([gk_rows, outfield], ignore_index=True)

        roles['segment_id'] = seg['seg_id']
        roles['segment_start'] = seg['seg_start']
        roles['segment_end'] = seg['seg_end']
        all_roles.append(roles[['playerId', 'role', 'start_x', 'start_y', 'segment_id', 'segment_start', 'segment_end']])

    result = pd.concat(all_roles, ignore_index=True)
    result['matchId'] = match_events['matchId'].iloc[0]
    return result

def assign_roles_for_season(events_df, raw_matches, players_df, team_id):
    """Apply assign_roles_for_match() across every match the team played in."""
    matches_by_id = {m['wyId']: m for m in raw_matches}
    all_results = []
    for match_id, match_events in events_df.groupby('matchId'):
        if not (match_events['teamId'] == team_id).any():
            continue
        raw_match = matches_by_id[match_id]
        roles = assign_roles_for_match(match_events, raw_match, players_df, team_id)
        if not roles.empty:
            all_results.append(roles)
    return pd.concat(all_results, ignore_index=True)