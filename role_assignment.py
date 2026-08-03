import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

#What this does, for every match, it splits play into time segments at each substitution, and within each segment, matches the 10 outfield players' average positions to 10 fixed reference role slots using optimal 1-to-1 assignment (so that no two players share the same role). The goalkeeper is handled separately.

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

def assign_roles_via_template(segment_events, players_df, team_id):
    """
    Assign each outfield player to one of 10 fixed role slots using optimal (Hungarian algorithm) bipartite matching between each player's average       position and the reference template, guaranteeing a unique role per player. The goalkeeper is identified directly via players_df, not by position.
    """
    ev = segment_events[(segment_events['teamId'] == team_id) & (segment_events['playerId'] != 0)].copy()
    ev['start_x'] = ev['positions'].apply(lambda p: p[0]['x'] if isinstance(p, list) and len(p) > 0 else None)
    ev['start_y'] = ev['positions'].apply(lambda p: p[0]['y'] if isinstance(p, list) and len(p) > 0 else None)
    ev = ev.dropna(subset=['start_x', 'start_y'])
    if ev.empty:
        return pd.DataFrame(columns=['playerId', 'role', 'start_x', 'start_y'])

    avg_pos = ev.groupby('playerId')[['start_x', 'start_y']].mean().reset_index()
    avg_pos = avg_pos.merge(players_df[['wyId', 'role.name']], left_on='playerId', right_on='wyId', how='left')

    gk_mask = avg_pos['role.name'] == 'Goalkeeper'
    goalkeepers = avg_pos[gk_mask].copy()
    goalkeepers['role'] = 'Goalkeeper'

    outfield = avg_pos[~gk_mask].copy()
    if len(outfield) < 2:
        outfield['role'] = 'Unclassified'
        return pd.concat([goalkeepers, outfield], ignore_index=True)[['playerId', 'role', 'start_x', 'start_y']]

    #Now we normalize the positions of each player for this segment

    x_min, x_max = outfield['start_x'].min(), outfield['start_x'].max()
    y_min, y_max = outfield['start_y'].min(), outfield['start_y'].max()
    outfield['norm_x'] = (outfield['start_x'] - x_min) / (x_max - x_min + 1e-9)
    outfield['norm_y'] = (outfield['start_y'] - y_min) / (y_max - y_min + 1e-9)

    player_coords = outfield[['norm_x', 'norm_y']].to_numpy()
    cost = np.linalg.norm(player_coords[:, None, :] - TEMPLATE_COORDS[None, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(cost)

    outfield = outfield.iloc[row_ind].copy()
    outfield['role'] = [TEMPLATE_NAMES[c] for c in col_ind]

    result = pd.concat([goalkeepers, outfield], ignore_index=True)
    return result[['playerId', 'role', 'start_x', 'start_y']]


def get_time_boundaries(raw_match, team_id):
    """Return match-second boundaries (0, sub1, sub2, ..., inf) marking each personnel-change segment for team_id, based on substitution minutes."""
    team_data = raw_match['teamsData'][str(team_id)]
    subs = team_data.get('formation', {}).get('substitutions', [])
    minutes = sorted(s['minute'] for s in subs)
    return [0] + [m * 60 for m in minutes] + [float('inf')]

def assign_roles_for_match(match_events, raw_match, players_df, team_id):
    """
    Splits one match into time segments bounded by substitutions, and assigns template-based roles separately within each segment (so only the players actually on the pitch at that time compete for the 10 roles slots).
    """
    match_events = match_events.copy()
    match_events['match_seconds'] = np.where(
        match_events['matchPeriod'] == '1H', match_events['eventSec'], 2700 + match_events['eventSec']
    )
    boundaries = get_time_boundaries(raw_match, team_id)

    all_roles = []
    for seg_id, (seg_start, seg_end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        seg_events = match_events[
        (match_events['match_seconds'] >= seg_start) & (match_events['match_seconds'] < seg_end)
        ]
        if seg_events.empty:
            continue
        roles = assign_roles_via_template(seg_events, players_df, team_id)
        if roles.empty:
            continue
        roles['segment_id'] = seg_id
        roles['segment_start'] = seg_start
        roles['segment_end'] = seg_end
        all_roles.append(roles)
    if not all_roles:
        return pd.DataFrame()
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