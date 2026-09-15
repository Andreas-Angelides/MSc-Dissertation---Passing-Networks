import numpy as np
import pandas as pd
import networkx as nx

#Here we name all the events that might take action

ACTION_EVENTS = {'Pass', 'Shot', 'Free Kick', 'Others on the ball', 'Goalkeeper leaving line'}

#Here we fixed the set of 11 role nodes, used identically for every episode's network, 
#so all networks are directly comparable regardless of which roles happened 
#to touch the ball in a given episode.

ROLE_NODES = [
    'Goalkeeper', 'Defence-Left', 'Defence-CentreLeft', 'Defence-CentreRight', 'Defence-Right', 'Midfield-Left', 'Midfield-Centre', 'Midfield-Right', 'Attack-Left', 'Attack-Centre', 'Attack-Right'
]

#The following function isolates for a certain episode's slice of action events, then starts looking for Pass events, for a single episode. Then for each pass, the next action event is treated as the receiver and both players are identified by their roles via role_map. For consistency reasons, every network always has all 11 nodes present, even some roles have recorded zero passes.

def build_episode_network(episode, match_action_events, role_map):
    """
    Builds a directed, weighted role-based passing network for one possession episode. The receiver of each pass if inferred as the next action event by the same team within the episode. Passes with no following event, or where the same player performs both the pass and the next action are excluded.
    """
    ep_events = match_action_events[
    (match_action_events['matchPeriod'] == episode['period']) & 
    (match_action_events['eventSec'] >= episode['start']) & 
    (match_action_events['eventSec'] <= episode['end'])
    ].reset_index(drop=True)

    G = nx.DiGraph()
    G.add_nodes_from(ROLE_NODES)

    for i, row in ep_events.iterrows():
        if row['eventName'] != 'Pass' or i + 1 >= len(ep_events):
            continue
        next_row = ep_events.iloc[i + 1]
        if row['playerId'] == next_row['playerId']:
            continue
        passer_role = role_map.get(row['playerId'])
        receiver_role = role_map.get(ep_events.iloc[i + 1]['playerId'])
        if passer_role is None or receiver_role is None:
            continue
        if G.has_edge(passer_role, receiver_role):
            G[passer_role][receiver_role]['weight'] += 1
        else:
            G.add_edge(passer_role, receiver_role, weight=1)
    return G

#The following function converts the graph into a 11x11 matrix, using the ROLE_NODES ordering, then flattens it into a single-length vector. 

def network_to_vector(G):
    """Flatten a role-based network into a fixed-length weighted adjacency vector."""
    A = nx.to_numpy_array(G, nodelist=ROLE_NODES, weight='weight')
    return A.flatten()

#The following function loops through matches, then builds a network for every valid episode using the segment-specific role mapping. Returns a dictionary of the actual graph objects (useful for visualisations), the flattened feature matrix, and the list of episode IDs.

def build_season_networks(events_df, valid_episodes_df, role_assignments_df, team_id):
    """
    Builds one network per  valid possession episode across the season, plus a matching feature matrix for use in embedding/clustering.
    """
    networks = {}
    vectors = []
    episode_ids = []

    for match_id, match_episodes in valid_episodes_df.groupby('matchId'):
        match_action_events = events_df[
        (events_df['matchId'] == match_id) & (events_df['teamId'] == team_id) & 
        (events_df['eventName'].isin(ACTION_EVENTS))
        ].sort_values(['matchPeriod', 'eventSec']).reset_index(drop=True)
        match_roles = role_assignments_df[role_assignments_df['matchId'] == match_id]
        for _, ep in match_episodes.iterrows():
            seg_roles = match_roles[match_roles['segment_id'] == ep['segment_id']]
            role_map = dict(zip(seg_roles['playerId'], seg_roles['role']))
            G = build_episode_network(ep, match_action_events, role_map)
            networks[ep['episode_id']] = G
            vectors.append(network_to_vector(G))
            episode_ids.append(ep['episode_id'])

    vector_matrix = np.array(vectors)
    return networks, vector_matrix, episode_ids

DEPTH = {
    'Goalkeeper': 0,
    'Defence-Left':1, 'Defence-CentreLeft': 1, 'Defence-CentreRight': 1, 'Defence-Right': 1, 'Midfield-Left': 2, 'Midfield-Centre': 2, 'Midfield-Right': 2, 'Attack-Left': 3, 'Attack-Centre': 3, 'Attack-Right': 3,
}
DEPTH_ARR = np.array([DEPTH[r] for r in ROLE_NODES])

def network_to_summary_features(G):
    """
    Converts a role-based network into a compact set of interpretable summary features, rather than the full 121-dim raw adjacency vector: out-degree and in-degree share per role (how much each role contributes to/receives from ball circulation), overall verticality (average forward progression per pass, weighted by depth), density (proportion of possible role-pairs actually used), and goalkeeper involvement share.
    """
    A = nx.to_numpy_array(G, nodelist=ROLE_NODES, weight='weight')
    n = len(ROLE_NODES)
    total = A.sum()
    if total == 0:
        return np.zeros(2 * n + 3)

    out_deg = A.sum(axis=1) / total
    in_deg = A.sum(axis=0) / total

    i_idx, j_idx =np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    depth_change = DEPTH_ARR[j_idx] - DEPTH_ARR[i_idx]
    verticality = (A * depth_change).sum() / total

    density = (A>0).sum() / (n * n)
    gk_share = (A[0, :].sum() + A[:, 0].sum()) / total

    return np.concatenate([out_deg, in_deg, [verticality, density, gk_share]])


def episode_pass_quality(episode, match_passes):
    """
    Computes two pass-quality features for one episode: overall pass accuracy, and the proportion of passes that are "progressive" (advance the ball by at least 25% of the remaining distance to the opponent's goal - a standard definition in football analytics).
    """
    ep_passes = match_passes[
    (match_passes['matchPeriod'] == episode['period']) &
    (match_passes['eventSec'] >= episode['start']) & 
    (match_passes['eventSec'] <= episode['end'])
    ]
    if len(ep_passes) == 0:
        return 0.0, 0.0

    def has_tag(tags, tag_id):
        return any(t.get('id') == tag_id for t in tags) if isinstance(tags, list) else False

    def is_progressive(pos):
        if not isinstance(pos, list) or len(pos) < 2:
            return False
        start_x, end_x = pos[0]['x'], pos[1]['x']
        dist_start = 100 - start_x
        if dist_start <= 0:
            return False
        return (100 - end_x) <= 0.75 * dist_start

    accuracy = ep_passes['tags'].apply(lambda t: has_tag(t, 1801)).mean()
    progressive_pct = ep_passes['positions'].apply(is_progressive).mean()
    return accuracy, progressive_pct


def episode_tactical_features(episode, match_passes, structural_features):
    """
    Additional tactical features suggested by supervisor feedback:
    the proportion of passes originating in the middle and attacking thirds of the pitch (defensive third is impied by the other two and ommited to avoid redundancy), the total number of passes in the episode, and the number of distinct roles involved as either a passer or receiver.
    """
    ep_passes = match_passes[
    (match_passes['matchPeriod'] == episode['period']) &
    (match_passes['eventSec'] >= episode['start']) &
    (match_passes['eventSec'] <= episode['end'])
    ]
    n_passes = len(ep_passes)
    if n_passes == 0:
        return {'pct_mid_third': 0, 'pct_att_third': 0, 'n_passes_total': 0, 'n_roles_involved': 0}

    def start_x(pos):
        if isinstance(pos, list) and len(pos) > 0:
            return pos[0]['x']
        return None

    xs = ep_passes['positions'].apply(start_x).dropna()
    pct_mid = ((xs >= 33.33) & (xs < 66.67)).mean()
    pct_att = (xs >= 66.67).mean()

    out_in = np.array(structural_features[:22]).reshape(2, 11)
    n_roles_involved = int(((out_in[0] + out_in[1]) > 0).sum())

    return {'pct_mid_third': pct_mid, 'pct_att_third': pct_att,
    'n_passes_total': n_passes, 'n_roles_involved': n_roles_involved}