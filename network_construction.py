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
    Builds a directed, weighted role-based passing network for one possession episode. The receiver of each pass if inferred as the next action event by the same team within the episode; a pass with no following action event (i.e. the last event in the episode) is excluded, since it has no valid same-team receiver.
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
    