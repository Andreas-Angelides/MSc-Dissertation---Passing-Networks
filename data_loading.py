from pathlib import Path
import json
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent


def _load_json(filename: str) -> list:
    path = RAW_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Is {filename} in the same folder as this script?"
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_events(competition: str = "England") -> pd.DataFrame:
    data = _load_json(f"events_{competition}.json")
    return pd.json_normalize(data)


def load_matches(competition: str = "England") -> pd.DataFrame:
    data = _load_json(f"matches_{competition}.json")
    return pd.json_normalize(data)


def load_players() -> pd.DataFrame:
    data = _load_json("players.json")
    return pd.json_normalize(data)


def load_teams() -> pd.DataFrame:
    data = _load_json("teams.json")
    return pd.json_normalize(data)


def get_team_id(teams_df: pd.DataFrame, team_name: str) -> int:
    matches = teams_df[
        teams_df["name"].str.contains(team_name, case=False, na=False)
        | teams_df["officialName"].str.contains(team_name, case=False, na=False)
    ]
    if len(matches) == 0:
        raise ValueError(f"No team found matching '{team_name}'")
    if len(matches) > 1:
        raise ValueError(
            f"Multiple teams match '{team_name}':\n{matches[['wyId', 'name', 'officialName']]}"
        )
    return int(matches.iloc[0]["wyId"])


def filter_matches_for_team(matches_df: pd.DataFrame, team_id: int) -> pd.DataFrame:
    team_id_str = str(team_id)
    team_cols = [c for c in matches_df.columns if c.startswith(f"teamsData.{team_id_str}")]
    if not team_cols:
        raise ValueError(
            f"No columns found for team_id={team_id} in matches_df. "
            "Check that matches_df was loaded with pd.json_normalize and that "
            "team_id is correct."
        )
    mask = matches_df[team_cols[0]].notna()
    return matches_df[mask].reset_index(drop=True)


def filter_events_for_team(events_df: pd.DataFrame, team_id: int, match_ids=None) -> pd.DataFrame:
    mask = events_df["teamId"] == team_id
    if match_ids is not None:
        mask &= events_df["matchId"].isin(match_ids)
    return events_df[mask].reset_index(drop=True)


def sanity_check_matches(matches_df: pd.DataFrame, expected_n_matches: int = 38) -> None:
    n = len(matches_df)
    print(f"Matches found: {n} (expected: {expected_n_matches})")
    if n != expected_n_matches:
        print(
            "WARNING: match count does not match expectation. "
            "Check competition/team filtering before proceeding."
        )

def load_raw_matches(competition: str = "England") -> list:
    """Load matches as raw nested JSON (needed for formation/substitution lookups)."""
    return _load_json(f"matches_{competition}.json")