"""Generic Checkmarble scenario inspector.

Lists every scenario on the platform with its live version, rule count,
and per-rule detail. Useful for verifying a jurisdiction pack was
applied correctly, or for exporting the current rule state for audit.
"""

import json
import os
import sys
import urllib.request
import urllib.error

FIREBASE_URL = os.environ.get("FIREBASE_URL", "http://localhost:9099")
API_URL = os.environ.get("CHECKMARBLE_API_URL", "http://localhost:8080")
EMAIL = os.environ.get("CREATE_ORG_ADMIN_EMAIL", "admin@example.com")
PASSWORD = os.environ.get("ADMIN_PASSWORD")
if not PASSWORD:
    sys.exit("ERROR: ADMIN_PASSWORD env var is required.")

def post_json(url, data=None, headers=None):
    body = json.dumps(data).encode("utf-8") if data else b"{}"
    req = urllib.request.Request(url, data=body, headers=headers or {}, method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def get_json(url, headers=None):
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

# Step 1: Firebase Auth
print("=" * 80)
print("STEP 1: Authenticating with Firebase Auth emulator...")
firebase_signin_url = (
    f"{FIREBASE_URL}/identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key=fake-api-key"
)
firebase_resp = post_json(firebase_signin_url, {
    "email": EMAIL,
    "password": PASSWORD,
    "returnSecureToken": True
})
id_token = firebase_resp.get("idToken")
print(f"  Firebase ID token obtained.")

# Step 2: Checkmarble JWT
print("STEP 2: Exchanging for Checkmarble JWT...")
token_resp = post_json(
    f"{API_URL}/token", data=None,
    headers={"Authorization": f"Bearer {id_token}"}
)
access_token = token_resp.get("access_token")
print(f"  Access token obtained.")

auth_headers = {"Authorization": f"Bearer {access_token}"}

# Step 3: List all scenarios
print("STEP 3: Fetching all scenarios...")
scenarios_resp = get_json(f"{API_URL}/scenarios", auth_headers)

# The /scenarios endpoint may return a list or dict with a key
if isinstance(scenarios_resp, list):
    scenarios = scenarios_resp
elif isinstance(scenarios_resp, dict):
    # Try common wrapper keys
    for key in ("scenarios", "data", "items"):
        if key in scenarios_resp:
            scenarios = scenarios_resp[key]
            break
    else:
        scenarios = scenarios_resp
else:
    scenarios = []

print(f"  Found {len(scenarios)} scenarios.\n")

# Step 4: For each scenario, get iteration details with embedded rules
print("=" * 80)
print(f"{'FULL SCENARIO REPORT':^80}")
print("=" * 80)

total_rules = 0
live_count = 0
not_live_count = 0
results = []

for i, scenario in enumerate(scenarios, 1):
    scenario_id = scenario.get("id")
    scenario_name = scenario.get("name", "UNKNOWN")

    # Get scenario detail
    try:
        detail = get_json(f"{API_URL}/scenarios/{scenario_id}", auth_headers)
    except Exception as e:
        detail = scenario

    live_version_id = detail.get("live_version_id") or detail.get("liveVersionId")
    is_live = bool(live_version_id)

    if is_live:
        live_count += 1
    else:
        not_live_count += 1

    # Get rules from scenario iteration body
    rules = []
    if live_version_id:
        try:
            iteration = get_json(
                f"{API_URL}/scenario-iterations/{live_version_id}",
                auth_headers
            )
            body = iteration.get("body", {})
            if body:
                rules = body.get("rules", [])
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            rules = []
            print(f"  ERROR for {scenario_name}: HTTP {e.code}")
        except Exception as e:
            rules = []
            print(f"  ERROR for {scenario_name}: {e}")

    rule_count = len(rules)
    total_rules += rule_count

    results.append({
        "index": i,
        "name": scenario_name,
        "id": scenario_id,
        "is_live": is_live,
        "live_version_id": live_version_id,
        "rule_count": rule_count,
        "rules": rules
    })

# Print results
for r in results:
    print(f"\n{'_' * 80}")
    print(f"  Scenario {r['index']:2d}: {r['name']}")
    print(f"  ID:            {r['id']}")
    print(f"  Live:          {'YES' if r['is_live'] else 'NO'}")
    if r['is_live']:
        print(f"  Live Version:  {r['live_version_id']}")
    print(f"  Rule Count:    {r['rule_count']}")

    if r['rules']:
        print(f"  Rules:")
        for j, rule in enumerate(r['rules'], 1):
            rule_name = rule.get("name") or rule.get("description") or "UNNAMED"
            rule_id = rule.get("id", "?")
            score_mod = rule.get("score_modifier")
            score_str = f", score_modifier: {score_mod}" if score_mod is not None else ""
            print(f"    {j:3d}. {rule_name}")
            print(f"         id: {rule_id}{score_str}")
    elif r['is_live']:
        print(f"  Rules:         (none)")
    else:
        print(f"  Rules:         (not live - skipped)")

# Summary
print(f"\n{'=' * 80}")
print(f"{'SUMMARY':^80}")
print(f"{'=' * 80}")
print(f"  Total scenarios:      {len(scenarios)}")
print(f"  Live scenarios:       {live_count}")
print(f"  Not live scenarios:   {not_live_count}")
print(f"  Total rules (all):    {total_rules}")
print()
print(f"  {'Scenario Name':<45} {'Live':<6} {'Rules':<6}")
print(f"  {'-'*45} {'-'*5} {'-'*5}")
for r in results:
    live_str = "YES" if r["is_live"] else "NO"
    print(f"  {r['name']:<45} {live_str:<6} {r['rule_count']:<6}")
print(f"{'=' * 80}")
