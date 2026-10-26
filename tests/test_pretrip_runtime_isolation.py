"""Real existing session/API regression, not a claim to test the separate reminder repo."""

import os
import subprocess
import sys
import unittest

from fastapi.testclient import TestClient

from happy_path.config import DB_PATH
from happy_path.core.graph import CorridorRegistry
from happy_path.main import app
from happy_path.pretrip import safely_evaluate


class TestPreTripRuntimeIsolation(unittest.TestCase):
    def test_existing_session_destination_and_api_survive_failed_enrichment(self):
        with TestClient(app) as client:
            response = client.post(
                "/v1/itineraries/plan",
                json={
                    "origin_query": "HCMC",
                    "destination_query": "Vung Tau",
                    "departure_iso": "06:00",
                },
            )
            self.assertEqual(response.status_code, 200, response.text)
            original = response.json()
            sid = original["session_id"]
            before = app.state.registry.load_session(sid)

            def failed_source():
                with open("/this-enrichment-source-does-not-exist.parquet", "rb"):
                    pass

            self.assertEqual(safely_evaluate(failed_source)["status"], "unavailable")
            restarted = CorridorRegistry(DB_PATH)
            self.assertEqual(restarted.load_session(sid), before)
            self.assertEqual(before.corridor_id, "hcm_vungtau")
            self.assertEqual(app.state.registry.load_session(sid), before)
            changed = client.post(
                "/v1/itineraries/mutate",
                json={
                    "session_id": sid,
                    "candidate_id": original["candidates"][0]["candidate_id"],
                    "action": "SHIFT_DEPARTURE",
                    "new_departure_iso": "07:00",
                },
            )
            self.assertEqual(changed.status_code, 200, changed.text)
            self.assertEqual(
                app.state.registry.load_session(sid).corridor_id, before.corridor_id
            )
            self.assertEqual(client.get("/v1/corridors/health").status_code, 200)

    def test_runtime_functions_when_entire_pretrip_package_unavailable(self):
        program = """
import importlib.abc,sys
class BlockOptional(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.startswith('happy_path.pretrip'):
   raise ImportError('optional enrichment deliberately unavailable')
sys.meta_path.insert(0,BlockOptional())
from happy_path.main import app
from fastapi.testclient import TestClient
with TestClient(app) as client:
 assert client.get('/v1/corridors/health').status_code==200
 response=client.post('/v1/itineraries/plan',json={'origin_query':'HCMC','destination_query':'Da Lat','departure_iso':'06:00'})
 assert response.status_code==200,response.text
 session=app.state.registry.load_session(response.json()['session_id'])
 assert session.corridor_id=='hcm_dalat'
 assert not any(k.startswith('happy_path.pretrip') for k in sys.modules)
"""
        result = subprocess.run(
            [sys.executable, "-B", "-c", program],
            capture_output=True,
            text=True,
            timeout=30,
            env=dict(os.environ, GOONG_API_KEY="", GOONG_REST_KEY=""),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
