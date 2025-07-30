import pytest
import requests


def test_health_endpoint():
    """Test the health check endpoint"""
    url = "http://0.0.0.0:9910/health"
    resp = requests.get(url)

    assert resp.status_code == 200
    resp_json = resp.json()
    assert resp_json["status"] == "healthy"
    assert "model_loaded" in resp_json


def test_quarc_batch_prediction():
    """Test batch prediction with multiple SMILES"""
    url = "http://0.0.0.0:9910/condition_prediction"

    data = {
        "smiles": [
            "[CH3:1][O:2][C:3](=[O:4])[c:5]1[cH:6][cH:7][c:8]2[nH:9][c:10]([c:11]([c:12]2[cH:13]1)[CH2:14][c:15]1[cH:16][cH:17][c:18]([cH:19][c:20]1[Cl:21])I)[CH3:22].[CH3:23][C:24]([CH3:25])([CH3:26])[SH:27]>c1ccc(cc1)[P](c1ccccc1)(c1ccccc1)[Pd]([P](c1ccccc1)(c1ccccc1)c1ccccc1)([P](c1ccccc1)(c1ccccc1)c1ccccc1)[P](c1ccccc1)(c1ccccc1)c1ccccc1.CCCCN(CCCC)CCCC.CN(C)C=O>[CH3:1][O:2][C:3](=[O:4])[c:5]1[cH:6][cH:7][c:8]2[nH:9][c:10]([c:11]([c:12]2[cH:13]1)[CH2:14][c:15]1[cH:16][cH:17][c:18]([cH:19][c:20]1[Cl:21])[S:27][C:24]([CH3:23])([CH3:25])[CH3:26])[CH3:22]",
            "Cl.[O:1]=[C:2]1[CH2:3][CH2:4][CH:5]([C:6]([NH:7]1)=[O:8])[N:9]1[CH2:10][c:11]2[c:12]([cH:13][cH:14][cH:15][c:16]2[O:17][CH2:18][c:19]2[cH:20][cH:21][cH:22][c:23]([cH:24]2)[CH2:25]Br)[C:26]1=[O:27].[F:28][c:29]1[cH:30][cH:31][c:32]([cH:33][cH:34]1)[CH:35]1[CH2:36][CH2:37][NH:38][CH2:39][CH2:40]1>CC(C)N(CC)C(C)C.CC#N>[O:1]=[C:2]1[CH2:3][CH2:4][CH:5]([C:6]([NH:7]1)=[O:8])[N:9]1[CH2:10][c:11]2[c:16]([cH:15][cH:14][cH:13][c:12]2[C:26]1=[O:27])[O:17][CH2:18][c:19]1[cH:20][cH:21][cH:22][c:23]([cH:24]1)[CH2:25][N:38]1[CH2:37][CH2:36][CH:35]([CH2:40][CH2:39]1)[c:32]1[cH:31][cH:30][c:29]([cH:34][cH:33]1)[F:28]",  # Simple oxidation
        ],
        "top_k": 3,
    }

    resp = requests.post(url, json=data)

    assert resp.status_code == 200
    resp_json = resp.json()
    assert resp_json["status"] == "SUCCESS"

    results = resp_json["results"]
    assert len(results) == 2  # Two results for two input SMILES


if __name__ == "__main__":
    pytest.main([__file__])
